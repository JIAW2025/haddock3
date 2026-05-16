#!/usr/bin/env python3
"""
CaSR 受体 - 酪蛋白水解多肽 HADDOCK3 虚拟对接脚本。

这个脚本面向“会使用命令行，但不熟悉编程”的用户，所以代码里包含较多中文注释。
它做四件事：

1. 读取 CaSR 受体结构和多肽候选序列；
2. 如果没有提供多肽 PDB，则用 PeptideBuilder 自动生成多肽初始构象：
   - loop
   - alpha helix
   - beta sheet / extended sheet-like
3. 为每条多肽自动创建 HADDOCK3 docking workflow：
   - topoaa      : 准备拓扑和参数
   - rigidbody   : 刚体对接，大量采样
   - seletop     : 选择刚体阶段最好的模型
   - flexref     : 半柔性精修，多肽设为 fully flexible
   - emref/mdref : 最终精修
   - clustfcc    : 按接触模式聚类
   - seletopclusts : 每个 cluster 选最佳模型
   - caprieval   : 输出最终评价表
4. 可选直接运行 HADDOCK3，并自动汇总最终 HADDOCK score/ranking。

推荐 CSV 输入格式：

    id,sequence,pdb
    casein_pep_001,RELEEL,
    casein_pep_002,FFVAPFPEVFGK,data/peptides/casein_pep_002.pdb

说明：
- `pdb` 列可以留空；留空时脚本会用 PeptideBuilder 生成 loop/alpha/sheet 构象。
- 如果你已经有更可靠的多肽 PDB，也可以填入 `pdb` 路径，脚本会直接使用。

运行前安装依赖：

    pip install PeptideBuilder biopython

示例：

    python scripts/casr_peptide_docking.py \
        --receptor data/CaSR.pdb \
        --peptides peptides.csv \
        --pocket-residues 170,171,172,173,300,301 \
        --activation-residues 170,173,301 \
        --outdir casr_casein_screen

如果要真正运行 HADDOCK3，加上：

    --run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =============================================================================
# 1. 官方蛋白-多肽 docking 推荐参数集中写在这里
# =============================================================================
# 这些默认值主要参考 HADDOCK3 官方 examples/docking-protein-peptide/*-full.cfg。
# 你以后想改参数，优先改这里；不用去读下面复杂函数。

# 默认链 ID：CaSR 受体用 A，多肽用 B。你的 PDB 如果不是这个链号，可以命令行覆盖。
DEFAULT_RECEPTOR_CHAIN = "A"
DEFAULT_PEPTIDE_CHAIN = "B"

# HADDOCK3 执行方式：local 适合本机；batch/mpi 适合集群环境。
DEFAULT_MODE = "local"
DEFAULT_NCORES = 8

# rigidbody：刚体对接阶段生成模型数。官方蛋白-多肽 full 示例是 3000。
DEFAULT_RIGIDBODY_SAMPLING = 3000

# seletop：从 rigidbody 阶段选多少模型进入 flexref。官方蛋白-多肽 full 示例是 400。
DEFAULT_SELECT_TOP = 400

# tolerance：允许多少比例的模型失败。官方 full 示例中常用 5。
DEFAULT_TOLERANCE = 5

# flexref：蛋白-多肽官方示例中增加了精修步数，让柔性多肽有更多调整空间。
DEFAULT_MDSTEPS_RIGID = 2000
DEFAULT_MDSTEPS_COOL1 = 2000
DEFAULT_MDSTEPS_COOL2 = 4000
DEFAULT_MDSTEPS_COOL3 = 4000

# 多肽二级结构二面角约束。官方蛋白-多肽示例使用 alphabeta。
DEFAULT_SSDIHED = "alphabeta"

# 最终 refinement：官方普通蛋白-多肽 full 示例使用 emref；mdref 更慢但更充分。
DEFAULT_USE_MDREF = False

# 每个 cluster 最后保留几个模型。官方示例常用 4。
DEFAULT_TOP_MODELS_PER_CLUSTER = 4

# clustfcc 最小 cluster 数量。筛选任务中设为 1 可以避免小规模测试时无 cluster 报错。
DEFAULT_MIN_CLUSTER_POPULATION = 1

# 激活关键残基在 AIR 约束中重复几次。重复越多，越偏向让多肽靠近这些关键残基。
# 这是一个启发式参数，不是 HADDOCK 官方 scoring 权重。
DEFAULT_ACTIVATION_WEIGHT = 2

# PeptideBuilder 默认生成的多肽构象分布。
# 这里表示每条多肽生成 1 个 loop、1 个 alpha helix、1 个 beta sheet-like 构象。
DEFAULT_PEPTIDE_CONFORMATION_DISTRIBUTION = {
    "loop": 1,
    "alpha": 1,
    "sheet": 1,
}

# 三类多肽构象的主链二面角近似值。
# PeptideBuilder 根据这些 phi/psi 角生成理想化初始构象。
# 注意：这只是 docking 起点，不代表真实最终构象；后续 flexref 会继续优化。
PEPTIDE_BACKBONE_ANGLES = {
    "loop": (-75.0, 145.0),      # 松散 loop / extended-like 起点
    "alpha": (-57.0, -47.0),    # alpha helix 常见 phi/psi
    "sheet": (-135.0, 135.0),   # beta sheet 常见 phi/psi
}


# =============================================================================
# 2. 简单数据结构：每条多肽的信息放在这里
# =============================================================================

@dataclass(frozen=True)
class PeptideEntry:
    """一条多肽候选。

    peptide_id: 多肽名字，例如 casein_pep_001
    sequence:   氨基酸序列，例如 RELEEL
    pdb:        如果用户提供了 PDB，就保存路径；如果没有，则为 None
    """

    peptide_id: str
    sequence: str
    pdb: Path | None


# =============================================================================
# 3. 工具函数：解析输入文件和残基列表
# =============================================================================

def parse_residue_list(text: str) -> list[int]:
    """解析残基编号列表。

    支持两种写法：
    - 单个编号：170,171,300
    - 范围：300-305

    例子：
    "170,171,300-302" -> [170, 171, 300, 301, 302]
    """
    if not text:
        return []

    residues: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid residue range: {item}")
            residues.extend(range(start, end + 1))
        else:
            residues.append(int(item))

    # 去重并排序，避免用户重复写残基编号。
    return sorted(set(residues))


def read_peptides_csv(csv_file: Path) -> list[PeptideEntry]:
    """读取多肽 CSV 文件。

    CSV 至少需要两列：
    - id
    - sequence

    第三列 pdb 可选：
    - 如果有 pdb 路径，脚本直接使用该 PDB；
    - 如果 pdb 为空，脚本会用 PeptideBuilder 自动生成多构象 PDB。
    """
    peptides: list[PeptideEntry] = []
    with csv_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "sequence"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            peptide_id = row["id"].strip()
            sequence = row["sequence"].strip().upper()
            pdb_text = row.get("pdb", "").strip()
            pdb = Path(pdb_text) if pdb_text else None

            if not peptide_id:
                raise ValueError("Peptide id cannot be empty")
            if not sequence:
                raise ValueError(f"Peptide {peptide_id} has empty sequence")
            if pdb is not None and not pdb.exists():
                raise FileNotFoundError(f"Peptide PDB not found for {peptide_id}: {pdb}")

            peptides.append(PeptideEntry(peptide_id, sequence, pdb))

    return peptides


def infer_peptide_residue_numbers(sequence: str) -> list[int]:
    """根据序列长度推断多肽 residue 编号。

    这里假设多肽 PDB residue 从 1 开始连续编号：1..N。
    PeptideBuilder 生成的 PDB 正好符合这个假设。
    如果你自己提供的 PDB 编号不是 1..N，需要手动检查生成的 ambig.tbl。
    """
    return list(range(1, len(sequence) + 1))


# =============================================================================
# 4. PeptideBuilder：自动生成 loop / alpha / sheet 多肽构象
# =============================================================================

def build_single_peptide_model(sequence: str, conformation: str, chain_id: str):
    """用 PeptideBuilder 生成一个单独的多肽构象。

    函数调用逻辑：
    1. 根据 conformation 选择 phi/psi 二面角；
    2. 用 PeptideBuilder.initialize_res() 创建第一个残基；
    3. 用 PeptideBuilder.add_residue() 逐个添加后续残基；
    4. 把链 ID 改成用户指定的 peptide chain，例如 B。
    """
    try:
        import PeptideBuilder
        from PeptideBuilder import Geometry
    except ImportError as exc:
        raise ImportError(
            "PeptideBuilder is required to generate peptide conformations. "
            "Install it with: pip install PeptideBuilder biopython"
        ) from exc

    if conformation not in PEPTIDE_BACKBONE_ANGLES:
        raise ValueError(f"Unknown peptide conformation: {conformation}")

    phi, psi = PEPTIDE_BACKBONE_ANGLES[conformation]

    # 创建第一个氨基酸。
    first_geo = Geometry.geometry(sequence[0])
    structure = PeptideBuilder.initialize_res(first_geo)

    # 从第二个氨基酸开始逐个添加。
    for aa in sequence[1:]:
        geo = Geometry.geometry(aa)
        # PeptideBuilder 中：
        # - phi 是当前残基 phi 角
        # - psi_im1 是前一个残基 psi 角
        geo.phi = phi
        geo.psi_im1 = psi
        PeptideBuilder.add_residue(structure, geo)

    # PeptideBuilder 默认链 ID 未必是 B；这里统一改成 peptide_chain。
    for model in structure:
        for chain in model:
            chain.id = chain_id

    return structure


def structure_to_pdb_atom_lines(structure) -> list[str]:
    """把 Bio.PDB structure 转成 PDB 文本行，只保留 ATOM/TER。

    这个函数用于把多个构象合并成一个 multi-model PDB。
    HADDOCK3 的 topoaa 会自动拆分 multi-model PDB ensemble。
    """
    from io import StringIO

    from Bio.PDB import PDBIO

    buffer = StringIO()
    io = PDBIO()
    io.set_structure(structure)
    io.save(buffer)

    lines = []
    for line in buffer.getvalue().splitlines():
        if line.startswith(("ATOM", "HETATM", "TER")):
            lines.append(line)
    return lines


def generate_peptide_ensemble_pdb(
    sequence: str,
    output_pdb: Path,
    chain_id: str,
    conformation_distribution: dict[str, int],
) -> None:
    """生成多肽 multi-model PDB。

    文件创建逻辑：
    - MODEL 1: loop
    - MODEL 2: alpha
    - MODEL 3: sheet
    如果某个构象数量设为 0，则不生成。
    """
    model_index = 1
    output_lines: list[str] = []

    for conformation, count in conformation_distribution.items():
        for copy_index in range(count):
            structure = build_single_peptide_model(sequence, conformation, chain_id)
            atom_lines = structure_to_pdb_atom_lines(structure)

            output_lines.append(f"MODEL     {model_index:4d}")
            output_lines.append(
                f"REMARK peptide_conformation {conformation} copy {copy_index + 1}"
            )
            output_lines.extend(atom_lines)
            output_lines.append("ENDMDL")
            model_index += 1

    if model_index == 1:
        raise ValueError("No peptide conformations were generated")

    output_lines.append("END")
    output_pdb.write_text("\n".join(output_lines) + "\n")


# =============================================================================
# 5. 创建 HADDOCK AIR 约束文件 ambig.tbl
# =============================================================================

def write_air_restraints(
    output_file: Path,
    receptor_chain: str,
    peptide_chain: str,
    pocket_residues: Iterable[int],
    activation_residues: Iterable[int],
    peptide_residues: Iterable[int],
    activation_weight: int = DEFAULT_ACTIVATION_WEIGHT,
    distance: float = 2.0,
    lower_bound: float = 2.0,
    upper_bound: float = 0.0,
) -> None:
    """写 HADDOCK ambiguous interaction restraints，即 ambig.tbl。

    生物学逻辑：
    - CaSR 口袋残基：作为 docking 引导信息；
    - CaSR 激活关键残基：重复写入几次，让 docking 更偏向这些关键位置；
    - 多肽所有残基：作为 passive partner，因为短肽通常整条都可能接触蛋白。

    HADDOCK/CNS 约束格式大致是：

        assign (蛋白某残基)
               (多肽残基1 or 多肽残基2 or ...)
               2.0 2.0 0.0

    它不是指定唯一接触对，而是说：蛋白这个残基应靠近多肽候选残基集合中的某些残基。
    """
    pocket = list(pocket_residues)
    activation = list(activation_residues)
    peptide = list(peptide_residues)

    if not pocket and not activation:
        raise ValueError("At least one pocket or activation residue must be provided")
    if not peptide:
        raise ValueError("Peptide residue list is empty")

    receptor_restraints: list[int] = []
    receptor_restraints.extend(pocket)

    # 激活关键残基重复写入，用于增强其影响。
    for _ in range(max(1, activation_weight)):
        receptor_restraints.extend(activation)

    lines: list[str] = [
        "! HADDOCK AIR restraints generated for CaSR-peptide docking",
        "! 中文说明：这些约束把 CaSR 口袋/激活残基引导到多肽附近",
        "!",
        f"! Receptor chain/segid: {receptor_chain}",
        f"! Peptide chain/segid: {peptide_chain}",
        f"! Pocket residues: {','.join(map(str, pocket))}",
        f"! Activation-important residues: {','.join(map(str, activation))}",
        "!",
    ]

    for receptor_resid in receptor_restraints:
        lines.append(f"assign ( resid {receptor_resid} and segid {receptor_chain} )")
        lines.append("       (")
        for idx, peptide_resid in enumerate(peptide):
            prefix = "        " if idx == 0 else "     or "
            lines.append(f"{prefix}( resid {peptide_resid} and segid {peptide_chain} )")
        lines.append(f"       ) {distance:.1f} {lower_bound:.1f} {upper_bound:.1f}")
        lines.append("!")

    output_file.write_text("\n".join(lines) + "\n")


# =============================================================================
# 6. 创建 HADDOCK3 workflow 配置文件 docking.cfg
# =============================================================================

def write_haddock_cfg(
    output_file: Path,
    receptor_pdb: str,
    peptide_pdb: str,
    peptide_length: int,
    peptide_chain: str,
    sampling: int,
    select_top: int,
    ncores: int,
    mode: str,
    use_mdref: bool,
    tolerance: int,
    mdsteps_rigid: int,
    mdsteps_cool1: int,
    mdsteps_cool2: int,
    mdsteps_cool3: int,
    ssdihed: str,
    top_models_per_cluster: int,
    min_cluster_population: int,
    receptor_hisd: list[int] | None = None,
    receptor_hise: list[int] | None = None,
) -> None:
    """写 HADDOCK3 配置文件 docking.cfg。

    这个文件就是 HADDOCK3 实际运行的 workflow。
    下面配置块中每个 [模块名] 都是 HADDOCK3 的一个步骤。
    """
    receptor_hisd = receptor_hisd or []
    receptor_hise = receptor_hise or []

    # topoaa.mol1 是第一个 molecule，也就是 receptor。
    # 这里允许用户指定 HISD/HISE，避免 CaSR 质子化状态完全自动判断。
    histidine_lines = [
        "[topoaa]",
        "# topoaa：准备 CNS 拓扑/参数；补氢、补缺失原子，生成 *_haddock.pdb 和 *.psf。",
        "autohis = false",
        "",
        "[topoaa.mol1]",
        "# topoaa.mol1：第 1 个分子，也就是 receptor/CaSR 的专属设置。",
        f"nhisd = {len(receptor_hisd)}",
    ]
    for idx, resid in enumerate(receptor_hisd, start=1):
        histidine_lines.append(f"hisd_{idx} = {resid}")
    histidine_lines.append(f"nhise = {len(receptor_hise)}")
    for idx, resid in enumerate(receptor_hise, start=1):
        histidine_lines.append(f"hise_{idx} = {resid}")

    refinement_module = "mdref" if use_mdref else "emref"
    refinement_comment = (
        "mdref：短分子动力学精修，更慢但 relaxation 更充分。"
        if use_mdref
        else "emref：能量最小化精修，官方蛋白-多肽普通 full 示例默认使用。"
    )

    cfg = f'''# ====================================================================
# CaSR receptor - peptide docking workflow
# 自动生成文件：不要手动改太多；建议改脚本顶部默认参数后重新生成。

run_dir = "haddock_run"

mode = "{mode}"
ncores = {ncores}
debug = true

# molecules 定义输入分子：
# 第 1 个是 CaSR receptor，第 2 个是 peptide。顺序很重要。
molecules = [
    "{receptor_pdb}",
    "{peptide_pdb}"
]

# ====================================================================
{chr(10).join(histidine_lines)}

# ====================================================================
[rigidbody]
# rigidbody：刚体对接，类似 HADDOCK2 的 it0。
# 做什么：把 receptor 和 peptide 当作刚体，随机旋转/平移，大量生成初始结合姿势。
# 由 ambig.tbl 中的 CaSR 口袋/激活残基约束把 peptide 拉向已知口袋。
tolerance = {tolerance}
ambig_fname = "ambig.tbl"
sampling = {sampling}

# ====================================================================
[seletop]
# seletop：从刚体对接模型中选择 HADDOCK score 最好的前 N 个进入 flexref。
select = {select_top}

# ====================================================================
[flexref]
# flexref：半柔性 simulated annealing，类似 HADDOCK2 的 it1。
# 做什么：让界面和多肽发生柔性调整，改善 clash、侧链 packing 和局部构象。
# 对蛋白-多肽尤其重要，因为多肽通常很柔。
tolerance = {tolerance}
ambig_fname = "ambig.tbl"

# 下面三行把整条多肽设为 fully flexible。
# 如果 peptide chain 是 B，长度是 11，就代表 B:1-11 全部可柔性调整。
fle_sta_1 = 1
fle_end_1 = {peptide_length}
fle_seg_1 = "{peptide_chain}"

# ssdihed：自动给 alpha/beta-like 主链区域加二面角约束，避免多肽精修时构象完全崩坏。
ssdihed = "{ssdihed}"

# flexref 的模拟退火步数。蛋白-多肽官方 full 示例使用 2000/2000/4000/4000。
mdsteps_rigid = {mdsteps_rigid}
mdsteps_cool1 = {mdsteps_cool1}
mdsteps_cool2 = {mdsteps_cool2}
mdsteps_cool3 = {mdsteps_cool3}

# ====================================================================
[{refinement_module}]
# {refinement_comment}
# 做什么：对 flexref 后的模型再做最终 relaxation，并重新计算 HADDOCK score。
tolerance = {tolerance}
ambig_fname = "ambig.tbl"

# 最终精修阶段仍然让整条 peptide flexible。
fle_sta_1 = 1
fle_end_1 = {peptide_length}
fle_seg_1 = "{peptide_chain}"

ssdihed = "{ssdihed}"

# ====================================================================
[clustfcc]
# clustfcc：按 Fraction of Common Contacts 聚类。
# 做什么：不是只看最低分单个模型，而是看哪些结合模式反复出现。
# 对 peptide docking 很重要，因为可信结果通常是一个稳定 cluster，而不是孤立低分模型。
min_population = {min_cluster_population}

# ====================================================================
[seletopclusts]
# seletopclusts：每个 cluster 选择前几个模型，减少最终结果数量。
top_models = {top_models_per_cluster}

# ====================================================================
[caprieval]
# caprieval：最终分析表。
# 如果没有 reference complex，它仍可输出模型 score/cluster 信息；
# 如果以后你有实验复合物结构，可添加：reference_fname = "reference.pdb"
allatoms = true

# ====================================================================
'''
    output_file.write_text(cfg)


# =============================================================================
# 7. 运行 HADDOCK3 和汇总最终 score/ranking
# =============================================================================

def run_haddock(workdir: Path, cfg_name: str = "docking.cfg") -> None:
    """在某个多肽目录中运行 HADDOCK3。"""
    # 等价于在命令行进入 workdir 后执行：haddock3 docking.cfg
    subprocess.run(["haddock3", cfg_name], cwd=workdir, check=True)


def find_latest_table(haddock_run: Path, table_name: str) -> Path | None:
    """寻找编号最大的 HADDOCK3 step 里的指定表格。

    HADDOCK3 每一步目录类似：0_topoaa、1_rigidbody、2_seletop ...
    最终 caprieval 通常是编号最大的 caprieval 目录。
    """
    candidates = sorted(haddock_run.glob(f"*_*/{table_name}"))
    if not candidates:
        return None

    def step_number(path: Path) -> int:
        try:
            return int(path.parent.name.split("_", 1)[0])
        except ValueError:
            return -1

    return max(candidates, key=step_number)


def read_best_score_from_tsv(tsv_file: Path) -> dict[str, str] | None:
    """从 TSV 表格读取 score 最低的一行。

    HADDOCK score 通常越低越好，所以这里按 score 升序排序。
    不同模块 TSV 列名可能略有差异，因此这里做了兼容处理。
    """
    with tsv_file.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    if not rows:
        return None

    # 优先找 score 列；如果没有，就尝试找包含 score 的列。
    fieldnames = rows[0].keys()
    score_col = "score" if "score" in fieldnames else None
    if score_col is None:
        for col in fieldnames:
            if "score" in col.lower():
                score_col = col
                break
    if score_col is None:
        return None

    def score_value(row: dict[str, str]) -> float:
        try:
            return float(row[score_col])
        except (TypeError, ValueError):
            return float("inf")

    best = min(rows, key=score_value)
    best["_score_col"] = score_col
    return best


def collect_final_scores(outdir: Path, output_csv: Path) -> None:
    """汇总每条多肽的最终 HADDOCK score，并生成总 ranking 表。

    输出文件：summary_best_scores.csv

    主要列：
    - rank: 总排名，1 是最好
    - peptide_id: 多肽 ID
    - best_score: 该多肽最佳模型 score
    - best_structure: 最佳模型文件名
    - source_table: score 来自哪个 HADDOCK3 输出表
    """
    summary_rows: list[dict[str, str]] = []

    for peptide_dir in sorted(outdir.glob("run_*")):
        peptide_id = peptide_dir.name.removeprefix("run_")
        haddock_run = peptide_dir / "haddock_run"
        if not haddock_run.exists():
            continue

        # 优先使用最终 caprieval 的 capri_ss.tsv。
        table = find_latest_table(haddock_run, "capri_ss.tsv")

        # 如果没有 capri_ss.tsv，就退而求其次，找最后一个普通 tsv。
        if table is None:
            all_tsv = sorted(haddock_run.glob("*_*/*.tsv"))
            table = all_tsv[-1] if all_tsv else None

        if table is None:
            summary_rows.append({
                "rank": "",
                "peptide_id": peptide_id,
                "best_score": "",
                "best_structure": "",
                "source_table": "",
                "note": "No score table found. Was --run used? Did HADDOCK finish?",
            })
            continue

        best = read_best_score_from_tsv(table)
        if best is None:
            summary_rows.append({
                "rank": "",
                "peptide_id": peptide_id,
                "best_score": "",
                "best_structure": "",
                "source_table": str(table),
                "note": "Score column not found in table",
            })
            continue

        score_col = best.pop("_score_col")
        structure = best.get("structure", best.get("model", best.get("name", "")))
        summary_rows.append({
            "rank": "",
            "peptide_id": peptide_id,
            "best_score": best.get(score_col, ""),
            "best_structure": structure,
            "source_table": str(table),
            "note": "",
        })

    # 按 best_score 从低到高排名；空值排最后。
    def sort_key(row: dict[str, str]) -> float:
        try:
            return float(row["best_score"])
        except (TypeError, ValueError):
            return float("inf")

    summary_rows.sort(key=sort_key)
    rank = 1
    for row in summary_rows:
        if row["best_score"]:
            row["rank"] = str(rank)
            rank += 1

    with output_csv.open("w", newline="") as handle:
        fieldnames = [
            "rank",
            "peptide_id",
            "best_score",
            "best_structure",
            "source_table",
            "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def collect_score_files(outdir: Path, output_csv: Path) -> None:
    """额外输出所有 score/analysis 文件位置，方便你手动检查。"""
    rows: list[dict[str, str]] = []

    for peptide_dir in sorted(outdir.glob("run_*")):
        peptide_id = peptide_dir.name.removeprefix("run_")
        haddock_run = peptide_dir / "haddock_run"
        if not haddock_run.exists():
            continue

        score_files = sorted(haddock_run.glob("*_*/capri_ss.tsv"))
        score_files.extend(sorted(haddock_run.glob("*_*/capri_clt.tsv")))
        score_files.extend(sorted(haddock_run.glob("*_*/clustfcc.tsv")))

        for score_file in score_files:
            rows.append({"peptide_id": peptide_id, "score_file": str(score_file)})

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["peptide_id", "score_file"])
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 8. 主程序：命令行参数、文件创建、函数调用总入口
# =============================================================================

def main() -> None:
    """脚本入口函数。

    执行顺序：
    1. 读取命令行参数；
    2. 读取多肽 CSV；
    3. 对每条多肽创建一个 run_xxx 文件夹；
    4. 复制 receptor.pdb；
    5. 复制或生成 peptide.pdb；
    6. 创建 ambig.tbl；
    7. 创建 docking.cfg；
    8. 如果指定 --run，则运行 haddock3；
    9. 汇总 score/ranking。
    """
    parser = argparse.ArgumentParser(
        description="Generate and optionally run HADDOCK3 CaSR-peptide docking workflows."
    )

    # 基本输入文件。
    parser.add_argument("--receptor", required=True, type=Path, help="Prepared CaSR receptor PDB file.")
    parser.add_argument("--peptides", required=True, type=Path, help="CSV file with columns: id,sequence,pdb(optional).")

    # 链 ID 和 docking 先验知识。
    parser.add_argument("--receptor-chain", default=DEFAULT_RECEPTOR_CHAIN, help="CaSR receptor chain/segid used in AIR restraints.")
    parser.add_argument("--peptide-chain", default=DEFAULT_PEPTIDE_CHAIN, help="Peptide chain/segid used in AIR restraints.")
    parser.add_argument("--pocket-residues", required=True, help="Comma-separated CaSR pocket residues, e.g. 170,171,300-305.")
    parser.add_argument("--activation-residues", default="", help="Comma-separated activation-important CaSR residues.")
    parser.add_argument("--activation-weight", default=DEFAULT_ACTIVATION_WEIGHT, type=int, help="How many times activation residues are repeated in AIR restraints.")

    # 可选 HIS 质子化设置。
    parser.add_argument("--receptor-hisd", default="", help="Comma-separated receptor HID/HISD residue numbers.")
    parser.add_argument("--receptor-hise", default="", help="Comma-separated receptor HIE/HISE residue numbers.")

    # HADDOCK 官方蛋白-多肽 workflow 关键参数；默认值已经在脚本顶部集中定义。
    parser.add_argument("--sampling", default=DEFAULT_RIGIDBODY_SAMPLING, type=int, help="Number of rigid-body docking models per peptide.")
    parser.add_argument("--select-top", default=DEFAULT_SELECT_TOP, type=int, help="Number of rigid-body models selected for flexible refinement.")
    parser.add_argument("--tolerance", default=DEFAULT_TOLERANCE, type=int, help="Allowed failure tolerance percentage.")
    parser.add_argument("--ncores", default=DEFAULT_NCORES, type=int, help="Number of CPU cores for local execution.")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["local", "batch", "mpi"], help="HADDOCK3 execution mode.")
    parser.add_argument("--mdref", action="store_true", default=DEFAULT_USE_MDREF, help="Use mdref instead of emref for final refinement.")

    # 输出目录和运行开关。
    parser.add_argument("--outdir", default=Path("casr_peptide_screen"), type=Path, help="Output directory for generated workflows.")
    parser.add_argument("--run", action="store_true", help="Actually run haddock3 for each peptide.")

    args = parser.parse_args()

    if not args.receptor.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {args.receptor}")

    # 读取输入。
    peptides = read_peptides_csv(args.peptides)
    pocket_residues = parse_residue_list(args.pocket_residues)
    activation_residues = parse_residue_list(args.activation_residues)
    receptor_hisd = parse_residue_list(args.receptor_hisd)
    receptor_hise = parse_residue_list(args.receptor_hise)

    # 创建总输出目录。
    args.outdir.mkdir(parents=True, exist_ok=True)

    for peptide in peptides:
        # 每条多肽一个独立目录，方便并行和排错。
        peptide_dir = args.outdir / f"run_{peptide.peptide_id}"
        peptide_dir.mkdir(parents=True, exist_ok=True)

        receptor_dst = peptide_dir / "receptor.pdb"
        peptide_dst = peptide_dir / "peptide.pdb"
        air_dst = peptide_dir / "ambig.tbl"
        cfg_dst = peptide_dir / "docking.cfg"

        # 复制 receptor 文件。
        shutil.copy2(args.receptor, receptor_dst)

        # 如果 CSV 提供了 peptide PDB，则复制；否则自动生成 loop/alpha/sheet ensemble。
        if peptide.pdb is not None:
            shutil.copy2(peptide.pdb, peptide_dst)
            print(f"[PEPTIDE] Using provided PDB for {peptide.peptide_id}: {peptide.pdb}")
        else:
            generate_peptide_ensemble_pdb(
                sequence=peptide.sequence,
                output_pdb=peptide_dst,
                chain_id=args.peptide_chain,
                conformation_distribution=DEFAULT_PEPTIDE_CONFORMATION_DISTRIBUTION,
            )
            print(f"[PEPTIDE] Generated loop/alpha/sheet PDB ensemble for {peptide.peptide_id}")

        # 创建 AIR 约束文件 ambig.tbl。
        peptide_residues = infer_peptide_residue_numbers(peptide.sequence)
        write_air_restraints(
            output_file=air_dst,
            receptor_chain=args.receptor_chain,
            peptide_chain=args.peptide_chain,
            pocket_residues=pocket_residues,
            activation_residues=activation_residues,
            peptide_residues=peptide_residues,
            activation_weight=args.activation_weight,
        )

        # 创建 HADDOCK3 workflow 配置文件 docking.cfg。
        write_haddock_cfg(
            output_file=cfg_dst,
            receptor_pdb="receptor.pdb",
            peptide_pdb="peptide.pdb",
            peptide_length=len(peptide.sequence),
            peptide_chain=args.peptide_chain,
            sampling=args.sampling,
            select_top=args.select_top,
            ncores=args.ncores,
            mode=args.mode,
            use_mdref=args.mdref,
            tolerance=args.tolerance,
            mdsteps_rigid=DEFAULT_MDSTEPS_RIGID,
            mdsteps_cool1=DEFAULT_MDSTEPS_COOL1,
            mdsteps_cool2=DEFAULT_MDSTEPS_COOL2,
            mdsteps_cool3=DEFAULT_MDSTEPS_COOL3,
            ssdihed=DEFAULT_SSDIHED,
            top_models_per_cluster=DEFAULT_TOP_MODELS_PER_CLUSTER,
            min_cluster_population=DEFAULT_MIN_CLUSTER_POPULATION,
            receptor_hisd=receptor_hisd,
            receptor_hise=receptor_hise,
        )

        print(f"[OK] Generated workflow for {peptide.peptide_id}: {peptide_dir}")

        # 可选：真正运行 HADDOCK3。
        if args.run:
            print(f"[RUN] haddock3 docking.cfg in {peptide_dir}")
            run_haddock(peptide_dir)

    # 汇总输出文件位置和最终 score ranking。
    score_index = args.outdir / "score_files.csv"
    summary_scores = args.outdir / "summary_best_scores.csv"
    collect_score_files(args.outdir, score_index)
    collect_final_scores(args.outdir, summary_scores)

    print(f"[DONE] Workflows generated in: {args.outdir}")
    print(f"[DONE] Score file index: {score_index}")
    print(f"[DONE] Final ranking table: {summary_scores}")


if __name__ == "__main__":
    main()
