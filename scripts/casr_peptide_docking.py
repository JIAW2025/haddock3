#!/usr/bin/env python3
"""
Batch HADDOCK3 workflow generator/runner for CaSR receptor - peptide docking.

This helper script is intended for virtual screening of casein hydrolysate
peptides against a CaSR receptor structure using knowledge-driven HADDOCK3
protein-peptide docking.

For each peptide candidate, the script creates a HADDOCK3 workflow directory
containing:

    receptor.pdb
    peptide.pdb
    ambig.tbl
    docking.cfg

The generated AIR restraints connect user-provided CaSR pocket / activation
residues to all residues of the peptide. The peptide is then treated as fully
flexible during flexible refinement.

Example peptides.csv:

    id,sequence,pdb
    casein_pep_001,RELEEL,data/peptides/casein_pep_001.pdb
    casein_pep_002,FFVAPFPEVFGK,data/peptides/casein_pep_002.pdb

Example usage:

    python scripts/casr_peptide_docking.py \
        --receptor data/CaSR.pdb \
        --receptor-chain A \
        --peptide-chain B \
        --peptides peptides.csv \
        --pocket-residues 170,171,172,173,300,301 \
        --activation-residues 170,173,301 \
        --activation-weight 3 \
        --sampling 3000 \
        --select-top 400 \
        --ncores 32 \
        --outdir casr_casein_screen \
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


@dataclass(frozen=True)
class PeptideEntry:
    """One peptide candidate."""

    peptide_id: str
    sequence: str
    pdb: Path


def parse_residue_list(text: str) -> list[int]:
    """Parse comma-separated residue numbers and simple ranges.

    Examples
    --------
    "170,171,300-305" -> [170, 171, 300, 301, 302, 303, 304, 305]
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

    return sorted(set(residues))


def read_peptides_csv(csv_file: Path) -> list[PeptideEntry]:
    """Read peptide candidates from CSV file."""
    peptides: list[PeptideEntry] = []
    with csv_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "sequence", "pdb"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            peptide_id = row["id"].strip()
            sequence = row["sequence"].strip().upper()
            pdb = Path(row["pdb"].strip())

            if not peptide_id:
                raise ValueError("Peptide id cannot be empty")
            if not sequence:
                raise ValueError(f"Peptide {peptide_id} has empty sequence")
            if not pdb.exists():
                raise FileNotFoundError(f"Peptide PDB not found for {peptide_id}: {pdb}")

            peptides.append(PeptideEntry(peptide_id, sequence, pdb))

    return peptides


def infer_peptide_residue_numbers(sequence: str) -> list[int]:
    """Infer peptide residue numbering from sequence length.

    This assumes the peptide PDB is numbered 1..N. If a peptide PDB uses
    different residue numbering, edit the generated ambig.tbl or extend this
    script with PDB residue parsing.
    """
    return list(range(1, len(sequence) + 1))


def write_air_restraints(
    output_file: Path,
    receptor_chain: str,
    peptide_chain: str,
    pocket_residues: Iterable[int],
    activation_residues: Iterable[int],
    peptide_residues: Iterable[int],
    activation_weight: int = 2,
    distance: float = 2.0,
    lower_bound: float = 2.0,
    upper_bound: float = 0.0,
) -> None:
    """Write HADDOCK ambiguous interaction restraints.

    The normal pocket residues are written once. Activation-important residues
    are repeated `activation_weight` times to bias the AIR term toward contacts
    involving those residues.
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
    for _ in range(max(1, activation_weight)):
        receptor_restraints.extend(activation)

    lines: list[str] = [
        "! HADDOCK AIR restraints generated for CaSR-peptide docking",
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
    receptor_hisd: list[int] | None = None,
    receptor_hise: list[int] | None = None,
) -> None:
    """Write a HADDOCK3 protein-peptide docking workflow."""
    receptor_hisd = receptor_hisd or []
    receptor_hise = receptor_hise or []

    histidine_lines = [
        "[topoaa]",
        "autohis = false",
        "",
        "[topoaa.mol1]",
        f"nhisd = {len(receptor_hisd)}",
    ]
    for idx, resid in enumerate(receptor_hisd, start=1):
        histidine_lines.append(f"hisd_{idx} = {resid}")
    histidine_lines.append(f"nhise = {len(receptor_hise)}")
    for idx, resid in enumerate(receptor_hise, start=1):
        histidine_lines.append(f"hise_{idx} = {resid}")

    refinement_module = "mdref" if use_mdref else "emref"

    cfg = f'''# ====================================================================
# CaSR receptor - peptide docking workflow
# Generated by scripts/casr_peptide_docking.py

run_dir = "haddock_run"

mode = "{mode}"
ncores = {ncores}
debug = true

molecules = [
    "{receptor_pdb}",
    "{peptide_pdb}"
]

# ====================================================================
{chr(10).join(histidine_lines)}

# ====================================================================
# Rigid-body docking / HADDOCK it0-like sampling.
[rigidbody]
tolerance = 20
ambig_fname = "ambig.tbl"
sampling = {sampling}

# ====================================================================
# Select top rigid-body models for flexible refinement.
[seletop]
select = {select_top}

# ====================================================================
# Semi-flexible refinement / HADDOCK it1-like refinement.
# The entire peptide is treated as fully flexible.
[flexref]
tolerance = 20
ambig_fname = "ambig.tbl"

fle_sta_1 = 1
fle_end_1 = {peptide_length}
fle_seg_1 = "{peptide_chain}"

# Automatically define backbone dihedral restraints for alpha/beta-like regions.
ssdihed = "alphabeta"

# More refinement steps are useful for flexible peptides.
mdsteps_rigid = 5000
mdsteps_cool1 = 5000
mdsteps_cool2 = 10000
mdsteps_cool3 = 10000

# ====================================================================
# Final refinement.
[{refinement_module}]
tolerance = 20
ambig_fname = "ambig.tbl"

fle_sta_1 = 1
fle_end_1 = {peptide_length}
fle_seg_1 = "{peptide_chain}"

ssdihed = "alphabeta"

# ====================================================================
# Cluster models by fraction of common contacts.
[clustfcc]
min_population = 1

# Select best models per cluster.
[seletopclusts]
top_models = 4

# Final analysis without a reference complex. If a reference is available,
# add reference_fname = "reference.pdb" below.
[caprieval]
allatoms = true

# ====================================================================
'''
    output_file.write_text(cfg)


def run_haddock(workdir: Path, cfg_name: str = "docking.cfg") -> None:
    """Run haddock3 in a working directory."""
    subprocess.run(["haddock3", cfg_name], cwd=workdir, check=True)


def collect_score_files(outdir: Path, output_csv: Path) -> None:
    """Collect locations of generated HADDOCK score/analysis files."""
    rows: list[dict[str, str]] = []

    for peptide_dir in sorted(outdir.glob("run_*")):
        peptide_id = peptide_dir.name.removeprefix("run_")
        haddock_run = peptide_dir / "haddock_run"
        if not haddock_run.exists():
            continue

        score_files = sorted(haddock_run.glob("*_*/capri_ss.tsv"))
        if not score_files:
            score_files = sorted(haddock_run.glob("*_*/*.tsv"))

        for score_file in score_files:
            rows.append({"peptide_id": peptide_id, "score_file": str(score_file)})

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["peptide_id", "score_file"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run HADDOCK3 CaSR-peptide docking workflows."
    )
    parser.add_argument("--receptor", required=True, type=Path, help="Prepared CaSR receptor PDB file.")
    parser.add_argument("--receptor-chain", default="A", help="CaSR receptor chain/segid used in AIR restraints.")
    parser.add_argument("--peptide-chain", default="B", help="Peptide chain/segid used in AIR restraints.")
    parser.add_argument("--peptides", required=True, type=Path, help="CSV file with columns: id,sequence,pdb.")
    parser.add_argument("--pocket-residues", required=True, help="Comma-separated CaSR pocket residues, e.g. 170,171,300-305.")
    parser.add_argument("--activation-residues", default="", help="Comma-separated activation-important CaSR residues.")
    parser.add_argument("--activation-weight", default=2, type=int, help="How many times activation residues are repeated in AIR restraints.")
    parser.add_argument("--receptor-hisd", default="", help="Comma-separated receptor HID/HISD residue numbers.")
    parser.add_argument("--receptor-hise", default="", help="Comma-separated receptor HIE/HISE residue numbers.")
    parser.add_argument("--sampling", default=3000, type=int, help="Number of rigid-body docking models per peptide.")
    parser.add_argument("--select-top", default=400, type=int, help="Number of rigid-body models selected for flexible refinement.")
    parser.add_argument("--ncores", default=8, type=int, help="Number of CPU cores for local execution.")
    parser.add_argument("--mode", default="local", choices=["local", "batch", "mpi"], help="HADDOCK3 execution mode.")
    parser.add_argument("--mdref", action="store_true", help="Use mdref instead of emref for final refinement.")
    parser.add_argument("--outdir", default=Path("casr_peptide_screen"), type=Path, help="Output directory for generated workflows.")
    parser.add_argument("--run", action="store_true", help="Actually run haddock3 for each peptide.")

    args = parser.parse_args()

    if not args.receptor.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {args.receptor}")

    peptides = read_peptides_csv(args.peptides)
    pocket_residues = parse_residue_list(args.pocket_residues)
    activation_residues = parse_residue_list(args.activation_residues)
    receptor_hisd = parse_residue_list(args.receptor_hisd)
    receptor_hise = parse_residue_list(args.receptor_hise)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for peptide in peptides:
        peptide_dir = args.outdir / f"run_{peptide.peptide_id}"
        peptide_dir.mkdir(parents=True, exist_ok=True)

        receptor_dst = peptide_dir / "receptor.pdb"
        peptide_dst = peptide_dir / "peptide.pdb"
        air_dst = peptide_dir / "ambig.tbl"
        cfg_dst = peptide_dir / "docking.cfg"

        shutil.copy2(args.receptor, receptor_dst)
        shutil.copy2(peptide.pdb, peptide_dst)

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
            receptor_hisd=receptor_hisd,
            receptor_hise=receptor_hise,
        )

        print(f"[OK] Generated workflow for {peptide.peptide_id}: {peptide_dir}")

        if args.run:
            print(f"[RUN] haddock3 docking.cfg in {peptide_dir}")
            run_haddock(peptide_dir)

    score_index = args.outdir / "score_files.csv"
    collect_score_files(args.outdir, score_index)
    print(f"[DONE] Workflows generated in: {args.outdir}")
    print(f"[DONE] Score file index: {score_index}")


if __name__ == "__main__":
    main()
