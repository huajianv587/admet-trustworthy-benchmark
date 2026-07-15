from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (
    LOCKED_CACHE_SPLIT_ALGORITHM,
    LOCKED_CACHE_SPLIT_SOURCE,
    PYTDC_SPLIT_SOURCE,
    SPLIT_SOURCE_CHOICES,
    TDC_TASKS,
    extract_xy,
    load_tdc_official,
)
from admet_benchmark.smiles_compat import model_input_smiles


FIELDS = [
    "task",
    "model",
    "seed",
    "split_source",
    "split_algorithm",
    "rank",
    "feature",
    "bit_id",
    "mean_abs_shap",
    "example_split",
    "example_index",
    "example_smiles",
    "bit_atom_index",
    "bit_radius",
    "substructure_smiles",
]


def bit_id_from_feature(feature: str) -> int:
    match = re.search(r"morgan_bit_(\d+)", feature)
    if not match:
        raise ValueError(f"Cannot parse Morgan bit id from feature: {feature}")
    return int(match.group(1))


def task_to_tdc_name(task: str) -> str:
    mapping = {item.lower(): item for item in TDC_TASKS}
    try:
        return mapping[task.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown task in SHAP table: {task}") from exc


def fragment_smiles(mol, atom_idx: int, radius: int) -> str:
    from rdkit import Chem

    if radius == 0:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[atom_idx], canonical=True)
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms: set[int] = set()
    for bond_idx in env:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    if not atoms:
        atoms.add(atom_idx)
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=list(env), canonical=True)


def find_example(smiles_rows: list[tuple[str, int, str]], bit_id: int, radius: int, n_bits: int) -> dict | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for split_name, index, smiles in smiles_rows:
        mol = Chem.MolFromSmiles(model_input_smiles(smiles))
        if mol is None:
            continue
        bit_info: dict[int, list[tuple[int, int]]] = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
        if bit_id not in bit_info:
            continue
        atom_idx, bit_radius = bit_info[bit_id][0]
        return {
            "example_split": split_name,
            "example_index": index,
            "example_smiles": smiles,
            "bit_atom_index": atom_idx,
            "bit_radius": bit_radius,
            "substructure_smiles": fragment_smiles(mol, atom_idx, bit_radius),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Map top SHAP Morgan bits to representative substructures.")
    parser.add_argument("--shap", default=str(ROOT / "results" / "tables" / "shap_selected_local_prelim.csv"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-k-per-task", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tdc_official"))
    parser.add_argument(
        "--split-source", choices=SPLIT_SOURCE_CHOICES, default=PYTDC_SPLIT_SOURCE
    )
    parser.add_argument("--out", default=str(ROOT / "results" / "tables" / "shap_morgan_bit_substructures_local_prelim.csv"))
    args = parser.parse_args()

    shap_df = pd.read_csv(args.shap)
    if "split_source" in shap_df.columns:
        observed_sources = set(shap_df["split_source"].astype(str))
        if observed_sources != {args.split_source}:
            raise ValueError(
                f"SHAP split_source mismatch: expected {args.split_source}, "
                f"observed {sorted(observed_sources)}"
            )
    elif args.split_source == LOCKED_CACHE_SPLIT_SOURCE:
        raise ValueError(
            "Locked-cache SHAP mapping requires split_source provenance in the SHAP table"
        )
    observed_seeds = set(pd.to_numeric(shap_df["seed"], errors="raise").astype(int))
    if observed_seeds != {args.seed}:
        raise ValueError(
            f"SHAP seed mismatch: expected {args.seed}, observed {sorted(observed_seeds)}"
        )
    selected = shap_df[shap_df["rank"] <= args.top_k_per_task].copy()
    rows: list[dict] = []
    for task_key, group in selected.groupby("task", sort=False):
        tdc_task = task_to_tdc_name(task_key)
        split = load_tdc_official(
            tdc_task,
            data_dir=args.data_dir,
            seed=args.seed,
            split_source=args.split_source,
        )
        train_smiles, _ = extract_xy(split.train)
        test_smiles, _ = extract_xy(split.test)
        smiles_rows = [("test", idx, smi) for idx, smi in enumerate(test_smiles)]
        smiles_rows.extend(("train", idx, smi) for idx, smi in enumerate(train_smiles))
        for item in group.itertuples(index=False):
            bit_id = bit_id_from_feature(item.feature)
            example = find_example(smiles_rows, bit_id, args.radius, args.n_bits) or {}
            rows.append(
                {
                    "task": item.task,
                    "model": item.model,
                    "seed": item.seed,
                    "split_source": args.split_source,
                    "split_algorithm": (
                        LOCKED_CACHE_SPLIT_ALGORITHM
                        if args.split_source == LOCKED_CACHE_SPLIT_SOURCE
                        else "pytdc_default_task_specific_runtime"
                    ),
                    "rank": item.rank,
                    "feature": item.feature,
                    "bit_id": bit_id,
                    "mean_abs_shap": item.mean_abs_shap,
                    "example_split": example.get("example_split", ""),
                    "example_index": example.get("example_index", ""),
                    "example_smiles": example.get("example_smiles", ""),
                    "bit_atom_index": example.get("bit_atom_index", ""),
                    "bit_radius": example.get("bit_radius", ""),
                    "substructure_smiles": example.get("substructure_smiles", ""),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows: {len(rows)}")
    print(f"mapped: {sum(1 for row in rows if row['substructure_smiles'])}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
