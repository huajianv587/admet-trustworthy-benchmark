from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from admet_benchmark.data_loader import OfficialSplit
from admet_benchmark.smiles_compat import LEGACY_VALENCE_REPAIRS, model_input_smiles
from experiments import analyze_calibration
from experiments import analyze_posthoc_ablations
from experiments import map_shap_morgan_bits
from experiments import run_shap_selected
from experiments import run_y_randomization_all
from experiments.analyze_reliability_regret import RunPrediction


def test_y_randomization_accepts_validation_selection_table(tmp_path: Path) -> None:
    rows = [
        {"task": task, "validation_selected_model": "rf"}
        for task in run_y_randomization_all.TDC_TASKS
    ]
    path = tmp_path / "selection.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    selected = run_y_randomization_all.load_best_models(path)
    assert set(selected) == set(run_y_randomization_all.TDC_TASKS)
    assert set(selected.values()) == {"rf"}


def test_y_randomization_repairs_only_model_inputs(monkeypatch) -> None:
    legacy = next(iter(LEGACY_VALENCE_REPAIRS))
    split = OfficialSplit(
        name="Solubility_AqSolDB",
        train=pd.DataFrame({"Drug": [legacy, "CC"], "Y": [1.0, 2.0]}),
        valid=pd.DataFrame({"Drug": ["CCC"], "Y": [1.5]}),
        test=pd.DataFrame({"Drug": [legacy, "CO"], "Y": [1.1, 1.9]}),
        task_type="regression",
        metric="mae",
        category="absorption",
    )
    observed: list[list[str]] = []

    def fake_morgan(smiles):
        observed.append(list(smiles))
        return np.ones((len(smiles), 2)), list(range(len(smiles)))

    monkeypatch.setattr(run_y_randomization_all, "load_tdc_official", lambda *a, **k: split)
    monkeypatch.setattr(run_y_randomization_all, "morgan_count_matrix", fake_morgan)
    monkeypatch.setattr(run_y_randomization_all, "make_traditional_model", lambda *a, **k: object())
    monkeypatch.setattr(
        run_y_randomization_all,
        "y_randomization_test",
        lambda *a, **k: {
            "real_score": 0.1,
            "random_mean": 0.5,
            "random_std": 0.1,
            "random_scores": [0.5],
        },
    )
    summary, _ = run_y_randomization_all.run_one(
        "Solubility_AqSolDB",
        "rf",
        1,
        1,
        "unused",
        1,
        "locked_cache",
    )
    assert observed[0][0] == model_input_smiles(legacy)
    assert observed[1][0] == model_input_smiles(legacy)
    assert summary["split_source"] == "locked_cache"
    assert summary["train_n"] == 2
    assert summary["test_n"] == 2


def test_calibration_writes_uncalibrated_bins_once() -> None:
    frame = pd.DataFrame(
        {
            "task": ["AMES"] * 8,
            "task_type": ["classification"] * 8,
            "official_metric": ["auroc"] * 8,
            "model": ["rf"] * 8,
            "seed": [1] * 8,
            "split": ["valid"] * 4 + ["test"] * 4,
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "y_pred": [0.1, 0.8, 0.2, 0.9, 0.15, 0.75, 0.3, 0.85],
        }
    )
    rows, bins = analyze_calibration.analyze_frame(frame)
    assert len(rows) == 2
    methods = pd.Series([row["method"] for row in bins]).value_counts().to_dict()
    assert methods == {"uncalibrated": 10, "isotonic": 10, "platt": 10}


def test_shap_selection_and_mapping_preserve_source_smiles(tmp_path: Path) -> None:
    selection = pd.DataFrame(
        [
            {"task": task, "model": "rf"}
            for task, _ in run_shap_selected.SELECTED
        ]
    )
    path = tmp_path / "best.csv"
    selection.to_csv(path, index=False)
    assert set(model for _, model in run_shap_selected.selected_pairs(path)) == {"rf"}

    from rdkit import Chem
    from rdkit.Chem import AllChem

    legacy = next(iter(LEGACY_VALENCE_REPAIRS))
    mol = Chem.MolFromSmiles(model_input_smiles(legacy))
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=bit_info)
    bit_id = next(iter(bit_info))
    example = map_shap_morgan_bits.find_example(
        [("test", 0, legacy)], bit_id, 2, 2048
    )
    assert example is not None
    assert example["example_smiles"] == legacy


def test_equal_mass_ece_is_tie_and_row_order_invariant() -> None:
    labels = np.asarray([0, 1, 0, 1, 1, 0])
    probabilities = np.asarray([0.2, 0.2, 0.5, 0.5, 0.8, 0.8])
    first = analyze_posthoc_ablations.equal_mass_ece(labels, probabilities, 4)
    order = np.asarray([5, 0, 4, 1, 3, 2])
    second = analyze_posthoc_ablations.equal_mass_ece(
        labels[order], probabilities[order], 4
    )
    assert first == second


def test_zero_fit_ensemble_ablation_enumerates_all_2_3_5_seed_subsets() -> None:
    runs = {}
    for seed in range(1, 6):
        valid = pd.DataFrame(
            {
                "sample_index": [0, 1, 2, 3],
                "smiles": ["C", "CC", "CCC", "CCCC"],
                "y_true": [0, 1, 0, 1],
                "y_pred": [0.1, 0.8, 0.2, 0.9],
            }
        )
        test = valid.copy()
        test["y_pred"] = np.asarray([0.1, 0.8, 0.2, 0.9]) + seed * 0.001
        runs[("ames", "rf", seed)] = RunPrediction(
            task="ames",
            model="rf",
            seed=seed,
            task_type="classification",
            official_metric="auroc",
            valid=valid,
            test=test,
            source=f"seed{seed}.csv",
        )
    ensemble = analyze_posthoc_ablations.ensemble_size_rows(
        runs, [2, 3, 5], [1, 2, 3, 4, 5]
    )
    assert ensemble.groupby("ensemble_size").size().to_dict() == {2: 10, 3: 10, 5: 1}
    calibration = analyze_posthoc_ablations.calibration_binning_rows(runs, [5, 10])
    assert len(calibration) == 5 * 3 * 2 * 2
