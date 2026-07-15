import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_reliability_regret import (
    EXPECTED_TASKS,
    FINAL_EXPECTED_MODELS,
    ModelEnsemble,
    RunPrediction,
    _load_similarity_for_test,
    _normalize_prediction_file,
    _similarity_cache_path,
    audit_ensemble_test_alignment,
    audit_run_sample_alignment,
    aurc_from_curve,
    build_gate,
    build_regrets,
    holm_adjust,
    risk_coverage_curve,
    select_validation_models,
)


def test_locked_similarity_cache_never_falls_back_to_legacy_name(tmp_path: Path):
    legacy = tmp_path / "ames_seed1_test_similarity.csv"
    legacy.write_text("sample_index,max_tanimoto\n0,0.5\n", encoding="utf-8")
    resolved = _similarity_cache_path(tmp_path, "ames", 1, "locked_cache")
    assert resolved == tmp_path / "ames_seed1_locked_cache_test_similarity.csv"
    assert not resolved.exists()
    assert _similarity_cache_path(tmp_path, "ames", 1, "pytdc") == legacy


def test_similarity_cache_requires_exact_test_smiles(tmp_path: Path):
    path = tmp_path / "ames_seed1_locked_cache_test_similarity.csv"
    pd.DataFrame(
        {
            "sample_index": [0, 1],
            "smiles": ["C", "wrong"],
            "max_tanimoto": [0.4, 0.8],
        }
    ).to_csv(path, index=False)
    test = pd.DataFrame(
        {
            "sample_index": [0, 1],
            "smiles": ["C", "CC"],
            "y_true": [0.0, 1.0],
            "y_pred": [0.1, 0.9],
        }
    )
    similarity, reason = _load_similarity_for_test(path, test)
    assert similarity is None
    assert reason == "similarity_cache_test_smiles_mismatch"


def test_formal_similarity_cache_requires_split_provenance(tmp_path: Path):
    path = tmp_path / "ames_seed1_locked_cache_test_similarity.csv"
    pd.DataFrame(
        {
            "sample_index": [0],
            "smiles": ["C"],
            "max_tanimoto": [0.4],
        }
    ).to_csv(path, index=False)
    test = pd.DataFrame(
        {"sample_index": [0], "smiles": ["C"], "y_true": [0.0], "y_pred": [0.1]}
    )
    similarity, reason = _load_similarity_for_test(
        path, test, expected_split_source="locked_cache"
    )
    assert similarity is None
    assert reason == "similarity_cache_provenance_missing"


def _neural_prediction_payload() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["valid", "valid", "test", "test"],
            "smiles": ["C", "CC", "CCC", "CCCC"],
            "label": [0, 1, 0, 1],
            "prediction": [0.2, 0.8, 0.1, 0.9],
        }
    )


def _neural_metrics(model: str) -> dict:
    return {
        "status": "complete",
        "task": "AMES",
        "model": model,
        "seed": 1,
        "task_type": "classification",
        "metric_name": "auroc",
        "valid_metric": 1.0,
        "test_metric": 1.0,
        "prediction_rows": 4,
    }


def _chemprop_config() -> dict:
    return {
        "status": "complete",
        "task": "AMES",
        "model": "chemprop_dmpnn",
        "seed": 1,
        "task_type": "classification",
        "official_metric": "auroc",
        "software": {"chemprop": "2.2.1"},
        "training": {"data_seed": 1, "pytorch_seed": 1},
        "architecture": {
            "directed_bond_messages": True,
            "atom_messages": False,
            "undirected": False,
        },
        "data": {
            "valid_n": 2,
            "test_n": 2,
            "official_test_preserved": True,
            "train_sha256": "a" * 64,
            "valid_sha256": "b" * 64,
            "test_sha256": "c" * 64,
        },
    }


def _ensemble(model: str, validation_scores: dict[int, float], prediction: float):
    return ModelEnsemble(
        task="ames",
        model=model,
        task_type="classification",
        official_metric="auroc",
        seeds=tuple(validation_scores),
        validation_scores=validation_scores,
        test=pd.DataFrame(
            {
                "sample_index": [0, 1],
                "y_true": [0, 1],
                "prediction": [prediction, prediction],
                "uncertainty": [0.1, 0.2],
                "max_tanimoto": [0.3, 0.8],
                "n_seeds": [5, 5],
            }
        ),
        conformal_coverage=float("nan"),
    )


def test_validation_selection_does_not_depend_on_test_predictions():
    rng = np.random.default_rng(7)
    ensembles = [
        _ensemble("a", {seed: 0.8 for seed in range(1, 6)}, prediction=0.99),
        _ensemble("b", {seed: 0.7 for seed in range(1, 6)}, prediction=0.01),
    ]
    selected, _ = select_validation_models(
        ensembles,
        seed_bootstrap_reps=50,
        absolute_tolerance=0.005,
        relative_tolerance=0.01,
        rng=rng,
    )
    assert selected.loc[0, "validation_selected_model"] == "a"

    # Reversing every test prediction cannot change a validation-only choice.
    ensembles[0].test["prediction"] = 0.0
    ensembles[1].test["prediction"] = 1.0
    selected_reversed, _ = select_validation_models(
        ensembles,
        seed_bootstrap_reps=50,
        absolute_tolerance=0.005,
        relative_tolerance=0.01,
        rng=np.random.default_rng(7),
    )
    assert selected_reversed.loc[0, "validation_selected_model"] == "a"


def test_risk_coverage_and_excess_aurc_are_exact_and_nonnegative():
    losses = np.array([3.0, 1.0, 2.0])
    uncertainty = np.array([0.1, 0.2, 0.3])
    curve = risk_coverage_curve(losses, uncertainty)
    assert np.allclose(curve["coverage"], [1 / 3, 2 / 3, 1.0])
    assert np.allclose(curve["cumulative_risk"], [3.0, 2.0, 2.0])
    assert np.allclose(curve["oracle_cumulative_risk"], [1.0, 1.5, 2.0])
    aurc, oracle, excess = aurc_from_curve(curve)
    assert np.isclose(aurc, 7 / 3)
    assert np.isclose(oracle, 1.5)
    assert np.isclose(excess, 5 / 6)
    assert excess >= 0

    tied = risk_coverage_curve(np.array([0.0, 2.0]), np.array([0.5, 0.5]))
    tied_reversed = risk_coverage_curve(
        np.array([2.0, 0.0]), np.array([0.5, 0.5])
    )
    assert np.allclose(tied["cumulative_risk"], [1.0, 1.0])
    assert np.allclose(
        tied["cumulative_risk"], tied_reversed["cumulative_risk"]
    )


def test_regret_uses_validation_choice_and_test_oracle_only_as_counterfactual():
    reliability = pd.DataFrame(
        [
            {
                "task": "ames",
                "model": "a",
                "task_type": "classification",
                "official_metric": "auroc",
                "dimension": "brier",
                "value": 0.20,
                "higher_is_better": False,
                "risk_metric": "brier",
            },
            {
                "task": "ames",
                "model": "b",
                "task_type": "classification",
                "official_metric": "auroc",
                "dimension": "brier",
                "value": 0.10,
                "higher_is_better": False,
                "risk_metric": "brier",
            },
        ]
    )
    selections = pd.DataFrame(
        [
            {
                "task": "ames",
                "validation_selected_model": "a",
                "validation_tie_models": "a",
                "validation_tie_count": 1,
                "selected_exact_bootstrap_probability": 1.0,
                "selected_tie_bootstrap_probability": 1.0,
            }
        ]
    )
    regret = build_regrets(reliability, selections).iloc[0]
    assert regret["validation_selected_model"] == "a"
    assert regret["test_reliability_oracle_model"] == "b"
    assert np.isclose(regret["reliability_regret"], 0.10)
    assert regret["reliability_regret"] >= 0


def test_neural_metrics_json_is_required_as_completion_marker(tmp_path: Path):
    prediction_dir = tmp_path / "predictions"
    metrics_dir = tmp_path / "metrics"
    checkpoint_dir = tmp_path / "checkpoints"
    prediction_dir.mkdir()
    metrics_dir.mkdir()
    checkpoint_dir.mkdir()
    path = prediction_dir / "AMES_gin_seed1.csv"
    _neural_prediction_payload().to_csv(path, index=False)
    frame, reason = _normalize_prediction_file(path)
    assert frame is None
    assert reason == "missing_neural_metrics_marker"

    (metrics_dir / "AMES_gin_seed1.json").write_text(
        json.dumps(_neural_metrics("gin")), encoding="utf-8"
    )
    frame, reason = _normalize_prediction_file(path)
    assert frame is None
    assert reason == "missing_or_empty_neural_checkpoint"

    (checkpoint_dir / "AMES_gin_seed1.pt").write_bytes(b"checkpoint")
    frame, reason = _normalize_prediction_file(path)
    assert reason is None
    assert frame is not None
    assert set(frame["task"]) == {"ames"}


def test_chemprop_name_is_preserved_and_legacy_dmpnn_is_honestly_renamed(
    tmp_path: Path,
):
    prediction_dir = tmp_path / "predictions"
    metrics_dir = tmp_path / "metrics"
    checkpoint_dir = tmp_path / "checkpoints"
    config_dir = tmp_path / "configs"
    prediction_dir.mkdir()
    metrics_dir.mkdir()
    checkpoint_dir.mkdir()
    config_dir.mkdir()
    payload = _neural_prediction_payload()
    for model in ("chemprop_dmpnn", "dmpnn"):
        stem = f"AMES_{model}_seed1"
        path = prediction_dir / f"{stem}.csv"
        payload.to_csv(path, index=False)
        (checkpoint_dir / f"{stem}.pt").write_bytes(b"checkpoint")
        (metrics_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    **_neural_metrics(model),
                    **(
                        {
                            "chemprop_version": "2.2.1",
                            "official_test_preserved": True,
                        }
                        if model == "chemprop_dmpnn"
                        else {}
                    ),
                }
            ),
            encoding="utf-8",
        )
        if model == "chemprop_dmpnn":
            (config_dir / f"{stem}.json").write_text(
                json.dumps(_chemprop_config()), encoding="utf-8"
            )
        frame, reason = _normalize_prediction_file(path)
        assert reason is None
        assert frame is not None
        expected = "chemprop_dmpnn" if model == "chemprop_dmpnn" else "legacy_node_mpnn"
        assert set(frame["model"]) == {expected}


def test_holm_adjustment_is_monotone_and_not_smaller_than_raw_pvalues():
    raw = pd.Series([0.01, 0.04, 0.03])
    adjusted = holm_adjust(raw)
    ordered = adjusted.loc[raw.sort_values().index].to_numpy()
    assert np.all(np.diff(ordered) >= 0)
    assert np.all(adjusted.to_numpy() >= raw.to_numpy())


def test_final_gate_cannot_be_completed_by_traditional_models_alone():
    empty_test = pd.DataFrame()
    ensembles = [
        ModelEnsemble(
            task=task,
            model=model,
            task_type="classification",
            official_metric="auroc",
            seeds=(1, 2, 3, 4, 5),
            validation_scores={seed: 0.5 for seed in range(1, 6)},
            test=empty_test,
            conformal_coverage=float("nan"),
        )
        for task in EXPECTED_TASKS
        for model in ("rf", "xgboost", "lightgbm")
    ]
    selections = pd.DataFrame(
        {
            "task": EXPECTED_TASKS,
            "model_pool": ["lightgbm;rf;xgboost"] * len(EXPECTED_TASKS),
        }
    )
    summary = pd.DataFrame(
        {
            "dimension": ["ece", "applicability_alignment"],
            "practical_effect_signal": [True, True],
            "tie_robust_effect_signal": [True, True],
            "random_baseline_retention_signal": [True, True],
        }
    )
    gate = build_gate(
        ensembles,
        selections,
        summary,
        minimum_seeds=5,
        expected_models=FINAL_EXPECTED_MODELS,
        gate_scope="final",
    )
    assert gate["status"] == "HOLD"
    assert not gate["coverage"]["complete_expected_model_endpoint_matrix"]
    assert set(gate["coverage"]["missing_expected_models"]) == {
        "gin",
        "chemprop_dmpnn",
    }


def test_cross_model_test_sample_misalignment_is_detected():
    left = _ensemble("rf", {seed: 0.8 for seed in range(1, 6)}, prediction=0.8)
    right = _ensemble("gin", {seed: 0.7 for seed in range(1, 6)}, prediction=0.7)
    for item, smiles in ((left, ["C", "CC"]), (right, ["C", "CCC"])):
        item.test["smiles"] = smiles
    audit, aligned = audit_ensemble_test_alignment([left, right])
    assert not aligned
    assert not audit["task_models_aligned"].any()


def test_cross_model_validation_alignment_is_checked_within_task_seed():
    def run(model: str, valid_smiles: list[str]) -> RunPrediction:
        valid = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "smiles": valid_smiles,
                "y_true": [0.0, 1.0],
                "y_pred": [0.2, 0.8],
            }
        )
        test = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "smiles": ["CCC", "CCCC"],
                "y_true": [0.0, 1.0],
                "y_pred": [0.1, 0.9],
                "max_tanimoto": [0.2, 0.7],
            }
        )
        return RunPrediction(
            task="ames",
            model=model,
            seed=1,
            task_type="classification",
            official_metric="auroc",
            valid=valid,
            test=test,
            source=model,
        )

    runs = {
        ("ames", "rf", 1): run("rf", ["C", "CC"]),
        ("ames", "gin", 1): run("gin", ["C", "CCN"]),
    }
    audit, aligned = audit_run_sample_alignment(runs)
    assert not aligned
    assert not audit["task_seed_models_aligned"].any()


def test_final_gate_requires_exact_public_seed_set_not_only_five_seeds():
    ensembles = [
        ModelEnsemble(
            task=task,
            model=model,
            task_type="classification",
            official_metric="auroc",
            seeds=(2, 3, 4, 5, 6),
            validation_scores={seed: 0.5 for seed in (2, 3, 4, 5, 6)},
            test=pd.DataFrame(),
            conformal_coverage=float("nan"),
        )
        for task in EXPECTED_TASKS
        for model in FINAL_EXPECTED_MODELS
    ]
    pool = ";".join(sorted(FINAL_EXPECTED_MODELS))
    selections = pd.DataFrame({"task": EXPECTED_TASKS, "model_pool": [pool] * 22})
    summary = pd.DataFrame(
        {
            "dimension": ["ece", "brier"],
            "practical_effect_signal": [True, True],
            "tie_robust_effect_signal": [True, True],
            "random_baseline_retention_signal": [True, True],
        }
    )
    gate = build_gate(
        ensembles,
        selections,
        summary,
        minimum_seeds=5,
        expected_models=FINAL_EXPECTED_MODELS,
        gate_scope="final",
    )
    assert gate["status"] == "HOLD"
    assert not gate["coverage"]["final_seed_sets_exact"]
