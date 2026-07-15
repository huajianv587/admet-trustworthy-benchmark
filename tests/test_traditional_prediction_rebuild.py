from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_benchmark.data_loader import OfficialSplit
from admet_benchmark.metrics import classification_metrics
from experiments import run_traditional_grid as grid
from experiments import run_traditional_predictions as predictions
from experiments.summarize_traditional_grid import summarize_frame


def _frame(smiles: list[str], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"Drug": smiles, "Y": labels})


def _split() -> OfficialSplit:
    return OfficialSplit(
        name="AMES",
        train=_frame(["C", "CC", "CCC", "CCCC"], [0, 1, 0, 1]),
        valid=_frame(["CO", "CN", "CCO", "CCN"], [0, 1, 0, 1]),
        test=_frame(["CCl", "CBr", "CCCl", "CCBr"], [0, 1, 0, 1]),
        task_type="classification",
        metric="auroc",
        category="toxicity",
    )


def _write_prediction(root: Path, *, bad_label: bool = False) -> Path:
    path = root / "ames" / "rf_seed1.csv"
    path.parent.mkdir(parents=True)
    valid_scores = [0.1, 0.8, 0.3, 0.9]
    test_scores = [0.2, 0.7, 0.4, 0.6]
    rows = []
    split = _split()
    for split_name, source, scores in (
        ("valid", split.valid, valid_scores),
        ("test", split.test, test_scores),
    ):
        for sample_index, (smiles, label, score) in enumerate(
            zip(source["Drug"], source["Y"], scores)
        ):
            rows.append(
                {
                    "task": "AMES",
                    "task_type": "classification",
                    "official_metric": "auroc",
                    "model": "rf",
                    "seed": 1,
                    "split": split_name,
                    "sample_index": sample_index,
                    "smiles": smiles,
                    "y_true": 9 if bad_label and split_name == "test" and sample_index == 0 else label,
                    "y_pred": score,
                }
            )
    pd.DataFrame(rows, columns=grid.PREDICTION_FIELDNAMES).to_csv(path, index=False)
    return path


def _rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grid, "load_tdc_official", lambda *args, **kwargs: _split())
    prediction_dir = tmp_path / "predictions"
    output = tmp_path / "grid.csv"
    audit = tmp_path / "audit.json"
    result = grid.rebuild_grid_from_predictions(
        prediction_dir=prediction_dir,
        output_path=output,
        audit_path=audit,
        tasks=["AMES"],
        models=["rf"],
        seeds=[1],
        radius=2,
        n_bits=2048,
        data_dir="unused",
        split_source="locked_cache",
    )
    return prediction_dir, output, audit, result


def test_rebuild_strictly_recomputes_grid_and_uses_timing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction = _write_prediction(tmp_path / "predictions")
    predictions.write_prediction_manifest(
        prediction,
        task="AMES",
        task_type="classification",
        official_metric="auroc",
        model="rf",
        seed=1,
        split_source="locked_cache",
        radius=2,
        n_bits=2048,
        train_n=4,
        valid_n=4,
        test_n=4,
        fit_seconds=1.23456,
        representation_repair_count=0,
    )

    _, output, audit_path, audit = _rebuild(tmp_path, monkeypatch)
    assert audit["status"] == "PASS"
    assert audit["timing_manifest_runs"] == 1
    row = pd.read_csv(output).iloc[0]
    assert pd.read_csv(output, nrows=0).columns.tolist() == grid.FIELDNAMES
    assert row["train_n"] == 4
    assert row["valid_n"] == 4
    assert row["test_n"] == 4
    assert row["fit_seconds"] == pytest.approx(1.235)

    expected_valid = classification_metrics(
        np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.8, 0.3, 0.9])
    )
    expected_test = classification_metrics(
        np.asarray([0, 1, 0, 1]), np.asarray([0.2, 0.7, 0.4, 0.6])
    )
    for key, value in expected_valid.items():
        assert row[f"valid_{key}"] == pytest.approx(value)
    for key, value in expected_test.items():
        assert row[f"test_{key}"] == pytest.approx(value)
    assert json.loads(audit_path.read_text())["validated_runs"] == 1


def test_legacy_prediction_without_manifest_marks_fit_seconds_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_prediction(tmp_path / "predictions")
    _, output, _, audit = _rebuild(tmp_path, monkeypatch)
    assert audit["fit_seconds_na_runs"] == 1
    rebuilt = pd.read_csv(output)
    assert rebuilt["fit_seconds"].isna().all()

    summary, best = summarize_frame(rebuilt)
    assert summary.loc[0, "n_fit_seconds"] == 0
    assert summary.loc[0, "fit_seconds_coverage"] == 0
    assert np.isnan(summary.loc[0, "mean_fit_seconds"])
    assert best.loc[0, "model"] == "rf"


def test_rebuild_holds_on_label_identity_mismatch_without_writing_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_prediction(tmp_path / "predictions", bad_label=True)
    monkeypatch.setattr(grid, "load_tdc_official", lambda *args, **kwargs: _split())
    output = tmp_path / "grid.csv"
    audit = tmp_path / "audit.json"
    with pytest.raises(grid.PredictionAuditError, match="rebuild HOLD"):
        grid.rebuild_grid_from_predictions(
            prediction_dir=tmp_path / "predictions",
            output_path=output,
            audit_path=audit,
            tasks=["AMES"],
            models=["rf"],
            seeds=[1],
            radius=2,
            n_bits=2048,
            data_dir="unused",
            split_source="locked_cache",
        )
    assert not output.exists()
    payload = json.loads(audit.read_text())
    assert payload["status"] == "HOLD"
    assert "labels are not aligned" in payload["errors"][0]["detail"]


def test_resume_keys_are_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    path.write_text("task,model,seed\nAMES,rf,1\n", encoding="utf-8")
    assert grid.existing_keys(path) == {("ames", "rf", 1)}


def test_best_summary_selects_on_validation_not_test() -> None:
    frame = pd.DataFrame(
        [
            {
                "task": "AMES",
                "task_type": "classification",
                "official_metric": "auroc",
                "model": "rf",
                "seed": 1,
                "valid_auroc": 0.90,
                "test_auroc": 0.60,
                "fit_seconds": "NA",
            },
            {
                "task": "AMES",
                "task_type": "classification",
                "official_metric": "auroc",
                "model": "xgboost",
                "seed": 1,
                "valid_auroc": 0.80,
                "test_auroc": 0.95,
                "fit_seconds": "NA",
            },
        ]
    )
    _, best = summarize_frame(frame)
    assert best.loc[0, "model"] == "rf"
