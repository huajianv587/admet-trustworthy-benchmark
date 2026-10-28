import hashlib
import json
import typing
from pathlib import Path

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import numpy as np
import pandas as pd
import pytest

from experiments.analyze_conditional_reliability import holm_adjust
from experiments.analyze_reviewer_defense_sensitivities import (
    CONDITIONAL_METRICS,
    DEFAULT_CONDITIONAL_DIR,
    DEFAULT_LOCKED_TEST_ROOT,
    DEFAULT_RELIABILITY_DIR,
    EXPECTED_ENDPOINT_UNION,
    EXPECTED_ENDPOINT_UNIQUE_SUM,
    EXPECTED_OVERLAPPING_PAIRS,
    EXPECTED_PAIR_COUNT,
    FORBIDDEN_PUBLIC_COLUMNS,
    MODEL_FAMILIES,
    conditional_family_sensitivity,
    cross_endpoint_overlap_audit,
    leave_one_model_out_sensitivity,
    run_analysis,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_paths() -> list[Path]:
    return [
        *sorted(DEFAULT_LOCKED_TEST_ROOT.glob("*/test.csv")),
        DEFAULT_RELIABILITY_DIR / "validation_seed_scores.csv",
        DEFAULT_RELIABILITY_DIR / "test_reliability_by_task_model.csv",
        DEFAULT_CONDITIONAL_DIR / "conditional_reliability_by_task_model.csv",
    ]


pytestmark = pytest.mark.skipif(
    len(list(DEFAULT_LOCKED_TEST_ROOT.glob("*/test.csv"))) != 22
    or not all(path.is_file() for path in _source_paths()[22:]),
    reason=(
        "requires locally obtained label-bearing TDC splits and complete locked "
        "result tables, which are intentionally absent from the public source clone"
    ),
)


def test_locked_cross_endpoint_overlap_headlines_are_exact_and_aggregate_only() -> None:
    before = {_path: _sha256(_path) for _path in _source_paths()[:22]}
    endpoint_rows, pairwise, components, summary, paths = cross_endpoint_overlap_audit(
        DEFAULT_LOCKED_TEST_ROOT
    )
    after = {_path: _sha256(_path) for _path in _source_paths()[:22]}

    assert before == after
    assert len(paths) == 22
    assert len(endpoint_rows) == 22
    assert len(pairwise) == EXPECTED_PAIR_COUNT == 231
    assert len(components) == 1
    assert components.loc[0, "endpoint_count"] == 22
    assert summary["status"] == "GO"
    assert summary["endpoint_unique_count_sum"] == EXPECTED_ENDPOINT_UNIQUE_SUM == 16082
    assert summary["global_unique_union_count"] == EXPECTED_ENDPOINT_UNION == 13818
    assert summary["pairs_with_overlap"] == EXPECTED_OVERLAPPING_PAIRS == 209
    assert summary["maximum_pair_intersection_count"] == 538
    # RDKit canonical SMILES are version-sensitive.  The project pins the
    # canonicalization runtime so this aggregate digest remains reproducible.
    assert summary["global_whole_set_sha256"] == (
        "b04546d16ad7a7a644fd1a1f1eb3829287c4b293d9a3b68abf57d811111d97b6"
    )
    for frame in (endpoint_rows, pairwise, components):
        assert not ({column.lower() for column in frame.columns} & FORBIDDEN_PUBLIC_COLUMNS)
        serialized = frame.to_csv(index=False)
        assert "CCO" not in serialized


def test_leave_one_model_out_reselects_and_recomputes_every_four_model_pool() -> None:
    validation = pd.read_csv(DEFAULT_RELIABILITY_DIR / "validation_seed_scores.csv")
    reliability = pd.read_csv(
        DEFAULT_RELIABILITY_DIR / "test_reliability_by_task_model.csv"
    )
    validation_before = validation.copy(deep=True)
    reliability_before = reliability.copy(deep=True)
    first = leave_one_model_out_sensitivity(
        validation,
        reliability,
        bootstrap_reps=31,
        permutation_reps=101,
        selection_bootstrap_reps=29,
        random_seed=777,
    )
    second = leave_one_model_out_sensitivity(
        validation,
        reliability,
        bootstrap_reps=31,
        permutation_reps=101,
        selection_bootstrap_reps=29,
        random_seed=777,
    )

    pd.testing.assert_frame_equal(validation, validation_before, check_exact=True)
    pd.testing.assert_frame_equal(reliability, reliability_before, check_exact=True)
    for left, right in zip(first[:4], second[:4]):
        pd.testing.assert_frame_equal(left, right, check_exact=True)

    selections, probabilities, endpoints, dimensions, gate = first
    assert len(selections) == 110
    assert len(probabilities) == 440
    assert len(endpoints) == 460
    assert len(dimensions) == 30
    assert set(selections["n_models"]) == {4}
    assert set(endpoints["n_models"]) == {4}
    assert all(
        omitted not in pool.split(";")
        for omitted, pool in zip(selections["omitted_model"], selections["model_pool"])
    )
    assert np.allclose(
        endpoints["random_selection_expected_regret"],
        endpoints["exact_uniform_random_expected_regret"],
        rtol=1e-12,
        atol=1e-12,
    )
    expected_normalized = np.divide(
        endpoints["reliability_regret"],
        endpoints["pool_reliability_loss_range"],
        out=np.zeros(len(endpoints), dtype=float),
        where=endpoints["pool_reliability_loss_range"].to_numpy() > 1e-12,
    )
    assert np.allclose(endpoints["normalized_regret"], expected_normalized)
    assert endpoints.groupby("omitted_model")["dimension"].nunique().eq(6).all()
    assert gate["checks"]["validation_reselected_in_each_pool"] is True
    assert gate["checks"]["oracle_and_pool_range_recomputed"] is True
    # Reduced test resampling is deliberately non-formal.
    assert gate["status"] == "HOLD"


def test_conditional_family_stratification_preserves_eligibility_and_holm_family() -> None:
    source = pd.read_csv(
        DEFAULT_CONDITIONAL_DIR / "conditional_reliability_by_task_model.csv"
    )
    before = source.copy(deep=True)
    first_rows, first_summary, first_gate = conditional_family_sensitivity(
        source,
        bootstrap_reps=37,
        permutation_reps=101,
        random_seed=123,
    )
    second_rows, second_summary, second_gate = conditional_family_sensitivity(
        source,
        bootstrap_reps=37,
        permutation_reps=101,
        random_seed=123,
    )
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    pd.testing.assert_frame_equal(first_rows, second_rows, check_exact=True)
    pd.testing.assert_frame_equal(first_summary, second_summary, check_exact=True)
    assert first_gate == second_gate
    assert len(first_rows) == 330
    assert len(first_summary) == 36
    assert np.array_equal(first_rows["eligible"].to_numpy(), before["eligible"].to_numpy())
    assert dict(
        first_rows[["model", "model_family"]].drop_duplicates().to_numpy()
    ) == MODEL_FAMILIES
    assert first_summary.groupby(["model_family", "threshold"])["metric"].nunique().eq(
        len(CONDITIONAL_METRICS)
    ).all()
    for _, group in first_summary.groupby(["model_family", "threshold"], sort=True):
        expected = holm_adjust(group["sign_flip_p"])
        assert np.allclose(
            group["holm_p_within_family_threshold"], expected, equal_nan=True
        )
    assert first_gate["checks"]["frozen_eligibility_preserved"] is True
    assert first_gate["status"] == "HOLD"


def test_bound_run_is_deterministic_immutable_and_contains_no_raw_structures(
    tmp_path: Path,
) -> None:
    input_hashes = {path: _sha256(path) for path in _source_paths()}
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    kwargs = {
        "bootstrap_reps": 19,
        "permutation_reps": 47,
        "selection_bootstrap_reps": 17,
        "random_seed": 2468,
    }
    first_gate = run_analysis(out_dir=first_dir, **kwargs)
    second_gate = run_analysis(out_dir=second_dir, **kwargs)
    assert first_gate == second_gate
    assert input_hashes == {path: _sha256(path) for path in _source_paths()}

    first_files = sorted(path.name for path in first_dir.iterdir() if not path.name.startswith("."))
    second_files = sorted(path.name for path in second_dir.iterdir() if not path.name.startswith("."))
    assert first_files == second_files
    assert len(first_files) == 17
    for name in first_files:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()

    manifest = json.loads(
        (first_dir / "reviewer_defense_sensitivities_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["input_files"]) == 25
    assert len(manifest["output_files"]) == 16
    assert all(not item["path"].startswith(".") for item in manifest["output_files"])
    for path in first_dir.iterdir():
        if path.name.startswith(".") or not path.is_file():
            continue
        payload = path.read_bytes()
        assert b"CCO" not in payload
    assert first_gate["checks"]["source_inputs_sha256_and_size_unchanged"] is True
    assert first_gate["checks"]["zero_refit"] is True
    assert first_gate["status"] == "HOLD"
