import hashlib
import typing
from pathlib import Path

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import pandas as pd
import pytest

from experiments.analyze_crossfit_influence_sensitivity import (
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_RANDOM_SEED,
    build_influence_tables,
    compare_reference_dimension_summary,
    normalize_endpoint_summary,
    run_analysis,
)


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CROSSFIT_DIR = (
    ROOT / "results" / "tables" / "crossfit_oracle_sensitivity_final"
)
requires_locked_crossfit_tables = pytest.mark.skipif(
    not (
        (LOCKED_CROSSFIT_DIR / "crossfit_oracle_endpoint_summary.csv").is_file()
        and (LOCKED_CROSSFIT_DIR / "crossfit_oracle_dimension_summary.csv").is_file()
    ),
    reason=(
        "requires the full release result tables; the source-only public clone "
        "still runs all self-contained cross-fit tests"
    ),
)


def _synthetic_endpoint_summary() -> pd.DataFrame:
    tasks = (
        ("AMES", "toxicity", 0.010),
        ("HIA_Hou", "absorption", 0.020),
        ("BBB_Martins", "distribution", 0.030),
        ("CYP2C9_Veith", "metabolism", 0.040),
    )
    dimensions = ("ece_10bin", "ece_5bin", "applicability_alignment")
    rows = []
    for scheme, role, shift in (
        ("bemis_murcko", "primary", 0.0),
        ("canonical_identity", "secondary", 0.003),
    ):
        for dimension_index, dimension in enumerate(dimensions):
            for task, _category, value in tasks:
                rows.append(
                    {
                        "grouping_scheme": scheme,
                        "scheme_role": role,
                        "task": task,
                        "task_type": "classification",
                        "dimension": dimension,
                        "evidence_tier": (
                            "sensitivity" if dimension == "ece_5bin" else "primary"
                        ),
                        "mean_signed_d": value + shift + 0.005 * dimension_index,
                    }
                )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        [
            "grouping_scheme",
            "dimension",
            "full_mean_signed_d",
            "full_ci_low",
            "full_ci_high",
        ]
    ].rename(
        columns={
            "full_mean_signed_d": "mean_endpoint_signed_d",
            "full_ci_low": "mean_endpoint_signed_d_ci_low",
            "full_ci_high": "mean_endpoint_signed_d_ci_high",
        }
    )


def test_determinism_counts_and_sign_fields() -> None:
    source = _synthetic_endpoint_summary()
    first = build_influence_tables(source, bootstrap_reps=137, random_seed=42)
    second = build_influence_tables(source, bootstrap_reps=137, random_seed=42)
    for first_frame, second_frame in zip(first, second):
        pd.testing.assert_frame_equal(first_frame, second_frame, check_exact=True)

    loeo, loco, summary = first
    assert len(loeo) == len(source)
    assert len(loco) == 2 * 3 * 4  # schemes x dimensions x present categories
    assert len(summary) == 2 * 3
    assert set(loeo["sign_stability"]) == {"preserved"}
    assert loeo["mean_sign_stable"].all()
    assert loco["mean_sign_stable"].all()
    assert (summary["loeo_ci_low_gt_zero_count"] <= summary["loeo_omission_count"]).all()
    assert (summary["loco_ci_low_gt_zero_count"] <= summary["loco_omission_count"]).all()


def test_category_mapping_is_locked_and_input_frame_is_not_mutated() -> None:
    source = _synthetic_endpoint_summary()
    before = source.copy(deep=True)
    normalized = normalize_endpoint_summary(source)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    mapped = dict(zip(normalized["task"], normalized["category"]))
    assert mapped["ames"] == "toxicity"
    assert mapped["hia_hou"] == "absorption"
    assert mapped["bbb_martins"] == "distribution"
    assert mapped["cyp2c9_veith"] == "metabolism"

    unknown = source.copy(deep=True)
    unknown.loc[0, "task"] = "not_a_locked_tdc_endpoint"
    with pytest.raises(ValueError, match="absent from locked TDC category metadata"):
        normalize_endpoint_summary(unknown)


def test_file_bound_run_does_not_mutate_input_and_writes_complete_gate(
    tmp_path: Path,
) -> None:
    source = _synthetic_endpoint_summary()
    input_path = tmp_path / "endpoint_summary.csv"
    reference_path = tmp_path / "dimension_summary.csv"
    out_dir = tmp_path / "influence"
    source.to_csv(input_path, index=False)
    _, _, full_summary = build_influence_tables(
        source,
        bootstrap_reps=DEFAULT_BOOTSTRAP_REPS,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    _reference_from_summary(full_summary).to_csv(reference_path, index=False)
    before_hash = _sha256(input_path)

    gate = run_analysis(
        input_path,
        out_dir,
        reference_summary_path=reference_path,
    )

    assert _sha256(input_path) == before_hash
    assert gate["status"] == "GO"
    assert gate["checks"]["source_hash_and_size_unchanged"] is True
    assert gate["checks"]["locked_full_dimension_intervals_reproduced"] is True
    assert set(path.name for path in out_dir.iterdir()) == {
        "leave_one_endpoint_out.csv",
        "leave_one_category_out.csv",
        "influence_dimension_summary.csv",
        "table_supp_crossfit_influence_primary.tex",
        "crossfit_influence_sensitivity_gate.json",
    }


@requires_locked_crossfit_tables
def test_locked_full_intervals_reproduce_all_fourteen_rows_and_expected_support() -> None:
    source_dir = LOCKED_CROSSFIT_DIR
    endpoint_rows = pd.read_csv(source_dir / "crossfit_oracle_endpoint_summary.csv")
    reference_rows = pd.read_csv(source_dir / "crossfit_oracle_dimension_summary.csv")
    _, _, summary = build_influence_tables(endpoint_rows)
    audit = compare_reference_dimension_summary(summary, reference_rows)
    assert audit["passes"] is True
    assert audit["reference_row_count"] == 14
    assert audit["observed_row_count"] == 14
    assert audit["maximum_absolute_error"] <= 1e-11

    primary = summary.loc[summary["grouping_scheme"] == "bemis_murcko"].set_index(
        "dimension"
    )
    applicability = primary.loc["applicability_alignment"]
    assert applicability["loeo_ci_low_gt_zero_count"] == 22
    assert applicability["loeo_omission_count"] == 22
    assert applicability["loco_ci_low_gt_zero_count"] == 5
    assert applicability["loco_omission_count"] == 5
    ece_10bin = primary.loc["ece_10bin"]
    assert ece_10bin["loeo_ci_low_gt_zero_count"] == 4
    assert ece_10bin["loeo_omission_count"] == 13
    assert ece_10bin["loco_ci_low_gt_zero_count"] == 1
    assert ece_10bin["loco_omission_count"] == 4
