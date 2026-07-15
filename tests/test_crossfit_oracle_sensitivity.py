import typing
from pathlib import Path

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import numpy as np
import pytest

from experiments.analyze_crossfit_oracle_sensitivity import (
    bidirectional_crossfit_records,
    evaluate_crossfit_direction,
    exact_two_sided_sign_flip_pvalue,
    fast_aurc,
    holm_adjust,
    make_grouped_twofold_split,
    molecule_group_keys,
    summarize_dimensions,
)
import pandas as pd
from experiments.analyze_reliability_regret import (
    aurc_from_curve,
    risk_coverage_curve,
)


def test_grouped_split_has_no_group_leakage_and_is_deterministic():
    y_true = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    groups = np.asarray(["g1", "g1", "g2", "g2", "g3", "g4", "g5", "g6"])
    first_a, first_b, first_strata = make_grouped_twofold_split(
        y_true, groups, "classification", random_state=2026
    )
    second_a, second_b, second_strata = make_grouped_twofold_split(
        y_true, groups, "classification", random_state=2026
    )
    assert np.array_equal(first_a, second_a)
    assert np.array_equal(first_b, second_b)
    assert np.array_equal(first_strata, second_strata)
    assert not np.any(first_a & first_b)
    assert np.all(first_a | first_b)
    assert set(groups[first_a]).isdisjoint(set(groups[first_b]))


def test_signed_negative_difference_is_preserved_without_truncation():
    # Model index 1 wins on selection A but loses to the validation-selected
    # model (index 0) on evaluation B.  Signed D must therefore be negative.
    selection_losses = np.asarray([0.4, 0.1, 0.5, 0.6, 0.7])
    evaluation_losses = np.asarray([0.2, 0.8, 0.5, 0.6, 0.7])
    result = evaluate_crossfit_direction(
        selection_losses, evaluation_losses, selected_model_index=0
    )
    assert result["usable"] is True
    assert result["oracle_index"] == 1
    assert np.isclose(result["signed_d"], -0.6, rtol=0.0, atol=1e-15)


def test_bidirectional_records_are_ab_symmetric_and_complete():
    losses_a = np.asarray([0.5, 0.1, 0.4, 0.8, 0.9])
    losses_b = np.asarray([0.2, 0.7, 0.3, 0.8, 0.9])
    records = bidirectional_crossfit_records(
        losses_a, losses_b, selected_model_index=0
    )
    assert [record["direction"] for record in records] == ["A_to_B", "B_to_A"]
    assert records[0]["selection_fold"] == records[1]["evaluation_fold"] == "A"
    assert records[0]["evaluation_fold"] == records[1]["selection_fold"] == "B"
    assert records[0]["oracle_index"] == 1
    assert records[1]["oracle_index"] == 0
    assert records[0]["selected_selection_loss"] == losses_a[0]
    assert records[0]["selected_evaluation_loss"] == losses_b[0]
    assert records[1]["selected_selection_loss"] == losses_b[0]
    assert records[1]["selected_evaluation_loss"] == losses_a[0]


def test_exact_oracle_ties_follow_locked_model_order():
    losses_a = np.asarray([0.2, 0.2, 0.3, 0.4, 0.5])
    losses_b = np.asarray([0.5, 0.1, 0.3, 0.4, 0.5])
    result = evaluate_crossfit_direction(losses_a, losses_b, selected_model_index=2)
    assert result["oracle_index"] == 0


def test_exact_oracle_tie_average_sensitivity_uses_all_selection_fold_minima():
    selection_losses = np.asarray([0.1, 0.1, 0.3, 0.4, 0.5])
    evaluation_losses = np.asarray([0.8, 0.2, 0.5, 0.4, 0.3])
    result = evaluate_crossfit_direction(
        selection_losses, evaluation_losses, selected_model_index=2
    )
    assert result["oracle_index"] == 0
    assert result["oracle_tie_count"] == 2
    assert result["oracle_tie_indices"] == "0;1"
    assert np.isclose(result["signed_d"], -0.3)
    assert np.isclose(result["tie_averaged_oracle_evaluation_loss"], 0.5)
    assert np.isclose(result["tie_averaged_signed_d"], 0.0)


def test_scaffold_group_is_stereo_insensitive_and_acyclic_falls_back_to_identity():
    smiles = ["N[C@H](C)c1ccccc1", "N[C@@H](C)c1ccccc1", "CCO", "CCN"]
    canonical, scaffold = molecule_group_keys(smiles)
    assert canonical[0] != canonical[1]
    assert scaffold[0] == scaffold[1]
    assert scaffold[0].startswith("SCAFFOLD::")
    assert scaffold[2].startswith("ACYCLIC::")
    assert scaffold[3].startswith("ACYCLIC::")
    assert scaffold[2] != scaffold[3]


def test_fast_aurc_matches_existing_tie_safe_implementation():
    losses = np.asarray([0.8, 0.2, 0.4, 1.1, 0.6, 0.3], dtype=float)
    uncertainty = np.asarray([0.2, 0.1, 0.1, 0.4, 0.2, 0.3], dtype=float)
    sample_index = np.asarray([8, 2, 7, 1, 3, 5], dtype=int)
    expected = aurc_from_curve(
        risk_coverage_curve(losses, uncertainty, sample_index)
    )
    observed = fast_aurc(losses, uncertainty, sample_index)
    assert np.allclose(observed, expected, rtol=0.0, atol=1e-14)

    mask = np.asarray([True, False, True, True, False, True])
    expected_masked = aurc_from_curve(
        risk_coverage_curve(losses[mask], uncertainty[mask], sample_index[mask])
    )
    observed_masked = fast_aurc(losses, uncertainty, sample_index, mask=mask)
    assert np.allclose(observed_masked, expected_masked, rtol=0.0, atol=1e-14)


def test_exact_sign_flip_enumerates_all_assignments_and_holm_is_monotone():
    pvalue, assignments, extreme_count = exact_two_sided_sign_flip_pvalue(
        np.asarray([1.0, 2.0, 3.0])
    )
    assert assignments == 8
    assert extreme_count == 2
    assert np.isclose(pvalue, 0.25, rtol=0.0, atol=0.0)

    adjusted = holm_adjust(pd.Series([0.04, 0.01, 0.03], index=["c", "a", "b"]))
    assert np.allclose(
        adjusted.loc[["a", "b", "c"]].to_numpy(),
        np.asarray([0.03, 0.06, 0.06]),
        rtol=0.0,
        atol=1e-15,
    )


def test_locked_crossfit_exact_sign_flip_and_zero_redundancy():
    root = Path(__file__).resolve().parents[1]
    endpoint_path = (
        root
        / "results/tables/crossfit_oracle_sensitivity_final/"
        "crossfit_oracle_endpoint_summary.csv"
    )
    if not endpoint_path.is_file():
        pytest.skip("locked cross-fit result is absent from the code-only source staging")
    endpoint_rows = pd.read_csv(endpoint_path)
    summary = summarize_dimensions(endpoint_rows, bootstrap_reps=17, random_seed=20260715)
    primary = summary[summary["grouping_scheme"].eq("bemis_murcko")].set_index(
        "dimension"
    )
    expected = {
        "brier": (0.3310546875, 0.79025030136108398),
        "ece_5bin": (0.079345703125, 0.47607421875),
        "ece_10bin": (0.107666015625, 0.47607421875),
        "applicability_alignment": (
            0.00035953521728515625,
            0.0025167465209960938,
        ),
        "selective_relative_excess_aurc": (
            0.26341676712036133,
            0.79025030136108398,
        ),
        "selective_relative_excess_aurc_log_loss": (
            0.0849609375,
            0.47607421875,
        ),
        "conformal_coverage_error": (0.578125, 0.79025030136108398),
    }
    for dimension, (raw_p, holm_p) in expected.items():
        assert np.isclose(primary.loc[dimension, "sign_flip_p_exact"], raw_p)
        assert np.isclose(primary.loc[dimension, "holm_p_within_grouping_7"], holm_p)
    assert primary["holm_reject_0_05"].sum() == 1
    assert bool(primary.loc["applicability_alignment", "holm_reject_0_05"])

    brier_values = endpoint_rows[
        endpoint_rows["grouping_scheme"].eq("bemis_murcko")
        & endpoint_rows["dimension"].eq("brier")
    ]["mean_signed_d"].to_numpy(dtype=float)
    with_zeros = exact_two_sided_sign_flip_pvalue(brier_values)
    without_zeros = exact_two_sided_sign_flip_pvalue(brier_values[brier_values != 0])
    assert np.isclose(with_zeros[0], without_zeros[0], rtol=0.0, atol=0.0)
    assert with_zeros[1] == 4 * without_zeros[1]
    assert with_zeros[2] == 4 * without_zeros[2]


def test_supplementary_crossfit_table_distinguishes_heuristic_threshold_from_validation_tolerance():
    root = Path(__file__).resolve().parents[1]
    table_path = (
        root
        / "results/tables/manuscript_final/"
        "table_supp_crossfit_oracle_summary.tex"
    )
    if not table_path.is_file():
        pytest.skip("locked supplementary cross-fit table is absent")
    table = table_path.read_text(encoding="utf-8")
    assert "Above heuristic threshold" in table
    assert "Above tolerance" not in table
