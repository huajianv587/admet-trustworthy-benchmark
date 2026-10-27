"""Zero-retraining cross-fitted reliability-oracle sensitivity analysis.

The validation-selected model for each endpoint is frozen from the formal
reliability-regret analysis.  Repeated grouped two-fold splits are then made
inside the locked official test set.  For each split, fold A selects a
dimension-specific candidate oracle and fold B evaluates the signed difference

    D = L_B(validation-selected model) - L_B(oracle selected on A),

followed by the reverse B-to-A direction.  Negative D values are scientifically
meaningful and are never truncated.  Repeated folds are averaged within each
endpoint before endpoints are bootstrapped.

Because seven reliability dimensions are summarized for each grouping, the
endpoint means also receive an exact two-sided sign-flip diagnostic. Holm
adjustment is applied across all seven dimensions within each grouping. This
post-hoc diagnostic is descriptive rather than confirmatory, especially
because endpoint test sets can share molecules.

The primary grouping uses Bemis--Murcko scaffolds, with acyclic molecules
falling back to canonical molecular identity.  Canonical molecular identity is
reported as a secondary grouping sensitivity.  All models and dimensions share
the same split mask for a task, scheme, and repeat.
"""

from __future__ import annotations

# The portable Python 3.9 runtime in this workspace contains the dataclasses
# backport.  Its pandas import path expects typing._ClassVar, which CPython 3.9
# no longer exposes.  Apply the narrow compatibility alias before importing
# pandas/sklearn; it is a no-op on runtimes that already expose the attribute.
import typing

typing._ClassVar = getattr(typing, "_ClassVar", typing.ClassVar)  # type: ignore[attr-defined]

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import (  # noqa: E402
    TDC_TASK_META,
    infer_label_column,
    infer_smiles_column,
)
from admet_benchmark.smiles_compat import model_input_smiles  # noqa: E402
from experiments.analyze_reliability_regret import (  # noqa: E402
    EXPECTED_TASKS,
    FINAL_EXPECTED_MODELS,
    FINAL_EXPECTED_SEEDS,
    ModelEnsemble,
    RunPrediction,
    audit_ensemble_test_alignment,
    audit_run_sample_alignment,
    build_ensembles,
    conformal_quantile,
    load_runs,
    sample_fingerprint,
    sample_rows_match,
)


GROUPING_SCHEMES = ("bemis_murcko", "canonical_identity")
PRIMARY_GROUPING_SCHEME = "bemis_murcko"
DEFAULT_REPEATS = 500
DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_RANDOM_SEED = 20260715
MAX_EXACT_SIGN_FLIP_ENDPOINTS = 24

DIMENSION_ORDER = (
    "brier",
    "ece_5bin",
    "ece_10bin",
    "applicability_alignment",
    "selective_relative_excess_aurc",
    "selective_relative_excess_aurc_log_loss",
    "conformal_coverage_error",
)

DIMENSION_META = {
    "brier": {
        "population": "classification",
        "evidence_tier": "primary",
        "practical_tolerance": 0.005,
        "definition": "mean Brier loss",
    },
    "ece_5bin": {
        "population": "classification",
        "evidence_tier": "sensitivity",
        "practical_tolerance": 0.010,
        "definition": "five equal-width-bin expected calibration error",
    },
    "ece_10bin": {
        "population": "classification",
        "evidence_tier": "primary",
        "practical_tolerance": 0.010,
        "definition": "ten equal-width-bin expected calibration error",
    },
    "applicability_alignment": {
        "population": "all",
        "evidence_tier": "primary",
        "practical_tolerance": 0.050,
        "definition": "negative Spearman(max Tanimoto, negative absolute error)",
    },
    "selective_relative_excess_aurc": {
        "population": "all",
        "evidence_tier": "primary",
        "practical_tolerance": 0.010,
        "definition": (
            "tie-safe excess AURC divided by mean primary sample loss; "
            "Brier for classification and absolute error for regression"
        ),
    },
    "selective_relative_excess_aurc_log_loss": {
        "population": "classification",
        "evidence_tier": "secondary",
        "practical_tolerance": 0.010,
        "definition": "tie-safe log-loss excess AURC divided by mean log loss",
    },
    "conformal_coverage_error": {
        "population": "regression",
        "evidence_tier": "descriptive",
        "practical_tolerance": 0.010,
        "definition": (
            "absolute validation-residual empirical coverage error from nominal "
            "coverage, using the split-conformal quantile formula without a standard "
            "finite-sample guarantee"
        ),
    },
}


@dataclass(frozen=True)
class AURCCache:
    """Pre-sorted arrays for fast masked AURC evaluation."""

    losses: np.ndarray
    uncertainty: np.ndarray
    uncertainty_order: np.ndarray
    oracle_order: np.ndarray
    finite: np.ndarray

    @classmethod
    def build(
        cls,
        losses: np.ndarray,
        uncertainty: np.ndarray,
        sample_index: np.ndarray | None = None,
    ) -> "AURCCache":
        losses = np.asarray(losses, dtype=float)
        uncertainty = np.asarray(uncertainty, dtype=float)
        if losses.shape != uncertainty.shape:
            raise ValueError("losses and uncertainty must have identical shape")
        if sample_index is None:
            sample_index = np.arange(len(losses), dtype=int)
        sample_index = np.asarray(sample_index)
        if sample_index.shape != losses.shape:
            raise ValueError("sample_index must match losses")
        finite = np.isfinite(losses) & np.isfinite(uncertainty)
        uncertainty_order = np.lexsort((sample_index, uncertainty))
        oracle_order = np.lexsort((sample_index, losses))
        return cls(losses, uncertainty, uncertainty_order, oracle_order, finite)

    def evaluate(self, mask: np.ndarray | None = None) -> tuple[float, float, float]:
        if mask is None:
            mask = np.ones(len(self.losses), dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.losses.shape:
            raise ValueError("AURC mask must match cached arrays")
        keep = mask & self.finite
        uncertainty_order = self.uncertainty_order[keep[self.uncertainty_order]]
        oracle_order = self.oracle_order[keep[self.oracle_order]]
        return _aurc_from_preordered(
            self.losses,
            self.uncertainty,
            uncertainty_order,
            oracle_order,
        )


@dataclass
class PreparedTask:
    task: str
    task_type: str
    model_order: tuple[str, ...]
    sample_index: np.ndarray
    smiles: np.ndarray
    y_true: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    max_tanimoto: np.ndarray
    absolute_error: np.ndarray
    primary_aurc: tuple[AURCCache, ...]
    log_loss_aurc: tuple[AURCCache, ...] | None
    conformal_covered: np.ndarray | None
    canonical_groups: np.ndarray
    scaffold_groups: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_records(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def stable_random_state(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


@lru_cache(maxsize=None)
def canonical_identity(smiles: str) -> str:
    """Return an isomeric canonical identity, applying audited parser repair."""
    from rdkit import Chem

    source = model_input_smiles(str(smiles))
    molecule = Chem.MolFromSmiles(source)
    if molecule is None:
        raise ValueError(f"RDKit could not parse locked SMILES: {smiles!r}")
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


@lru_cache(maxsize=None)
def bemis_murcko_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    source = model_input_smiles(str(smiles))
    molecule = Chem.MolFromSmiles(source)
    if molecule is None:
        raise ValueError(f"RDKit could not parse locked SMILES: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    # Standard Bemis--Murcko grouping is non-isomeric: stereoisomers sharing
    # the same graph scaffold must remain in the same fold.
    return str(Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False))


def molecule_group_keys(smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    canonical = np.asarray([canonical_identity(str(value)) for value in smiles], dtype=object)
    scaffolds: list[str] = []
    for source, identity in zip(smiles, canonical):
        scaffold = bemis_murcko_scaffold(str(source))
        # Treating every acyclic molecule as one empty scaffold would create an
        # artificial mega-group.  The locked policy therefore falls back to
        # canonical molecular identity only when Murcko returns an empty graph.
        key = f"SCAFFOLD::{scaffold}" if scaffold else f"ACYCLIC::{identity}"
        scaffolds.append(key)
    return canonical, np.asarray(scaffolds, dtype=object)


def regression_quantile_strata(y_true: np.ndarray, n_bins: int = 10) -> np.ndarray:
    values = np.asarray(y_true, dtype=float)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Regression stratification labels must be finite")
    if len(values) < 2 or np.unique(values).size < 2:
        return np.zeros(len(values), dtype=int)
    bins = min(int(n_bins), len(values), int(np.unique(values).size))
    categorized = pd.qcut(pd.Series(values), q=bins, labels=False, duplicates="drop")
    result = categorized.to_numpy(dtype=float)
    if not np.isfinite(result).all() or np.unique(result).size < 2:
        return np.zeros(len(values), dtype=int)
    return result.astype(int)


def make_grouped_twofold_split(
    y_true: np.ndarray,
    groups: np.ndarray,
    task_type: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one deterministic stratified group-disjoint A/B split."""
    y_true = np.asarray(y_true, dtype=float)
    groups = np.asarray(groups, dtype=object)
    if len(y_true) != len(groups):
        raise ValueError("labels and groups must have the same length")
    if np.unique(groups).size < 2:
        raise ValueError("At least two distinct groups are required for two-fold splitting")
    if task_type == "classification":
        strata = y_true.astype(int)
        if not np.allclose(strata, y_true) or not set(np.unique(strata)).issubset({0, 1}):
            raise ValueError("Classification labels must be binary")
    elif task_type == "regression":
        strata = regression_quantile_strata(y_true)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    splitter = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=int(random_state),
    )
    a_indices, b_indices = next(
        splitter.split(np.zeros(len(y_true), dtype=float), strata, groups)
    )
    a_mask = np.zeros(len(y_true), dtype=bool)
    b_mask = np.zeros(len(y_true), dtype=bool)
    a_mask[a_indices] = True
    b_mask[b_indices] = True
    if not a_mask.any() or not b_mask.any() or np.any(a_mask & b_mask):
        raise AssertionError("Grouped two-fold splitter produced invalid sample masks")
    if not np.all(a_mask | b_mask):
        raise AssertionError("Grouped two-fold splitter did not cover every sample")
    overlap = set(groups[a_mask]).intersection(set(groups[b_mask]))
    if overlap:
        raise AssertionError(f"Group leakage detected: {len(overlap)} shared groups")
    return a_mask, b_mask, strata


def _expected_tie_cumulative_loss(
    losses: np.ndarray,
    uncertainty: np.ndarray,
    uncertainty_order: np.ndarray,
) -> np.ndarray:
    """Vectorized expected cumulative loss under random within-tie order."""
    ordered_losses = losses[uncertainty_order]
    ordered_uncertainty = uncertainty[uncertainty_order]
    n = len(ordered_losses)
    if n == 0:
        return np.asarray([], dtype=float)
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_uncertainty) != 0) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    sums = np.add.reduceat(ordered_losses, starts)
    means = sums / counts
    cumulative_before = np.r_[0.0, np.cumsum(sums[:-1])]
    within_group = np.arange(n) - np.repeat(starts, counts) + 1
    return np.repeat(cumulative_before, counts) + within_group * np.repeat(means, counts)


def _aurc_from_preordered(
    losses: np.ndarray,
    uncertainty: np.ndarray,
    uncertainty_order: np.ndarray,
    oracle_order: np.ndarray,
) -> tuple[float, float, float]:
    n = len(uncertainty_order)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    retained = np.arange(1, n + 1, dtype=float)
    expected_cumulative = _expected_tie_cumulative_loss(
        losses, uncertainty, uncertainty_order
    )
    aurc = float(np.mean(expected_cumulative / retained))
    oracle = float(np.mean(np.cumsum(losses[oracle_order]) / retained))
    # This clamp is part of the metric definition and matches the existing
    # implementation; the cross-fitted signed D below is never clamped.
    excess = max(0.0, aurc - oracle)
    return aurc, oracle, excess


def fast_aurc(
    losses: np.ndarray,
    uncertainty: np.ndarray,
    sample_index: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Tie-safe AURC equivalent to the existing curve implementation.

    The implementation has no Python loop over uncertainty ties.  Repeated
    cross-fit calls use :class:`AURCCache`, which additionally avoids re-sorting
    full arrays for every half-fold.
    """
    return AURCCache.build(losses, uncertainty, sample_index).evaluate(mask)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    average = (starts + 1 + ends) / 2.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(average, counts)
    return ranks


def fast_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 3:
        return float("nan")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    denominator = math.sqrt(
        float(np.dot(left_rank, left_rank) * np.dot(right_rank, right_rank))
    )
    if denominator <= 0:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)


def ece_loss(y_true: np.ndarray, probability: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if len(y_true) == 0:
        return float("nan")
    bin_index = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins).astype(float)
    sum_y = np.bincount(bin_index, weights=y_true, minlength=n_bins)
    sum_p = np.bincount(bin_index, weights=probability, minlength=n_bins)
    occupied = counts > 0
    return float(
        np.sum(np.abs(sum_y[occupied] - sum_p[occupied])) / len(y_true)
    )


def _relative_excess_aurc(cache: AURCCache, mask: np.ndarray) -> float:
    _, _, excess = cache.evaluate(mask)
    valid = mask & cache.finite
    if not valid.any():
        return float("nan")
    mean_loss = float(np.mean(cache.losses[valid]))
    return float(excess / max(mean_loss, 1e-12))


def prepare_conformal_coverage(
    runs: Mapping[tuple[str, str, int], RunPrediction],
    ensembles: Sequence[ModelEnsemble],
    alpha: float,
) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for ensemble in ensembles:
        if ensemble.task_type != "regression":
            continue
        coverage_parts: list[np.ndarray] = []
        reference = ensemble.test.sort_values("sample_index", kind="mergesort")
        for seed in ensemble.seeds:
            run = runs[(ensemble.task, ensemble.model, int(seed))]
            test = run.test.sort_values("sample_index", kind="mergesort")
            if not sample_rows_match(test, reference):
                raise ValueError(
                    f"Conformal test rows disagree for {(ensemble.task, ensemble.model, seed)}"
                )
            residual = np.abs(
                run.valid["y_true"].to_numpy(dtype=float)
                - run.valid["y_pred"].to_numpy(dtype=float)
            )
            quantile = conformal_quantile(residual, alpha=alpha)
            covered = (
                np.abs(
                    test["y_true"].to_numpy(dtype=float)
                    - test["y_pred"].to_numpy(dtype=float)
                )
                <= quantile
            ).astype(float)
            coverage_parts.append(covered)
        result[(ensemble.task, ensemble.model)] = np.mean(
            np.vstack(coverage_parts), axis=0
        )
    return result


def prepare_task(
    task: str,
    ensembles: Sequence[ModelEnsemble],
    conformal: Mapping[tuple[str, str], np.ndarray],
) -> PreparedTask:
    lookup = {item.model: item for item in ensembles if item.task == task}
    if set(lookup) != set(FINAL_EXPECTED_MODELS):
        raise ValueError(
            f"{task}: expected exactly {FINAL_EXPECTED_MODELS}, found {sorted(lookup)}"
        )
    ordered = [lookup[model] for model in FINAL_EXPECTED_MODELS]
    reference = ordered[0].test.sort_values("sample_index", kind="mergesort")
    if any(not sample_rows_match(item.test, reference) for item in ordered):
        raise ValueError(f"{task}: model test rows are not aligned")
    sample_index = reference["sample_index"].to_numpy(dtype=int)
    smiles = reference["smiles"].astype(str).to_numpy()
    y_true = reference["y_true"].to_numpy(dtype=float)
    prediction = np.vstack(
        [
            item.test.sort_values("sample_index", kind="mergesort")["prediction"].to_numpy(
                dtype=float
            )
            for item in ordered
        ]
    )
    uncertainty = np.vstack(
        [
            item.test.sort_values("sample_index", kind="mergesort")["uncertainty"].to_numpy(
                dtype=float
            )
            for item in ordered
        ]
    )
    max_tanimoto = np.vstack(
        [
            item.test.sort_values("sample_index", kind="mergesort")["max_tanimoto"].to_numpy(
                dtype=float
            )
            for item in ordered
        ]
    )
    absolute_error = np.abs(prediction - y_true[None, :])
    task_type = ordered[0].task_type
    if task_type == "classification":
        probability = np.clip(prediction, 1e-12, 1 - 1e-12)
        primary_losses = (probability - y_true[None, :]) ** 2
        log_losses = -(
            y_true[None, :] * np.log(probability)
            + (1 - y_true[None, :]) * np.log(1 - probability)
        )
        log_loss_aurc = tuple(
            AURCCache.build(log_losses[index], uncertainty[index], sample_index)
            for index in range(len(ordered))
        )
        conformal_covered = None
    else:
        primary_losses = absolute_error
        log_loss_aurc = None
        conformal_covered = np.vstack(
            [conformal[(task, model)] for model in FINAL_EXPECTED_MODELS]
        )
    primary_aurc = tuple(
        AURCCache.build(primary_losses[index], uncertainty[index], sample_index)
        for index in range(len(ordered))
    )
    canonical_groups, scaffold_groups = molecule_group_keys(smiles.tolist())
    return PreparedTask(
        task=task,
        task_type=task_type,
        model_order=FINAL_EXPECTED_MODELS,
        sample_index=sample_index,
        smiles=smiles,
        y_true=y_true,
        prediction=prediction,
        uncertainty=uncertainty,
        max_tanimoto=max_tanimoto,
        absolute_error=absolute_error,
        primary_aurc=primary_aurc,
        log_loss_aurc=log_loss_aurc,
        conformal_covered=conformal_covered,
        canonical_groups=canonical_groups,
        scaffold_groups=scaffold_groups,
    )


def compute_fold_losses(
    prepared: PreparedTask,
    mask: np.ndarray,
    target_coverage: float,
) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != prepared.y_true.shape or not mask.any():
        raise ValueError("Fold mask is empty or misaligned")
    model_count = len(prepared.model_order)
    result: dict[str, np.ndarray] = {}
    if prepared.task_type == "classification":
        y_true = prepared.y_true[mask]
        probability = np.clip(prepared.prediction[:, mask], 1e-12, 1 - 1e-12)
        result["brier"] = np.mean((probability - y_true[None, :]) ** 2, axis=1)
        result["ece_5bin"] = np.asarray(
            [ece_loss(y_true, probability[index], 5) for index in range(model_count)]
        )
        result["ece_10bin"] = np.asarray(
            [ece_loss(y_true, probability[index], 10) for index in range(model_count)]
        )
    result["applicability_alignment"] = np.asarray(
        [
            -fast_spearman(
                prepared.max_tanimoto[index, mask],
                -prepared.absolute_error[index, mask],
            )
            for index in range(model_count)
        ]
    )
    result["selective_relative_excess_aurc"] = np.asarray(
        [_relative_excess_aurc(cache, mask) for cache in prepared.primary_aurc]
    )
    if prepared.log_loss_aurc is not None:
        result["selective_relative_excess_aurc_log_loss"] = np.asarray(
            [_relative_excess_aurc(cache, mask) for cache in prepared.log_loss_aurc]
        )
    if prepared.conformal_covered is not None:
        result["conformal_coverage_error"] = np.abs(
            np.mean(prepared.conformal_covered[:, mask], axis=1) - target_coverage
        )
    return result


def select_candidate_oracle_indices(losses: np.ndarray) -> np.ndarray:
    """Return every finite exact minimum in locked model order."""
    losses = np.asarray(losses, dtype=float)
    if len(losses) != len(FINAL_EXPECTED_MODELS):
        raise ValueError("Oracle loss vector does not match FINAL_EXPECTED_MODELS")
    finite = np.isfinite(losses)
    if not finite.any():
        return np.asarray([], dtype=int)
    best = float(np.min(losses[finite]))
    return np.flatnonzero(finite & (losses == best)).astype(int)


def select_candidate_oracle(losses: np.ndarray) -> int | None:
    """Select the first exact minimum in ``FINAL_EXPECTED_MODELS`` order."""
    indices = select_candidate_oracle_indices(losses)
    return int(indices[0]) if len(indices) else None


def evaluate_crossfit_direction(
    selection_losses: np.ndarray,
    evaluation_losses: np.ndarray,
    selected_model_index: int,
) -> dict[str, object]:
    """Evaluate one direction without consulting the evaluation fold to select."""
    selection_losses = np.asarray(selection_losses, dtype=float)
    evaluation_losses = np.asarray(evaluation_losses, dtype=float)
    oracle_indices = select_candidate_oracle_indices(selection_losses)
    if not len(oracle_indices):
        return {
            "usable": False,
            "reason": "no_finite_candidate_on_selection_fold",
            "oracle_index": None,
            "oracle_tie_count": 0,
            "oracle_tie_indices": "",
            "signed_d": float("nan"),
            "tie_average_usable": False,
            "tie_averaged_signed_d": float("nan"),
        }
    oracle_index = int(oracle_indices[0])
    selected_evaluation = float(evaluation_losses[selected_model_index])
    oracle_evaluation = float(evaluation_losses[oracle_index])
    usable = np.isfinite(selected_evaluation) and np.isfinite(oracle_evaluation)
    # Deliberately no max(0, ...): negative values must survive.
    signed_d = selected_evaluation - oracle_evaluation if usable else float("nan")
    tied_evaluation = evaluation_losses[oracle_indices]
    tie_average_usable = bool(
        np.isfinite(selected_evaluation) and np.isfinite(tied_evaluation).all()
    )
    tie_averaged_oracle_evaluation = (
        float(np.mean(tied_evaluation)) if tie_average_usable else float("nan")
    )
    tie_averaged_signed_d = (
        selected_evaluation - tie_averaged_oracle_evaluation
        if tie_average_usable
        else float("nan")
    )
    return {
        "usable": bool(usable),
        "reason": "complete" if usable else "nonfinite_evaluation_loss",
        "oracle_index": int(oracle_index),
        "oracle_tie_count": int(len(oracle_indices)),
        "oracle_tie_indices": ";".join(str(int(index)) for index in oracle_indices),
        "signed_d": float(signed_d),
        "tie_average_usable": tie_average_usable,
        "tie_averaged_oracle_evaluation_loss": tie_averaged_oracle_evaluation,
        "tie_averaged_signed_d": float(tie_averaged_signed_d),
        "fixed_minus_tie_averaged_signed_d": (
            float(signed_d - tie_averaged_signed_d)
            if usable and tie_average_usable
            else float("nan")
        ),
        "selected_selection_loss": float(selection_losses[selected_model_index]),
        "selected_evaluation_loss": selected_evaluation,
        "oracle_selection_loss": float(selection_losses[oracle_index]),
        "oracle_evaluation_loss": oracle_evaluation,
        "finite_candidate_count": int(np.isfinite(selection_losses).sum()),
    }


def bidirectional_crossfit_records(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    selected_model_index: int,
) -> list[dict[str, object]]:
    a_to_b = evaluate_crossfit_direction(losses_a, losses_b, selected_model_index)
    b_to_a = evaluate_crossfit_direction(losses_b, losses_a, selected_model_index)
    return [
        {**a_to_b, "direction": "A_to_B", "selection_fold": "A", "evaluation_fold": "B"},
        {**b_to_a, "direction": "B_to_A", "selection_fold": "B", "evaluation_fold": "A"},
    ]


def _counts_json(values: np.ndarray) -> str:
    unique, counts = np.unique(values, return_counts=True)
    payload = {str(value): int(count) for value, count in zip(unique, counts)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _split_hash(values: np.ndarray) -> str:
    return sha256_records([int(value) for value in np.sort(np.asarray(values, dtype=int))])


def _group_hash(values: np.ndarray) -> str:
    return sha256_records(sorted(str(value) for value in set(values)))


def load_locked_selections(
    path: Path,
    ensembles: Sequence[ModelEnsemble],
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(path)
    required = {"task", "validation_selected_model", "model_pool"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Selection file lacks {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["task"] = frame["task"].astype(str).str.lower()
    frame["validation_selected_model"] = (
        frame["validation_selected_model"].astype(str).str.lower()
    )
    if frame["task"].duplicated().any() or set(frame["task"]) != set(EXPECTED_TASKS):
        raise ValueError("Formal selection file must contain each locked endpoint once")
    expected_pool = set(FINAL_EXPECTED_MODELS)
    for row in frame.itertuples(index=False):
        pool = set(str(row.model_pool).split(";"))
        if pool != expected_pool or row.validation_selected_model not in expected_pool:
            raise ValueError(f"Invalid fixed model pool for {row.task}")
    # Recompute only the exact validation winner from the currently loaded
    # ensembles, ensuring the frozen table belongs to these prediction inputs.
    by_task: dict[str, list[ModelEnsemble]] = {}
    for item in ensembles:
        by_task.setdefault(item.task, []).append(item)
    selected = dict(zip(frame["task"], frame["validation_selected_model"]))
    for task, items in by_task.items():
        means = {
            item.model: float(np.mean(list(item.validation_scores.values())))
            for item in items
        }
        recomputed = sorted(means, key=lambda model: (-means[model], model))[0]
        if selected.get(task) != recomputed:
            raise ValueError(
                f"Frozen validation selection mismatch for {task}: "
                f"file={selected.get(task)}, recomputed={recomputed}"
            )
    return frame.sort_values("task").reset_index(drop=True), selected


def run_crossfit(
    prepared_tasks: Sequence[PreparedTask],
    selected_models: Mapping[str, str],
    repeats: int,
    target_coverage: float,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prepared in sorted(prepared_tasks, key=lambda item: item.task):
        selected_model = selected_models[prepared.task]
        selected_index = prepared.model_order.index(selected_model)
        full_losses = compute_fold_losses(
            prepared,
            np.ones(len(prepared.y_true), dtype=bool),
            target_coverage,
        )
        full_oracles = {
            dimension: select_candidate_oracle(values)
            for dimension, values in full_losses.items()
        }
        for scheme in GROUPING_SCHEMES:
            groups = (
                prepared.scaffold_groups
                if scheme == "bemis_murcko"
                else prepared.canonical_groups
            )
            for repeat in range(repeats):
                split_seed = stable_random_state(
                    random_seed, prepared.task, scheme, repeat
                )
                a_mask, b_mask, strata = make_grouped_twofold_split(
                    prepared.y_true,
                    groups,
                    prepared.task_type,
                    split_seed,
                )
                loss_a = compute_fold_losses(prepared, a_mask, target_coverage)
                loss_b = compute_fold_losses(prepared, b_mask, target_coverage)
                dimensions = [
                    dimension
                    for dimension in DIMENSION_ORDER
                    if dimension in loss_a and dimension in loss_b
                ]
                a_indices = prepared.sample_index[a_mask]
                b_indices = prepared.sample_index[b_mask]
                a_sample_hash = _split_hash(a_indices)
                b_sample_hash = _split_hash(b_indices)
                a_group_hash = _group_hash(groups[a_mask])
                b_group_hash = _group_hash(groups[b_mask])
                group_overlap = set(groups[a_mask]).intersection(set(groups[b_mask]))
                fold_id = sha256_records(
                    [prepared.task, scheme, repeat, a_sample_hash, b_sample_hash]
                )
                split_base = {
                    "task": prepared.task,
                    "task_type": prepared.task_type,
                    "grouping_scheme": scheme,
                    "scheme_role": (
                        "primary" if scheme == PRIMARY_GROUPING_SCHEME else "secondary"
                    ),
                    "repeat": int(repeat),
                    "split_random_state": int(split_seed),
                    "fold_id": fold_id,
                    "test_n": len(prepared.y_true),
                    "a_n": int(a_mask.sum()),
                    "b_n": int(b_mask.sum()),
                    "a_group_count": int(np.unique(groups[a_mask]).size),
                    "b_group_count": int(np.unique(groups[b_mask]).size),
                    "group_overlap_count": len(group_overlap),
                    "a_sample_index_sha256": a_sample_hash,
                    "b_sample_index_sha256": b_sample_hash,
                    "a_group_sha256": a_group_hash,
                    "b_group_sha256": b_group_hash,
                    "a_stratum_counts": _counts_json(strata[a_mask]),
                    "b_stratum_counts": _counts_json(strata[b_mask]),
                    "validation_selected_model": selected_model,
                    "model_order": ";".join(prepared.model_order),
                }
                for dimension in dimensions:
                    full_oracle_index = full_oracles[dimension]
                    for record in bidirectional_crossfit_records(
                        loss_a[dimension], loss_b[dimension], selected_index
                    ):
                        oracle_index = record.get("oracle_index")
                        rows.append(
                            {
                                **split_base,
                                "dimension": dimension,
                                "evidence_tier": DIMENSION_META[dimension][
                                    "evidence_tier"
                                ],
                                "practical_tolerance": DIMENSION_META[dimension][
                                    "practical_tolerance"
                                ],
                                **record,
                                "candidate_oracle_model": (
                                    prepared.model_order[int(oracle_index)]
                                    if oracle_index is not None
                                    else ""
                                ),
                                "full_test_oracle_model": (
                                    prepared.model_order[int(full_oracle_index)]
                                    if full_oracle_index is not None
                                    else ""
                                ),
                                "candidate_matches_full_test_oracle": bool(
                                    oracle_index is not None
                                    and full_oracle_index is not None
                                    and int(oracle_index) == int(full_oracle_index)
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def summarize_endpoints(fold_rows: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, task, dimension), group in fold_rows.groupby(
        ["grouping_scheme", "task", "dimension"], sort=True
    ):
        usable = group[group["usable"] & np.isfinite(group["signed_d"])].copy()
        repeat_means = usable.groupby("repeat", sort=True)["signed_d"].mean()
        tie_usable = group[
            group["tie_average_usable"]
            & np.isfinite(group["tie_averaged_signed_d"])
        ].copy()
        tie_repeat_means = tie_usable.groupby("repeat", sort=True)[
            "tie_averaged_signed_d"
        ].mean()
        oracle_counts = usable["candidate_oracle_model"].value_counts().to_dict()
        probabilities = np.asarray(
            [oracle_counts.get(model, 0) / max(len(usable), 1) for model in FINAL_EXPECTED_MODELS],
            dtype=float,
        )
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else float("nan")
        normalized_entropy = (
            entropy / math.log(len(FINAL_EXPECTED_MODELS))
            if np.isfinite(entropy) and len(FINAL_EXPECTED_MODELS) > 1
            else float("nan")
        )
        values = repeat_means.to_numpy(dtype=float)
        tolerance = float(DIMENSION_META[dimension]["practical_tolerance"])
        row: dict[str, object] = {
            "grouping_scheme": scheme,
            "scheme_role": (
                "primary" if scheme == PRIMARY_GROUPING_SCHEME else "secondary"
            ),
            "task": task,
            "task_type": str(group["task_type"].iloc[0]),
            "dimension": dimension,
            "evidence_tier": DIMENSION_META[dimension]["evidence_tier"],
            "validation_selected_model": str(
                group["validation_selected_model"].iloc[0]
            ),
            "full_test_oracle_model": str(group["full_test_oracle_model"].iloc[0]),
            "expected_repeats": int(repeats),
            "complete_repeats": int(len(values)),
            "expected_direction_rows": int(2 * repeats),
            "usable_direction_rows": int(len(usable)),
            "tie_average_usable_direction_rows": int(len(tie_usable)),
            "mean_signed_d": float(np.mean(values)) if len(values) else float("nan"),
            "mean_tie_averaged_signed_d": (
                float(tie_repeat_means.mean())
                if len(tie_repeat_means)
                else float("nan")
            ),
            "mean_fixed_minus_tie_averaged_signed_d": (
                float(np.mean(values) - tie_repeat_means.mean())
                if len(values) and len(tie_repeat_means)
                else float("nan")
            ),
            "oracle_tie_direction_count": int(
                np.sum(usable["oracle_tie_count"].to_numpy(dtype=int) > 1)
            ),
            "oracle_tie_direction_fraction": (
                float(np.mean(usable["oracle_tie_count"].to_numpy(dtype=int) > 1))
                if len(usable)
                else float("nan")
            ),
            "median_repeat_signed_d": (
                float(np.median(values)) if len(values) else float("nan")
            ),
            "repeat_signed_d_q025": (
                float(np.quantile(values, 0.025)) if len(values) else float("nan")
            ),
            "repeat_signed_d_q975": (
                float(np.quantile(values, 0.975)) if len(values) else float("nan")
            ),
            "mean_absolute_signed_d": (
                float(np.mean(np.abs(values))) if len(values) else float("nan")
            ),
            "positive_repeat_fraction": (
                float(np.mean(values > 0)) if len(values) else float("nan")
            ),
            "negative_repeat_fraction": (
                float(np.mean(values < 0)) if len(values) else float("nan")
            ),
            "zero_repeat_fraction": (
                float(np.mean(values == 0)) if len(values) else float("nan")
            ),
            "practical_tolerance": tolerance,
            "repeat_fraction_above_practical_tolerance": (
                float(np.mean(values > tolerance)) if len(values) else float("nan")
            ),
            "candidate_full_oracle_agreement_fraction": (
                float(usable["candidate_matches_full_test_oracle"].mean())
                if len(usable)
                else float("nan")
            ),
            "candidate_selection_entropy_nats": entropy,
            "candidate_selection_entropy_normalized": normalized_entropy,
            "minimum_direction_signed_d": (
                float(usable["signed_d"].min()) if len(usable) else float("nan")
            ),
            "maximum_direction_signed_d": (
                float(usable["signed_d"].max()) if len(usable) else float("nan")
            ),
        }
        for model in FINAL_EXPECTED_MODELS:
            row[f"candidate_oracle_count_{model}"] = int(oracle_counts.get(model, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_interval(
    values: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_two_sided_sign_flip_pvalue(
    values: np.ndarray,
    *,
    chunk_size: int = 262_144,
) -> tuple[float, int, int]:
    """Enumerate every endpoint-sign assignment for a two-sided mean test.

    The largest analysis family has 22 endpoint means, so all ``2**22`` sign
    assignments are tractable. Chunking bounds peak memory without replacing
    the exact enumeration with Monte Carlo sampling.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), 0, 0
    if len(values) > MAX_EXACT_SIGN_FLIP_ENDPOINTS:
        raise ValueError(
            "Exact sign-flip enumeration supports at most "
            f"{MAX_EXACT_SIGN_FLIP_ENDPOINTS} values"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    assignment_count = 1 << len(values)
    observed = abs(float(np.sum(values, dtype=np.float64)))
    threshold = np.nextafter(observed, -np.inf)
    bit_positions = np.arange(len(values), dtype=np.uint64)
    extreme_count = 0
    for start in range(0, assignment_count, chunk_size):
        stop = min(start + chunk_size, assignment_count)
        codes = np.arange(start, stop, dtype=np.uint64)
        negative = ((codes[:, None] >> bit_positions[None, :]) & 1).astype(
            np.float64
        )
        signed_sums = np.sum((1.0 - 2.0 * negative) * values, axis=1)
        extreme_count += int(np.count_nonzero(np.abs(signed_sums) >= threshold))
    return (
        float(extreme_count / assignment_count),
        int(assignment_count),
        int(extreme_count),
    )


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Return monotone Holm-adjusted values while preserving input indices."""
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().astype(float).sort_values(kind="mergesort")
    running_max = 0.0
    family_size = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        candidate = min(1.0, (family_size - rank) * float(value))
        running_max = max(running_max, candidate)
        adjusted.loc[index] = running_max
    return adjusted


def summarize_dimensions(
    endpoint_rows: pd.DataFrame,
    bootstrap_reps: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scheme, dimension), group in endpoint_rows.groupby(
        ["grouping_scheme", "dimension"], sort=True
    ):
        values = group["mean_signed_d"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        tie_values = group["mean_tie_averaged_signed_d"].to_numpy(dtype=float)
        tie_values = tie_values[np.isfinite(tie_values)]
        rng = np.random.default_rng(
            stable_random_state(random_seed, "endpoint_bootstrap", scheme, dimension)
        )
        ci_low, ci_high = bootstrap_mean_interval(values, bootstrap_reps, rng)
        tie_rng = np.random.default_rng(
            stable_random_state(
                random_seed,
                "endpoint_tie_average_bootstrap",
                scheme,
                dimension,
            )
        )
        tie_ci_low, tie_ci_high = bootstrap_mean_interval(
            tie_values, bootstrap_reps, tie_rng
        )
        (
            sign_flip_p,
            sign_flip_assignments,
            sign_flip_extreme_count,
        ) = exact_two_sided_sign_flip_pvalue(values)
        (
            tie_sign_flip_p,
            tie_sign_flip_assignments,
            tie_sign_flip_extreme_count,
        ) = exact_two_sided_sign_flip_pvalue(tie_values)
        tolerance = float(DIMENSION_META[dimension]["practical_tolerance"])
        rows.append(
            {
                "grouping_scheme": scheme,
                "scheme_role": (
                    "primary" if scheme == PRIMARY_GROUPING_SCHEME else "secondary"
                ),
                "dimension": dimension,
                "evidence_tier": DIMENSION_META[dimension]["evidence_tier"],
                "metric_definition": DIMENSION_META[dimension]["definition"],
                "endpoint_count": int(len(values)),
                "mean_endpoint_signed_d": (
                    float(np.mean(values)) if len(values) else float("nan")
                ),
                "median_endpoint_signed_d": (
                    float(np.median(values)) if len(values) else float("nan")
                ),
                "mean_endpoint_signed_d_ci_low": ci_low,
                "mean_endpoint_signed_d_ci_high": ci_high,
                "mean_endpoint_tie_averaged_signed_d": (
                    float(np.mean(tie_values)) if len(tie_values) else float("nan")
                ),
                "mean_endpoint_tie_averaged_signed_d_ci_low": tie_ci_low,
                "mean_endpoint_tie_averaged_signed_d_ci_high": tie_ci_high,
                "mean_fixed_minus_tie_averaged_signed_d": float(
                    group["mean_fixed_minus_tie_averaged_signed_d"].mean()
                ),
                "mean_oracle_tie_direction_fraction": float(
                    group["oracle_tie_direction_fraction"].mean()
                ),
                "positive_endpoint_count": int(np.sum(values > 0)),
                "negative_endpoint_count": int(np.sum(values < 0)),
                "zero_endpoint_count": int(np.sum(values == 0)),
                "positive_endpoint_fraction": (
                    float(np.mean(values > 0)) if len(values) else float("nan")
                ),
                "practical_tolerance": tolerance,
                "endpoint_count_above_practical_tolerance": int(
                    np.sum(values > tolerance)
                ),
                "endpoint_fraction_above_practical_tolerance": (
                    float(np.mean(values > tolerance)) if len(values) else float("nan")
                ),
                "mean_candidate_full_oracle_agreement_fraction": float(
                    group["candidate_full_oracle_agreement_fraction"].mean()
                ),
                "mean_candidate_selection_entropy_normalized": float(
                    group["candidate_selection_entropy_normalized"].mean()
                ),
                "bootstrap_unit": "endpoint",
                "bootstrap_reps": int(bootstrap_reps),
                "sign_flip_unit": "endpoint_mean",
                "sign_flip_method": "exact_enumeration_two_sided",
                "sign_flip_assignments": int(sign_flip_assignments),
                "sign_flip_extreme_count": int(sign_flip_extreme_count),
                "sign_flip_p_exact": float(sign_flip_p),
                "holm_family": "all_7_dimensions_within_grouping",
                "holm_family_size": len(DIMENSION_ORDER),
                "tie_averaged_sign_flip_assignments": int(
                    tie_sign_flip_assignments
                ),
                "tie_averaged_sign_flip_extreme_count": int(
                    tie_sign_flip_extreme_count
                ),
                "tie_averaged_sign_flip_p_exact": float(tie_sign_flip_p),
            }
        )
    summary = pd.DataFrame(rows)
    summary["holm_p_within_grouping_7"] = (
        summary.groupby("grouping_scheme", group_keys=False)["sign_flip_p_exact"]
        .apply(holm_adjust)
        .reindex(summary.index)
    )
    summary["holm_reject_0_05"] = summary["holm_p_within_grouping_7"] < 0.05
    summary["tie_averaged_holm_p_within_grouping_7"] = (
        summary.groupby("grouping_scheme", group_keys=False)[
            "tie_averaged_sign_flip_p_exact"
        ]
        .apply(holm_adjust)
        .reindex(summary.index)
    )
    summary["tie_averaged_holm_reject_0_05"] = (
        summary["tie_averaged_holm_p_within_grouping_7"] < 0.05
    )
    return summary


def locked_split_overlap_audit(
    data_dir: Path,
    prepared_tasks: Sequence[PreparedTask],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_by_task = {item.task: item for item in prepared_tasks}
    audit_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for task in EXPECTED_TASKS:
        task_dir = data_dir / "admet_group" / task
        train_val_path = task_dir / "train_val.csv"
        test_path = task_dir / "test.csv"
        if not train_val_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing locked source files for {task}")
        train_val = pd.read_csv(train_val_path)
        test = pd.read_csv(test_path)
        train_smiles_column = infer_smiles_column(train_val)
        test_smiles_column = infer_smiles_column(test)
        train_label_column = infer_label_column(train_val)
        test_label_column = infer_label_column(test)
        train_smiles = train_val[train_smiles_column].astype(str).tolist()
        test_smiles = test[test_smiles_column].astype(str).tolist()
        train_labels = train_val[train_label_column].to_numpy(dtype=float)
        test_labels = test[test_label_column].to_numpy(dtype=float)
        train_canonical, train_scaffold_groups = molecule_group_keys(train_smiles)
        test_canonical, test_scaffold_groups = molecule_group_keys(test_smiles)
        canonical_overlap = set(train_canonical).intersection(set(test_canonical))
        scaffold_overlap = set(train_scaffold_groups).intersection(
            set(test_scaffold_groups)
        )
        train_nonempty_scaffolds = {
            value.removeprefix("SCAFFOLD::")
            for value in train_scaffold_groups
            if str(value).startswith("SCAFFOLD::")
        }
        test_nonempty_scaffolds = {
            value.removeprefix("SCAFFOLD::")
            for value in test_scaffold_groups
            if str(value).startswith("SCAFFOLD::")
        }
        nonempty_overlap = train_nonempty_scaffolds.intersection(
            test_nonempty_scaffolds
        )
        prepared = prepared_by_task[task]
        prediction_matches = bool(
            len(prepared.smiles) == len(test_smiles)
            and prepared.smiles.astype(str).tolist() == test_smiles
            and np.allclose(
                prepared.y_true,
                test_labels,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False,
            )
        )
        source_test_frame = pd.DataFrame(
            {
                "sample_index": np.arange(len(test_smiles), dtype=int),
                "smiles": test_smiles,
            }
        )
        prediction_test_frame = pd.DataFrame(
            {
                "sample_index": prepared.sample_index,
                "smiles": prepared.smiles.astype(str),
            }
        )
        audit_rows.append(
            {
                "task": task,
                "task_type": prepared.task_type,
                "train_val_rows": len(train_val),
                "test_rows": len(test),
                "train_val_unique_canonical": len(set(train_canonical)),
                "test_unique_canonical": len(set(test_canonical)),
                "canonical_overlap_count": len(canonical_overlap),
                "canonical_overlap_sha256": sha256_records(sorted(canonical_overlap)),
                "train_val_unique_scaffold_groups": len(set(train_scaffold_groups)),
                "test_unique_scaffold_groups": len(set(test_scaffold_groups)),
                "scaffold_group_overlap_count": len(scaffold_overlap),
                "scaffold_group_overlap_sha256": sha256_records(
                    sorted(scaffold_overlap)
                ),
                "nonempty_bemis_murcko_overlap_count": len(nonempty_overlap),
                "nonempty_bemis_murcko_overlap_sha256": sha256_records(
                    sorted(nonempty_overlap)
                ),
                "train_val_acyclic_fallback_rows": int(
                    np.sum(
                        [str(value).startswith("ACYCLIC::") for value in train_scaffold_groups]
                    )
                ),
                "test_acyclic_fallback_rows": int(
                    np.sum(
                        [str(value).startswith("ACYCLIC::") for value in test_scaffold_groups]
                    )
                ),
                "prediction_test_matches_locked_source_order_labels": prediction_matches,
                "prediction_test_sample_fingerprint": sample_fingerprint(
                    prediction_test_frame
                ),
                "locked_test_sample_fingerprint": sample_fingerprint(source_test_frame),
                "train_val_file_sha256": sha256_file(train_val_path),
                "test_file_sha256": sha256_file(test_path),
                "train_val_ordered_smiles_sha256": sha256_records(train_smiles),
                "test_ordered_smiles_sha256": sha256_records(test_smiles),
                "train_val_ordered_labels_sha256": sha256_records(
                    [float(value) for value in train_labels]
                ),
                "test_ordered_labels_sha256": sha256_records(
                    [float(value) for value in test_labels]
                ),
                "scaffold_policy": (
                    "Bemis-Murcko canonical scaffold; empty scaffold falls back "
                    "to ACYCLIC::canonical_identity"
                ),
            }
        )
        for split, path, frame, smiles_values, label_values in (
            ("train_val", train_val_path, train_val, train_smiles, train_labels),
            ("test", test_path, test, test_smiles, test_labels),
        ):
            source_rows.append(
                {
                    "task": task,
                    "split": split,
                    "path": str(path.relative_to(ROOT)),
                    "file_size_bytes": path.stat().st_size,
                    "row_count": len(frame),
                    "file_sha256": sha256_file(path),
                    "ordered_smiles_sha256": sha256_records(smiles_values),
                    "ordered_labels_sha256": sha256_records(
                        [float(value) for value in label_values]
                    ),
                }
            )
    return pd.DataFrame(audit_rows), pd.DataFrame(source_rows)
def _expected_endpoint_dimension_count() -> int:
    classification = sum(
        meta["task_type"] == "classification" for meta in TDC_TASK_META.values()
    )
    regression = len(TDC_TASK_META) - classification
    return classification * 6 + regression * 3


def build_gate(
    fold_rows: pd.DataFrame,
    endpoint_rows: pd.DataFrame,
    dimension_rows: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    input_audit: pd.DataFrame,
    run_alignment_ok: bool,
    ensemble_alignment_ok: bool,
    repeats: int,
    bootstrap_reps: int,
    selection_path: Path,
    source_hashes: pd.DataFrame,
    output_hashes: Mapping[str, str],
) -> dict[str, object]:
    expected_endpoint_dimensions = _expected_endpoint_dimension_count()
    expected_fold_rows = (
        len(GROUPING_SCHEMES) * expected_endpoint_dimensions * repeats * 2
    )
    all_usable = bool(len(fold_rows) and fold_rows["usable"].all())
    no_group_leakage = bool(
        len(fold_rows) and (fold_rows["group_overlap_count"] == 0).all()
    )
    fold_reuse_counts = (
        fold_rows.groupby(
            ["task", "grouping_scheme", "repeat", "direction"], sort=False
        )[
            [
                "a_sample_index_sha256",
                "b_sample_index_sha256",
                "a_group_sha256",
                "b_group_sha256",
            ]
        ]
        .nunique()
        .to_numpy()
    )
    all_dimensions_share_folds = bool(
        len(fold_reuse_counts) and np.all(fold_reuse_counts == 1)
    )
    complete_repeats = bool(
        len(endpoint_rows)
        and (endpoint_rows["complete_repeats"] == repeats).all()
        and (endpoint_rows["usable_direction_rows"] == 2 * repeats).all()
        and (
            endpoint_rows["tie_average_usable_direction_rows"] == 2 * repeats
        ).all()
    )
    overlap_zero = bool(
        len(overlap_audit) == len(EXPECTED_TASKS)
        and (overlap_audit["canonical_overlap_count"] == 0).all()
        and (overlap_audit["scaffold_group_overlap_count"] == 0).all()
        and (overlap_audit["nonempty_bemis_murcko_overlap_count"] == 0).all()
    )
    prediction_source_aligned = bool(
        len(overlap_audit) == len(EXPECTED_TASKS)
        and overlap_audit[
            "prediction_test_matches_locked_source_order_labels"
        ].all()
    )
    formal_sources_complete = bool(
        len(input_audit[input_audit["included"]])
        == len(EXPECTED_TASKS) * len(FINAL_EXPECTED_MODELS) * len(FINAL_EXPECTED_SEEDS)
    )
    exact_sign_flip_holm_complete = bool(
        len(dimension_rows) == len(GROUPING_SCHEMES) * len(DIMENSION_ORDER)
        and (dimension_rows["sign_flip_unit"] == "endpoint_mean").all()
        and (
            dimension_rows["sign_flip_method"]
            == "exact_enumeration_two_sided"
        ).all()
        and (
            dimension_rows["sign_flip_assignments"]
            == (2 ** dimension_rows["endpoint_count"].astype(int))
        ).all()
        and (dimension_rows["sign_flip_extreme_count"] >= 0).all()
        and (
            dimension_rows["sign_flip_extreme_count"]
            <= dimension_rows["sign_flip_assignments"]
        ).all()
        and np.allclose(
            dimension_rows["sign_flip_p_exact"],
            dimension_rows["sign_flip_extreme_count"]
            / dimension_rows["sign_flip_assignments"],
            rtol=0.0,
            atol=0.0,
        )
        and (
            dimension_rows["holm_family"]
            == "all_7_dimensions_within_grouping"
        ).all()
        and (dimension_rows["holm_family_size"] == len(DIMENSION_ORDER)).all()
        and dimension_rows["sign_flip_p_exact"].between(0.0, 1.0).all()
        and dimension_rows["holm_p_within_grouping_7"].between(0.0, 1.0).all()
        and (
            dimension_rows["holm_p_within_grouping_7"]
            >= dimension_rows["sign_flip_p_exact"]
        ).all()
        and (
            dimension_rows["tie_averaged_sign_flip_assignments"]
            == (2 ** dimension_rows["endpoint_count"].astype(int))
        ).all()
        and (
            dimension_rows["tie_averaged_sign_flip_extreme_count"] >= 0
        ).all()
        and (
            dimension_rows["tie_averaged_sign_flip_extreme_count"]
            <= dimension_rows["tie_averaged_sign_flip_assignments"]
        ).all()
        and dimension_rows["tie_averaged_sign_flip_p_exact"].between(0.0, 1.0).all()
        and dimension_rows["tie_averaged_holm_p_within_grouping_7"].between(
            0.0, 1.0
        ).all()
    )
    contract_passes = bool(
        len(fold_rows) == expected_fold_rows
        and len(endpoint_rows)
        == len(GROUPING_SCHEMES) * expected_endpoint_dimensions
        and all_usable
        and no_group_leakage
        and all_dimensions_share_folds
        and complete_repeats
        and overlap_zero
        and prediction_source_aligned
        and formal_sources_complete
        and exact_sign_flip_holm_complete
        and run_alignment_ok
        and ensemble_alignment_ok
    )
    main_summary = dimension_rows[
        dimension_rows["grouping_scheme"] == PRIMARY_GROUPING_SCHEME
    ]
    evidence = {
        str(row.dimension): {
            "endpoint_count": int(row.endpoint_count),
            "mean_signed_d": float(row.mean_endpoint_signed_d),
            "ci_low": float(row.mean_endpoint_signed_d_ci_low),
            "ci_high": float(row.mean_endpoint_signed_d_ci_high),
            "positive_endpoint_count": int(row.positive_endpoint_count),
            "negative_endpoint_count": int(row.negative_endpoint_count),
            "above_practical_tolerance_count": int(
                row.endpoint_count_above_practical_tolerance
            ),
            "above_practical_tolerance_fraction": float(
                row.endpoint_fraction_above_practical_tolerance
            ),
            "mean_candidate_full_oracle_agreement_fraction": float(
                row.mean_candidate_full_oracle_agreement_fraction
            ),
            "mean_candidate_selection_entropy_normalized": float(
                row.mean_candidate_selection_entropy_normalized
            ),
            "sign_flip_p_exact": float(row.sign_flip_p_exact),
            "holm_p_within_grouping_7": float(row.holm_p_within_grouping_7),
            "holm_reject_0_05": bool(row.holm_reject_0_05),
            "tie_averaged_mean_signed_d": float(
                row.mean_endpoint_tie_averaged_signed_d
            ),
            "tie_averaged_ci_low": float(
                row.mean_endpoint_tie_averaged_signed_d_ci_low
            ),
            "tie_averaged_ci_high": float(
                row.mean_endpoint_tie_averaged_signed_d_ci_high
            ),
            "mean_oracle_tie_direction_fraction": float(
                row.mean_oracle_tie_direction_fraction
            ),
            "tie_averaged_sign_flip_p_exact": float(
                row.tie_averaged_sign_flip_p_exact
            ),
            "tie_averaged_holm_p_within_grouping_7": float(
                row.tie_averaged_holm_p_within_grouping_7
            ),
            "evidence_tier": str(row.evidence_tier),
        }
        for row in main_summary.itertuples(index=False)
    }
    signed_values = fold_rows.loc[fold_rows["usable"], "signed_d"].to_numpy(float)
    return {
        "status": "GO" if contract_passes else "HOLD",
        "scope": "zero_retraining_crossfit_oracle_sensitivity_final_locked",
        "interpretation": (
            "GO certifies analysis completeness, split integrity, input alignment, and "
            "hash-bound scaffold-disjoint provenance. It does not force the signed "
            "sensitivity estimate to be positive. Scientific conclusions must follow "
            "the reported endpoint-bootstrap intervals and the descriptive exact "
            "sign-flip/Holm sensitivity."
        ),
        "analysis_contract": {
            "passes": contract_passes,
            "expected_models_in_tie_order": list(FINAL_EXPECTED_MODELS),
            "expected_seeds": list(FINAL_EXPECTED_SEEDS),
            "expected_tasks": len(EXPECTED_TASKS),
            "grouping_schemes": list(GROUPING_SCHEMES),
            "primary_grouping_scheme": PRIMARY_GROUPING_SCHEME,
            "repeats_per_endpoint_scheme": int(repeats),
            "directions_per_repeat": 2,
            "expected_fold_rows": int(expected_fold_rows),
            "observed_fold_rows": int(len(fold_rows)),
            "expected_endpoint_dimension_rows": int(
                len(GROUPING_SCHEMES) * expected_endpoint_dimensions
            ),
            "observed_endpoint_dimension_rows": int(len(endpoint_rows)),
            "observed_dimension_summary_rows": int(len(dimension_rows)),
            "all_rows_usable": all_usable,
            "no_group_leakage": no_group_leakage,
            "all_models_dimensions_share_fold": all_dimensions_share_folds,
            "all_repeats_bidirectionally_complete": complete_repeats,
            "formal_prediction_sources_complete": formal_sources_complete,
            "run_sample_alignment_exact": run_alignment_ok,
            "ensemble_sample_alignment_exact": ensemble_alignment_ok,
            "prediction_test_matches_locked_source": prediction_source_aligned,
            "canonical_and_scaffold_overlap_zero_all_endpoints": overlap_zero,
            "exact_sign_flip_holm_complete": exact_sign_flip_holm_complete,
        },
        "signed_difference_policy": {
            "formula": "L_evaluation(validation_selected)-L_evaluation(oracle_selection_half)",
            "negative_values_never_truncated": True,
            "minimum_observed_signed_d": (
                float(np.min(signed_values)) if len(signed_values) else None
            ),
            "maximum_observed_signed_d": (
                float(np.max(signed_values)) if len(signed_values) else None
            ),
            "negative_direction_count": int(np.sum(signed_values < 0)),
            "positive_direction_count": int(np.sum(signed_values > 0)),
            "zero_direction_count": int(np.sum(signed_values == 0)),
        },
        "inference_policy": {
            "within_endpoint": (
                "Average A-to-B and B-to-A within repeat, then average the 500 "
                "repeat means within endpoint. Repeated folds are not independent units."
            ),
            "across_endpoints": (
                f"Percentile bootstrap of endpoint means with {bootstrap_reps} replicates."
            ),
            "multiplicity_sensitivity": (
                "Exact two-sided sign-flip enumeration of endpoint means, with Holm "
                f"adjustment across all {len(DIMENSION_ORDER)} dimensions separately "
                "within each grouping. This post-hoc diagnostic is descriptive, not "
                "confirmatory, and does not remove shared-molecule dependence."
            ),
            "exact_candidate_tie_sensitivity": (
                "The locked estimator takes the first exact minimum in model order. "
                "A zero-refit sensitivity instead averages evaluation-fold losses "
                "across every exact minimum selected on the opposite fold."
            ),
            "aurc": (
                "Full risk-coverage metric is recomputed within each half-fold. "
                "Uncertainty ties use expected random within-tie cumulative loss."
            ),
            "secondary_dimensions": [
                dimension
                for dimension, meta in DIMENSION_META.items()
                if meta["evidence_tier"] != "primary"
            ],
        },
        "primary_bemis_murcko_results": evidence,
        "overlap_audit": {
            "endpoints": int(len(overlap_audit)),
            "canonical_overlap_total": int(
                overlap_audit["canonical_overlap_count"].sum()
            ),
            "scaffold_group_overlap_total": int(
                overlap_audit["scaffold_group_overlap_count"].sum()
            ),
            "nonempty_bemis_murcko_overlap_total": int(
                overlap_audit["nonempty_bemis_murcko_overlap_count"].sum()
            ),
            "acyclic_fallback": "ACYCLIC::canonical_identity",
        },
        "provenance": {
            "selection_file": str(selection_path.relative_to(ROOT)),
            "selection_file_sha256": sha256_file(selection_path),
            "locked_source_file_count": int(len(source_hashes)),
            "locked_source_hash_manifest_sha256": sha256_records(
                source_hashes.sort_values(["task", "split"])[
                    ["task", "split", "file_sha256"]
                ].to_dict("records")
            ),
            "output_sha256": dict(output_hashes),
        },
    }


def write_supplementary_summary_table(
    dimension_rows: pd.DataFrame,
    out_dir: Path,
    source_summary_path: Path | None = None,
) -> dict[str, str]:
    """Write a compact 14-row CSV/TeX table directly consumable by LaTeX."""
    expected_rows = len(GROUPING_SCHEMES) * len(DIMENSION_ORDER)
    if len(dimension_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} dimension-summary rows, found {len(dimension_rows)}"
        )
    scheme_rank = {scheme: index for index, scheme in enumerate(GROUPING_SCHEMES)}
    dimension_rank = {
        dimension: index for index, dimension in enumerate(DIMENSION_ORDER)
    }
    ordered = dimension_rows.copy()
    ordered["_scheme_rank"] = ordered["grouping_scheme"].map(scheme_rank)
    ordered["_dimension_rank"] = ordered["dimension"].map(dimension_rank)
    ordered = ordered.sort_values(["_scheme_rank", "_dimension_rank"])
    if ordered[["_scheme_rank", "_dimension_rank"]].isna().any().any():
        raise ValueError("Unexpected grouping scheme or dimension in summary")

    table = pd.DataFrame(
        {
            "grouping": ordered["grouping_scheme"].to_numpy(),
            "dimension": ordered["dimension"].to_numpy(),
            "n_endpoints": ordered["endpoint_count"].to_numpy(dtype=int),
            "mean_signed_d": ordered["mean_endpoint_signed_d"].to_numpy(float),
            "ci_low": ordered["mean_endpoint_signed_d_ci_low"].to_numpy(float),
            "ci_high": ordered["mean_endpoint_signed_d_ci_high"].to_numpy(float),
            "mean_signed_d_95ci": [
                f"{mean:.4f} [{low:.4f}, {high:.4f}]"
                for mean, low, high in zip(
                    ordered["mean_endpoint_signed_d"],
                    ordered["mean_endpoint_signed_d_ci_low"],
                    ordered["mean_endpoint_signed_d_ci_high"],
                )
            ],
            "positive_endpoints": ordered["positive_endpoint_count"].to_numpy(
                dtype=int
            ),
            "above_tolerance_endpoints": ordered[
                "endpoint_count_above_practical_tolerance"
            ].to_numpy(dtype=int),
            "oracle_agreement": ordered[
                "mean_candidate_full_oracle_agreement_fraction"
            ].to_numpy(float),
            "sign_flip_p_exact": ordered["sign_flip_p_exact"].to_numpy(float),
            "holm_p_within_grouping_7": ordered[
                "holm_p_within_grouping_7"
            ].to_numpy(float),
            "holm_reject_0_05": ordered["holm_reject_0_05"].to_numpy(bool),
            "selection_entropy_normalized": ordered[
                "mean_candidate_selection_entropy_normalized"
            ].to_numpy(float),
            "evidence_tier": ordered["evidence_tier"].to_numpy(),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_supp_crossfit_oracle_summary.csv"
    tex_path = out_dir / "table_supp_crossfit_oracle_summary.tex"
    manifest_path = out_dir / "table_supp_crossfit_oracle_summary_manifest.json"
    table.to_csv(csv_path, index=False, float_format="%.12g")

    scheme_label = {
        "bemis_murcko": "Bemis--Murcko",
        "canonical_identity": "Canonical identity",
    }
    dimension_label = {
        "brier": "Brier",
        "ece_5bin": "ECE (5 bins)",
        "ece_10bin": "ECE (10 bins)",
        "applicability_alignment": "Morgan support/error rank loss",
        "selective_relative_excess_aurc": "Primary relative excess AURC",
        "selective_relative_excess_aurc_log_loss": "Log-loss relative excess AURC",
        "conformal_coverage_error": "Validation-residual coverage error",
    }
    tex_lines = [
        "% Generated by experiments/analyze_crossfit_oracle_sensitivity.py",
        "\\begin{tabular}{llrlllll}",
        "\\toprule",
        (
            "Grouping & Dimension & $n$ & Mean signed $D$ [95\\% endpoint interval] & "
            "Positive endpoints & Above heuristic threshold & Descriptive Holm $p$ & Oracle agreement \\\\"
        ),
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        tex_lines.append(
            f"{scheme_label[str(row.grouping)]} & "
            f"{dimension_label[str(row.dimension)]} & "
            f"{int(row.n_endpoints)} & "
            f"{row.mean_signed_d_95ci} & "
            f"{int(row.positive_endpoints)}/{int(row.n_endpoints)} & "
            f"{int(row.above_tolerance_endpoints)}/{int(row.n_endpoints)} & "
            f"{float(row.holm_p_within_grouping_7):.4g} & "
            f"{float(row.oracle_agreement):.3f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema": "crossfit_oracle_supplementary_table_v2",
        "rows": len(table),
        "csv_path": str(csv_path.relative_to(ROOT)),
        "tex_path": str(tex_path.relative_to(ROOT)),
        "csv_sha256": sha256_file(csv_path),
        "tex_sha256": sha256_file(tex_path),
    }
    if source_summary_path is not None:
        manifest["source_summary_path"] = str(source_summary_path.relative_to(ROOT))
        manifest["source_summary_sha256"] = sha256_file(source_summary_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        str(csv_path.relative_to(ROOT)): sha256_file(csv_path),
        str(tex_path.relative_to(ROOT)): sha256_file(tex_path),
        str(manifest_path.relative_to(ROOT)): sha256_file(manifest_path),
    }


def remove_generated_appledouble_sidecars(paths: Iterable[Path]) -> list[str]:
    """Remove macOS AppleDouble metadata only for outputs generated here."""
    removed: list[str] = []
    for path in paths:
        sidecar = path.parent / f"._{path.name}"
        if sidecar.is_file():
            sidecar.unlink()
            removed.append(str(sidecar))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        default=str(ROOT / "results" / "predictions" / "traditional_locked_cache"),
    )
    parser.add_argument(
        "--additional-prediction-dirs",
        nargs="*",
        default=[
            str(ROOT / "results" / "neural" / "predictions"),
            str(ROOT / "results" / "neural_chemprop" / "predictions"),
        ],
    )
    parser.add_argument(
        "--similarity-dir",
        default=str(ROOT / "results" / "cache" / "applicability"),
    )
    parser.add_argument(
        "--selection-file",
        default=str(
            ROOT
            / "results"
            / "tables"
            / "reliability_regret_final"
            / "validation_selection_by_task.csv"
        ),
    )
    parser.add_argument(
        "--data-dir", default=str(ROOT / "data" / "tdc_official")
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            ROOT
            / "results"
            / "tables"
            / "crossfit_oracle_sensitivity_final"
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    if args.repeats <= 0 or args.bootstrap_reps <= 0:
        raise ValueError("repeats and bootstrap-reps must be positive")

    prediction_dirs = [
        Path(args.prediction_dir),
        *[Path(path) for path in args.additional_prediction_dirs],
    ]
    runs, input_audit = load_runs(
        prediction_dirs,
        Path(args.similarity_dir),
        similarity_split_source="locked_cache",
        require_similarity=True,
    )
    run_alignment, run_alignment_ok = audit_run_sample_alignment(runs)
    if not run_alignment_ok:
        raise ValueError("Formal run samples are misaligned")
    ensembles, validation_scores, availability = build_ensembles(
        runs, minimum_seeds=len(FINAL_EXPECTED_SEEDS), alpha=args.alpha
    )
    if any(set(item.seeds) != set(FINAL_EXPECTED_SEEDS) for item in ensembles):
        raise ValueError("Final cross-fit analysis requires exact seeds 1--5")
    ensemble_alignment, ensemble_alignment_ok = audit_ensemble_test_alignment(
        ensembles
    )
    if not ensemble_alignment_ok:
        raise ValueError("Formal ensemble samples are misaligned")
    selection_path = Path(args.selection_file)
    selections, selected_models = load_locked_selections(selection_path, ensembles)
    conformal = prepare_conformal_coverage(runs, ensembles, alpha=args.alpha)
    prepared_tasks = [
        prepare_task(task, ensembles, conformal) for task in EXPECTED_TASKS
    ]

    overlap_audit, source_hashes = locked_split_overlap_audit(
        Path(args.data_dir), prepared_tasks
    )
    fold_rows = run_crossfit(
        prepared_tasks,
        selected_models,
        repeats=args.repeats,
        target_coverage=1 - args.alpha,
        random_seed=args.random_seed,
    )
    endpoint_rows = summarize_endpoints(fold_rows, args.repeats)
    dimension_rows = summarize_dimensions(
        endpoint_rows,
        bootstrap_reps=args.bootstrap_reps,
        random_seed=args.random_seed,
    )

    # Normalize the output directory before writing manifests.  Callers often
    # provide a repository-relative path, while the manifest records paths
    # relative to ROOT; mixing the two makes pathlib.relative_to fail after the
    # expensive cross-fit computation has already completed.
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {
        "crossfit_oracle_fold_level.csv": fold_rows,
        "crossfit_oracle_endpoint_summary.csv": endpoint_rows,
        "crossfit_oracle_dimension_summary.csv": dimension_rows,
        "locked_split_overlap_audit.csv": overlap_audit,
        "locked_split_source_sha256.csv": source_hashes,
        "prediction_input_audit.csv": input_audit,
        "run_sample_alignment_audit.csv": run_alignment,
        "model_seed_availability.csv": availability,
        "ensemble_sample_alignment_audit.csv": ensemble_alignment,
        "validation_seed_scores.csv": validation_scores,
        "fixed_validation_selection.csv": selections,
    }
    output_hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        path = out_dir / name
        frame.to_csv(path, index=False, float_format="%.12g")
        output_hashes[name] = sha256_file(path)
    manuscript_hashes = write_supplementary_summary_table(
        dimension_rows,
        ROOT / "results" / "tables" / "manuscript_final",
        source_summary_path=out_dir / "crossfit_oracle_dimension_summary.csv",
    )
    output_hashes.update(manuscript_hashes)
    gate = build_gate(
        fold_rows,
        endpoint_rows,
        dimension_rows,
        overlap_audit,
        input_audit,
        run_alignment_ok,
        ensemble_alignment_ok,
        args.repeats,
        args.bootstrap_reps,
        selection_path,
        source_hashes,
        output_hashes,
    )
    gate_path = out_dir / "crossfit_oracle_sensitivity_gate.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    remove_generated_appledouble_sidecars(
        [
            *[out_dir / name for name in outputs],
            gate_path,
            *[ROOT / path for path in manuscript_hashes],
        ]
    )
    print(f"formal runs: {len(runs)}")
    print(f"task-model ensembles: {len(ensembles)}")
    print(f"fold rows: {len(fold_rows)}")
    print(f"endpoint summaries: {len(endpoint_rows)}")
    print(f"dimension summaries: {len(dimension_rows)}")
    print(f"status: {gate['status']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
