from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .data_loader import OfficialSplit, infer_label_column, infer_smiles_column
from .smiles_compat import SMILES_COMPATIBILITY_POLICY, model_input_smiles


Canonicalizer = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class CleaningPolicy:
    """Pre-specified, conservative cleaning used only for sensitivity analysis."""

    invalid_smiles: str = "drop"
    missing_or_invalid_label: str = "drop"
    within_split_duplicates: str = "keep_first_if_labels_agree"
    conflicting_duplicate_labels: str = "drop_entire_canonical_group"
    salts_or_mixtures: str = "exclude_multicomponent_records"
    split_precedence: tuple[str, str, str] = ("test", "valid", "train")
    regression_conflict_tolerance: float = 1e-8

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["split_precedence"] = list(self.split_precedence)
        return payload


@dataclass(frozen=True)
class CleanedSensitivitySplit:
    name: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    task_type: str
    metric: str
    category: str
    audit: dict[str, object]


def rdkit_canonicalize(smiles: str) -> str | None:
    """Canonicalize after the audited parser-only compatibility translation.

    The compatibility map preserves chemical identity and belongs to the model
    input representation.  Applying it here prevents an RDKit-version change
    from being misreported as an ``invalid_smiles`` cleaning deletion.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for cleaned-data sensitivity analysis.") from exc

    mol = Chem.MolFromSmiles(model_input_smiles(str(smiles)))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _valid_label(value: object, task_type: str) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric):
        return False
    if task_type == "classification":
        return numeric in (0.0, 1.0)
    return task_type == "regression"


def _labels_conflict(values: pd.Series, task_type: str, tolerance: float) -> bool:
    numeric = values.astype(float).to_numpy()
    if task_type == "classification":
        return np.unique(numeric).size > 1
    return bool(np.max(numeric) - np.min(numeric) > tolerance)


def clean_one_split(
    frame: pd.DataFrame,
    *,
    split_name: str,
    task_type: str,
    policy: CleaningPolicy | None = None,
    canonicalizer: Canonicalizer = rdkit_canonicalize,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Canonicalize and de-duplicate one split without consulting other splits.

    Conflicting labels for the same canonical molecule are dropped as a group.
    This avoids label-dependent averaging or majority-vote rules that could make
    the sensitivity analysis look artificially favorable.
    """
    policy = policy or CleaningPolicy()
    smiles_column = infer_smiles_column(frame)
    label_column = infer_label_column(frame)
    work = frame.copy(deep=True)
    work["_source_row"] = np.arange(len(work), dtype=int)

    canonical_values: list[str | None] = []
    invalid_smiles = 0
    parser_compatibility_repairs = 0
    salts_or_mixtures = 0
    multicomponent_values: list[bool] = []
    for raw in work[smiles_column].tolist():
        raw_text = str(raw)
        parser_compatibility_repairs += int(model_input_smiles(raw_text) != raw_text)
        is_multicomponent = "." in raw_text
        salts_or_mixtures += int(is_multicomponent)
        multicomponent_values.append(is_multicomponent)
        canonical = canonicalizer(raw_text)
        if canonical is None:
            invalid_smiles += 1
        canonical_values.append(canonical)
    work["_canonical_smiles"] = canonical_values
    work["_is_multicomponent"] = multicomponent_values

    valid_label_mask = work[label_column].map(lambda value: _valid_label(value, task_type))
    missing_or_invalid_label = int((~valid_label_mask).sum())
    if policy.salts_or_mixtures == "exclude_multicomponent_records":
        component_mask = ~work["_is_multicomponent"]
    elif policy.salts_or_mixtures == "retain_and_report":
        component_mask = pd.Series(True, index=work.index)
    else:
        raise ValueError(f"Unknown salts_or_mixtures policy: {policy.salts_or_mixtures}")
    usable = work[
        valid_label_mask & work["_canonical_smiles"].notna() & component_mask
    ].copy()

    duplicate_groups = 0
    duplicate_rows_beyond_first = 0
    conflicting_groups_dropped = 0
    conflicting_rows_dropped = 0
    kept_indices: list[int] = []
    for _, group in usable.groupby("_canonical_smiles", sort=False, dropna=False):
        group = group.sort_values("_source_row", kind="stable")
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_rows_beyond_first += len(group) - 1
        if _labels_conflict(
            group[label_column], task_type, policy.regression_conflict_tolerance
        ):
            conflicting_groups_dropped += 1
            conflicting_rows_dropped += len(group)
            continue
        kept_indices.append(int(group.index[0]))

    cleaned = usable.loc[kept_indices].sort_values("_source_row", kind="stable").copy()
    cleaned[smiles_column] = cleaned["_canonical_smiles"]
    cleaned = cleaned.drop(
        columns=["_source_row", "_canonical_smiles", "_is_multicomponent"]
    ).reset_index(drop=True)

    report: dict[str, object] = {
        "split": split_name,
        "input_rows": int(len(frame)),
        "invalid_smiles_dropped": invalid_smiles,
        "parser_compatibility_repairs_applied": parser_compatibility_repairs,
        "parser_compatibility_repairs_count_as_cleaning_deletions": False,
        "missing_or_invalid_labels_dropped": missing_or_invalid_label,
        "salts_or_mixtures_detected": salts_or_mixtures,
        "salts_or_mixtures_dropped": (
            salts_or_mixtures
            if policy.salts_or_mixtures == "exclude_multicomponent_records"
            else 0
        ),
        "duplicate_canonical_groups": duplicate_groups,
        "duplicate_rows_beyond_first": duplicate_rows_beyond_first,
        "conflicting_label_groups_dropped": conflicting_groups_dropped,
        "conflicting_label_rows_dropped": conflicting_rows_dropped,
        "output_rows_before_cross_split_filter": int(len(cleaned)),
        "output_rows": int(len(cleaned)),
    }
    return cleaned, report


def _canonical_ids(frame: pd.DataFrame) -> set[str]:
    smiles_column = infer_smiles_column(frame)
    return set(frame[smiles_column].astype(str))


def _remove_ids(frame: pd.DataFrame, excluded: set[str]) -> tuple[pd.DataFrame, int]:
    smiles_column = infer_smiles_column(frame)
    mask = ~frame[smiles_column].astype(str).isin(excluded)
    return frame.loc[mask].reset_index(drop=True), int((~mask).sum())


def clean_official_split_for_sensitivity(
    split: OfficialSplit,
    *,
    policy: CleaningPolicy | None = None,
    canonicalizer: Canonicalizer = rdkit_canonicalize,
) -> CleanedSensitivitySplit:
    """Create leakage-safe cleaned splits with fixed ``test > valid > train`` precedence.

    The official split object is never modified. Test molecules are retained,
    validation molecules overlapping test are removed, and training molecules
    overlapping either held-out split are removed. The returned test frame is
    the common evaluation set for both official-trained and cleaned-trained
    models in the paired sensitivity experiment.
    """
    policy = policy or CleaningPolicy()
    if policy.split_precedence != ("test", "valid", "train"):
        raise ValueError("Sensitivity protocol requires split_precedence test > valid > train.")

    train, train_report = clean_one_split(
        split.train,
        split_name="train",
        task_type=split.task_type,
        policy=policy,
        canonicalizer=canonicalizer,
    )
    valid, valid_report = clean_one_split(
        split.valid,
        split_name="valid",
        task_type=split.task_type,
        policy=policy,
        canonicalizer=canonicalizer,
    )
    test, test_report = clean_one_split(
        split.test,
        split_name="test",
        task_type=split.task_type,
        policy=policy,
        canonicalizer=canonicalizer,
    )

    train_ids_before = _canonical_ids(train)
    valid_ids_before = _canonical_ids(valid)
    test_ids = _canonical_ids(test)
    overlaps_before = {
        "train_valid": len(train_ids_before & valid_ids_before),
        "train_test": len(train_ids_before & test_ids),
        "valid_test": len(valid_ids_before & test_ids),
    }

    valid, valid_removed_for_test = _remove_ids(valid, test_ids)
    valid_ids = _canonical_ids(valid)
    train, train_removed_for_held_out = _remove_ids(train, test_ids | valid_ids)
    train_ids = _canonical_ids(train)

    overlaps_after = {
        "train_valid": len(train_ids & valid_ids),
        "train_test": len(train_ids & test_ids),
        "valid_test": len(valid_ids & test_ids),
    }
    if any(overlaps_after.values()):
        raise AssertionError(f"Cross-split molecule leakage remains: {overlaps_after}")
    if not len(train) or not len(valid) or not len(test):
        raise ValueError("Cleaning produced an empty train, validation, or test split.")

    train_report["cross_split_rows_removed"] = train_removed_for_held_out
    valid_report["cross_split_rows_removed"] = valid_removed_for_test
    test_report["cross_split_rows_removed"] = 0
    train_report["output_rows"] = int(len(train))
    valid_report["output_rows"] = int(len(valid))
    test_report["output_rows"] = int(len(test))

    audit: dict[str, object] = {
        "audit_schema_version": 2,
        "protocol": "paired-common-test-v1",
        "main_result_policy": "TDC official split remains the primary analysis",
        "model_input_representation_policy": SMILES_COMPATIBILITY_POLICY,
        "parser_compatibility_repairs_count_as_cleaning_deletions": False,
        "policy": policy.to_dict(),
        "splits": {
            "train": train_report,
            "valid": valid_report,
            "test": test_report,
        },
        "canonical_molecule_overlap_before_precedence": overlaps_before,
        "canonical_molecule_overlap_after_precedence": overlaps_after,
    }
    return CleanedSensitivitySplit(
        name=split.name,
        train=train,
        valid=valid,
        test=test,
        task_type=split.task_type,
        metric=split.metric,
        category=split.category,
        audit=audit,
    )
