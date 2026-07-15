#!/usr/bin/env python3
"""Build and audit the submission reproducibility release index.

The builder never copies raw data or model checkpoints.  It writes a compact
release index containing repository-relative paths, streaming SHA-256 hashes,
file sizes, seed/split provenance, packaging actions, and a conservative
GO/HOLD gate.  A missing, malformed, or incomplete run is always HOLD.

Submission models
-----------------
* ``rf``, ``xgboost``, ``lightgbm``: 330 traditional runs in total.
* ``gin``: 110 graph neural-network runs.
* ``chemprop_dmpnn``: 110 official Chemprop directed bond-message runs.

Artifacts produced by the historical in-repository model id ``dmpnn`` are
indexed only as ``legacy_node_mpnn`` and are excluded from the submission by
default.  They cannot satisfy the ``chemprop_dmpnn`` completeness gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admet_benchmark.data_loader import TDC_TASK_META, TDC_TASKS  # noqa: E402
from admet_benchmark.smiles_compat import (  # noqa: E402
    LEGACY_VALENCE_REPAIR_INCHIKEYS,
    LEGACY_VALENCE_REPAIRS,
    SMILES_COMPATIBILITY_POLICY,
)


SEEDS = (1, 2, 3, 4, 5)
TRADITIONAL_MODELS = ("rf", "xgboost", "lightgbm")
SUBMISSION_NEURAL_MODELS = ("gin", "chemprop_dmpnn")
LEGACY_MODEL = "legacy_node_mpnn"
LEGACY_SOURCE_MODEL_ID = "dmpnn"

# This is deliberately an explicit allowlist, not a repository glob.  It is
# the auditable code/config closure for the formal benchmark release: active
# training and cluster entrypoints, the locked post-processing chain, the
# analysis/figure/table builders it invokes, and their local library modules.
# Historical guides, transfer helpers, credentials, raw data, and unselected
# experiments therefore cannot enter the public package by directory accident.
REQUIRED_PUBLIC_CODE_PATHS = (
    ".gitignore",
    "README.md",
    "LICENSE",
    "LICENSE-DATA.md",
    "CITATION.cff",
    ".zenodo.json",
    "requirements.txt",
    "environment.yml",
    "pyproject.toml",
    "docs/MAIN_DATA_PROTOCOL.md",
    "experiments/config/tdc_official.yaml",
    "experiments/config/gin.yaml",
    "experiments/config/chemprop_dmpnn.yaml",
    "experiments/config/sensitivity.yaml",
    "scripts/run_final_posthoc_locked.sh",
    "scripts/run_locked_cache_traditional_full.sh",
    "scripts/run_locked_cache_cleaned_sensitivity_cpu.sh",
    "scripts/run_locked_cache_cpu_defenses.sh",
    "scripts/build_public_release_archive.py",
    "release/build_github_source_staging.py",
    "scripts/slurm_gin_resume_12h.sh",
    "scripts/slurm_chemprop_dmpnn_resume_12h.sh",
    "experiments/run_traditional_grid.py",
    "experiments/run_traditional_predictions.py",
    "experiments/run_neural_a40.py",
    "experiments/run_neural_a40_parallel.py",
    "experiments/run_chemprop_dmpnn.py",
    "experiments/run_chemprop_dmpnn_parallel.py",
    "experiments/run_cleaned_sensitivity_cpu.py",
    "experiments/summarize_traditional_grid.py",
    "experiments/precheck_final_neural_artifacts.py",
    "experiments/analyze_applicability_domain.py",
    "experiments/analyze_reliability_regret.py",
    "experiments/analyze_conditional_reliability.py",
    "experiments/analyze_calibration.py",
    "experiments/analyze_posthoc_ablations.py",
    "experiments/analyze_trustworthiness_alignment.py",
    "experiments/analyze_crossfit_oracle_sensitivity.py",
    "experiments/analyze_crossfit_influence_sensitivity.py",
    "experiments/analyze_reviewer_defense_sensitivities.py",
    "experiments/make_reliability_regret_figure.py",
    "experiments/make_trustworthiness_main_figures.py",
    "experiments/make_manuscript_tables.py",
    "experiments/summarize_cleaned_sensitivity.py",
    "experiments/audit_cleaned_sensitivity_predictions.py",
    "experiments/run_y_randomization_all.py",
    "experiments/run_shap_selected.py",
    "experiments/map_shap_morgan_bits.py",
    "experiments/summarize_locked_cache_cpu_defenses.py",
    "experiments/build_reproducibility_release.py",
    "src/admet_benchmark/__init__.py",
    "src/admet_benchmark/applicability.py",
    "src/admet_benchmark/calibration.py",
    "src/admet_benchmark/data_loader.py",
    "src/admet_benchmark/defense.py",
    "src/admet_benchmark/featurization.py",
    "src/admet_benchmark/metrics.py",
    "src/admet_benchmark/models.py",
    "src/admet_benchmark/neural_models.py",
    "src/admet_benchmark/sensitivity.py",
    "src/admet_benchmark/smiles_compat.py",
    "src/admet_benchmark/uncertainty.py",
    "src/admet_benchmark/utils.py",
)

# Public tests are enumerated independently from implementation paths so the
# release gate can prove that the README's ``pytest -q`` instruction is real.
# The author/submission readiness gate is deliberately not public: it binds
# private declarations, journal-day checks, and unpublished manuscript files.
REQUIRED_PUBLIC_TEST_PATHS = (
    "tests/test_applicability.py",
    "tests/test_chemprop_dmpnn_workflow.py",
    "tests/test_cleaned_sensitivity.py",
    "tests/test_cleaned_sensitivity_prediction_audit.py",
    "tests/test_cleaned_sensitivity_runner_hardening.py",
    "tests/test_cleaned_sensitivity_summary_gate.py",
    "tests/test_crossfit_influence_sensitivity.py",
    "tests/test_crossfit_oracle_sensitivity.py",
    "tests/test_data_loader_split_source.py",
    "tests/test_defense_workflows.py",
    "tests/test_final_neural_artifact_precheck.py",
    "tests/test_github_source_staging.py",
    "tests/test_locked_cache_cpu_defense_gate.py",
    "tests/test_make_manuscript_tables.py",
    "tests/test_public_release_archive.py",
    "tests/test_reliability_regret.py",
    "tests/test_reliability_regret_figure.py",
    "tests/test_reproducibility_release.py",
    "tests/test_reviewer_defense_sensitivities.py",
    "tests/test_traditional_prediction_rebuild.py",
    "tests/test_trustworthiness_analysis.py",
    "tests/test_trustworthiness_main_figures.py",
)
REQUIRED_PUBLIC_SOFTWARE_PATHS = (
    *REQUIRED_PUBLIC_CODE_PATHS,
    *REQUIRED_PUBLIC_TEST_PATHS,
)
PUBLIC_BUNDLE_README_NAME = "README.md"
PUBLIC_SEED_LEDGER_NAME = "run_seeds_and_hashes.csv"

# These are the three publication figures referenced directly by paper/main.tex.
# They are explicit required result artifacts so every rendered figure is
# dynamically rehashed at release time. Manuscript sources and PDFs are audited
# locally but excluded from the data/software archive until the author-controlled
# declarations and public DOI are final.
FINAL_MANUSCRIPT_FIGURES = {
    "paper_figure_01": (
        "results/figures/reliability_regret_final/"
        "fig01_validation_selected_reliability_regret.pdf",
        "reliability_regret_final",
    ),
    "paper_figure_02": (
        "results/figures/trustworthiness_submission_final/"
        "fig02_conditional_reliability.pdf",
        "trustworthiness_submission_final",
    ),
    "paper_figure_03": (
        "results/figures/trustworthiness_submission_final/"
        "fig03_calibration_gains_failures.pdf",
        "trustworthiness_submission_final",
    ),
}

FINAL_CROSSFIT_OUTPUTS = {
    "crossfit_fold_level": (
        "crossfit_oracle_fold_level.csv",
        "crossfit_oracle_fold_evidence",
    ),
    "crossfit_endpoint_summary": (
        "crossfit_oracle_endpoint_summary.csv",
        "crossfit_oracle_summary",
    ),
    "crossfit_dimension_summary": (
        "crossfit_oracle_dimension_summary.csv",
        "crossfit_oracle_summary",
    ),
    "crossfit_overlap_audit": (
        "locked_split_overlap_audit.csv",
        "locked_scaffold_overlap_audit",
    ),
    "crossfit_source_hashes": (
        "locked_split_source_sha256.csv",
        "locked_source_hash_manifest",
    ),
    "crossfit_prediction_input_audit": (
        "prediction_input_audit.csv",
        "crossfit_input_audit",
    ),
    "crossfit_run_alignment": (
        "run_sample_alignment_audit.csv",
        "crossfit_alignment_audit",
    ),
    "crossfit_model_seed_availability": (
        "model_seed_availability.csv",
        "crossfit_input_audit",
    ),
    "crossfit_ensemble_alignment": (
        "ensemble_sample_alignment_audit.csv",
        "crossfit_alignment_audit",
    ),
    "crossfit_validation_scores": (
        "validation_seed_scores.csv",
        "crossfit_input_audit",
    ),
    "crossfit_fixed_selection": (
        "fixed_validation_selection.csv",
        "crossfit_selection_provenance",
    ),
    "crossfit_gate": (
        "crossfit_oracle_sensitivity_gate.json",
        "crossfit_oracle_gate",
    ),
}
FINAL_CROSSFIT_DERIVED_OUTPUTS = {
    "crossfit_manuscript_summary_csv": (
        "results/tables/manuscript_final/table_supp_crossfit_oracle_summary.csv",
        "crossfit_manuscript_summary",
    ),
    "crossfit_manuscript_summary_tex": (
        "results/tables/manuscript_final/table_supp_crossfit_oracle_summary.tex",
        "crossfit_manuscript_summary",
    ),
    "crossfit_manuscript_summary_manifest": (
        "results/tables/manuscript_final/table_supp_crossfit_oracle_summary_manifest.json",
        "crossfit_manuscript_summary_manifest",
    ),
}
FINAL_CROSSFIT_GROUPING_SCHEMES = ("bemis_murcko", "canonical_identity")
FINAL_CROSSFIT_DIMENSIONS = (
    "brier",
    "ece_5bin",
    "ece_10bin",
    "applicability_alignment",
    "selective_relative_excess_aurc",
    "selective_relative_excess_aurc_log_loss",
    "conformal_coverage_error",
)
FINAL_CROSSFIT_REPEATS = 500
FINAL_CROSSFIT_BOOTSTRAP_REPS = 5000

FINAL_CROSSFIT_INFLUENCE_OUTPUTS = {
    "crossfit_influence_loeo": (
        "leave_one_endpoint_out.csv",
        "crossfit_influence_evidence",
    ),
    "crossfit_influence_loco": (
        "leave_one_category_out.csv",
        "crossfit_influence_evidence",
    ),
    "crossfit_influence_summary": (
        "influence_dimension_summary.csv",
        "crossfit_influence_summary",
    ),
    "crossfit_influence_table": (
        "table_supp_crossfit_influence_primary.tex",
        "crossfit_influence_supplementary_table",
    ),
    "crossfit_influence_gate": (
        "crossfit_influence_sensitivity_gate.json",
        "crossfit_influence_gate",
    ),
}
FINAL_CROSSFIT_INFLUENCE_SCOPE = (
    "post_hoc_zero_refit_crossfit_influence_reviewer_defense"
)
FINAL_CROSSFIT_INFLUENCE_ANALYSIS_SCOPE = (
    "post_hoc_zero_refit_reviewer_defense"
)
FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS = 5000
FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED = 20260715
FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED = 20260714
FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS = {
    "crossfit_influence_loeo": 210,
    "crossfit_influence_loco": 60,
    "crossfit_influence_summary": 14,
}
FINAL_CROSSFIT_INFLUENCE_FOCUS_DIMENSIONS = (
    "ece_10bin",
    "ece_5bin",
    "applicability_alignment",
)
FINAL_CROSSFIT_EVIDENCE_TIERS = {
    "brier": "primary",
    "ece_5bin": "sensitivity",
    "ece_10bin": "primary",
    "applicability_alignment": "primary",
    "selective_relative_excess_aurc": "primary",
    "selective_relative_excess_aurc_log_loss": "secondary",
    "conformal_coverage_error": "descriptive",
}

FINAL_REVIEWER_DEFENSE_OUTPUTS = {
    "reviewer_conditional_rows": (
        "conditional_reliability_by_family.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_conditional_gate": (
        "conditional_reliability_family_gate.json",
        "post_hoc_reviewer_defense_gate",
    ),
    "reviewer_conditional_summary": (
        "conditional_reliability_family_summary.csv",
        "post_hoc_reviewer_defense_summary",
    ),
    "reviewer_overlap_endpoints": (
        "cross_endpoint_test_endpoint_summary.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_overlap_components": (
        "cross_endpoint_test_overlap_components.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_overlap_pairs": (
        "cross_endpoint_test_overlap_pairwise.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_overlap_gate": (
        "cross_endpoint_test_overlap_summary.json",
        "post_hoc_reviewer_defense_gate",
    ),
    "reviewer_lomo_summary": (
        "leave_one_model_out_dimension_summary.csv",
        "post_hoc_reviewer_defense_summary",
    ),
    "reviewer_lomo_endpoints": (
        "leave_one_model_out_endpoint.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_lomo_gate": (
        "leave_one_model_out_gate.json",
        "post_hoc_reviewer_defense_gate",
    ),
    "reviewer_lomo_probabilities": (
        "leave_one_model_out_validation_probabilities.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_lomo_selections": (
        "leave_one_model_out_validation_selection.csv",
        "post_hoc_reviewer_defense_evidence",
    ),
    "reviewer_combined_gate": (
        "reviewer_defense_sensitivities_gate.json",
        "post_hoc_reviewer_defense_gate",
    ),
    "reviewer_manifest": (
        "reviewer_defense_sensitivities_manifest.json",
        "post_hoc_reviewer_defense_manifest",
    ),
    "reviewer_conditional_table": (
        "table_supp_conditional_reliability_families.tex",
        "post_hoc_reviewer_defense_table",
    ),
    "reviewer_overlap_table": (
        "table_supp_cross_endpoint_test_overlap.tex",
        "post_hoc_reviewer_defense_table",
    ),
    "reviewer_lomo_table": (
        "table_supp_leave_one_model_out.tex",
        "post_hoc_reviewer_defense_table",
    ),
}
FINAL_REVIEWER_DEFENSE_SCOPE = "post_hoc_reviewer_defense_sensitivities"
FINAL_REVIEWER_DEFENSE_BOOTSTRAP_REPS = 5000
FINAL_REVIEWER_DEFENSE_PERMUTATION_REPS = 20000
FINAL_REVIEWER_DEFENSE_SELECTION_BOOTSTRAP_REPS = 1000
FINAL_REVIEWER_DEFENSE_RANDOM_SEED = 20260715
FINAL_REVIEWER_DEFENSE_THRESHOLDS = (0.3, 0.5, 0.7)
FINAL_REVIEWER_DEFENSE_FAMILIES = {
    "fingerprint": ("rf", "xgboost", "lightgbm"),
    "neural": ("gin", "chemprop_dmpnn"),
}
FINAL_REVIEWER_DEFENSE_DIMENSIONS = (
    "applicability_alignment",
    "brier",
    "conformal_coverage_error",
    "ece",
    "selective_relative_excess_aurc",
    "selective_relative_excess_aurc_log_loss",
)
FINAL_REVIEWER_DEFENSE_CONDITIONAL_METRICS = (
    "log_error_ratio_outside_inside",
    "uq_correlation_advantage_inside",
    "selective_gain_advantage_inside",
    "ece_gap_outside_inside",
    "brier_gap_outside_inside",
    "coverage_error_gap_outside_inside",
)
FINAL_REVIEWER_DEFENSE_EXPECTED_ROWS = {
    "reviewer_overlap_endpoints": 22,
    "reviewer_overlap_components": 1,
    "reviewer_overlap_pairs": 231,
    "reviewer_lomo_summary": 30,
    "reviewer_lomo_endpoints": 460,
    "reviewer_lomo_probabilities": 440,
    "reviewer_lomo_selections": 110,
    "reviewer_conditional_rows": 330,
    "reviewer_conditional_summary": 36,
}

FINAL_PUBLIC_RESULT_OUTPUTS = {
    "applicability_domain_all_seeds": (
        "results/tables/applicability_domain_locked_cache_all_seeds.csv",
        "public_aggregate_result",
    ),
    "reliability_regret_gate": (
        "results/tables/reliability_regret_final/reliability_regret_gate.json",
        "public_aggregate_result_gate",
    ),
    "risk_coverage_by_sample": (
        "results/tables/reliability_regret_final/risk_coverage_by_sample.csv",
        "public_structure_free_curve_result",
    ),
    "method_effects_summary": (
        "results/tables/trustworthiness_alignment_final/method_effects_summary.csv",
        "public_aggregate_result",
    ),
    "rank_alignment_summary": (
        "results/tables/trustworthiness_alignment_final/rank_alignment_summary.csv",
        "public_aggregate_result",
    ),
}

FINAL_CLEANED_TASKS = (
    "Caco2_Wang",
    "BBB_Martins",
    "PPBR_AZ",
    "Solubility_AqSolDB",
    "CYP2C9_Veith",
    "DILI",
    "AMES",
    "LD50_Zhu",
)
FINAL_CLEANED_MODELS = TRADITIONAL_MODELS
FINAL_CLEANED_SEEDS = SEEDS
FINAL_CLEANED_PREDICTION_COUNT = (
    len(FINAL_CLEANED_TASKS) * len(FINAL_CLEANED_MODELS) * len(FINAL_CLEANED_SEEDS)
)
CLEANED_PREDICTION_AUDIT_TRUE_COLUMNS = (
    "row_count_matches_run",
    "finite_labels_and_predictions",
    "classification_probability_range_ok",
    "canonical_smiles_complete",
    "canonical_smiles_unique",
    "source_common_test_exact",
    "official_metric_matches_run",
    "cleaned_metric_matches_run",
    "paired_delta_matches_run",
)
CLEANED_PREDICTION_AUDIT_REQUIRED_COLUMNS = {
    "task",
    "model",
    "seed",
    "prediction_path",
    "prediction_sha256",
    "prediction_bytes",
    "rows",
    *CLEANED_PREDICTION_AUDIT_TRUE_COLUMNS,
}
SENSITIVITY_COMPLETE_STATUSES = {
    "SENSITIVITY_STABLE",
    "SENSITIVITY_MIXED",
    "SENSITIVITY_SENSITIVE",
}
INTEGER_TEXT_RE = re.compile(r"[+-]?(?:0|[1-9][0-9]*)")
PUBLIC_CONFIG_FORBIDDEN_KEY_RE = re.compile(
    r"(?:^|_)(?:command|commands|cmd|argv|cli|invocation|executable|binary|"
    r"python_?(?:bin|executable|path)|password|passwd|token|secret|api_?key)(?:$|_)",
    re.IGNORECASE,
)
PUBLIC_CONFIG_RAW_PAYLOAD_KEYS = frozenset(
    {
        "smiles",
        "source_smiles",
        "model_input_smiles",
        "canonical_smiles",
        "sample_index",
        "label",
        "y_true",
        # The complete repair list also carries structure-adjacent identifiers
        # (for example standard InChIKey and split/index detail).  Public
        # configs retain only the aggregate count, policy, and split hashes.
        "model_input_repairs",
    }
)
PUBLIC_RAW_PAYLOAD_JSON_KEY_RE = re.compile(
    r'"(?:'
    + "|".join(re.escape(key) for key in sorted(PUBLIC_CONFIG_RAW_PAYLOAD_KEYS))
    + r')"\s*:',
    re.IGNORECASE,
)
EMBEDDED_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s=:'\"])/(?:tmp|Users|Volumes|home|var|opt|usr|private)(?:/|$)"
)
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s=:'\"])[A-Za-z]:[\\/]")
SENSITIVE_CONFIG_VALUE_RE = re.compile(
    r"(?i)(?:password|passwd|token|secret|api[_-]?key|authorization|bearer)\s*[:=]"
)
PUBLIC_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_+./~-])/(?!/)(?=[^\s\"'<>])"
)
PUBLIC_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|"
    r"\\\\[A-Za-z0-9._$-]+\\[A-Za-z0-9._$ -]+"
    r"(?:\\[A-Za-z0-9._$ -]+)*)"
)
PRIVATE_HOST_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s=:'\"])/(?:Users|Volumes|home|private/var/folders)(?:/|$)"
)
PUBLIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:[\"']?\b(?:password|passwd|token|secret|api[_-]?key|"
    r"authorization|bearer)\b[\"']?\s*[:=])"
)
PUBLIC_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:password|passwd|token|secret|api_?key|authorization|bearer)(?:$|_)",
    re.IGNORECASE,
)
PUBLIC_PACKAGE_FORBIDDEN_DATA_KEYS = frozenset(
    {
        *PUBLIC_CONFIG_RAW_PAYLOAD_KEYS,
        "example_smiles",
        "substructure_smiles",
    }
)
REFERENCE_AUDIT_SELF_SUPERSEDED_RE = re.compile(
    r"^\s*(?:>\s*)?(?:(?:#{1,6}\s*)|(?:[-*]\s*))?"
    r"(?:\*{1,2})?(?:(?:audit\s+)?status\s*:\s*)?"
    r"(?:\*{1,2})?superseded\b",
    re.IGNORECASE | re.MULTILINE,
)

TRADITIONAL_PREDICTION_COLUMNS = {
    "task",
    "task_type",
    "official_metric",
    "model",
    "seed",
    "split",
    "sample_index",
    "smiles",
    "y_true",
    "y_pred",
}
NEURAL_PREDICTION_COLUMNS = {"split", "smiles", "label", "prediction"}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RESOLVED_POLICY_STATUSES = {"APPROVED", "COMPLETE", "NOT_APPLICABLE"}
TODO_MARKERS = {"", "TODO", "TBD", "UNKNOWN", "UNDECIDED"}
CHECKPOINT_NOT_RELEASED_DECISION = "NOT_RELEASED"
CHECKPOINT_NOT_RELEASED_STATUSES = {"APPROVED", "NOT_APPLICABLE"}
LABEL_ALIGNMENT_RTOL = 1e-6
LABEL_ALIGNMENT_ATOL = 1e-6
SOLUBILITY_TASK = "solubility_aqsoldb"
MODEL_INPUT_SPLIT_HASHES = ("train_sha256", "valid_sha256", "test_sha256")
SOURCE_SPLIT_HASHES = (
    "source_train_sha256",
    "source_valid_sha256",
    "source_test_sha256",
)
_SOLUBILITY_SOURCE_SMILES = (
    "CC1=CC=C[NH+2]([O-])[CH-]1",
    "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1",
)
SOLUBILITY_COMPATIBILITY_REPAIRS = tuple(
    {
        "split": "test",
        "sample_index": sample_index,
        "source_smiles": source_smiles,
        "model_input_smiles": LEGACY_VALENCE_REPAIRS[source_smiles],
        "standard_inchikey": LEGACY_VALENCE_REPAIR_INCHIKEYS[source_smiles],
        "policy": SMILES_COMPATIBILITY_POLICY,
    }
    for sample_index, source_smiles in zip((1159, 1160), _SOLUBILITY_SOURCE_SMILES)
)


@dataclass(frozen=True)
class PredictionAudit:
    valid: bool
    rows: int
    valid_rows: int
    test_rows: int
    split_sha256: str
    label_alignment_rows: tuple[tuple[str, int, float], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class FileInfo:
    path: str
    sha256: str
    bytes: int | None
    exists: bool


@dataclass(frozen=True)
class FileSnapshot:
    """One immutable byte read used for both parsing and release hashing."""

    path: Path
    relative_path: str
    data: bytes
    sha256: str

    @property
    def bytes(self) -> int:
        return len(self.data)

    @property
    def info(self) -> FileInfo:
        return FileInfo(self.relative_path, self.sha256, self.bytes, True)


@dataclass(frozen=True)
class SupplementalFile:
    info: FileInfo
    role: str
    package_action: str = "include"


@dataclass(frozen=True)
class CleanedPredictionSource:
    task: str
    model: str
    seed: int
    rows: int
    snapshot: FileSnapshot


@dataclass(frozen=True)
class FinalSupplementAudit:
    complete: bool
    failures: tuple[str, ...]
    files: tuple[SupplementalFile, ...]
    details: dict[str, Any]
    cleaned_predictions: tuple[CleanedPredictionSource, ...] = ()
    cleaned_source_data_sha256: str = ""
    public_metadata_payloads: tuple[tuple[str, bytes], ...] = ()


@dataclass(frozen=True)
class PublicPredictionViewAudit:
    complete: bool
    failures: tuple[str, ...]
    files: tuple[SupplementalFile, ...]
    details: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_task(value: str) -> str:
    return str(value).strip().lower()


def stable_json_hash(payload: Any) -> str:
    material = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    """Parse an integer without accepting booleans or truncating decimals."""

    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not INTEGER_TEXT_RE.fullmatch(text):
            raise ValueError(f"{label} must be a canonical integer, got {value!r}")
        result = int(text)
    elif isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} must be an integer, got {value!r}")
        result = int(numeric)
    else:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {result}")
    return result


def _strict_true(value: Any, label: str) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() == "true":
        return True
    raise ValueError(f"{label} must be true, got {value!r}")


def _is_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


class SupplementSnapshotCatalog:
    """Read each supplemental file once and retain its immutable snapshot."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._snapshots: dict[Path, FileSnapshot] = {}
        self._roles: defaultdict[Path, set[str]] = defaultdict(set)
        self._actions: defaultdict[Path, set[str]] = defaultdict(set)

    def _relative(self, path: Path) -> tuple[Path, str]:
        absolute = path.resolve()
        try:
            relative = absolute.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError(f"supplement path escapes project root: {path}") from exc
        if _is_appledouble(absolute):
            raise ValueError(f"AppleDouble sidecar is not releasable: {relative}")
        return absolute, relative

    def require(
        self,
        path: Path,
        role: str,
        failures: list[str],
        *,
        package_action: str = "include",
    ) -> FileSnapshot | None:
        try:
            absolute, relative = self._relative(path)
        except ValueError as exc:
            failures.append(str(exc))
            return None
        if absolute not in self._snapshots:
            try:
                data = absolute.read_bytes()
            except OSError as exc:
                failures.append(f"missing or unreadable supplement: {relative} ({exc})")
                return None
            self._snapshots[absolute] = FileSnapshot(
                path=absolute,
                relative_path=relative,
                data=data,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        self._roles[absolute].add(role)
        self._actions[absolute].add(package_action)
        return self._snapshots[absolute]

    def files(self) -> tuple[SupplementalFile, ...]:
        rows: list[SupplementalFile] = []
        for absolute, snapshot in sorted(
            self._snapshots.items(), key=lambda item: item[1].relative_path
        ):
            actions = self._actions[absolute]
            if "exclude_raw_structure_and_labels" in actions:
                action = "exclude_raw_structure_and_labels"
            elif "exclude_nonportable" in actions:
                action = "exclude_nonportable"
            elif "generate_public_prediction_view" in actions:
                action = "generate_public_prediction_view"
            else:
                action = "include"
            rows.append(
                SupplementalFile(
                    info=snapshot.info,
                    role=";".join(sorted(self._roles[absolute])),
                    package_action=action,
                )
            )
        return tuple(rows)


def _json_object(snapshot: FileSnapshot, label: str) -> dict[str, Any]:
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root is not an object")
    return value


def _csv_rows(snapshot: FileSnapshot, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    if any(not line.strip() for line in text.splitlines()):
        raise ValueError(f"{label} contains a blank row")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"{label} is malformed CSV: {exc}") from exc
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"{label} has a missing or duplicate header")
    if any(None in row for row in rows):
        raise ValueError(f"{label} contains a truncated or over-wide row")
    return columns, rows


def _csv_shape_streaming(snapshot: FileSnapshot, label: str) -> tuple[int, list[str]]:
    """Validate CSV width while counting large evidence tables without row storage."""

    try:
        text = io.TextIOWrapper(
            io.BytesIO(snapshot.data), encoding="utf-8-sig", newline=""
        )
        reader = csv.reader(text)
        columns = next(reader, [])
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"{label} has a missing or duplicate header")
        rows = 0
        for row in reader:
            if not row or all(not str(value).strip() for value in row):
                raise ValueError(f"{label} contains a blank row")
            if len(row) != len(columns):
                raise ValueError(
                    f"{label} row {reader.line_num} has {len(row)} columns, "
                    f"expected {len(columns)}"
                )
            rows += 1
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    except csv.Error as exc:
        raise ValueError(f"{label} is malformed CSV: {exc}") from exc
    return rows, list(columns)


def _strip_latex_comment(line: str) -> str:
    """Remove an unescaped TeX comment while preserving escaped percent signs."""

    for index, character in enumerate(line):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return line[:index]
    return line


def _split_latex_alignment_cells(row: str) -> list[str]:
    r"""Split one TeX row at alignment tabs, excluding escaped ``\&`` values."""

    cells: list[str] = []
    start = 0
    for index, character in enumerate(row):
        if character != "&":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and row[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 1:
            continue
        cells.append(row[start:index].strip())
        start = index + 1
    cells.append(row[start:].strip())
    return cells


def _latex_longtable_shape(snapshot: FileSnapshot, label: str) -> tuple[int, int]:
    r"""Return body-row and column counts for the locked standalone longtable.

    Longtable repeats its header and defines a continuation footer inside the
    environment.  Those rows are presentation metadata, not table data.  The
    locked serializer places data strictly after ``\endlastfoot``; requiring
    the complete marker sequence makes this parser fail closed if that contract
    changes instead of silently counting headers or footers as observations.
    """

    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc

    lines = [_strip_latex_comment(line).strip() for line in text.splitlines()]
    nonempty = [(line_number, line) for line_number, line in enumerate(lines, start=1) if line]
    if not nonempty:
        raise ValueError(f"{label} is empty")

    begin_matches: list[tuple[int, re.Match[str]]] = []
    begin_pattern = re.compile(r"\\begin\{longtable\}\{([lcr]+)\}")
    for line_number, line in nonempty:
        match = begin_pattern.fullmatch(line)
        if match is not None:
            begin_matches.append((line_number, match))
    if len(begin_matches) != 1:
        raise ValueError(f"{label} must contain exactly one supported longtable")

    markers = (
        r"\endfirsthead",
        r"\endhead",
        r"\endfoot",
        r"\endlastfoot",
        r"\end{longtable}",
    )
    marker_lines: dict[str, int] = {}
    for marker in markers:
        matches = [line_number for line_number, line in nonempty if line == marker]
        if len(matches) != 1:
            raise ValueError(f"{label} must contain exactly one {marker} marker")
        marker_lines[marker] = matches[0]

    begin_line, begin_match = begin_matches[0]
    ordered_lines = [
        begin_line,
        marker_lines[r"\endfirsthead"],
        marker_lines[r"\endhead"],
        marker_lines[r"\endfoot"],
        marker_lines[r"\endlastfoot"],
        marker_lines[r"\end{longtable}"],
    ]
    if ordered_lines != sorted(ordered_lines) or len(set(ordered_lines)) != len(ordered_lines):
        raise ValueError(f"{label} longtable markers are out of order")
    if nonempty[0][0] != begin_line or nonempty[-1][0] != marker_lines[r"\end{longtable}"]:
        raise ValueError(f"{label} contains content outside the longtable environment")

    expected_columns = len(begin_match.group(1))
    body_start = marker_lines[r"\endlastfoot"]
    body_end = marker_lines[r"\end{longtable}"]
    row_count = 0
    for line_number, line in nonempty:
        if not body_start < line_number < body_end:
            continue
        if not line.endswith(r"\\"):
            raise ValueError(
                f"{label} line {line_number} is not a complete longtable data row"
            )
        payload = line[:-2].rstrip()
        if not payload or r"\\" in payload or payload.startswith(r"\multicolumn"):
            raise ValueError(f"{label} line {line_number} is not a plain data row")
        cells = _split_latex_alignment_cells(payload)
        if any(not cell for cell in cells):
            raise ValueError(f"{label} line {line_number} contains an empty data cell")
        if len(cells) != expected_columns:
            raise ValueError(
                f"{label} line {line_number} has {len(cells)} columns, "
                f"expected {expected_columns}"
            )
        row_count += 1

    if row_count == 0:
        raise ValueError(f"{label} contains no longtable data rows")
    return row_count, expected_columns


def _manuscript_table_shape(snapshot: FileSnapshot, label: str) -> tuple[int, int]:
    """Validate a reader-visible table and return ``(rows, columns)``."""

    suffix = snapshot.path.suffix.lower()
    if suffix == ".csv":
        columns, rows = _csv_rows(snapshot, label)
        return len(rows), len(columns)
    if suffix == ".tex":
        return _latex_longtable_shape(snapshot, label)
    raise ValueError(f"{label} has unsupported table format {suffix or '<none>'}")


def _prediction_inventory_digest(records: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(record) for record in records), key=lambda row: row["filename"])
    return stable_json_hash(ordered)


_DROP_PUBLIC_METADATA_VALUE = object()


def _is_absolute_path_text(value: str) -> bool:
    """Return whether a complete string is a POSIX or Windows absolute path."""

    text = value.strip()
    if not text:
        return False
    return Path(text).is_absolute() or PureWindowsPath(text).is_absolute()


def _contains_absolute_path_text(value: str) -> bool:
    """Detect embedded POSIX/Windows absolute paths without flagging URLs."""

    text = value.strip()
    return bool(
        _is_absolute_path_text(text)
        or PUBLIC_POSIX_ABSOLUTE_PATH_RE.search(text)
        or PUBLIC_WINDOWS_ABSOLUTE_PATH_RE.search(text)
    )


def _repository_logical_path(value: str, root: Path) -> str | None:
    """Map one complete absolute path to a repository-relative logical path.

    Paths outside ``root`` deliberately return ``None``.  They must never be
    reduced to a basename because that would both leak provenance semantics and
    permit unrelated source paths to collide in a public hash mapping.
    """

    text = value.strip()
    if PureWindowsPath(text).is_absolute() and not Path(text).is_absolute():
        return None
    if not Path(text).is_absolute():
        return text
    try:
        relative = Path(text).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    logical = relative.as_posix()
    return logical or "."


def _sanitize_public_metadata_string(value: str, root: Path) -> str | object:
    """Logicalize in-repository paths and drop nonportable/sensitive strings."""

    text = value.strip()
    if not text:
        return value
    if PUBLIC_SECRET_ASSIGNMENT_RE.search(text):
        return _DROP_PUBLIC_METADATA_VALUE
    if _is_absolute_path_text(text):
        logical = _repository_logical_path(text, root)
        return logical if logical is not None else _DROP_PUBLIC_METADATA_VALUE

    # Some provenance strings embed a project path in a longer diagnostic.  A
    # project-root prefix can be made portable without losing its logical
    # meaning.  Any remaining absolute path is external/nonportable and the
    # complete string is omitted.
    root_text = root.resolve().as_posix()
    portable = value.replace(root_text + "/", "")
    if portable.strip() == root_text:
        portable = "."
    elif root_text in portable:
        portable = portable.replace(root_text, ".")
    if _contains_absolute_path_text(portable):
        return _DROP_PUBLIC_METADATA_VALUE
    return portable


def _sanitize_public_metadata_tree(
    value: Any, root: Path, *, location: str = "$"
) -> Any:
    """Recursively sanitize metadata values *and mapping keys*.

    Repository-internal absolute paths become repository-relative logical
    paths.  External absolute paths and secret assignments are omitted.  Key
    normalization is collision checked so two source paths can never be
    silently collapsed into one public hash entry.
    """

    if isinstance(value, list):
        output = []
        for index, item in enumerate(value):
            sanitized = _sanitize_public_metadata_tree(
                item, root, location=f"{location}[{index}]"
            )
            if sanitized is not _DROP_PUBLIC_METADATA_VALUE:
                output.append(sanitized)
        return output
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.strip().lower().replace("-", "_")
            if PUBLIC_SECRET_KEY_RE.search(normalized_key):
                continue
            if key == "ignored_appledouble_sidecars" and isinstance(item, list):
                public_key = "ignored_appledouble_sidecars_count"
                if public_key in output:
                    raise ValueError(
                        f"public metadata key collision at {location}: {public_key}"
                    )
                output[public_key] = len(item)
                continue
            public_key_value = _sanitize_public_metadata_string(key, root)
            if public_key_value is _DROP_PUBLIC_METADATA_VALUE:
                continue
            public_key = str(public_key_value)
            if public_key in output:
                raise ValueError(
                    f"public metadata key collision after path logicalization at {location}"
                )
            sanitized = _sanitize_public_metadata_tree(
                item, root, location=f"{location}.{public_key}"
            )
            if sanitized is not _DROP_PUBLIC_METADATA_VALUE:
                output[public_key] = sanitized
        return output
    if isinstance(value, str):
        return _sanitize_public_metadata_string(value, root)
    return value


def _sanitize_public_metadata_payload(name: str, payload: bytes, root: Path) -> bytes:
    """Sanitize a JSON/CSV metadata payload at the public-tree boundary."""

    suffix = Path(name).suffix.lower()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"public metadata {name} is not UTF-8") from exc
    if suffix == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"public metadata {name} is not valid JSON") from exc
        sanitized = _sanitize_public_metadata_tree(parsed, root)
        if sanitized is _DROP_PUBLIC_METADATA_VALUE:
            raise ValueError(f"public metadata {name} has no portable root payload")
        return (
            json.dumps(sanitized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        source_columns = list(reader.fieldnames or [])
        if not source_columns:
            raise ValueError(f"public metadata {name} has no CSV header")
        column_map: dict[str, str] = {}
        public_columns: list[str] = []
        for source_column in source_columns:
            public_column_value = _sanitize_public_metadata_string(source_column, root)
            if public_column_value is _DROP_PUBLIC_METADATA_VALUE:
                continue
            public_column = str(public_column_value)
            if public_column in public_columns:
                raise ValueError(
                    f"public metadata CSV column collision after path logicalization: {name}"
                )
            column_map[source_column] = public_column
            public_columns.append(public_column)
        if not public_columns:
            raise ValueError(f"public metadata {name} has no portable CSV columns")
        public_rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            public_row: dict[str, Any] = {}
            for source_column, public_column in column_map.items():
                sanitized = _sanitize_public_metadata_tree(
                    row.get(source_column, ""),
                    root,
                    location=f"{name}:{row_number}:{public_column}",
                )
                public_row[public_column] = (
                    "" if sanitized is _DROP_PUBLIC_METADATA_VALUE else sanitized
                )
            public_rows.append(public_row)
        return _serialize_public_csv(public_columns, public_rows)
    raise ValueError(f"unsupported public metadata payload type: {name}")


def _sanitize_public_config(value: Any, root: Path) -> Any:
    """Remove raw samples, commands, secrets, and machine paths from a config."""

    if isinstance(value, list):
        output = []
        for item in value:
            sanitized = _sanitize_public_config(item, root)
            if sanitized is not None:
                output.append(sanitized)
        return output
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if (
                normalized in PUBLIC_CONFIG_RAW_PAYLOAD_KEYS
                or PUBLIC_CONFIG_FORBIDDEN_KEY_RE.search(normalized)
            ):
                continue
            sanitized = _sanitize_public_config(item, root)
            if sanitized is not None:
                output[str(key)] = sanitized
        return output
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if Path(text).is_absolute():
            try:
                return Path(text).resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                return None
        if (
            EMBEDDED_ABSOLUTE_PATH_RE.search(text)
            or WINDOWS_ABSOLUTE_PATH_RE.search(text)
            or SENSITIVE_CONFIG_VALUE_RE.search(text)
        ):
            return None
        return value
    return value


def _public_config_is_clean(value: Any) -> bool:
    if isinstance(value, list):
        return all(_public_config_is_clean(item) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if (
                normalized in PUBLIC_CONFIG_RAW_PAYLOAD_KEYS
                or PUBLIC_CONFIG_FORBIDDEN_KEY_RE.search(normalized)
            ):
                return False
            if not _public_config_is_clean(item):
                return False
        return True
    if isinstance(value, str):
        return (
            not Path(value.strip()).is_absolute()
            and not bool(EMBEDDED_ABSOLUTE_PATH_RE.search(value))
            and not bool(WINDOWS_ABSOLUTE_PATH_RE.search(value))
            and not bool(SENSITIVE_CONFIG_VALUE_RE.search(value))
        )
    return True


class ArtifactHasher:
    """Streaming file hashing with a per-build cache."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._cache: dict[Path, FileInfo] = {}

    def relative(self, path: Path) -> str:
        absolute = path.resolve()
        try:
            return absolute.relative_to(self.root).as_posix()
        except ValueError:
            return Path(os.path.relpath(absolute, self.root)).as_posix()

    def inspect(self, path: Path | None) -> FileInfo:
        if path is None:
            return FileInfo("", "", None, False)
        absolute = path.resolve()
        if absolute in self._cache:
            return self._cache[absolute]
        relative = self.relative(absolute)
        if not absolute.is_file():
            info = FileInfo(relative, "", None, False)
        else:
            digest = hashlib.sha256()
            with absolute.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            info = FileInfo(relative, digest.hexdigest(), absolute.stat().st_size, True)
        self._cache[absolute] = info
        return info


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable JSON: {exc}"
    if not isinstance(value, dict):
        return None, "JSON root is not an object"
    return value, None


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _hash_split_rows(rows: Iterable[tuple[str, int, str]]) -> str:
    """Hash sample identity only; labels are checked numerically in the gate."""
    digest = hashlib.sha256()
    for split, sample_index, smiles in rows:
        record = [split, sample_index, smiles]
        digest.update(
            json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def audit_prediction(
    path: Path,
    *,
    model: str,
    task: str,
    seed: int,
    expected_valid_n: int | None,
    expected_test_n: int | None,
    task_type: str,
) -> PredictionAudit:
    """Validate schema, metadata, held-out row counts, and finite values."""
    errors: list[str] = []
    if not path.is_file():
        return PredictionAudit(False, 0, 0, 0, "", (), ("prediction missing",))

    traditional = model in TRADITIONAL_MODELS
    required = TRADITIONAL_PREDICTION_COLUMNS if traditional else NEURAL_PREDICTION_COLUMNS
    label_column = "y_true" if traditional else "label"
    prediction_column = "y_pred" if traditional else "prediction"
    split_counts: Counter[str] = Counter()
    split_ordinals: Counter[str] = Counter()
    split_material: list[tuple[str, int, str]] = []
    label_material: list[tuple[str, int, float]] = []
    row_count = 0
    sample_indices: defaultdict[str, set[str]] = defaultdict(set)

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = sorted(required - fields)
            if missing:
                return PredictionAudit(
                    False,
                    0,
                    0,
                    0,
                    "",
                    (),
                    (f"missing columns: {','.join(missing)}",),
                )
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                split = str(row.get("split", "")).strip().lower()
                if split not in {"valid", "test"}:
                    errors.append(f"line {line_number}: invalid split {split!r}")
                    continue
                split_counts[split] += 1
                ordinal = split_ordinals[split]
                split_ordinals[split] += 1
                sample_index = ordinal
                smiles = str(row.get("smiles", ""))
                if not smiles:
                    errors.append(f"line {line_number}: empty smiles")
                label = row.get(label_column)
                prediction = row.get(prediction_column)
                if not _finite_number(label):
                    errors.append(f"line {line_number}: non-finite label")
                    continue
                if not _finite_number(prediction):
                    errors.append(f"line {line_number}: non-finite prediction")
                    continue
                pred_value = float(prediction)
                if task_type == "classification" and not 0.0 <= pred_value <= 1.0:
                    errors.append(f"line {line_number}: classification prediction outside [0,1]")
                if traditional:
                    if normalize_task(row.get("task", "")) != normalize_task(task):
                        errors.append(f"line {line_number}: task metadata mismatch")
                    if str(row.get("model", "")).strip() != model:
                        errors.append(f"line {line_number}: model metadata mismatch")
                    try:
                        row_seed = int(row.get("seed", ""))
                    except (TypeError, ValueError):
                        row_seed = None
                    if row_seed != seed:
                        errors.append(f"line {line_number}: seed metadata mismatch")
                    try:
                        parsed_sample_index = int(row.get("sample_index", ""))
                    except (TypeError, ValueError):
                        parsed_sample_index = -1
                    if parsed_sample_index < 0:
                        errors.append(f"line {line_number}: invalid sample_index")
                    sample_index = parsed_sample_index
                    if str(sample_index) in sample_indices[split]:
                        errors.append(f"line {line_number}: duplicate sample_index in {split}")
                    sample_indices[split].add(str(sample_index))
                split_material.append((split, sample_index, smiles))
                label_material.append((split, sample_index, float(label)))
    except (OSError, csv.Error, UnicodeError) as exc:
        errors.append(f"unreadable CSV: {exc}")

    valid_n = split_counts["valid"]
    test_n = split_counts["test"]
    if expected_valid_n is not None and valid_n != expected_valid_n:
        errors.append(f"valid rows {valid_n} != expected {expected_valid_n}")
    if expected_test_n is not None and test_n != expected_test_n:
        errors.append(f"test rows {test_n} != expected {expected_test_n}")
    if valid_n == 0 or test_n == 0:
        errors.append("both valid and test splits are required")
    split_hash = _hash_split_rows(split_material) if len(split_material) == row_count else ""
    return PredictionAudit(
        not errors,
        row_count,
        valid_n,
        test_n,
        split_hash,
        tuple(label_material),
        tuple(errors),
    )


def count_csv_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            next(reader)
            return sum(1 for _ in reader)
    except (OSError, csv.Error, StopIteration, UnicodeError):
        return None


def source_data_info(root: Path, task: str, hasher: ArtifactHasher) -> dict[str, Any]:
    task_dir = root / "data" / "tdc_official" / "admet_group" / normalize_task(task)
    train_val = task_dir / "train_val.csv"
    test = task_dir / "test.csv"
    train_info = hasher.inspect(train_val)
    test_info = hasher.inspect(test)
    train_val_n = count_csv_rows(train_val)
    test_n = count_csv_rows(test)
    valid_n = math.ceil(train_val_n * 0.125) if train_val_n is not None else None
    source_hash = ""
    if train_info.exists and test_info.exists:
        source_hash = stable_json_hash(
            {
                "train_val": train_info.sha256,
                "test": test_info.sha256,
            }
        )
    return {
        "train_val_path": train_info.path,
        "train_val_sha256": train_info.sha256,
        "test_path": test_info.path,
        "test_sha256": test_info.sha256,
        "source_data_sha256": source_hash,
        "train_val_n": train_val_n,
        "valid_n": valid_n,
        "test_n": test_n,
    }


def load_traditional_metrics(
    path: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, str]], list[str]]:
    rows: dict[tuple[str, str, int], dict[str, str]] = {}
    errors: list[str] = []
    if not path.is_file():
        return rows, [f"traditional metrics missing: {path}"]
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"task", "model", "seed", "valid_n", "test_n", "official_metric"}
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                return rows, [f"traditional metrics missing columns: {','.join(missing)}"]
            for line_number, row in enumerate(reader, start=2):
                try:
                    key = (
                        normalize_task(row["task"]),
                        row["model"].strip(),
                        int(row["seed"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"traditional metrics line {line_number}: invalid key ({exc})")
                    continue
                if key in rows:
                    errors.append(f"traditional metrics duplicate key: {key}")
                    continue
                official = str(row.get("official_metric", "")).strip().lower()
                for prefix in ("valid", "test"):
                    metric_column = f"{prefix}_{official}"
                    if metric_column not in row or not _finite_number(row[metric_column]):
                        errors.append(
                            f"traditional metrics {key}: non-finite {metric_column}"
                        )
                rows[key] = row
    except (OSError, csv.Error, UnicodeError) as exc:
        errors.append(f"traditional metrics unreadable: {exc}")
    return rows, errors


def _run_artifact_paths(base: Path, task: str, model: str, seed: int) -> dict[str, Path]:
    stem = f"{task}_{model}_seed{seed}"
    return {
        "prediction": base / "predictions" / f"{stem}.csv",
        "metrics": base / "metrics" / f"{stem}.json",
        "config": base / "configs" / f"{stem}.json",
        "checkpoint": base / "checkpoints" / f"{stem}.pt",
    }


def choose_neural_root(
    roots: Sequence[Path], task: str, source_model: str, seed: int
) -> tuple[Path, dict[str, Path]]:
    """Choose one provenance root; never assemble a run across directories."""
    candidates = []
    for index, root in enumerate(roots):
        paths = _run_artifact_paths(root, task, source_model, seed)
        score = sum(paths[name].is_file() for name in ("prediction", "metrics", "checkpoint"))
        candidates.append((score, -index, root, paths))
    if not candidates:
        root = ROOT / "results" / "neural"
        return root, _run_artifact_paths(root, task, source_model, seed)
    _, _, root, paths = max(candidates, key=lambda item: (item[0], item[1]))
    return root, paths


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _valid_sha256(value: Any) -> bool:
    return bool(HASH_RE.fullmatch(str(value or "")))


def validate_gin_compatibility_config(path: Path) -> list[str]:
    """Require the release-indexed GIN config to disclose the parser policy."""

    if not path.is_file():
        return ["GIN global config missing"]
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"GIN global config unreadable: {type(exc).__name__}"]
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return ["GIN global config data manifest missing"]
    if data.get("model_input_representation_policy") != SMILES_COMPATIBILITY_POLICY:
        return ["GIN global config parser compatibility policy mismatch"]
    return []


def validate_chemprop_global_compatibility_config(path: Path) -> list[str]:
    """Require the release-indexed Chemprop YAML to lock the parser policy."""

    if not path.is_file():
        return ["Chemprop global config missing"]
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Chemprop global config unreadable: {type(exc).__name__}"]
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return ["Chemprop global config data manifest missing"]
    if data.get("model_input_representation_policy") != SMILES_COMPATIBILITY_POLICY:
        return ["Chemprop global config parser compatibility policy mismatch"]
    return []


def _repair_projection(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    try:
        sample_index = int(value.get("sample_index"))
    except (TypeError, ValueError):
        return None
    return {
        "split": value.get("split"),
        "sample_index": sample_index,
        "source_smiles": value.get("source_smiles"),
        "model_input_smiles": value.get("model_input_smiles"),
        "standard_inchikey": value.get("standard_inchikey"),
        "policy": value.get("policy"),
    }


def validate_chemprop_compatibility_manifest(
    task: str, data: object
) -> list[str]:
    """Audit exact Solubility repairs, zero elsewhere, and six split hashes."""

    if not isinstance(data, dict):
        return ["Chemprop config data manifest missing"]
    errors: list[str] = []

    missing_model_hashes = [
        name for name in MODEL_INPUT_SPLIT_HASHES if not _valid_sha256(data.get(name))
    ]
    if missing_model_hashes:
        errors.append(
            "Chemprop model-input split hashes missing or invalid: "
            + ",".join(missing_model_hashes)
        )

    compatibility_fields = (
        "model_input_representation_policy",
        "model_input_repairs_n",
        "model_input_repairs",
        *SOURCE_SPLIT_HASHES,
    )
    is_solubility = normalize_task(task) == SOLUBILITY_TASK
    field_presence = [name in data for name in compatibility_fields]
    if not is_solubility and not any(field_presence):
        return errors
    if not all(field_presence):
        errors.append("Chemprop partial compatibility manifest")
        return errors
    if data.get("model_input_representation_policy") != SMILES_COMPATIBILITY_POLICY:
        errors.append("Chemprop parser compatibility policy mismatch")
    missing_source_hashes = [
        name for name in SOURCE_SPLIT_HASHES if not _valid_sha256(data.get(name))
    ]
    if missing_source_hashes:
        errors.append(
            "Chemprop source split hashes missing or invalid: "
            + ",".join(missing_source_hashes)
        )

    repairs = data.get("model_input_repairs")
    try:
        repairs_n = int(data.get("model_input_repairs_n"))
    except (TypeError, ValueError):
        repairs_n = -1
    if not isinstance(repairs, list) or repairs_n != len(repairs):
        errors.append("Chemprop compatibility repair count/list mismatch")
        return errors

    if is_solubility:
        observed = [_repair_projection(value) for value in repairs]
        if repairs_n != 2 or any(value is None for value in observed):
            errors.append("Chemprop Solubility exact compatibility mapping mismatch")
        else:
            assert all(value is not None for value in observed)
            observed_sorted = sorted(
                observed, key=lambda value: int(value["sample_index"])
            )
            expected = [dict(value) for value in SOLUBILITY_COMPATIBILITY_REPAIRS]
            if observed_sorted != expected:
                errors.append("Chemprop Solubility exact compatibility mapping mismatch")
            elif (
                data["source_train_sha256"] != data["train_sha256"]
                or data["source_valid_sha256"] != data["valid_sha256"]
                or data["source_test_sha256"] == data["test_sha256"]
            ):
                errors.append("Chemprop Solubility source/model hash relation mismatch")
    elif repairs_n != 0 or repairs != []:
        errors.append("Chemprop unexpected compatibility repairs for non-Solubility task")
    elif not missing_model_hashes and not missing_source_hashes and any(
        data[f"source_{name}_sha256"] != data[f"{name}_sha256"]
        for name in ("train", "valid", "test")
    ):
        errors.append("Chemprop no-repair source/model split hashes differ")
    return errors


def _base_record(model: str, task: str, seed: int, *, included: bool) -> dict[str, Any]:
    meta = TDC_TASK_META[task]
    return {
        "model": model,
        "task": task,
        "task_type": meta["task_type"],
        "official_metric": meta["metric"],
        "seed": int(seed),
        "included_default": included,
        "expected_submission_run": included,
        "run_status": "missing",
        "completion_marker": False,
        "prediction_valid": False,
        "prediction_rows": 0,
        "valid_rows": 0,
        "test_rows": 0,
        "split_sha256": "",
        "label_alignment_within_tolerance": False,
        "max_abs_label_difference_within_task_seed": None,
        "_label_alignment_rows": (),
        "config_split_sha256": "",
        "model_input_split_sha256": "",
        "source_split_sha256": "",
        "source_data_sha256": "",
        "software_sha256": "",
        "execution_spec_sha256": "",
        "representation_compatibility_valid": None,
        "representation_compatibility_policy": "",
        "compatibility_manifest_schema": "not_applicable",
        "model_input_repairs_n": None,
        "errors": [],
        "notes": "",
    }


def _attach_file(record: dict[str, Any], role: str, info: FileInfo, action: str) -> None:
    record[f"{role}_path"] = info.path
    record[f"{role}_sha256"] = info.sha256
    record[f"{role}_bytes"] = info.bytes
    record[f"{role}_package_action"] = action if info.exists else "not_ready"


def _validate_metrics_identity(
    payload: Mapping[str, Any], task: str, model: str, seed: int
) -> list[str]:
    errors: list[str] = []
    if normalize_task(payload.get("task", "")) != normalize_task(task):
        errors.append("metrics task mismatch")
    if str(payload.get("model", "")) != model:
        errors.append("metrics model mismatch")
    if _int_or_none(payload.get("seed")) != seed:
        errors.append("metrics seed mismatch")
    for name in ("valid_metric", "test_metric"):
        if not _finite_number(payload.get(name)):
            errors.append(f"metrics {name} is not finite")
    return errors


def build_traditional_record(
    *,
    root: Path,
    task: str,
    model: str,
    seed: int,
    metrics_path: Path,
    metrics_rows: Mapping[tuple[str, str, int], dict[str, str]],
    prediction_dir: Path,
    common_config: Path,
    execution_spec_sha256: str,
    software_sha256: str,
    data_info: Mapping[str, Any],
    hasher: ArtifactHasher,
) -> dict[str, Any]:
    record = _base_record(model, task, seed, included=True)
    record["software_sha256"] = software_sha256
    record["execution_spec_sha256"] = execution_spec_sha256
    record["source_data_sha256"] = data_info["source_data_sha256"]
    key = (normalize_task(task), model, seed)
    metrics_row = metrics_rows.get(key)
    prediction_path = prediction_dir / normalize_task(task) / f"{model}_seed{seed}.csv"
    prediction_info = hasher.inspect(prediction_path)
    metrics_info = hasher.inspect(metrics_path)
    config_info = hasher.inspect(common_config)
    _attach_file(record, "prediction", prediction_info, "include")
    _attach_file(record, "metrics", metrics_info, "include_shared_table")
    _attach_file(record, "config", config_info, "include")
    _attach_file(record, "checkpoint", hasher.inspect(None), "not_applicable")
    record["checkpoint_package_action"] = "not_applicable_retrain_from_seed"

    errors: list[str] = []
    expected_valid = expected_test = None
    if metrics_row is None:
        errors.append("traditional metrics row missing")
    else:
        expected_valid = _int_or_none(metrics_row.get("valid_n"))
        expected_test = _int_or_none(metrics_row.get("test_n"))
        if expected_valid is None or expected_test is None:
            errors.append("traditional metrics row counts invalid")
        if str(metrics_row.get("task_type")) != TDC_TASK_META[task]["task_type"]:
            errors.append("traditional metrics task_type mismatch")
        if str(metrics_row.get("official_metric")) != TDC_TASK_META[task]["metric"]:
            errors.append("traditional metrics official_metric mismatch")
    audit = audit_prediction(
        prediction_path,
        model=model,
        task=task,
        seed=seed,
        expected_valid_n=expected_valid,
        expected_test_n=expected_test,
        task_type=TDC_TASK_META[task]["task_type"],
    )
    errors.extend(audit.errors)
    if not metrics_info.exists:
        errors.append("traditional metrics file missing")
    if not config_info.exists:
        errors.append("traditional common config missing")
    if not data_info["source_data_sha256"]:
        errors.append("source data hash unavailable")
    record.update(
        {
            "prediction_valid": audit.valid,
            "prediction_rows": audit.rows,
            "valid_rows": audit.valid_rows,
            "test_rows": audit.test_rows,
            "split_sha256": audit.split_sha256,
            "_label_alignment_rows": audit.label_alignment_rows,
            "completion_marker": bool(metrics_row) and audit.valid,
            "errors": errors,
            "run_status": "complete" if not errors else ("missing" if not prediction_info.exists else "invalid"),
            "notes": "No checkpoint was persisted; reproducibility uses the published config, code, and seed.",
        }
    )
    return record


def build_gin_record(
    *,
    root: Path,
    task: str,
    seed: int,
    neural_roots: Sequence[Path],
    gin_config: Path,
    execution_spec_sha256: str,
    software_sha256: str,
    data_info: Mapping[str, Any],
    hasher: ArtifactHasher,
) -> dict[str, Any]:
    model = "gin"
    record = _base_record(model, task, seed, included=True)
    record["software_sha256"] = software_sha256
    record["execution_spec_sha256"] = execution_spec_sha256
    record["source_data_sha256"] = data_info["source_data_sha256"]
    _, paths = choose_neural_root(neural_roots, task, model, seed)
    prediction_info = hasher.inspect(paths["prediction"])
    metrics_info = hasher.inspect(paths["metrics"])
    checkpoint_info = hasher.inspect(paths["checkpoint"])
    config_info = hasher.inspect(gin_config)
    _attach_file(record, "prediction", prediction_info, "include")
    _attach_file(record, "metrics", metrics_info, "include")
    _attach_file(record, "config", config_info, "include")
    _attach_file(record, "checkpoint", checkpoint_info, "publish_via_lfs_or_object_store")

    errors: list[str] = []
    compatibility_errors = validate_gin_compatibility_config(gin_config)
    errors.extend(compatibility_errors)
    record["representation_compatibility_valid"] = not compatibility_errors
    record["representation_compatibility_policy"] = (
        SMILES_COMPATIBILITY_POLICY if not compatibility_errors else ""
    )
    record["compatibility_manifest_schema"] = "global_policy_v1"
    payload = None
    if not metrics_info.exists:
        errors.append("GIN metrics missing")
    else:
        payload, error = read_json(paths["metrics"])
        if error:
            errors.append(error)
        elif payload is not None:
            errors.extend(_validate_metrics_identity(payload, task, model, seed))
    audit = audit_prediction(
        paths["prediction"],
        model=model,
        task=task,
        seed=seed,
        expected_valid_n=data_info["valid_n"],
        expected_test_n=data_info["test_n"],
        task_type=TDC_TASK_META[task]["task_type"],
    )
    errors.extend(audit.errors)
    if not checkpoint_info.exists:
        errors.append("GIN checkpoint missing")
    if not config_info.exists and "GIN global config missing" not in errors:
        errors.append("GIN global config missing")
    if not data_info["source_data_sha256"]:
        errors.append("source data hash unavailable")
    completion = bool(payload is not None and audit.valid and checkpoint_info.exists)
    record.update(
        {
            "prediction_valid": audit.valid,
            "prediction_rows": audit.rows,
            "valid_rows": audit.valid_rows,
            "test_rows": audit.test_rows,
            "split_sha256": audit.split_sha256,
            "_label_alignment_rows": audit.label_alignment_rows,
            "completion_marker": completion,
            "errors": errors,
            "run_status": "complete" if not errors else ("missing" if not metrics_info.exists else "invalid"),
            "notes": "GIN completion contract: metrics JSON is written after prediction and checkpoint companions.",
        }
    )
    return record


def build_chemprop_record(
    *,
    root: Path,
    task: str,
    seed: int,
    chemprop_dir: Path,
    global_config_errors: Sequence[str],
    software_sha256: str,
    data_info: Mapping[str, Any],
    hasher: ArtifactHasher,
) -> dict[str, Any]:
    model = "chemprop_dmpnn"
    record = _base_record(model, task, seed, included=True)
    record["software_sha256"] = software_sha256
    record["source_data_sha256"] = data_info["source_data_sha256"]
    paths = _run_artifact_paths(chemprop_dir, task, model, seed)
    infos = {role: hasher.inspect(path) for role, path in paths.items()}
    for role, action in {
        "prediction": "include",
        "metrics": "include",
        "config": "include",
        "checkpoint": "publish_via_lfs_or_object_store",
    }.items():
        _attach_file(record, role, infos[role], action)

    errors: list[str] = list(global_config_errors)
    metrics = config = None
    if infos["metrics"].exists:
        metrics, error = read_json(paths["metrics"])
        if error:
            errors.append(error)
        elif metrics is not None:
            errors.extend(_validate_metrics_identity(metrics, task, model, seed))
            if metrics.get("status") != "complete":
                errors.append("Chemprop metrics status is not complete")
            if metrics.get("official_test_preserved") is not True:
                errors.append("Chemprop metrics does not confirm official test preservation")
            if metrics.get("chemprop_version") != "2.2.1":
                errors.append("Chemprop metrics version is not locked to 2.2.1")
    else:
        errors.append("Chemprop metrics missing")

    expected_valid = data_info["valid_n"]
    expected_test = data_info["test_n"]
    if infos["config"].exists:
        config, error = read_json(paths["config"])
        if error:
            errors.append(error)
        elif config is not None:
            if config.get("status") != "complete":
                errors.append("Chemprop config status is not complete")
            if normalize_task(config.get("task", "")) != normalize_task(task):
                errors.append("Chemprop config task mismatch")
            if config.get("model") != model or _int_or_none(config.get("seed")) != seed:
                errors.append("Chemprop config model/seed mismatch")
            if config.get("task_type") != TDC_TASK_META[task]["task_type"]:
                errors.append("Chemprop config task_type mismatch")
            if config.get("official_metric") != TDC_TASK_META[task]["metric"]:
                errors.append("Chemprop config official_metric mismatch")
            software = config.get("software", {})
            if not isinstance(software, dict) or software.get("chemprop") != "2.2.1":
                errors.append("Chemprop config software version is not locked to 2.2.1")
            training = config.get("training", {})
            if not isinstance(training, dict) or (
                _int_or_none(training.get("data_seed")) != seed
                or _int_or_none(training.get("pytorch_seed")) != seed
            ):
                errors.append("Chemprop config training seeds do not match run seed")
            architecture = config.get("architecture", {})
            if not isinstance(architecture, dict) or not (
                architecture.get("directed_bond_messages") is True
                and architecture.get("atom_messages") is False
                and architecture.get("undirected") is False
            ):
                errors.append("Chemprop config is not directed bond-message D-MPNN")
            config_data = config.get("data", {})
            if not isinstance(config_data, dict):
                errors.append("Chemprop config data manifest missing")
            else:
                compatibility_fields = (
                    "model_input_representation_policy",
                    "model_input_repairs_n",
                    "model_input_repairs",
                    *SOURCE_SPLIT_HASHES,
                )
                legacy_no_repair_schema = (
                    normalize_task(task) != SOLUBILITY_TASK
                    and not any(name in config_data for name in compatibility_fields)
                )
                compatibility_errors = validate_chemprop_compatibility_manifest(
                    task, config_data
                )
                errors.extend(compatibility_errors)
                record["representation_compatibility_valid"] = not (
                    global_config_errors or compatibility_errors
                )
                record["representation_compatibility_policy"] = (
                    SMILES_COMPATIBILITY_POLICY
                    if record["representation_compatibility_valid"]
                    else str(
                        config_data.get("model_input_representation_policy", "")
                    )
                )
                record["compatibility_manifest_schema"] = (
                    "legacy_no_repair_hash_identity"
                    if legacy_no_repair_schema
                    else "explicit_v1"
                )
                record["model_input_repairs_n"] = (
                    0
                    if legacy_no_repair_schema
                    else _int_or_none(config_data.get("model_input_repairs_n"))
                )
                expected_valid = _int_or_none(config_data.get("valid_n"))
                expected_test = _int_or_none(config_data.get("test_n"))
                if config_data.get("official_test_preserved") is not True:
                    errors.append("Chemprop config does not confirm official test preservation")
                model_input_hashes = {
                    name: config_data.get(f"{name}_sha256")
                    for name in ("train", "valid", "test")
                }
                source_hashes = {
                    name: config_data.get(f"source_{name}_sha256")
                    for name in ("train", "valid", "test")
                }
                if legacy_no_repair_schema:
                    source_hashes = dict(model_input_hashes)
                if all(_valid_sha256(value) for value in model_input_hashes.values()):
                    record["model_input_split_sha256"] = stable_json_hash(
                        model_input_hashes
                    )
                if all(_valid_sha256(value) for value in source_hashes.values()):
                    record["source_split_sha256"] = stable_json_hash(source_hashes)
                if record["model_input_split_sha256"] and record["source_split_sha256"]:
                    record["config_split_sha256"] = stable_json_hash(
                        {"model_input": model_input_hashes, "source": source_hashes}
                    )
            record["execution_spec_sha256"] = infos["config"].sha256
    else:
        errors.append("Chemprop config missing")

    audit = audit_prediction(
        paths["prediction"],
        model=model,
        task=task,
        seed=seed,
        expected_valid_n=expected_valid,
        expected_test_n=expected_test,
        task_type=TDC_TASK_META[task]["task_type"],
    )
    errors.extend(audit.errors)
    if metrics is not None and _int_or_none(metrics.get("prediction_rows")) != audit.rows:
        errors.append("Chemprop metrics prediction_rows mismatch")
    if not infos["checkpoint"].exists:
        errors.append("Chemprop checkpoint missing")
    if not data_info["source_data_sha256"]:
        errors.append("source data hash unavailable")
    completion = bool(
        metrics is not None
        and metrics.get("status") == "complete"
        and config is not None
        and config.get("status") == "complete"
        and audit.valid
        and infos["checkpoint"].exists
    )
    record.update(
        {
            "prediction_valid": audit.valid,
            "prediction_rows": audit.rows,
            "valid_rows": audit.valid_rows,
            "test_rows": audit.test_rows,
            "split_sha256": audit.split_sha256,
            "_label_alignment_rows": audit.label_alignment_rows,
            "completion_marker": completion,
            "errors": errors,
            "run_status": "complete" if not errors else ("missing" if not infos["metrics"].exists else "invalid"),
            "notes": "Official Chemprop v2 directed bond-message D-MPNN; metrics status=complete is the durable marker.",
        }
    )
    return record


def build_legacy_record(
    *,
    task: str,
    seed: int,
    neural_roots: Sequence[Path],
    common_config: Path,
    execution_spec_sha256: str,
    software_sha256: str,
    data_info: Mapping[str, Any],
    hasher: ArtifactHasher,
) -> dict[str, Any] | None:
    _, paths = choose_neural_root(neural_roots, task, LEGACY_SOURCE_MODEL_ID, seed)
    infos = {role: hasher.inspect(paths[role]) for role in ("prediction", "metrics", "checkpoint")}
    if not any(info.exists for info in infos.values()):
        return None
    record = _base_record(LEGACY_MODEL, task, seed, included=False)
    record["expected_submission_run"] = False
    record["software_sha256"] = software_sha256
    record["execution_spec_sha256"] = execution_spec_sha256
    record["source_data_sha256"] = data_info["source_data_sha256"]
    _attach_file(record, "prediction", infos["prediction"], "exclude_legacy")
    _attach_file(record, "metrics", infos["metrics"], "exclude_legacy")
    _attach_file(record, "config", hasher.inspect(common_config), "exclude_legacy")
    _attach_file(record, "checkpoint", infos["checkpoint"], "exclude_legacy")
    record.update(
        {
            "run_status": "excluded_legacy",
            "completion_marker": False,
            "errors": [],
            "notes": "Source model id 'dmpnn' is a node-message MPNN without directed bond states; excluded from submission.",
        }
    )
    return record


def aggregate_file_hash(paths: Sequence[Path], hasher: ArtifactHasher) -> str:
    entries = []
    for path in paths:
        info = hasher.inspect(path)
        if info.exists:
            entries.append({"path": info.path, "sha256": info.sha256})
    return stable_json_hash(entries) if entries else ""


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "available": commit is not None,
    }


def default_release_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "items": {
            "code_license": {
                "status": "TODO",
                "blocking": True,
                "decision": "TODO",
                "note": "Choose and add the repository code license.",
            },
            "generated_predictions_and_metrics_license": {
                "status": "TODO",
                "blocking": True,
                "decision": "TODO",
                "note": "Choose the license for generated predictions, metrics, and figures.",
            },
            "model_checkpoint_release_license": {
                "status": "TODO",
                "blocking": True,
                "decision": "TODO",
                "note": (
                    "If publishing checkpoints, choose a license and verify third-party "
                    "dependency notices; otherwise explicitly use decision NOT_RELEASED "
                    "with status APPROVED or NOT_APPLICABLE."
                ),
            },
            "tdc_raw_data_redistribution": {
                "status": "NOT_APPLICABLE",
                "blocking": False,
                "decision": "NO_RAW_DATA_PACKAGED",
                "note": "No raw TDC data are copied; only hashes and source-relative identifiers are indexed.",
            },
        },
    }


def load_or_create_policy(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(default_release_policy(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    payload, error = read_json(path)
    if error or payload is None:
        return {}, [error or "release policy missing"]
    items = payload.get("items")
    if not isinstance(items, dict):
        return payload, ["release policy items must be an object"]
    errors = []
    required_items = set(default_release_policy()["items"])
    for name in sorted(required_items - set(items)):
        errors.append(f"release policy item missing: {name}")
    for name, item in items.items():
        if not isinstance(item, dict):
            errors.append(f"release policy {name} is not an object")
            continue
        if item.get("blocking") and str(item.get("status", "")).upper() not in RESOLVED_POLICY_STATUSES:
            errors.append(f"release policy unresolved: {name}")
            continue
        if item.get("blocking") and str(item.get("decision", "")).strip().upper() in TODO_MARKERS:
            errors.append(f"release policy decision missing: {name}")
    checkpoint = items.get("model_checkpoint_release_license")
    if isinstance(checkpoint, dict):
        checkpoint_status = str(checkpoint.get("status", "")).strip().upper()
        checkpoint_decision = str(checkpoint.get("decision", "")).strip().upper()
        if (
            checkpoint_decision == CHECKPOINT_NOT_RELEASED_DECISION
            and checkpoint_status in RESOLVED_POLICY_STATUSES
            and checkpoint_status not in CHECKPOINT_NOT_RELEASED_STATUSES
        ):
            errors.append(
                "release policy model_checkpoint_release_license: NOT_RELEASED "
                "requires status APPROVED or NOT_APPLICABLE"
            )
    return payload, errors


def checkpoint_release_metadata(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve checkpoint packaging without choosing a policy for the owner."""
    items = policy.get("items", {}) if isinstance(policy, Mapping) else {}
    item = items.get("model_checkpoint_release_license", {})
    if not isinstance(item, Mapping):
        item = {}
    status = str(item.get("status", "TODO")).strip().upper()
    decision = str(item.get("decision", "TODO")).strip().upper()
    excluded = bool(
        decision == CHECKPOINT_NOT_RELEASED_DECISION
        and status in CHECKPOINT_NOT_RELEASED_STATUSES
    )
    resolved = bool(
        status in RESOLVED_POLICY_STATUSES and decision not in TODO_MARKERS
    )
    return {
        "status": status,
        "decision": decision,
        "policy_resolved": resolved,
        "checkpoints_excluded_by_policy": excluded,
        "checkpoint_publication_planned": resolved and not excluded,
        "package_action": (
            "exclude_by_release_policy"
            if excluded
            else "publish_via_lfs_or_object_store"
        ),
    }


def apply_checkpoint_release_policy(
    records: Sequence[dict[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only an explicit, resolved NOT_RELEASED checkpoint decision."""
    metadata = checkpoint_release_metadata(policy)
    if metadata["package_action"] != "exclude_by_release_policy":
        return metadata
    for record in records:
        if (
            record.get("checkpoint_package_action")
            == "publish_via_lfs_or_object_store"
        ):
            record["checkpoint_package_action"] = "exclude_by_release_policy"
    return metadata


def build_package_plan(
    records: Sequence[dict[str, Any]],
    support_files: Sequence[FileInfo] = (),
    supplemental_files: Sequence[SupplementalFile] = (),
) -> list[dict[str, Any]]:
    """Collapse per-run references into a unique, non-copying package plan."""
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        run_key = f"{record['model']}/{record['task']}/seed{record['seed']}"
        for role in ("prediction", "metrics", "config", "checkpoint"):
            path = record.get(f"{role}_path", "")
            if not path or not record.get(f"{role}_sha256"):
                continue
            item = grouped.setdefault(
                path,
                {
                    "relative_path": path,
                    "sha256": record[f"{role}_sha256"],
                    "bytes": record[f"{role}_bytes"],
                    "artifact_roles": set(),
                    "models": set(),
                    "run_keys": set(),
                    "included_run_keys": set(),
                    "ready_included_run_keys": set(),
                    "actions": set(),
                    "copied_by_builder": False,
                },
            )
            item["artifact_roles"].add(role)
            item["models"].add(record["model"])
            item["run_keys"].add(run_key)
            if record["included_default"]:
                item["included_run_keys"].add(run_key)
                if record["run_status"] == "complete":
                    item["ready_included_run_keys"].add(run_key)
            item["actions"].add(record[f"{role}_package_action"])

    plan = []
    for item in grouped.values():
        actions = item.pop("actions")
        run_keys = item.pop("run_keys")
        included_keys = item.pop("included_run_keys")
        ready_keys = item.pop("ready_included_run_keys")
        if "exclude_raw_structure_and_labels" in actions:
            action = "exclude_raw_structure_and_labels"
        elif "exclude_nonportable" in actions:
            action = "exclude_nonportable"
        elif "exclude_by_release_policy" in actions:
            action = "exclude_by_release_policy"
        elif included_keys and not ready_keys:
            action = "hold_not_ready"
        elif actions.intersection({"include", "include_shared_table"}):
            action = "include"
        elif "publish_via_lfs_or_object_store" in actions:
            action = "publish_via_lfs_or_object_store"
        else:
            action = "exclude_legacy"
        item.update(
            {
                "artifact_roles": ";".join(sorted(item["artifact_roles"])),
                "models": ";".join(sorted(item["models"])),
                "referencing_runs": len(run_keys),
                "included_referencing_runs": len(included_keys),
                "package_action": action,
            }
        )
        plan.append(item)
    existing = {row["relative_path"]: row for row in plan}

    def add_external(
        info: FileInfo, *, role: str, model_scope: str, package_action: str
    ) -> None:
        if not info.exists:
            return
        prior = existing.get(info.path)
        if prior is not None:
            roles = set(str(prior["artifact_roles"]).split(";"))
            roles.update(role.split(";"))
            prior["artifact_roles"] = ";".join(sorted(value for value in roles if value))
            models = set(str(prior["models"]).split(";"))
            models.add(model_scope)
            prior["models"] = ";".join(sorted(value for value in models if value))
            if package_action in {
                "exclude_raw_structure_and_labels",
                "exclude_nonportable",
            }:
                prior["package_action"] = package_action
            elif (
                package_action == "include"
                and role == "software_or_code"
                and prior["package_action"] == "exclude_nonportable"
            ):
                # An explicitly enumerated source/config dependency is part of
                # the reviewed code allowlist even when the same path also
                # served as a run config. Per-run configs are not support files
                # and remain excluded.
                prior["package_action"] = "include"
            return
        row = {
            "relative_path": info.path,
            "sha256": info.sha256,
            "bytes": info.bytes,
            "artifact_roles": role,
            "models": model_scope,
            "referencing_runs": 0,
            "included_referencing_runs": 0,
            "copied_by_builder": False,
            "package_action": package_action,
        }
        plan.append(row)
        existing[info.path] = row

    for info in support_files:
        add_external(
            info,
            role="software_or_code",
            model_scope="shared",
            package_action="include",
        )
    for item in supplemental_files:
        add_external(
            item.info,
            role=item.role,
            model_scope="supplementary_analysis",
            package_action=item.package_action,
        )
    return sorted(plan, key=lambda row: row["relative_path"])


def _local_python_module_inventory(root: Path) -> dict[str, str]:
    """Map importable in-repository Python modules to project-relative files."""

    root = root.resolve()
    inventory: dict[str, str] = {}
    search_roots = (
        (root / "experiments", root),
        (root / "src" / "admet_benchmark", root / "src"),
    )
    for search_root, import_root in search_roots:
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.rglob("*.py")):
            relative_parts = path.relative_to(root).parts
            if (
                _is_appledouble(path)
                or "__pycache__" in relative_parts
                or any(part.startswith("._") for part in relative_parts)
            ):
                continue
            module_parts = list(path.relative_to(import_root).with_suffix("").parts)
            if module_parts[-1] == "__init__":
                module_parts.pop()
            if not module_parts:
                continue
            inventory[".".join(module_parts)] = path.relative_to(root).as_posix()
    return inventory


def _module_name_for_local_path(relative: str) -> tuple[str, bool] | None:
    """Return (module name, is-package-init) for a local Python path."""

    path = Path(relative)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts[:1] == ["experiments"]:
        pass
    elif parts[:2] == ["src", "admet_benchmark"]:
        parts = parts[1:]
    elif parts[:1] == ["tests"]:
        pass
    else:
        return None
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    if not parts:
        return None
    return ".".join(parts), is_package


def _resolve_relative_import(
    *, importer_module: str, importer_is_package: bool, module: str | None, level: int
) -> str:
    """Resolve an ``ast.ImportFrom`` target without importing project code."""

    if level == 0:
        return str(module or "")
    package = (
        importer_module
        if importer_is_package
        else importer_module.rpartition(".")[0]
    )
    package_parts = package.split(".") if package else []
    parent_hops = level - 1
    if parent_hops >= len(package_parts):
        raise ValueError("relative import escapes its local package")
    base_parts = package_parts[: len(package_parts) - parent_hops]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def audit_public_code_package(
    *,
    root: Path,
    package_plan: Sequence[Mapping[str, Any]],
    public_readme_path: str,
    required: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Fail closed on an incomplete formal public code dependency closure."""

    if not required:
        return (
            {
                "schema_version": 1,
                "status": "NOT_REQUIRED",
                "required_code_paths": list(REQUIRED_PUBLIC_CODE_PATHS),
                "required_test_paths": list(REQUIRED_PUBLIC_TEST_PATHS),
                "required_code_paths_present": True,
                "required_test_paths_present": True,
                "local_import_closure_complete": True,
                "public_readme_path": public_readme_path,
                "public_readme_present": True,
                "appledouble_paths_included": [],
                "failures": [],
            },
            (),
        )

    root = root.resolve()
    included_paths = {
        str(row.get("relative_path", ""))
        for row in package_plan
        if row.get("package_action") == "include"
    }
    code_paths = set(REQUIRED_PUBLIC_CODE_PATHS)
    test_paths = set(REQUIRED_PUBLIC_TEST_PATHS)
    missing_on_disk = sorted(
        relative
        for relative in REQUIRED_PUBLIC_SOFTWARE_PATHS
        if not (root / relative).is_file()
    )
    missing_from_plan = sorted(set(REQUIRED_PUBLIC_SOFTWARE_PATHS) - included_paths)
    zero_byte_required_paths = sorted(
        relative
        for relative in REQUIRED_PUBLIC_SOFTWARE_PATHS
        if (root / relative).is_file() and (root / relative).stat().st_size == 0
    )
    missing_code_on_disk = sorted(code_paths.intersection(missing_on_disk))
    missing_tests_on_disk = sorted(test_paths.intersection(missing_on_disk))
    missing_code_from_plan = sorted(code_paths.intersection(missing_from_plan))
    missing_tests_from_plan = sorted(test_paths.intersection(missing_from_plan))
    empty_code_paths = sorted(code_paths.intersection(zero_byte_required_paths))
    empty_test_paths = sorted(test_paths.intersection(zero_byte_required_paths))
    appledouble_paths = sorted(
        relative
        for relative in included_paths
        if any(part.startswith("._") for part in Path(relative).parts)
    )
    public_readme_file = root / public_readme_path
    public_readme_present = (
        public_readme_file.is_file()
        and public_readme_file.stat().st_size > 0
        and public_readme_path in included_paths
    )

    failures: list[str] = []
    if missing_code_on_disk:
        failures.append(
            f"required public code paths missing on disk: {len(missing_code_on_disk)}"
        )
    if missing_tests_on_disk:
        failures.append(
            f"required public test paths missing on disk: {len(missing_tests_on_disk)}"
        )
    if missing_code_from_plan:
        failures.append(
            "required public code paths absent from include plan: "
            f"{len(missing_code_from_plan)}"
        )
    if missing_tests_from_plan:
        failures.append(
            "required public test paths absent from include plan: "
            f"{len(missing_tests_from_plan)}"
        )
    if empty_code_paths:
        failures.append(
            f"required public code paths are empty: {len(empty_code_paths)}"
        )
    if empty_test_paths:
        failures.append(
            f"required public test paths are empty: {len(empty_test_paths)}"
        )
    if appledouble_paths:
        failures.append(
            f"AppleDouble sidecars entered the include plan: {len(appledouble_paths)}"
        )
    if not public_readme_present:
        failures.append("public bundle README is missing, empty, or not allowlisted")

    module_inventory = _local_python_module_inventory(root)
    included_python_paths = sorted(
        relative
        for relative in included_paths
        if _module_name_for_local_path(relative) is not None
    )
    import_edges: list[dict[str, str]] = []
    import_failures: list[str] = []
    namespace_packages = {"experiments"}
    for relative in included_python_paths:
        module_info = _module_name_for_local_path(relative)
        if module_info is None:
            continue
        importer_module, importer_is_package = module_info
        path = root / relative
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            import_failures.append(f"{relative}: cannot parse Python source: {exc}")
            continue
        requested_modules: set[tuple[str, int]] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "admet_benchmark" or alias.name.startswith(
                        "admet_benchmark."
                    ) or alias.name == "experiments" or alias.name.startswith(
                        "experiments."
                    ):
                        requested_modules.add((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                try:
                    target = _resolve_relative_import(
                        importer_module=importer_module,
                        importer_is_package=importer_is_package,
                        module=node.module,
                        level=node.level,
                    )
                except ValueError as exc:
                    import_failures.append(
                        f"{relative}:{node.lineno}: {exc}"
                    )
                    continue
                if not (
                    target == "admet_benchmark"
                    or target.startswith("admet_benchmark.")
                    or target == "experiments"
                    or target.startswith("experiments.")
                ):
                    continue
                requested_modules.add((target, node.lineno))
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{target}.{alias.name}" if target else alias.name
                    if candidate in module_inventory:
                        requested_modules.add((candidate, node.lineno))

        for imported_module, line_number in sorted(requested_modules):
            dependency = module_inventory.get(imported_module)
            if dependency is None:
                if imported_module in namespace_packages:
                    continue
                import_failures.append(
                    f"{relative}:{line_number}: unresolved local import {imported_module}"
                )
                continue
            import_edges.append(
                {
                    "importer": relative,
                    "module": imported_module,
                    "dependency": dependency,
                }
            )
            if dependency not in included_paths:
                import_failures.append(
                    f"{relative}:{line_number}: local dependency not allowlisted: {dependency}"
                )

    if import_failures:
        failures.append(
            f"public Python local-import closure failures: {len(import_failures)}"
        )
    unique_failures = tuple(dict.fromkeys(failures))
    unique_import_failures = list(dict.fromkeys(import_failures))
    required_paths_present = not (
        missing_on_disk or missing_from_plan or zero_byte_required_paths
    )
    details = {
        "schema_version": 1,
        "status": "PASS" if not unique_failures else "HOLD",
        "required_code_paths": list(REQUIRED_PUBLIC_CODE_PATHS),
        "required_code_path_count": len(REQUIRED_PUBLIC_CODE_PATHS),
        "required_test_paths": list(REQUIRED_PUBLIC_TEST_PATHS),
        "required_test_path_count": len(REQUIRED_PUBLIC_TEST_PATHS),
        "required_code_paths_present": not (
            missing_code_on_disk or missing_code_from_plan or empty_code_paths
        ),
        "required_test_paths_present": not (
            missing_tests_on_disk or missing_tests_from_plan or empty_test_paths
        ),
        "required_software_paths_present": required_paths_present,
        "missing_required_code_paths_on_disk": missing_code_on_disk,
        "missing_required_code_paths_from_include_plan": missing_code_from_plan,
        "empty_required_code_paths": empty_code_paths,
        "missing_required_test_paths_on_disk": missing_tests_on_disk,
        "missing_required_test_paths_from_include_plan": missing_tests_from_plan,
        "empty_required_test_paths": empty_test_paths,
        "python_files_audited": len(included_python_paths),
        "local_import_edges": sorted(
            import_edges,
            key=lambda row: (row["importer"], row["module"], row["dependency"]),
        ),
        "local_import_closure_complete": not unique_import_failures,
        "local_import_failures": unique_import_failures,
        "public_readme_path": public_readme_path,
        "public_readme_present": public_readme_present,
        "appledouble_paths_included": appledouble_paths,
        "failures": list(unique_failures),
    }
    return details, unique_failures


def audit_included_public_package(
    *, root: Path, package_plan: Sequence[Mapping[str, Any]], required: bool
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Fail closed on raw fields, credentials, paths, and unsafe filenames.

    Unlike the generated-public-subtree audit, this scans every file that the
    final package plan marks ``include``. Structured raw-field checks are
    limited to CSV headers and JSON keys so analysis source code may discuss
    those fields without being mistaken for released row-level data.
    """

    if not required:
        return (
            {
                "schema_version": 1,
                "status": "NOT_REQUIRED",
                "included_files_scanned": 0,
                "failures": [],
            },
            (),
        )

    root = root.resolve()
    failures: list[str] = []
    files_scanned = 0
    utf8_files_scanned = 0
    binary_files_skipped = 0
    csv_files_scanned = 0
    json_files_scanned = 0
    forbidden_csv_columns: dict[str, list[str]] = {}
    forbidden_json_keys: dict[str, list[str]] = {}
    absolute_path_matches = 0
    secret_assignment_matches = 0

    def forbidden_json_paths(value: Any, prefix: str = "") -> list[str]:
        matches: list[str] = []
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.strip().lower().replace("-", "_")
                child_path = f"{prefix}/{key}" if prefix else key
                if normalized in PUBLIC_PACKAGE_FORBIDDEN_DATA_KEYS:
                    matches.append(child_path)
                matches.extend(forbidden_json_paths(child, child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{prefix}/{index}" if prefix else str(index)
                matches.extend(forbidden_json_paths(child, child_path))
        return matches

    included = [
        str(row.get("relative_path", ""))
        for row in package_plan
        if row.get("package_action") == "include"
    ]
    for relative in sorted(included):
        logical = Path(relative)
        if (
            not relative
            or logical.is_absolute()
            or ".." in logical.parts
            or any(part.startswith("._") for part in logical.parts)
        ):
            failures.append(f"unsafe included package path: {relative}")
            continue
        path = root / logical
        if not path.is_file():
            failures.append(f"included package file is missing: {relative}")
            continue
        files_scanned += 1
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            binary_files_skipped += 1
            continue
        utf8_files_scanned += 1
        # A standard interpreter shebang is executable metadata, not a leaked
        # project or host path. All subsequent lines remain fully scanned.
        scan_text = text.split("\n", 1)[1] if text.startswith("#!") and "\n" in text else text
        suffix = logical.suffix.lower()
        if suffix == ".py":
            # Scan Python string *values*, not their source-code escape syntax.
            # Otherwise harmless LaTeX such as ``"\\\\toprule\\n"`` looks
            # like a Windows UNC path even though its runtime value starts with
            # only one backslash.  Non-string tokens and comments remain in the
            # scan view, while raw/f-string literal fragments are evaluated and
            # scanned separately, so real embedded paths still fail closed.
            token_stream: list[tuple[int, str]] = []
            literal_fragments: list[str] = []
            try:
                for token in tokenize.generate_tokens(io.StringIO(scan_text).readline):
                    if token.type != tokenize.STRING:
                        token_stream.append((token.type, token.string))
                        continue
                    token_stream.append((token.type, '""'))
                    try:
                        expression = ast.parse(token.string, mode="eval")
                    except SyntaxError:
                        # A malformed public source file is handled by the code
                        # audit; retain the token here so this scan also fails
                        # conservatively if it contains a path or credential.
                        literal_fragments.append(token.string)
                        continue
                    for node in ast.walk(expression):
                        if isinstance(node, ast.Constant):
                            if isinstance(node.value, str):
                                literal_fragments.append(node.value)
                            elif isinstance(node.value, bytes):
                                literal_fragments.append(
                                    node.value.decode("utf-8", errors="replace")
                                )
                scan_text = tokenize.untokenize(token_stream) + "\n" + "\n".join(
                    literal_fragments
                )
            except (IndentationError, tokenize.TokenError):
                # Keep the original fail-closed view if tokenization itself is
                # impossible; syntactically valid release code takes the path
                # above and avoids escape-syntax false positives.
                pass
        path_count = len(PRIVATE_HOST_ABSOLUTE_PATH_RE.findall(scan_text)) + len(
            PUBLIC_WINDOWS_ABSOLUTE_PATH_RE.findall(scan_text)
        )
        secret_count = len(PUBLIC_SECRET_ASSIGNMENT_RE.findall(scan_text))
        absolute_path_matches += path_count
        secret_assignment_matches += secret_count
        if path_count:
            failures.append(
                f"included package payload contains {path_count} absolute path token(s): {relative}"
            )
        if secret_count:
            failures.append(
                f"included package payload contains {secret_count} secret assignment token(s): {relative}"
            )

        if suffix == ".csv":
            csv_files_scanned += 1
            try:
                header = next(csv.reader(io.StringIO(text, newline="")), [])
            except csv.Error as exc:
                failures.append(f"included CSV header is malformed: {relative} ({exc})")
                continue
            bad = sorted(
                {
                    str(column).strip()
                    for column in header
                    if str(column).strip().lower().replace("-", "_")
                    in PUBLIC_PACKAGE_FORBIDDEN_DATA_KEYS
                }
            )
            if bad:
                forbidden_csv_columns[relative] = bad
                failures.append(
                    f"included CSV exposes raw structure/label/index columns: {relative}"
                )
        elif suffix == ".json":
            json_files_scanned += 1
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                failures.append(f"included JSON is malformed: {relative} ({exc})")
                continue
            bad_paths = sorted(set(forbidden_json_paths(value)))
            if bad_paths:
                forbidden_json_keys[relative] = bad_paths
                failures.append(
                    f"included JSON exposes raw structure/label/index keys: {relative}"
                )

    unique_failures = tuple(dict.fromkeys(failures))
    details = {
        "schema_version": 1,
        "status": "PASS" if not unique_failures else "HOLD",
        "included_files_scanned": files_scanned,
        "utf8_files_scanned": utf8_files_scanned,
        "binary_files_skipped_from_text_scan": binary_files_skipped,
        "csv_files_scanned": csv_files_scanned,
        "json_files_scanned": json_files_scanned,
        "absolute_path_matches": absolute_path_matches,
        "secret_assignment_matches": secret_assignment_matches,
        "forbidden_csv_columns": forbidden_csv_columns,
        "forbidden_json_keys": forbidden_json_keys,
        "failures": list(unique_failures),
    }
    return details, unique_failures


def manifest_columns(records: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "model",
        "task",
        "task_type",
        "official_metric",
        "seed",
        "included_default",
        "expected_submission_run",
        "run_status",
        "completion_marker",
        "prediction_valid",
        "prediction_rows",
        "valid_rows",
        "test_rows",
        "split_sha256",
        "label_alignment_within_tolerance",
        "max_abs_label_difference_within_task_seed",
        "config_split_sha256",
        "model_input_split_sha256",
        "source_split_sha256",
        "representation_compatibility_valid",
        "representation_compatibility_policy",
        "compatibility_manifest_schema",
        "model_input_repairs_n",
        "source_data_sha256",
        "software_sha256",
        "execution_spec_sha256",
    ]
    for role in ("prediction", "metrics", "config", "checkpoint"):
        preferred.extend(
            [
                f"{role}_path",
                f"{role}_sha256",
                f"{role}_bytes",
                f"{role}_package_action",
            ]
        )
    preferred.extend(["errors", "notes"])
    extras = sorted(set().union(*(record.keys() for record in records)) - set(preferred))
    return preferred + extras


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key, value in output.items():
                if isinstance(value, (list, dict, tuple)):
                    output[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            writer.writerow(output)


def _build_gate(
    records: Sequence[dict[str, Any]],
    *,
    tasks: Sequence[str],
    seeds: Sequence[int],
    traditional_models: Sequence[str],
    traditional_metrics_errors: Sequence[str],
    policy_errors: Sequence[str],
) -> dict[str, Any]:
    submission_models = list(traditional_models) + list(SUBMISSION_NEURAL_MODELS)
    expected_by_model = {
        model: len(tasks) * len(seeds) for model in submission_models
    }
    expected_total = sum(expected_by_model.values())
    included = [record for record in records if record["included_default"]]
    complete_by_model = {
        model: sum(
            record["model"] == model and record["run_status"] == "complete"
            for record in included
        )
        for model in submission_models
    }
    failures: list[str] = list(traditional_metrics_errors) + list(policy_errors)

    if len(included) != expected_total:
        failures.append(f"included manifest records {len(included)} != expected {expected_total}")
    for model, expected in expected_by_model.items():
        if complete_by_model[model] != expected:
            failures.append(
                f"{model}: complete {complete_by_model[model]}/{expected}"
            )

    expected_keys = {
        (model, task, int(seed))
        for model in submission_models
        for task in tasks
        for seed in seeds
    }
    observed_keys = {
        (record["model"], record["task"], int(record["seed"])) for record in included
    }
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if missing_keys:
        failures.append(f"missing expected run keys: {len(missing_keys)}")
    if extra_keys:
        failures.append(f"unexpected included run keys: {len(extra_keys)}")

    invalid_records = [record for record in included if record["run_status"] != "complete"]
    for record in invalid_records:
        key = f"{record['model']}/{record['task']}/seed{record['seed']}"
        details = "; ".join(record.get("errors", [])) or record["run_status"]
        failures.append(f"{key}: {details}")

    seed_sets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for record in included:
        if record["run_status"] == "complete":
            seed_sets[(record["model"], record["task"])].add(int(record["seed"]))
    required_seeds = set(map(int, seeds))
    seed_ok = True
    for model in submission_models:
        for task in tasks:
            if seed_sets[(model, task)] != required_seeds:
                seed_ok = False

    split_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    split_complete = True
    for record in included:
        if record["run_status"] != "complete" or not record.get("split_sha256"):
            split_complete = False
            continue
        split_groups[(record["task"], int(record["seed"]))].add(record["split_sha256"])
    inconsistent_splits = {
        f"{task}/seed{seed}": sorted(values)
        for (task, seed), values in split_groups.items()
        if len(values) != 1
    }
    split_consistent = split_complete and not inconsistent_splits
    if inconsistent_splits:
        failures.append(
            f"cross-model split/sample identity hash mismatches: {len(inconsistent_splits)}"
        )
    if not split_complete:
        failures.append("split/sample identity hash coverage is incomplete")

    # Labels are deliberately not part of the identity hash. The GIN writer
    # serializes labels after a float32 batch round trip, whereas traditional
    # and Chemprop artifacts retain float64 source labels. Compare labels
    # pairwise with the locked tolerance, while keeping split/index/SMILES and
    # row count exact through split_sha256 above.
    complete_groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in included:
        if record["run_status"] == "complete":
            complete_groups[(record["task"], int(record["seed"]))].append(record)
    expected_group_models = set(submission_models)
    label_alignment_complete = True
    inconsistent_labels: dict[str, dict[str, Any]] = {}
    global_max_abs_label_difference = 0.0
    for task in tasks:
        for seed in seeds:
            group = complete_groups[(task, int(seed))]
            group_models = {record["model"] for record in group}
            group_key = f"{task}/seed{seed}"
            if group_models != expected_group_models:
                label_alignment_complete = False
                continue
            vectors: dict[str, tuple[list[tuple[str, int]], np.ndarray]] = {}
            malformed = False
            for record in group:
                rows = tuple(record.get("_label_alignment_rows", ()))
                keys = [(str(split), int(index)) for split, index, _ in rows]
                if not rows or len(keys) != len(set(keys)):
                    malformed = True
                    break
                order = sorted(range(len(keys)), key=lambda idx: keys[idx])
                ordered_keys = [keys[idx] for idx in order]
                values = np.asarray([float(rows[idx][2]) for idx in order], dtype=float)
                vectors[record["model"]] = (ordered_keys, values)
            pair_failures: list[str] = []
            group_max = 0.0
            if malformed or set(vectors) != expected_group_models:
                pair_failures.append("missing_or_duplicate_label_alignment_rows")
            else:
                models = sorted(vectors)
                for left_index, left_model in enumerate(models):
                    left_keys, left_values = vectors[left_model]
                    for right_model in models[left_index + 1 :]:
                        right_keys, right_values = vectors[right_model]
                        if left_keys != right_keys or len(left_values) != len(right_values):
                            pair_failures.append(
                                f"{left_model}!={right_model}:split_or_sample_index_keys"
                            )
                            continue
                        difference = np.abs(left_values - right_values)
                        pair_max = float(np.max(difference)) if len(difference) else 0.0
                        group_max = max(group_max, pair_max)
                        if not np.allclose(
                            left_values,
                            right_values,
                            rtol=LABEL_ALIGNMENT_RTOL,
                            atol=LABEL_ALIGNMENT_ATOL,
                            equal_nan=False,
                        ):
                            pair_failures.append(f"{left_model}!={right_model}:labels")
            group_ok = not pair_failures
            global_max_abs_label_difference = max(
                global_max_abs_label_difference, group_max
            )
            for record in group:
                record["label_alignment_within_tolerance"] = group_ok
                record["max_abs_label_difference_within_task_seed"] = group_max
            if not group_ok:
                inconsistent_labels[group_key] = {
                    "models": sorted(group_models),
                    "max_abs_label_difference": group_max,
                    "pair_failures": pair_failures,
                }
    labels_consistent = label_alignment_complete and not inconsistent_labels
    if inconsistent_labels:
        failures.append(
            f"cross-model label alignment HARD FAIL: {len(inconsistent_labels)} task/seed groups"
        )
    if not label_alignment_complete:
        failures.append("cross-model label alignment coverage is incomplete")

    legacy = [record for record in records if record["model"] == LEGACY_MODEL]
    legacy_exclusion_ok = all(not record["included_default"] for record in legacy)
    if not legacy_exclusion_ok:
        failures.append("legacy_node_mpnn is not fully excluded")

    neural_records = [
        record for record in included if record["model"] in SUBMISSION_NEURAL_MODELS
    ]
    representation_compatibility_complete = bool(neural_records) and all(
        record.get("representation_compatibility_valid") is True
        for record in neural_records
    )
    if not representation_compatibility_complete:
        failures.append(
            "formal neural parser compatibility policy/mapping/hash audit incomplete"
        )

    unique_failures = list(dict.fromkeys(failures))
    status = "GO" if not unique_failures else "HOLD"
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": status,
        "release_ready": status == "GO",
        "expected_total_runs": expected_total,
        "expected_by_model": expected_by_model,
        "complete_by_model": complete_by_model,
        "legacy_node_mpnn_discovered": len(legacy),
        "legacy_exclusion_ok": legacy_exclusion_ok,
        "checks": {
            "expected_key_set_exact": not missing_keys and not extra_keys,
            "seeds_exact": seed_ok,
            "prediction_schema_rows_finite": not invalid_records,
            "split_hash_coverage_complete": split_complete,
            "cross_model_sample_identity_hashes_consistent": split_consistent,
            "cross_model_labels_aligned_with_tolerance": labels_consistent,
            "cross_model_split_hashes_consistent": split_consistent
            and labels_consistent,
            "formal_neural_representation_compatibility_audited": (
                representation_compatibility_complete
            ),
            "license_and_redistribution_policy_resolved": not policy_errors,
        },
        "missing_keys": [list(key) for key in missing_keys],
        "extra_keys": [list(key) for key in extra_keys],
        "inconsistent_split_hashes": inconsistent_splits,
        "label_alignment": {
            "rtol": LABEL_ALIGNMENT_RTOL,
            "atol": LABEL_ALIGNMENT_ATOL,
            "coverage_complete": label_alignment_complete,
            "max_abs_difference_across_task_seed_groups": global_max_abs_label_difference,
            "inconsistent_task_seed_groups": inconsistent_labels,
        },
        "representation_compatibility": {
            "policy": SMILES_COMPATIBILITY_POLICY,
            "solubility_required_test_indices": [1159, 1160],
            "solubility_required_repairs": [
                dict(value) for value in SOLUBILITY_COMPATIBILITY_REPAIRS
            ],
            "all_formal_neural_records_valid": representation_compatibility_complete,
        },
        "failure_count": len(unique_failures),
        "failures": unique_failures,
    }


def _expect_int(
    payload: Mapping[str, Any],
    key: str,
    expected: int,
    label: str,
    failures: list[str],
) -> None:
    try:
        observed = _strict_int(payload.get(key), f"{label}.{key}")
    except ValueError as exc:
        failures.append(str(exc))
        return
    if observed != expected:
        failures.append(f"{label}.{key} is {observed}, expected {expected}")


def _expect_true(
    payload: Mapping[str, Any], key: str, label: str, failures: list[str]
) -> None:
    try:
        _strict_true(payload.get(key), f"{label}.{key}")
    except ValueError as exc:
        failures.append(str(exc))


def _validate_snapshot_binding(
    binding: Any,
    snapshot: FileSnapshot,
    label: str,
    failures: list[str],
    *,
    row_count: int | None = None,
    row_key: str = "rows",
) -> None:
    if not isinstance(binding, Mapping):
        failures.append(f"{label} binding is missing")
        return
    if binding.get("sha256") != snapshot.sha256:
        failures.append(f"{label} SHA-256 is stale")
    try:
        recorded_bytes = _strict_int(binding.get("bytes"), f"{label}.bytes", minimum=0)
    except ValueError as exc:
        failures.append(str(exc))
    else:
        if recorded_bytes != snapshot.bytes:
            failures.append(f"{label} byte count is stale")
    if row_count is not None:
        try:
            recorded_rows = _strict_int(
                binding.get(row_key), f"{label}.{row_key}", minimum=0
            )
        except ValueError as exc:
            failures.append(str(exc))
        else:
            if recorded_rows != row_count:
                failures.append(f"{label} row count is stale")


def _audit_crossfit_release_evidence(
    *,
    root: Path,
    catalog: SupplementSnapshotCatalog,
    fixed_files: Mapping[str, tuple[Path, str]],
    snapshots: Mapping[str, FileSnapshot],
    failures: list[str],
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    """Audit the hash-bound cross-fit/scaffold evidence and public portability."""

    def serialized_mean_matches(value: Any, expected: float) -> bool:
        """Allow only the rounding implied by the locked 12-significant-digit CSVs."""

        try:
            observed = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(observed) and math.isclose(
            observed,
            float(expected),
            rel_tol=5e-12,
            abs_tol=5e-13,
        )

    failure_start = len(failures)
    public_payloads: list[tuple[str, bytes]] = []
    required_names = set(FINAL_CROSSFIT_OUTPUTS) | set(
        FINAL_CROSSFIT_DERIVED_OUTPUTS
    )
    missing = sorted(required_names - set(snapshots))
    if missing:
        failures.append(f"cross-fit release evidence is missing {len(missing)} files")

    classification_tasks = sum(
        meta["task_type"] == "classification" for meta in TDC_TASK_META.values()
    )
    regression_tasks = len(TDC_TASK_META) - classification_tasks
    endpoint_dimensions = classification_tasks * 6 + regression_tasks * 3
    expected_rows = {
        "crossfit_fold_level": (
            len(FINAL_CROSSFIT_GROUPING_SCHEMES)
            * endpoint_dimensions
            * FINAL_CROSSFIT_REPEATS
            * 2
        ),
        "crossfit_endpoint_summary": (
            len(FINAL_CROSSFIT_GROUPING_SCHEMES) * endpoint_dimensions
        ),
        "crossfit_dimension_summary": (
            len(FINAL_CROSSFIT_GROUPING_SCHEMES)
            * len(FINAL_CROSSFIT_DIMENSIONS)
        ),
        "crossfit_overlap_audit": len(TDC_TASKS),
        "crossfit_source_hashes": 2 * len(TDC_TASKS),
        "crossfit_run_alignment": (
            len(TDC_TASKS)
            * len(SEEDS)
            * (len(TRADITIONAL_MODELS) + len(SUBMISSION_NEURAL_MODELS))
        ),
        "crossfit_model_seed_availability": (
            len(TDC_TASKS)
            * (len(TRADITIONAL_MODELS) + len(SUBMISSION_NEURAL_MODELS))
        ),
        "crossfit_ensemble_alignment": (
            len(TDC_TASKS)
            * (len(TRADITIONAL_MODELS) + len(SUBMISSION_NEURAL_MODELS))
        ),
        "crossfit_validation_scores": (
            len(TDC_TASKS)
            * len(SEEDS)
            * (len(TRADITIONAL_MODELS) + len(SUBMISSION_NEURAL_MODELS))
        ),
        "crossfit_fixed_selection": len(TDC_TASKS),
    }
    required_columns = {
        "crossfit_fold_level": {
            "task",
            "grouping_scheme",
            "repeat",
            "direction",
            "dimension",
            "usable",
            "signed_d",
            "oracle_tie_count",
            "tie_average_usable",
            "tie_averaged_signed_d",
            "group_overlap_count",
        },
        "crossfit_endpoint_summary": {
            "grouping_scheme",
            "task",
            "dimension",
            "complete_repeats",
            "usable_direction_rows",
            "mean_signed_d",
            "tie_average_usable_direction_rows",
            "mean_tie_averaged_signed_d",
            "oracle_tie_direction_fraction",
        },
        "crossfit_dimension_summary": {
            "grouping_scheme",
            "dimension",
            "endpoint_count",
            "mean_endpoint_signed_d",
            "bootstrap_unit",
            "bootstrap_reps",
            "sign_flip_unit",
            "sign_flip_method",
            "sign_flip_assignments",
            "sign_flip_extreme_count",
            "sign_flip_p_exact",
            "holm_family",
            "holm_family_size",
            "holm_p_within_grouping_7",
            "holm_reject_0_05",
            "mean_endpoint_tie_averaged_signed_d",
            "mean_oracle_tie_direction_fraction",
            "tie_averaged_sign_flip_assignments",
            "tie_averaged_sign_flip_extreme_count",
            "tie_averaged_sign_flip_p_exact",
            "tie_averaged_holm_p_within_grouping_7",
            "tie_averaged_holm_reject_0_05",
        },
        "crossfit_overlap_audit": {
            "task",
            "canonical_overlap_count",
            "scaffold_group_overlap_count",
            "nonempty_bemis_murcko_overlap_count",
            "prediction_test_matches_locked_source_order_labels",
        },
        "crossfit_source_hashes": {
            "task",
            "split",
            "path",
            "file_sha256",
            "ordered_smiles_sha256",
            "ordered_labels_sha256",
        },
        "crossfit_prediction_input_audit": {"path", "included", "reason"},
    }
    observed_rows: dict[str, int] = {}
    observed_columns: dict[str, list[str]] = {}
    for logical_name, (filename, _) in FINAL_CROSSFIT_OUTPUTS.items():
        if logical_name == "crossfit_gate" or logical_name not in snapshots:
            continue
        try:
            rows, columns = _csv_shape_streaming(
                snapshots[logical_name], f"cross-fit evidence {filename}"
            )
            observed_rows[logical_name] = rows
            observed_columns[logical_name] = columns
            required = required_columns.get(logical_name, set())
            if not required.issubset(columns):
                raise ValueError(
                    f"cross-fit evidence {filename} lacks columns {sorted(required - set(columns))}"
                )
            expected = expected_rows.get(logical_name)
            if expected is not None and rows != expected:
                raise ValueError(
                    f"cross-fit evidence {filename} has {rows} rows, expected {expected}"
                )
        except ValueError as exc:
            failures.append(str(exc))

    if "crossfit_prediction_input_audit" in snapshots:
        try:
            _, input_rows = _csv_rows(
                snapshots["crossfit_prediction_input_audit"],
                "cross-fit prediction input audit",
            )
            included_count = sum(
                str(row.get("included", "")).strip().lower() == "true"
                for row in input_rows
            )
            expected_included = (
                len(TDC_TASKS)
                * len(SEEDS)
                * (len(TRADITIONAL_MODELS) + len(SUBMISSION_NEURAL_MODELS))
            )
            if included_count != expected_included:
                raise ValueError(
                    "cross-fit prediction input audit includes "
                    f"{included_count}/{expected_included} formal runs"
                )
        except ValueError as exc:
            failures.append(str(exc))

    if "crossfit_overlap_audit" in snapshots:
        try:
            _, overlap_rows = _csv_rows(
                snapshots["crossfit_overlap_audit"], "locked split overlap audit"
            )
            for row_number, row in enumerate(overlap_rows, start=2):
                for column in (
                    "canonical_overlap_count",
                    "scaffold_group_overlap_count",
                    "nonempty_bemis_murcko_overlap_count",
                ):
                    if _strict_int(
                        row.get(column),
                        f"locked split overlap audit line {row_number} {column}",
                        minimum=0,
                    ) != 0:
                        raise ValueError(
                            f"locked split overlap audit line {row_number} has nonzero {column}"
                        )
                _strict_true(
                    row.get("prediction_test_matches_locked_source_order_labels"),
                    "locked split prediction/source alignment",
                )
        except ValueError as exc:
            failures.append(str(exc))

    if "crossfit_source_hashes" in snapshots:
        try:
            _, source_rows = _csv_rows(
                snapshots["crossfit_source_hashes"], "locked split source hash manifest"
            )
            for row_number, row in enumerate(source_rows, start=2):
                relative = Path(str(row.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"locked split source hash manifest line {row_number} has unsafe path"
                    )
                for column in (
                    "file_sha256",
                    "ordered_smiles_sha256",
                    "ordered_labels_sha256",
                ):
                    if not _valid_sha256(str(row.get(column, ""))):
                        raise ValueError(
                            f"locked split source hash manifest line {row_number} has invalid {column}"
                        )
        except ValueError as exc:
            failures.append(str(exc))

    crossfit_endpoint_rows: list[dict[str, str]] = []
    crossfit_dimension_rows: list[dict[str, str]] = []
    if "crossfit_endpoint_summary" in snapshots:
        try:
            _, endpoint_rows = _csv_rows(
                snapshots["crossfit_endpoint_summary"], "cross-fit endpoint summary"
            )
            crossfit_endpoint_rows = endpoint_rows
            if any(
                _strict_int(
                    row.get("complete_repeats"),
                    "cross-fit endpoint complete_repeats",
                    minimum=0,
                )
                != FINAL_CROSSFIT_REPEATS
                or _strict_int(
                    row.get("usable_direction_rows"),
                    "cross-fit endpoint usable_direction_rows",
                    minimum=0,
                )
                != 2 * FINAL_CROSSFIT_REPEATS
                or _strict_int(
                    row.get("tie_average_usable_direction_rows"),
                    "cross-fit endpoint tie-average usable_direction_rows",
                    minimum=0,
                )
                != 2 * FINAL_CROSSFIT_REPEATS
                for row in endpoint_rows
            ):
                raise ValueError("cross-fit endpoint summary is not repeat-complete")
        except ValueError as exc:
            failures.append(str(exc))

    if "crossfit_dimension_summary" in snapshots:
        try:
            _, dimension_rows = _csv_rows(
                snapshots["crossfit_dimension_summary"], "cross-fit dimension summary"
            )
            crossfit_dimension_rows = dimension_rows
            observed_pairs = {
                (str(row.get("grouping_scheme", "")), str(row.get("dimension", "")))
                for row in dimension_rows
            }
            expected_pairs = {
                (scheme, dimension)
                for scheme in FINAL_CROSSFIT_GROUPING_SCHEMES
                for dimension in FINAL_CROSSFIT_DIMENSIONS
            }
            if observed_pairs != expected_pairs:
                raise ValueError("cross-fit dimension summary pair inventory is incomplete")
            for row in dimension_rows:
                if (
                    row.get("bootstrap_unit") != "endpoint"
                    or _strict_int(
                        row.get("bootstrap_reps"),
                        "cross-fit dimension bootstrap_reps",
                        minimum=1,
                    )
                    != FINAL_CROSSFIT_BOOTSTRAP_REPS
                ):
                    raise ValueError("cross-fit dimension bootstrap contract is stale")
                endpoint_count = _strict_int(
                    row.get("endpoint_count"),
                    "cross-fit dimension endpoint_count",
                    minimum=1,
                )
                if (
                    row.get("sign_flip_unit") != "endpoint_mean"
                    or row.get("sign_flip_method")
                    != "exact_enumeration_two_sided"
                    or _strict_int(
                        row.get("sign_flip_assignments"),
                        "cross-fit exact sign-flip assignment count",
                        minimum=1,
                    )
                    != 2**endpoint_count
                    or _strict_int(
                        row.get("sign_flip_extreme_count"),
                        "cross-fit exact sign-flip extreme count",
                        minimum=0,
                    )
                    > 2**endpoint_count
                    or row.get("holm_family")
                    != "all_7_dimensions_within_grouping"
                    or _strict_int(
                        row.get("holm_family_size"),
                        "cross-fit Holm family size",
                        minimum=1,
                    )
                    != len(FINAL_CROSSFIT_DIMENSIONS)
                    or _strict_int(
                        row.get("tie_averaged_sign_flip_assignments"),
                        "cross-fit tie-averaged sign-flip assignment count",
                        minimum=1,
                    )
                    != 2**endpoint_count
                    or _strict_int(
                        row.get("tie_averaged_sign_flip_extreme_count"),
                        "cross-fit tie-averaged sign-flip extreme count",
                        minimum=0,
                    )
                    > 2**endpoint_count
                ):
                    raise ValueError(
                        "cross-fit exact sign-flip/Holm contract is stale"
                    )
        except ValueError as exc:
            failures.append(str(exc))

    if crossfit_endpoint_rows and crossfit_dimension_rows:
        try:
            endpoint_values: dict[tuple[str, str], list[float]] = defaultdict(list)
            tie_endpoint_values: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in crossfit_endpoint_rows:
                key = (
                    str(row.get("grouping_scheme", "")),
                    str(row.get("dimension", "")),
                )
                endpoint_values[key].append(float(row["mean_signed_d"]))
                tie_endpoint_values[key].append(
                    float(row["mean_tie_averaged_signed_d"])
                )

            recomputed: dict[tuple[str, str], tuple[float, int, int]] = {}
            tie_recomputed: dict[tuple[str, str], tuple[float, int, int]] = {}
            for target, source in (
                (recomputed, endpoint_values),
                (tie_recomputed, tie_endpoint_values),
            ):
                for key, values in source.items():
                    array = np.asarray(values, dtype=np.float64)
                    assignment_count = 1 << len(array)
                    observed = abs(float(np.sum(array, dtype=np.float64)))
                    threshold = np.nextafter(observed, -np.inf)
                    bit_positions = np.arange(len(array), dtype=np.uint64)
                    extreme_count = 0
                    chunk_size = 262_144
                    for start in range(0, assignment_count, chunk_size):
                        stop = min(start + chunk_size, assignment_count)
                        codes = np.arange(start, stop, dtype=np.uint64)
                        negative = (
                            (codes[:, None] >> bit_positions[None, :]) & 1
                        ).astype(np.float64)
                        signed_sums = np.sum(
                            (1.0 - 2.0 * negative) * array,
                            axis=1,
                        )
                        extreme_count += int(
                            np.count_nonzero(np.abs(signed_sums) >= threshold)
                        )
                    target[key] = (
                        float(extreme_count / assignment_count),
                        assignment_count,
                        extreme_count,
                    )

            holm: dict[tuple[str, str], float] = {}
            tie_holm: dict[tuple[str, str], float] = {}
            for target, source in ((holm, recomputed), (tie_holm, tie_recomputed)):
                for scheme in FINAL_CROSSFIT_GROUPING_SCHEMES:
                    family = sorted(
                        (
                            (key, value[0])
                            for key, value in source.items()
                            if key[0] == scheme
                        ),
                        key=lambda item: (item[1], item[0][1]),
                    )
                    running_max = 0.0
                    for rank, (key, pvalue) in enumerate(family):
                        candidate = min(1.0, (len(family) - rank) * pvalue)
                        running_max = max(running_max, candidate)
                        target[key] = running_max

            for row in crossfit_dimension_rows:
                key = (
                    str(row.get("grouping_scheme", "")),
                    str(row.get("dimension", "")),
                )
                if key not in recomputed or key not in holm:
                    raise ValueError(
                        "cross-fit dimension summary has no endpoint-source recomputation"
                    )
                values = endpoint_values[key]
                tie_values = tie_endpoint_values[key]
                pvalue, assignment_count, extreme_count = recomputed[key]
                (
                    tie_pvalue,
                    tie_assignment_count,
                    tie_extreme_count,
                ) = tie_recomputed[key]
                if _strict_int(
                    row.get("endpoint_count"),
                    "cross-fit recomputed endpoint count",
                    minimum=1,
                ) != len(values):
                    raise ValueError(
                        "cross-fit dimension endpoint count disagrees with endpoint summary"
                    )
                if not serialized_mean_matches(
                    row.get("mean_endpoint_signed_d"), float(np.mean(values))
                ):
                    raise ValueError(
                        "cross-fit dimension mean disagrees with endpoint summary"
                    )
                if not serialized_mean_matches(
                    row.get("mean_endpoint_tie_averaged_signed_d"),
                    float(np.mean(tie_values)),
                ):
                    raise ValueError(
                        "cross-fit tie-averaged dimension mean disagrees with endpoint summary"
                    )
                if (
                    row.get("sign_flip_p_exact") != f"{pvalue:.12g}"
                    or _strict_int(
                        row.get("sign_flip_assignments"),
                        "cross-fit recomputed assignment count",
                        minimum=1,
                    )
                    != assignment_count
                    or _strict_int(
                        row.get("sign_flip_extreme_count"),
                        "cross-fit recomputed extreme count",
                        minimum=0,
                    )
                    != extreme_count
                ):
                    raise ValueError(
                        "cross-fit exact sign-flip value disagrees with endpoint summary"
                    )
                if row.get("holm_p_within_grouping_7") != f"{holm[key]:.12g}":
                    raise ValueError(
                        "cross-fit Holm value disagrees with recomputed family"
                    )
                if (
                    row.get("tie_averaged_sign_flip_p_exact")
                    != f"{tie_pvalue:.12g}"
                    or _strict_int(
                        row.get("tie_averaged_sign_flip_assignments"),
                        "cross-fit recomputed tie-averaged assignment count",
                        minimum=1,
                    )
                    != tie_assignment_count
                    or _strict_int(
                        row.get("tie_averaged_sign_flip_extreme_count"),
                        "cross-fit recomputed tie-averaged extreme count",
                        minimum=0,
                    )
                    != tie_extreme_count
                    or row.get("tie_averaged_holm_p_within_grouping_7")
                    != f"{tie_holm[key]:.12g}"
                ):
                    raise ValueError(
                        "cross-fit tie-averaged sign-flip/Holm value disagrees "
                        "with endpoint summary"
                    )
                expected_reject = holm[key] < 0.05
                observed_reject = str(row.get("holm_reject_0_05", "")).lower()
                if observed_reject != str(expected_reject).lower():
                    raise ValueError(
                        "cross-fit Holm rejection flag disagrees with adjusted value"
                    )
                expected_tie_reject = tie_holm[key] < 0.05
                observed_tie_reject = str(
                    row.get("tie_averaged_holm_reject_0_05", "")
                ).lower()
                if observed_tie_reject != str(expected_tie_reject).lower():
                    raise ValueError(
                        "cross-fit tie-averaged Holm rejection flag disagrees "
                        "with adjusted value"
                    )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))

    if "crossfit_manuscript_summary_csv" in snapshots:
        try:
            rows, columns = _csv_shape_streaming(
                snapshots["crossfit_manuscript_summary_csv"],
                "cross-fit manuscript summary CSV",
            )
            if rows != expected_rows["crossfit_dimension_summary"]:
                raise ValueError(
                    "cross-fit manuscript summary CSV row count is stale"
                )
            required = {
                "grouping",
                "dimension",
                "n_endpoints",
                "mean_signed_d",
                "sign_flip_p_exact",
                "holm_p_within_grouping_7",
            }
            if not required.issubset(columns):
                raise ValueError(
                    "cross-fit manuscript summary CSV lacks required columns"
                )
        except ValueError as exc:
            failures.append(str(exc))

    if "crossfit_manuscript_summary_manifest" in snapshots:
        try:
            summary_manifest = _json_object(
                snapshots["crossfit_manuscript_summary_manifest"],
                "cross-fit manuscript summary manifest",
            )
            if summary_manifest.get("schema") != "crossfit_oracle_supplementary_table_v2":
                raise ValueError("cross-fit manuscript summary schema is stale")
            _expect_int(
                summary_manifest,
                "rows",
                expected_rows["crossfit_dimension_summary"],
                "cross-fit manuscript summary manifest",
                failures,
            )
            expected_bindings = {
                "csv_path": (
                    FINAL_CROSSFIT_DERIVED_OUTPUTS[
                        "crossfit_manuscript_summary_csv"
                    ][0],
                    "csv_sha256",
                    snapshots.get("crossfit_manuscript_summary_csv"),
                ),
                "tex_path": (
                    FINAL_CROSSFIT_DERIVED_OUTPUTS[
                        "crossfit_manuscript_summary_tex"
                    ][0],
                    "tex_sha256",
                    snapshots.get("crossfit_manuscript_summary_tex"),
                ),
                "source_summary_path": (
                    "results/tables/crossfit_oracle_sensitivity_final/"
                    "crossfit_oracle_dimension_summary.csv",
                    "source_summary_sha256",
                    snapshots.get("crossfit_dimension_summary"),
                ),
            }
            for path_key, (expected_path, hash_key, snapshot) in expected_bindings.items():
                if summary_manifest.get(path_key) != expected_path:
                    failures.append(
                        f"cross-fit manuscript summary {path_key} is stale"
                    )
                if snapshot is None or summary_manifest.get(hash_key) != snapshot.sha256:
                    failures.append(
                        f"cross-fit manuscript summary {hash_key} is stale"
                    )
        except ValueError as exc:
            failures.append(str(exc))

    gate: dict[str, Any] = {}
    if "crossfit_gate" in snapshots:
        try:
            gate = _json_object(snapshots["crossfit_gate"], "cross-fit oracle gate")
            if gate.get("status") != "GO":
                raise ValueError("cross-fit oracle gate status is not GO")
            if gate.get("scope") != "zero_retraining_crossfit_oracle_sensitivity_final_locked":
                raise ValueError("cross-fit oracle gate scope is not final locked")
            contract = gate.get("analysis_contract")
            if not isinstance(contract, Mapping):
                raise ValueError("cross-fit oracle analysis contract is missing")
            _expect_true(contract, "passes", "cross-fit oracle contract", failures)
            if contract.get("expected_models_in_tie_order") != [
                *TRADITIONAL_MODELS,
                *SUBMISSION_NEURAL_MODELS,
            ]:
                failures.append("cross-fit oracle model tie order is stale")
            if contract.get("expected_seeds") != list(SEEDS):
                failures.append("cross-fit oracle seed inventory is stale")
            _expect_int(
                contract,
                "expected_tasks",
                len(TDC_TASKS),
                "cross-fit oracle contract",
                failures,
            )
            if contract.get("grouping_schemes") != list(
                FINAL_CROSSFIT_GROUPING_SCHEMES
            ):
                failures.append("cross-fit oracle grouping schemes are stale")
            if contract.get("primary_grouping_scheme") != "bemis_murcko":
                failures.append("cross-fit oracle primary grouping is not Bemis-Murcko")
            for key, expected in (
                ("repeats_per_endpoint_scheme", FINAL_CROSSFIT_REPEATS),
                ("directions_per_repeat", 2),
                ("expected_fold_rows", expected_rows["crossfit_fold_level"]),
                ("observed_fold_rows", expected_rows["crossfit_fold_level"]),
                (
                    "expected_endpoint_dimension_rows",
                    expected_rows["crossfit_endpoint_summary"],
                ),
                (
                    "observed_endpoint_dimension_rows",
                    expected_rows["crossfit_endpoint_summary"],
                ),
                (
                    "observed_dimension_summary_rows",
                    expected_rows["crossfit_dimension_summary"],
                ),
            ):
                _expect_int(contract, key, expected, "cross-fit oracle contract", failures)
            for key in (
                "all_rows_usable",
                "no_group_leakage",
                "all_models_dimensions_share_fold",
                "all_repeats_bidirectionally_complete",
                "formal_prediction_sources_complete",
                "run_sample_alignment_exact",
                "ensemble_sample_alignment_exact",
                "prediction_test_matches_locked_source",
                "canonical_and_scaffold_overlap_zero_all_endpoints",
                "exact_sign_flip_holm_complete",
            ):
                _expect_true(contract, key, "cross-fit oracle contract", failures)
            signed_policy = gate.get("signed_difference_policy")
            if not isinstance(signed_policy, Mapping):
                raise ValueError("cross-fit signed-difference policy is missing")
            _expect_true(
                signed_policy,
                "negative_values_never_truncated",
                "cross-fit signed-difference policy",
                failures,
            )
            overlap_gate = gate.get("overlap_audit")
            if not isinstance(overlap_gate, Mapping):
                raise ValueError("cross-fit overlap gate is missing")
            for key, expected in (
                ("endpoints", len(TDC_TASKS)),
                ("canonical_overlap_total", 0),
                ("scaffold_group_overlap_total", 0),
                ("nonempty_bemis_murcko_overlap_total", 0),
            ):
                _expect_int(overlap_gate, key, expected, "cross-fit overlap gate", failures)
            provenance = gate.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("cross-fit provenance is missing")
            _expect_int(
                provenance,
                "locked_source_file_count",
                2 * len(TDC_TASKS),
                "cross-fit provenance",
                failures,
            )
            if not _valid_sha256(
                str(provenance.get("locked_source_hash_manifest_sha256", ""))
            ):
                failures.append("cross-fit locked source aggregate hash is invalid")
            output_hashes = provenance.get("output_sha256")
            expected_hashes = {
                filename: snapshots[logical_name].sha256
                for logical_name, (filename, _) in FINAL_CROSSFIT_OUTPUTS.items()
                if logical_name != "crossfit_gate" and logical_name in snapshots
            }
            expected_hashes.update(
                {
                    relative: snapshots[logical_name].sha256
                    for logical_name, (relative, _) in FINAL_CROSSFIT_DERIVED_OUTPUTS.items()
                    if logical_name in snapshots
                }
            )
            if output_hashes != expected_hashes:
                failures.append("cross-fit gate output SHA-256 inventory is stale")
        except ValueError as exc:
            failures.append(str(exc))

    nonportable_files: list[str] = []
    for logical_name, (filename, role) in FINAL_CROSSFIT_OUTPUTS.items():
        snapshot = snapshots.get(logical_name)
        if snapshot is None:
            continue
        try:
            text = snapshot.data.decode("utf-8-sig")
        except UnicodeDecodeError:
            failures.append(f"cross-fit evidence {filename} is not UTF-8")
            continue
        if not _contains_absolute_path_text(text):
            continue
        nonportable_files.append(filename)
        catalog.require(
            snapshot.path,
            role,
            failures,
            package_action="exclude_nonportable",
        )
        try:
            public_name = f"crossfit_{filename}"
            public_payloads.append(
                (
                    public_name,
                    _sanitize_public_metadata_payload(
                        public_name, snapshot.data, root
                    ),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    for logical_name, (relative, role) in FINAL_CROSSFIT_DERIVED_OUTPUTS.items():
        snapshot = snapshots.get(logical_name)
        if snapshot is None:
            continue
        try:
            text = snapshot.data.decode("utf-8-sig")
        except UnicodeDecodeError:
            failures.append(f"cross-fit evidence {relative} is not UTF-8")
            continue
        if not _contains_absolute_path_text(text):
            continue
        nonportable_files.append(relative)
        catalog.require(
            snapshot.path,
            role,
            failures,
            package_action="exclude_nonportable",
        )
        try:
            public_name = f"crossfit_{Path(relative).name}"
            public_payloads.append(
                (
                    public_name,
                    _sanitize_public_metadata_payload(
                        public_name, snapshot.data, root
                    ),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    complete = len(failures) == failure_start
    details = {
        "status": "PASS" if complete else "HOLD",
        "required_files": len(FINAL_CROSSFIT_OUTPUTS)
        + len(FINAL_CROSSFIT_DERIVED_OUTPUTS),
        "missing_files": missing,
        "observed_rows": observed_rows,
        "expected_fold_rows": expected_rows["crossfit_fold_level"],
        "expected_endpoint_summary_rows": expected_rows[
            "crossfit_endpoint_summary"
        ],
        "expected_dimension_summary_rows": expected_rows[
            "crossfit_dimension_summary"
        ],
        "repeats": FINAL_CROSSFIT_REPEATS,
        "bootstrap_reps": FINAL_CROSSFIT_BOOTSTRAP_REPS,
        "fold_level_evidence_directly_releasable": (
            "crossfit_oracle_fold_level.csv" not in nonportable_files
        ),
        "raw_nonportable_files": nonportable_files,
        "sanitized_public_metadata_files": [name for name, _ in public_payloads],
        "gate_status": gate.get("status") if gate else None,
    }
    return details, public_payloads


def _audit_crossfit_influence_release_evidence(
    *,
    snapshots: Mapping[str, FileSnapshot],
    failures: list[str],
) -> dict[str, Any]:
    """Fail closed on the hash-bound, zero-refit LOEO/LOCO defense outputs."""

    failure_start = len(failures)
    required_names = {
        *FINAL_CROSSFIT_INFLUENCE_OUTPUTS,
        "crossfit_influence_script",
        "crossfit_endpoint_summary",
        "crossfit_dimension_summary",
    }
    missing = sorted(required_names - set(snapshots))
    if missing:
        failures.append(
            "cross-fit influence release evidence is missing: " + ", ".join(missing)
        )

    observed_rows: dict[str, int] = {}

    def _number(row: Mapping[str, Any], column: str, label: str) -> float:
        value = row.get(column)
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} {column} is not numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"{label} {column} is not finite")
        return result

    def _integer(row: Mapping[str, Any], column: str, label: str) -> int:
        return _strict_int(row.get(column), f"{label} {column}", minimum=0)

    def _boolean(row: Mapping[str, Any], column: str, label: str) -> bool:
        value = row.get(column)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"{label} {column} must be true or false, got {value!r}")

    def _close(observed: float, expected: float) -> bool:
        return math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-11)

    def _sign(value: float) -> str:
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "zero"

    def _stable_seed(base_seed: int, *parts: object) -> int:
        payload = "|".join([str(base_seed), *[str(part) for part in parts]])
        return int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little"
        )

    try:
        if missing:
            raise ValueError("cross-fit influence inputs are incomplete")

        schemes = tuple(FINAL_CROSSFIT_GROUPING_SCHEMES)
        dimensions = tuple(FINAL_CROSSFIT_DIMENSIONS)
        metadata = {
            normalize_task(task): value for task, value in TDC_TASK_META.items()
        }
        expected_tasks = {normalize_task(task) for task in TDC_TASKS}
        if set(metadata) != expected_tasks:
            raise ValueError("locked TDC metadata/task inventories disagree")
        classification_tasks = {
            task
            for task, value in metadata.items()
            if value.get("task_type") == "classification"
        }
        regression_tasks = expected_tasks - classification_tasks
        task_categories = {
            task: str(value.get("category", "")).strip().lower()
            for task, value in metadata.items()
        }
        if any(not category for category in task_categories.values()):
            raise ValueError("locked TDC category metadata is incomplete")

        classification_dimensions = {
            "brier",
            "ece_5bin",
            "ece_10bin",
            "selective_relative_excess_aurc_log_loss",
        }
        regression_dimensions = {"conformal_coverage_error"}
        tasks_by_dimension = {
            dimension: (
                classification_tasks
                if dimension in classification_dimensions
                else regression_tasks
                if dimension in regression_dimensions
                else expected_tasks
            )
            for dimension in dimensions
        }
        expected_endpoint_keys = {
            (scheme, dimension, task)
            for scheme in schemes
            for dimension in dimensions
            for task in tasks_by_dimension[dimension]
        }
        expected_dimension_keys = {
            (scheme, dimension) for scheme in schemes for dimension in dimensions
        }
        expected_category_keys = {
            (scheme, dimension, category)
            for scheme in schemes
            for dimension in dimensions
            for category in {
                task_categories[task] for task in tasks_by_dimension[dimension]
            }
        }
        if len(expected_endpoint_keys) != FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS[
            "crossfit_influence_loeo"
        ]:
            raise ValueError("locked LOEO expectation no longer equals 210 rows")
        if len(expected_category_keys) != FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS[
            "crossfit_influence_loco"
        ]:
            raise ValueError("locked LOCO expectation no longer equals 60 rows")

        endpoint_columns, endpoint_rows = _csv_rows(
            snapshots["crossfit_endpoint_summary"],
            "cross-fit endpoint source for influence analysis",
        )
        endpoint_required = {
            "grouping_scheme",
            "scheme_role",
            "task",
            "dimension",
            "evidence_tier",
            "mean_signed_d",
        }
        if not endpoint_required.issubset(endpoint_columns):
            raise ValueError(
                "cross-fit endpoint source for influence analysis lacks columns "
                f"{sorted(endpoint_required - set(endpoint_columns))}"
            )
        endpoint_values: dict[tuple[str, str, str], float] = {}
        for row_number, row in enumerate(endpoint_rows, start=2):
            key = (
                str(row.get("grouping_scheme", "")).strip().lower(),
                str(row.get("dimension", "")).strip().lower(),
                normalize_task(str(row.get("task", ""))),
            )
            if key in endpoint_values:
                raise ValueError(
                    f"cross-fit endpoint source line {row_number} duplicates {key}"
                )
            endpoint_values[key] = _number(
                row, "mean_signed_d", f"cross-fit endpoint source line {row_number}"
            )
            expected_role = "primary" if key[0] == "bemis_murcko" else "secondary"
            if str(row.get("scheme_role", "")).strip().lower() != expected_role:
                raise ValueError(
                    f"cross-fit endpoint source line {row_number} has stale scheme role"
                )
            if (
                str(row.get("evidence_tier", "")).strip().lower()
                != FINAL_CROSSFIT_EVIDENCE_TIERS.get(key[1])
            ):
                raise ValueError(
                    f"cross-fit endpoint source line {row_number} has stale evidence tier"
                )
        if set(endpoint_values) != expected_endpoint_keys:
            raise ValueError(
                "cross-fit endpoint source key inventory is not the exact locked "
                "scheme/dimension/endpoint matrix"
            )

        dimension_columns, dimension_rows = _csv_rows(
            snapshots["crossfit_dimension_summary"],
            "cross-fit dimension source for influence analysis",
        )
        dimension_required = {
            "grouping_scheme",
            "dimension",
            "endpoint_count",
            "mean_endpoint_signed_d",
            "mean_endpoint_signed_d_ci_low",
            "mean_endpoint_signed_d_ci_high",
            "bootstrap_unit",
            "bootstrap_reps",
        }
        if not dimension_required.issubset(dimension_columns):
            raise ValueError(
                "cross-fit dimension source for influence analysis lacks columns "
                f"{sorted(dimension_required - set(dimension_columns))}"
            )
        dimension_values: dict[tuple[str, str], dict[str, float]] = {}
        for row_number, row in enumerate(dimension_rows, start=2):
            key = (
                str(row.get("grouping_scheme", "")).strip().lower(),
                str(row.get("dimension", "")).strip().lower(),
            )
            if key in dimension_values:
                raise ValueError(
                    f"cross-fit dimension source line {row_number} duplicates {key}"
                )
            values = {
                "mean": _number(
                    row,
                    "mean_endpoint_signed_d",
                    f"cross-fit dimension source line {row_number}",
                ),
                "ci_low": _number(
                    row,
                    "mean_endpoint_signed_d_ci_low",
                    f"cross-fit dimension source line {row_number}",
                ),
                "ci_high": _number(
                    row,
                    "mean_endpoint_signed_d_ci_high",
                    f"cross-fit dimension source line {row_number}",
                ),
            }
            expected_count = len(tasks_by_dimension.get(key[1], ()))
            if _integer(
                row, "endpoint_count", f"cross-fit dimension source line {row_number}"
            ) != expected_count:
                raise ValueError(
                    f"cross-fit dimension source line {row_number} endpoint count is stale"
                )
            if row.get("bootstrap_unit") != "endpoint" or _integer(
                row, "bootstrap_reps", f"cross-fit dimension source line {row_number}"
            ) != FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS:
                raise ValueError(
                    f"cross-fit dimension source line {row_number} bootstrap contract is stale"
                )
            source_mean = sum(
                endpoint_values[(key[0], key[1], task)]
                for task in tasks_by_dimension[key[1]]
            ) / expected_count
            if not _close(values["mean"], source_mean):
                raise ValueError(
                    f"cross-fit dimension source line {row_number} mean is stale"
                )
            if values["ci_low"] > values["ci_high"]:
                raise ValueError(
                    f"cross-fit dimension source line {row_number} has reversed CI"
                )
            dimension_values[key] = values
        if set(dimension_values) != expected_dimension_keys:
            raise ValueError("cross-fit dimension source key inventory is incomplete")

        loeo_required = {
            "grouping_scheme",
            "scheme_role",
            "dimension",
            "evidence_tier",
            "full_endpoint_count",
            "remaining_endpoint_count",
            "full_mean_signed_d",
            "full_ci_low",
            "full_ci_high",
            "full_mean_sign",
            "reduced_mean_signed_d",
            "reduced_ci_low",
            "reduced_ci_high",
            "reduced_mean_sign",
            "mean_sign_stable",
            "sign_stability",
            "ci_low_gt_zero",
            "ci_high_lt_zero",
            "ci_excludes_zero",
            "ci_supports_full_mean_sign",
            "bootstrap_unit",
            "bootstrap_reps",
            "bootstrap_seed",
            "full_bootstrap_seed",
            "analysis_scope",
            "omitted_endpoint",
            "omitted_endpoint_category",
        }
        loco_required = (loeo_required - {"omitted_endpoint", "omitted_endpoint_category"}) | {
            "omitted_category",
            "omitted_endpoint_count",
        }
        summary_required = {
            "grouping_scheme",
            "scheme_role",
            "dimension",
            "evidence_tier",
            "full_endpoint_count",
            "full_mean_signed_d",
            "full_ci_low",
            "full_ci_high",
            "full_ci_low_gt_zero",
            "loeo_omission_count",
            "loeo_reduced_mean_min",
            "loeo_reduced_mean_max",
            "loeo_ci_low_min",
            "loeo_ci_low_max",
            "loeo_ci_high_min",
            "loeo_ci_high_max",
            "loeo_mean_sign_stable_count",
            "loeo_mean_sign_stable_fraction",
            "loeo_ci_low_gt_zero_count",
            "loeo_ci_low_gt_zero_fraction",
            "loco_omission_count",
            "loco_reduced_mean_min",
            "loco_reduced_mean_max",
            "loco_ci_low_min",
            "loco_ci_low_max",
            "loco_ci_high_min",
            "loco_ci_high_max",
            "loco_mean_sign_stable_count",
            "loco_mean_sign_stable_fraction",
            "loco_ci_low_gt_zero_count",
            "loco_ci_low_gt_zero_fraction",
            "bootstrap_unit",
            "bootstrap_reps",
            "reduced_random_seed",
            "full_random_seed",
            "analysis_scope",
            "independent_validation",
        }

        parsed_reductions: dict[
            str, dict[tuple[str, str, str], dict[str, Any]]
        ] = {"loeo": {}, "loco": {}}

        for kind, logical_name, required_columns, expected_keys in (
            (
                "loeo",
                "crossfit_influence_loeo",
                loeo_required,
                expected_endpoint_keys,
            ),
            (
                "loco",
                "crossfit_influence_loco",
                loco_required,
                expected_category_keys,
            ),
        ):
            columns, rows = _csv_rows(
                snapshots[logical_name], f"cross-fit influence {kind.upper()}"
            )
            observed_rows[logical_name] = len(rows)
            if not required_columns.issubset(columns):
                raise ValueError(
                    f"cross-fit influence {kind.upper()} lacks columns "
                    f"{sorted(required_columns - set(columns))}"
                )
            if len(rows) != FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS[logical_name]:
                raise ValueError(
                    f"cross-fit influence {kind.upper()} has {len(rows)} rows, expected "
                    f"{FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS[logical_name]}"
                )
            for row_number, row in enumerate(rows, start=2):
                scheme = str(row.get("grouping_scheme", "")).strip().lower()
                dimension = str(row.get("dimension", "")).strip().lower()
                omitted = normalize_task(
                    str(
                        row.get("omitted_endpoint", "")
                        if kind == "loeo"
                        else row.get("omitted_category", "")
                    )
                )
                key = (scheme, dimension, omitted)
                if key in parsed_reductions[kind]:
                    raise ValueError(
                        f"cross-fit influence {kind.upper()} line {row_number} duplicates {key}"
                    )
                if key not in expected_keys:
                    raise ValueError(
                        f"cross-fit influence {kind.upper()} line {row_number} has unexpected key {key}"
                    )
                label = f"cross-fit influence {kind.upper()} line {row_number}"
                expected_role = "primary" if scheme == "bemis_murcko" else "secondary"
                if str(row.get("scheme_role", "")).strip().lower() != expected_role:
                    raise ValueError(f"{label} has stale scheme role")
                if (
                    str(row.get("evidence_tier", "")).strip().lower()
                    != FINAL_CROSSFIT_EVIDENCE_TIERS[dimension]
                ):
                    raise ValueError(f"{label} has stale evidence tier")
                tasks = tasks_by_dimension[dimension]
                full_count = len(tasks)
                if kind == "loeo":
                    omitted_tasks = {omitted}
                    if (
                        str(row.get("omitted_endpoint_category", "")).strip().lower()
                        != task_categories[omitted]
                    ):
                        raise ValueError(f"{label} has stale endpoint category")
                else:
                    omitted_tasks = {
                        task for task in tasks if task_categories[task] == omitted
                    }
                    if _integer(row, "omitted_endpoint_count", label) != len(
                        omitted_tasks
                    ):
                        raise ValueError(f"{label} has stale omitted endpoint count")
                remaining_tasks = tasks - omitted_tasks
                if _integer(row, "full_endpoint_count", label) != full_count or _integer(
                    row, "remaining_endpoint_count", label
                ) != len(remaining_tasks):
                    raise ValueError(f"{label} has stale full/remaining endpoint counts")
                reference = dimension_values[(scheme, dimension)]
                full_mean = _number(row, "full_mean_signed_d", label)
                full_ci_low = _number(row, "full_ci_low", label)
                full_ci_high = _number(row, "full_ci_high", label)
                if not all(
                    (
                        _close(full_mean, reference["mean"]),
                        _close(full_ci_low, reference["ci_low"]),
                        _close(full_ci_high, reference["ci_high"]),
                    )
                ):
                    raise ValueError(f"{label} full estimate/CI is stale")
                reduced_mean = _number(row, "reduced_mean_signed_d", label)
                expected_reduced_mean = sum(
                    endpoint_values[(scheme, dimension, task)]
                    for task in remaining_tasks
                ) / len(remaining_tasks)
                if not _close(reduced_mean, expected_reduced_mean):
                    raise ValueError(f"{label} reduced mean is not source-derived")
                reduced_ci_low = _number(row, "reduced_ci_low", label)
                reduced_ci_high = _number(row, "reduced_ci_high", label)
                if reduced_ci_low > reduced_ci_high:
                    raise ValueError(f"{label} has reversed reduced CI")
                full_sign = _sign(full_mean)
                reduced_sign = _sign(reduced_mean)
                sign_stable = full_sign == reduced_sign
                if row.get("full_mean_sign") != full_sign or row.get(
                    "reduced_mean_sign"
                ) != reduced_sign:
                    raise ValueError(f"{label} sign labels are stale")
                if _boolean(row, "mean_sign_stable", label) != sign_stable or row.get(
                    "sign_stability"
                ) != ("preserved" if sign_stable else "changed"):
                    raise ValueError(f"{label} sign-stability fields are stale")
                low_positive = reduced_ci_low > 0
                high_negative = reduced_ci_high < 0
                ci_excludes_zero = low_positive or high_negative
                if full_sign == "positive":
                    ci_supports_sign = low_positive
                elif full_sign == "negative":
                    ci_supports_sign = high_negative
                else:
                    ci_supports_sign = not ci_excludes_zero
                for column, expected in (
                    ("ci_low_gt_zero", low_positive),
                    ("ci_high_lt_zero", high_negative),
                    ("ci_excludes_zero", ci_excludes_zero),
                    ("ci_supports_full_mean_sign", ci_supports_sign),
                ):
                    if _boolean(row, column, label) != expected:
                        raise ValueError(f"{label} {column} is stale")
                if (
                    row.get("bootstrap_unit") != "endpoint"
                    or _integer(row, "bootstrap_reps", label)
                    != FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS
                    or row.get("analysis_scope")
                    != FINAL_CROSSFIT_INFLUENCE_ANALYSIS_SCOPE
                ):
                    raise ValueError(f"{label} analysis/bootstrap contract is stale")
                expected_full_seed = _stable_seed(
                    FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED,
                    "endpoint_bootstrap",
                    scheme,
                    dimension,
                )
                expected_reduced_seed = _stable_seed(
                    FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED,
                    "endpoint_bootstrap",
                    scheme,
                    dimension,
                    "leave_one_endpoint_out"
                    if kind == "loeo"
                    else "leave_one_category_out",
                    omitted,
                )
                if _integer(row, "full_bootstrap_seed", label) != expected_full_seed:
                    raise ValueError(f"{label} full bootstrap seed is stale")
                if _integer(row, "bootstrap_seed", label) != expected_reduced_seed:
                    raise ValueError(f"{label} reduced bootstrap seed is stale")
                parsed_reductions[kind][key] = {
                    "reduced_mean": reduced_mean,
                    "ci_low": reduced_ci_low,
                    "ci_high": reduced_ci_high,
                    "mean_sign_stable": sign_stable,
                    "ci_low_gt_zero": low_positive,
                }
            if set(parsed_reductions[kind]) != expected_keys:
                raise ValueError(
                    f"cross-fit influence {kind.upper()} key inventory is incomplete"
                )

        summary_columns, summary_rows = _csv_rows(
            snapshots["crossfit_influence_summary"],
            "cross-fit influence dimension summary",
        )
        observed_rows["crossfit_influence_summary"] = len(summary_rows)
        if not summary_required.issubset(summary_columns):
            raise ValueError(
                "cross-fit influence summary lacks columns "
                f"{sorted(summary_required - set(summary_columns))}"
            )
        if len(summary_rows) != FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS[
            "crossfit_influence_summary"
        ]:
            raise ValueError("cross-fit influence summary does not have 14 rows")
        parsed_summary: dict[tuple[str, str], dict[str, Any]] = {}
        for row_number, row in enumerate(summary_rows, start=2):
            scheme = str(row.get("grouping_scheme", "")).strip().lower()
            dimension = str(row.get("dimension", "")).strip().lower()
            key = (scheme, dimension)
            label = f"cross-fit influence summary line {row_number}"
            if key in parsed_summary or key not in expected_dimension_keys:
                raise ValueError(f"{label} has duplicate or unexpected key {key}")
            expected_role = "primary" if scheme == "bemis_murcko" else "secondary"
            if str(row.get("scheme_role", "")).strip().lower() != expected_role:
                raise ValueError(f"{label} has stale scheme role")
            if (
                str(row.get("evidence_tier", "")).strip().lower()
                != FINAL_CROSSFIT_EVIDENCE_TIERS[dimension]
            ):
                raise ValueError(f"{label} has stale evidence tier")
            reference = dimension_values[key]
            full_mean = _number(row, "full_mean_signed_d", label)
            full_ci_low = _number(row, "full_ci_low", label)
            full_ci_high = _number(row, "full_ci_high", label)
            if not all(
                (
                    _close(full_mean, reference["mean"]),
                    _close(full_ci_low, reference["ci_low"]),
                    _close(full_ci_high, reference["ci_high"]),
                )
            ):
                raise ValueError(f"{label} full estimate/CI is stale")
            if _integer(row, "full_endpoint_count", label) != len(
                tasks_by_dimension[dimension]
            ):
                raise ValueError(f"{label} full endpoint count is stale")
            if _boolean(row, "full_ci_low_gt_zero", label) != (full_ci_low > 0):
                raise ValueError(f"{label} full CI sign flag is stale")
            if (
                row.get("bootstrap_unit") != "endpoint"
                or _integer(row, "bootstrap_reps", label)
                != FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS
                or _integer(row, "reduced_random_seed", label)
                != FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED
                or _integer(row, "full_random_seed", label)
                != FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED
                or row.get("analysis_scope")
                != FINAL_CROSSFIT_INFLUENCE_ANALYSIS_SCOPE
                or _boolean(row, "independent_validation", label)
            ):
                raise ValueError(f"{label} analysis/bootstrap contract is stale")
            for prefix, reduction_kind in (("loeo", "loeo"), ("loco", "loco")):
                subset = [
                    value
                    for reduction_key, value in parsed_reductions[
                        reduction_kind
                    ].items()
                    if reduction_key[:2] == key
                ]
                count = len(subset)
                stable_count = sum(value["mean_sign_stable"] for value in subset)
                positive_count = sum(value["ci_low_gt_zero"] for value in subset)
                expected_numeric = {
                    f"{prefix}_reduced_mean_min": min(
                        value["reduced_mean"] for value in subset
                    ),
                    f"{prefix}_reduced_mean_max": max(
                        value["reduced_mean"] for value in subset
                    ),
                    f"{prefix}_ci_low_min": min(value["ci_low"] for value in subset),
                    f"{prefix}_ci_low_max": max(value["ci_low"] for value in subset),
                    f"{prefix}_ci_high_min": min(
                        value["ci_high"] for value in subset
                    ),
                    f"{prefix}_ci_high_max": max(
                        value["ci_high"] for value in subset
                    ),
                    f"{prefix}_mean_sign_stable_fraction": stable_count / count,
                    f"{prefix}_ci_low_gt_zero_fraction": positive_count / count,
                }
                if _integer(row, f"{prefix}_omission_count", label) != count:
                    raise ValueError(f"{label} {prefix} omission count is stale")
                if _integer(
                    row, f"{prefix}_mean_sign_stable_count", label
                ) != stable_count or _integer(
                    row, f"{prefix}_ci_low_gt_zero_count", label
                ) != positive_count:
                    raise ValueError(f"{label} {prefix} aggregate counts are stale")
                for column, expected in expected_numeric.items():
                    if not _close(_number(row, column, label), expected):
                        raise ValueError(f"{label} {column} is stale")
            parsed_summary[key] = {
                "full_mean_signed_d": full_mean,
                "full_ci_low": full_ci_low,
                "full_ci_high": full_ci_high,
                "loeo_ci_low_gt_zero_count": _integer(
                    row, "loeo_ci_low_gt_zero_count", label
                ),
                "loeo_omission_count": _integer(row, "loeo_omission_count", label),
                "loco_ci_low_gt_zero_count": _integer(
                    row, "loco_ci_low_gt_zero_count", label
                ),
                "loco_omission_count": _integer(row, "loco_omission_count", label),
            }
        if set(parsed_summary) != expected_dimension_keys:
            raise ValueError("cross-fit influence summary key inventory is incomplete")

        table_text = snapshots["crossfit_influence_table"].data.decode("utf-8")
        for required_text in (
            "Post-hoc, zero-refit influence sensitivity",
            "not a new primary result or independent validation",
            "ECE (10 bins)",
            "ECE (5 bins)",
            "Applicability alignment",
        ):
            if required_text not in table_text:
                raise ValueError(
                    f"cross-fit influence LaTeX table omits required text {required_text!r}"
                )

        gate = _json_object(
            snapshots["crossfit_influence_gate"], "cross-fit influence gate"
        )
        if gate.get("schema") != "crossfit_influence_sensitivity_v1":
            raise ValueError("cross-fit influence gate schema is stale")
        if gate.get("status") != "GO" or gate.get("scope") != FINAL_CROSSFIT_INFLUENCE_SCOPE:
            raise ValueError("cross-fit influence gate is not final GO")
        contract = gate.get("analysis_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("cross-fit influence analysis contract is missing")
        exact_contract = {
            "model_refits": 0,
            "input_unit": "cross-fit endpoint mean signed D",
            "bootstrap_unit": "endpoint",
            "bootstrap_method": "percentile_95pct",
            "bootstrap_reps": FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS,
            "full_reference_random_seed": FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED,
            "reduced_subset_random_seed": FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED,
            "seed_derivation": "sha256_first_4_bytes_little_endian",
            "multiple_testing_adjustment": "not_applicable_descriptive_sensitivity",
            "confirmatory_inference": False,
            "independent_validation": False,
        }
        for key, expected in exact_contract.items():
            if contract.get(key) != expected:
                raise ValueError(f"cross-fit influence analysis contract {key} is stale")
        checks = gate.get("checks")
        required_checks = {
            "source_hash_and_size_unchanged",
            "all_input_tasks_map_to_locked_tdc_categories",
            "loeo_has_one_row_per_input_endpoint_mean",
            "loco_has_one_row_per_present_category",
            "summary_has_every_grouping_dimension",
            "reduced_subsets_are_nonempty",
            "all_reported_estimates_and_intervals_are_finite",
            "bootstrap_contract_is_endpoint_5000_full_seed_20260715_reduced_seed_20260714",
            "locked_full_dimension_intervals_reproduced",
            "primary_bemis_murcko_focus_complete",
            "analysis_is_explicitly_post_hoc_and_not_independent_validation",
        }
        if not isinstance(checks, Mapping) or not required_checks.issubset(checks):
            raise ValueError("cross-fit influence gate checks are incomplete")
        for key in required_checks:
            _strict_true(checks.get(key), f"cross-fit influence gate check {key}")
        counts = gate.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("cross-fit influence gate counts are missing")
        expected_counts = {
            "input_endpoint_rows": len(expected_endpoint_keys),
            "grouping_schemes": len(schemes),
            "dimensions": len(dimensions),
            "grouping_dimension_cells": len(expected_dimension_keys),
            "loeo_rows": len(expected_endpoint_keys),
            "loco_rows": len(expected_category_keys),
            "summary_rows": len(expected_dimension_keys),
        }
        for key, expected in expected_counts.items():
            if _strict_int(counts.get(key), f"cross-fit influence counts {key}") != expected:
                raise ValueError(f"cross-fit influence gate count {key} is stale")
        reproduction = gate.get("reference_reproduction")
        if not isinstance(reproduction, Mapping):
            raise ValueError("cross-fit influence reference reproduction is missing")
        for key in ("reference_row_count", "observed_row_count"):
            if _strict_int(
                reproduction.get(key), f"cross-fit influence reproduction {key}"
            ) != len(expected_dimension_keys):
                raise ValueError(f"cross-fit influence reproduction {key} is stale")
        _strict_true(
            reproduction.get("keys_match"),
            "cross-fit influence reproduction keys_match",
        )
        _strict_true(
            reproduction.get("passes"), "cross-fit influence reproduction passes"
        )
        maximum_error = float(reproduction.get("maximum_absolute_error", math.inf))
        tolerance = float(reproduction.get("absolute_error_tolerance", math.nan))
        if (
            not math.isfinite(maximum_error)
            or not math.isfinite(tolerance)
            or tolerance <= 0
            or tolerance > 1e-11
            or maximum_error > tolerance
        ):
            raise ValueError("cross-fit influence reference reproduction tolerance failed")
        focus = gate.get("primary_bemis_murcko_focus")
        if not isinstance(focus, Mapping) or set(focus) != set(
            FINAL_CROSSFIT_INFLUENCE_FOCUS_DIMENSIONS
        ):
            raise ValueError("cross-fit influence primary focus inventory is stale")
        for dimension in FINAL_CROSSFIT_INFLUENCE_FOCUS_DIMENSIONS:
            record = focus.get(dimension)
            if not isinstance(record, Mapping):
                raise ValueError(f"cross-fit influence focus {dimension} is missing")
            expected = parsed_summary[("bemis_murcko", dimension)]
            for key, value in expected.items():
                if key.endswith("_count"):
                    if _strict_int(record.get(key), f"focus {dimension} {key}") != value:
                        raise ValueError(f"cross-fit influence focus {dimension} {key} is stale")
                elif not _close(float(record.get(key)), float(value)):
                    raise ValueError(f"cross-fit influence focus {dimension} {key} is stale")

        provenance = gate.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("cross-fit influence provenance is missing")
        endpoint_snapshot = snapshots["crossfit_endpoint_summary"]
        dimension_snapshot = snapshots["crossfit_dimension_summary"]
        script_snapshot = snapshots["crossfit_influence_script"]
        expected_input_path = (
            "results/tables/crossfit_oracle_sensitivity_final/"
            "crossfit_oracle_endpoint_summary.csv"
        )
        expected_reference_path = (
            "results/tables/crossfit_oracle_sensitivity_final/"
            "crossfit_oracle_dimension_summary.csv"
        )
        if provenance.get("input_path") != expected_input_path:
            raise ValueError("cross-fit influence input logical path is stale")
        for key in ("input_sha256_before", "input_sha256_after"):
            if provenance.get(key) != endpoint_snapshot.sha256:
                raise ValueError(f"cross-fit influence provenance {key} is stale")
        for key in ("input_bytes_before", "input_bytes_after"):
            if _strict_int(provenance.get(key), f"cross-fit influence {key}") != endpoint_snapshot.bytes:
                raise ValueError(f"cross-fit influence provenance {key} is stale")
        if provenance.get("reference_dimension_summary_path") != expected_reference_path:
            raise ValueError("cross-fit influence reference logical path is stale")
        if provenance.get("reference_dimension_summary_sha256") != dimension_snapshot.sha256:
            raise ValueError("cross-fit influence reference hash is stale")
        if provenance.get("analysis_script_path") != (
            "experiments/analyze_crossfit_influence_sensitivity.py"
        ):
            raise ValueError("cross-fit influence script logical path is stale")
        if provenance.get("analysis_script_sha256") != script_snapshot.sha256:
            raise ValueError("cross-fit influence script hash is stale")
        expected_output_hashes = {
            filename: snapshots[logical_name].sha256
            for logical_name, (filename, _) in FINAL_CROSSFIT_INFLUENCE_OUTPUTS.items()
            if logical_name != "crossfit_influence_gate"
        }
        if provenance.get("output_sha256") != expected_output_hashes:
            raise ValueError("cross-fit influence output SHA-256 inventory is stale")
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        failures.append(str(exc))

    complete = len(failures) == failure_start
    return {
        "status": "PASS" if complete else "HOLD",
        "required_files": len(FINAL_CROSSFIT_INFLUENCE_OUTPUTS),
        "missing_files": missing,
        "observed_rows": observed_rows,
        "expected_rows": dict(FINAL_CROSSFIT_INFLUENCE_EXPECTED_ROWS),
        "zero_refit": True,
        "post_hoc_not_independent_validation": True,
        "bootstrap_reps": FINAL_CROSSFIT_INFLUENCE_BOOTSTRAP_REPS,
        "full_random_seed": FINAL_CROSSFIT_INFLUENCE_FULL_RANDOM_SEED,
        "reduced_random_seed": FINAL_CROSSFIT_INFLUENCE_REDUCED_RANDOM_SEED,
        "directly_releasable": complete,
    }


def _audit_reviewer_defense_release_evidence(
    *,
    snapshots: Mapping[str, FileSnapshot],
    failures: list[str],
) -> dict[str, Any]:
    """Fail closed on aggregate-only, zero-refit reviewer-defense outputs."""

    failure_start = len(failures)
    required_names = {*FINAL_REVIEWER_DEFENSE_OUTPUTS, "reviewer_defense_script"}
    missing = sorted(required_names - set(snapshots))
    if missing:
        failures.append(
            "reviewer-defense release evidence is missing: " + ", ".join(missing)
        )

    observed_rows: dict[str, int] = {}
    forbidden_columns = {
        "smiles",
        "canonical_smiles",
        "molecule_id",
        "sample_index",
        "y_true",
        "label",
        "prediction",
        "y_pred",
    }

    def _number(value: Any, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return result

    def _boolean(value: Any, label: str) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"{label} must be true or false, got {value!r}")

    def _close(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-11)

    def _require_true_checks(
        payload: Mapping[str, Any], keys: set[str], label: str
    ) -> None:
        checks = payload.get("checks")
        if not isinstance(checks, Mapping) or not keys.issubset(checks):
            raise ValueError(f"{label} checks are incomplete")
        for key in keys:
            _strict_true(checks.get(key), f"{label} check {key}")

    def _require_post_hoc_flags(payload: Mapping[str, Any], label: str) -> None:
        _strict_true(payload.get("post_hoc"), f"{label} post_hoc")
        if _boolean(payload.get("primary_claim"), f"{label} primary_claim"):
            raise ValueError(f"{label} primary_claim must be false")

    def _finite_optional(value: Any, label: str) -> float | None:
        if value is None or str(value).strip() == "":
            return None
        return _number(value, label)

    try:
        if missing:
            raise ValueError("reviewer-defense inputs are incomplete")

        csv_payloads: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
        csv_required_columns = {
            "reviewer_overlap_endpoints": {
                "endpoint",
                "source_row_count",
                "unique_molecule_count",
                "whole_set_sha256",
                "source_file_sha256",
            },
            "reviewer_overlap_components": {
                "component_id",
                "endpoint_count",
                "endpoints",
            },
            "reviewer_overlap_pairs": {
                "endpoint_a",
                "endpoint_b",
                "endpoint_a_unique_count",
                "endpoint_b_unique_count",
                "intersection_unique_count",
                "union_unique_count",
                "jaccard",
                "has_overlap",
            },
            "reviewer_lomo_selections": {
                "omitted_model",
                "task",
                "n_models",
                "model_pool",
                "validation_selected_model",
                "validation_tie_models",
                "validation_tie_count",
            },
            "reviewer_lomo_probabilities": {
                "omitted_model",
                "task",
                "model",
                "exact_selection_probability",
                "validation_tie_probability",
            },
            "reviewer_lomo_endpoints": {
                "omitted_model",
                "task",
                "dimension",
                "n_models",
                "model_pool",
                "validation_selected_model",
                "test_reliability_oracle_model",
                "selected_reliability_value",
                "oracle_reliability_value",
                "higher_is_better",
                "reliability_regret",
                "random_selection_expected_regret",
                "regret_minus_random_expected",
                "normalized_regret",
                "validation_tie_models",
                "validation_tie_count",
                "tie_best_case_regret",
                "tie_best_case_normalized_regret",
                "tie_worst_case_regret",
                "pool_reliability_loss_min",
                "pool_reliability_loss_max",
                "pool_reliability_loss_range",
                "uniform_random_mean_reliability_loss",
                "exact_uniform_random_expected_regret",
            },
            "reviewer_lomo_summary": {
                "omitted_model",
                "dimension",
                "n_endpoints",
                "practical_effect_signal",
                "tie_robust_effect_signal",
                "random_baseline_retention_signal",
            },
            "reviewer_conditional_rows": {
                "task",
                "model",
                "model_family",
                "task_type",
                "n_seeds",
                "threshold",
                "eligible",
                *FINAL_REVIEWER_DEFENSE_CONDITIONAL_METRICS,
            },
            "reviewer_conditional_summary": {
                "model_family",
                "threshold",
                "metric",
                "n_pairs",
                "n_endpoints",
                "pair_positive_count",
                "pair_positive_fraction",
                "positive_count",
                "positive_fraction",
                "mean",
                "median",
                "mean_ci_low",
                "mean_ci_high",
                "sign_flip_p",
                "holm_p_within_family_threshold",
            },
        }
        for logical_name, required_columns in csv_required_columns.items():
            columns, rows = _csv_rows(
                snapshots[logical_name],
                f"reviewer-defense {FINAL_REVIEWER_DEFENSE_OUTPUTS[logical_name][0]}",
            )
            observed_rows[logical_name] = len(rows)
            if len(rows) != FINAL_REVIEWER_DEFENSE_EXPECTED_ROWS[logical_name]:
                raise ValueError(
                    f"reviewer-defense {logical_name} has {len(rows)} rows, expected "
                    f"{FINAL_REVIEWER_DEFENSE_EXPECTED_ROWS[logical_name]}"
                )
            if not required_columns.issubset(columns):
                raise ValueError(
                    f"reviewer-defense {logical_name} lacks columns "
                    f"{sorted(required_columns - set(columns))}"
                )
            raw_columns = {str(column).strip().lower() for column in columns}
            if raw_columns & forbidden_columns:
                raise ValueError(
                    f"reviewer-defense {logical_name} contains row-level molecular columns"
                )
            csv_payloads[logical_name] = (columns, rows)

        expected_tasks = {normalize_task(task) for task in TDC_TASKS}
        task_meta = {
            normalize_task(task): meta for task, meta in TDC_TASK_META.items()
        }
        classification_tasks = {
            task
            for task, meta in task_meta.items()
            if meta["task_type"] == "classification"
        }
        regression_tasks = expected_tasks - classification_tasks
        models = set(TRADITIONAL_MODELS) | set(SUBMISSION_NEURAL_MODELS)
        sorted_models = sorted(models)

        # Aggregate fixed-test overlap: exact endpoint/pair graph and arithmetic.
        endpoint_rows = csv_payloads["reviewer_overlap_endpoints"][1]
        endpoint_by_name: dict[str, dict[str, Any]] = {}
        for line, row in enumerate(endpoint_rows, start=2):
            endpoint = normalize_task(row.get("endpoint", ""))
            if endpoint in endpoint_by_name:
                raise ValueError(f"reviewer overlap endpoint line {line} is duplicated")
            source_rows = _strict_int(
                row.get("source_row_count"),
                f"reviewer overlap endpoint line {line} source_row_count",
                minimum=1,
            )
            unique_rows = _strict_int(
                row.get("unique_molecule_count"),
                f"reviewer overlap endpoint line {line} unique_molecule_count",
                minimum=1,
            )
            if unique_rows > source_rows:
                raise ValueError("reviewer overlap unique count exceeds source rows")
            for column in ("whole_set_sha256", "source_file_sha256"):
                if not _valid_sha256(str(row.get(column, ""))):
                    raise ValueError(f"reviewer overlap endpoint line {line} has invalid {column}")
            endpoint_by_name[endpoint] = {**row, "unique": unique_rows}
        if set(endpoint_by_name) != expected_tasks:
            raise ValueError("reviewer overlap endpoint key inventory is incomplete")
        if sum(int(row["unique"]) for row in endpoint_by_name.values()) != 16082:
            raise ValueError("reviewer overlap endpoint-unique sum is not 16082")

        pair_rows = csv_payloads["reviewer_overlap_pairs"][1]
        expected_pairs = set(itertools.combinations(sorted(expected_tasks), 2))
        observed_pairs: set[tuple[str, str]] = set()
        adjacency = {task: set() for task in expected_tasks}
        overlap_count = 0
        maximum_intersection = 0
        for line, row in enumerate(pair_rows, start=2):
            key = (
                normalize_task(row.get("endpoint_a", "")),
                normalize_task(row.get("endpoint_b", "")),
            )
            if key in observed_pairs or key not in expected_pairs:
                raise ValueError(f"reviewer overlap pair line {line} has invalid key {key}")
            observed_pairs.add(key)
            left = _strict_int(row.get("endpoint_a_unique_count"), f"pair {line} left", minimum=1)
            right = _strict_int(row.get("endpoint_b_unique_count"), f"pair {line} right", minimum=1)
            intersection = _strict_int(row.get("intersection_unique_count"), f"pair {line} intersection", minimum=0)
            union = _strict_int(row.get("union_unique_count"), f"pair {line} union", minimum=1)
            if left != endpoint_by_name[key[0]]["unique"] or right != endpoint_by_name[key[1]]["unique"]:
                raise ValueError(f"reviewer overlap pair line {line} endpoint counts are stale")
            if union != left + right - intersection:
                raise ValueError(f"reviewer overlap pair line {line} violates union arithmetic")
            expected_jaccard = intersection / union
            if not _close(_number(row.get("jaccard"), f"pair {line} jaccard"), expected_jaccard):
                raise ValueError(f"reviewer overlap pair line {line} has stale Jaccard")
            has_overlap = _boolean(row.get("has_overlap"), f"pair {line} has_overlap")
            if has_overlap != (intersection > 0):
                raise ValueError(f"reviewer overlap pair line {line} has stale overlap flag")
            if has_overlap:
                overlap_count += 1
                adjacency[key[0]].add(key[1])
                adjacency[key[1]].add(key[0])
            maximum_intersection = max(maximum_intersection, intersection)
        if observed_pairs != expected_pairs or overlap_count != 209:
            raise ValueError("reviewer overlap pair key/count contract failed")
        remaining = set(expected_tasks)
        graph_components: list[set[str]] = []
        while remaining:
            stack = [min(remaining)]
            component: set[str] = set()
            while stack:
                task = stack.pop()
                if task in component:
                    continue
                component.add(task)
                stack.extend(adjacency[task] - component)
            remaining -= component
            graph_components.append(component)
        if len(graph_components) != 1:
            raise ValueError("reviewer overlap graph is not one connected component")
        component_rows = csv_payloads["reviewer_overlap_components"][1]
        component = component_rows[0]
        if (
            _strict_int(component.get("component_id"), "reviewer component_id") != 1
            or _strict_int(component.get("endpoint_count"), "reviewer component count") != 22
            or set(str(component.get("endpoints", "")).split(";")) != expected_tasks
        ):
            raise ValueError("reviewer overlap component inventory is stale")

        overlap_gate = _json_object(snapshots["reviewer_overlap_gate"], "reviewer overlap gate")
        if overlap_gate.get("status") != "GO":
            raise ValueError("reviewer overlap gate is not GO")
        _require_post_hoc_flags(overlap_gate, "reviewer overlap gate")
        overlap_checks = {
            "endpoint_count_22",
            "pair_count_231",
            "endpoint_unique_sum_16082",
            "global_union_13818",
            "overlapping_pairs_209",
            "single_connected_component",
            "no_row_level_molecular_output",
        }
        _require_true_checks(overlap_gate, overlap_checks, "reviewer overlap gate")
        overlap_expected_ints = {
            "endpoint_count": 22,
            "pair_count": 231,
            "endpoint_unique_count_sum": 16082,
            "global_unique_union_count": 13818,
            "cross_endpoint_duplicate_memberships": 2264,
            "pairs_with_overlap": 209,
            "pairs_without_overlap": 22,
            "connected_component_count": 1,
            "largest_component_endpoint_count": 22,
            "maximum_pair_intersection_count": maximum_intersection,
        }
        for key, expected in overlap_expected_ints.items():
            if _strict_int(overlap_gate.get(key), f"reviewer overlap gate {key}") != expected:
                raise ValueError(f"reviewer overlap gate {key} is stale")
        if not _valid_sha256(str(overlap_gate.get("global_whole_set_sha256", ""))):
            raise ValueError("reviewer overlap global set hash is invalid")

        # Four-model leave-one-model-out pools: exact keys and algebra.
        expected_task_dimensions = {
            (task, dimension)
            for task in expected_tasks
            for dimension in FINAL_REVIEWER_DEFENSE_DIMENSIONS
            if (
                dimension in {"applicability_alignment", "selective_relative_excess_aurc"}
                or task in classification_tasks
                and dimension in {"brier", "ece", "selective_relative_excess_aurc_log_loss"}
                or task in regression_tasks
                and dimension == "conformal_coverage_error"
            )
        }
        if len(expected_task_dimensions) != 92:
            raise AssertionError("reviewer-defense task/dimension contract is internally stale")

        selection_rows = csv_payloads["reviewer_lomo_selections"][1]
        selection_by_key: dict[tuple[str, str], dict[str, str]] = {}
        for line, row in enumerate(selection_rows, start=2):
            omitted = str(row.get("omitted_model", ""))
            task = normalize_task(row.get("task", ""))
            key = (omitted, task)
            if key in selection_by_key or omitted not in models or task not in expected_tasks:
                raise ValueError(f"reviewer LOMO selection line {line} has invalid key {key}")
            pool = set(str(row.get("model_pool", "")).split(";"))
            expected_pool = models - {omitted}
            selected = str(row.get("validation_selected_model", ""))
            ties = set(str(row.get("validation_tie_models", "")).split(";"))
            if (
                _strict_int(row.get("n_models"), f"LOMO selection {line} n_models") != 4
                or pool != expected_pool
                or omitted in pool
                or selected not in pool
                or not ties
                or not ties.issubset(pool)
                or _strict_int(row.get("validation_tie_count"), f"LOMO selection {line} tie count") != len(ties)
            ):
                raise ValueError(f"reviewer LOMO selection line {line} is not a true four-model reselection")
            selection_by_key[key] = row
        expected_selection_keys = {
            (omitted, task) for omitted in models for task in expected_tasks
        }
        if set(selection_by_key) != expected_selection_keys:
            raise ValueError("reviewer LOMO selection key inventory is incomplete")

        probability_rows = csv_payloads["reviewer_lomo_probabilities"][1]
        probability_keys: set[tuple[str, str, str]] = set()
        exact_probability_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
        for line, row in enumerate(probability_rows, start=2):
            omitted = str(row.get("omitted_model", ""))
            task = normalize_task(row.get("task", ""))
            model = str(row.get("model", ""))
            key = (omitted, task, model)
            if key in probability_keys or omitted not in models or model not in models - {omitted}:
                raise ValueError(f"reviewer LOMO probability line {line} has invalid key")
            probability_keys.add(key)
            exact = _number(row.get("exact_selection_probability"), f"LOMO probability {line} exact")
            tie = _number(row.get("validation_tie_probability"), f"LOMO probability {line} tie")
            if not (0 <= exact <= 1 and 0 <= tie <= 1):
                raise ValueError(f"reviewer LOMO probability line {line} is outside [0,1]")
            exact_probability_sums[(omitted, task)] += exact
        expected_probability_keys = {
            (omitted, task, model)
            for omitted in models
            for task in expected_tasks
            for model in models - {omitted}
        }
        if probability_keys != expected_probability_keys or any(
            not _close(value, 1.0) for value in exact_probability_sums.values()
        ):
            raise ValueError("reviewer LOMO probability inventory is incomplete")

        lomo_rows = csv_payloads["reviewer_lomo_endpoints"][1]
        lomo_keys: set[tuple[str, str, str]] = set()
        for line, row in enumerate(lomo_rows, start=2):
            omitted = str(row.get("omitted_model", ""))
            task = normalize_task(row.get("task", ""))
            dimension = str(row.get("dimension", ""))
            key = (omitted, task, dimension)
            if key in lomo_keys or (task, dimension) not in expected_task_dimensions:
                raise ValueError(f"reviewer LOMO endpoint line {line} has invalid key {key}")
            lomo_keys.add(key)
            selection = selection_by_key[(omitted, task)]
            pool = set(str(row.get("model_pool", "")).split(";"))
            expected_pool = models - {omitted}
            selected = str(row.get("validation_selected_model", ""))
            oracle = str(row.get("test_reliability_oracle_model", ""))
            ties = set(str(row.get("validation_tie_models", "")).split(";"))
            if (
                _strict_int(row.get("n_models"), f"LOMO endpoint {line} n_models") != 4
                or pool != expected_pool
                or selected != selection.get("validation_selected_model")
                or str(row.get("validation_tie_models")) != str(selection.get("validation_tie_models"))
                or oracle not in pool
                or not ties.issubset(pool)
                or _strict_int(row.get("validation_tie_count"), f"LOMO endpoint {line} tie count") != len(ties)
            ):
                raise ValueError(f"reviewer LOMO endpoint line {line} has stale pool/selection")

            higher = _boolean(row.get("higher_is_better"), f"LOMO endpoint {line} orientation")
            selected_value = _number(row.get("selected_reliability_value"), f"LOMO endpoint {line} selected value")
            oracle_value = _number(row.get("oracle_reliability_value"), f"LOMO endpoint {line} oracle value")
            loss_min = _number(row.get("pool_reliability_loss_min"), f"LOMO endpoint {line} pool min")
            loss_max = _number(row.get("pool_reliability_loss_max"), f"LOMO endpoint {line} pool max")
            loss_range = _number(row.get("pool_reliability_loss_range"), f"LOMO endpoint {line} pool range")
            loss_mean = _number(row.get("uniform_random_mean_reliability_loss"), f"LOMO endpoint {line} pool mean")
            regret = _number(row.get("reliability_regret"), f"LOMO endpoint {line} regret")
            exact_random = _number(row.get("exact_uniform_random_expected_regret"), f"LOMO endpoint {line} exact random")
            original_random = _number(row.get("random_selection_expected_regret"), f"LOMO endpoint {line} random")
            normalized = _number(row.get("normalized_regret"), f"LOMO endpoint {line} normalized")
            selected_loss = -selected_value if higher else selected_value
            oracle_loss = -oracle_value if higher else oracle_value
            expected_regret = max(0.0, selected_loss - oracle_loss)
            expected_random = max(0.0, loss_mean - loss_min)
            expected_normalized = regret / loss_range if loss_range > 1e-12 else 0.0
            if (
                loss_range < 0
                or not _close(loss_range, loss_max - loss_min)
                or not _close(oracle_loss, loss_min)
                or not _close(regret, expected_regret)
                or not _close(exact_random, expected_random)
                or not _close(original_random, exact_random)
                or not _close(normalized, expected_normalized)
                or not _close(
                    _number(row.get("regret_minus_random_expected"), f"LOMO endpoint {line} regret-random"),
                    regret - exact_random,
                )
            ):
                raise ValueError(f"reviewer LOMO endpoint line {line} fails four-model regret algebra")
            tie_best = _number(row.get("tie_best_case_regret"), f"LOMO endpoint {line} tie best")
            tie_worst = _number(row.get("tie_worst_case_regret"), f"LOMO endpoint {line} tie worst")
            tie_normalized = _number(row.get("tie_best_case_normalized_regret"), f"LOMO endpoint {line} tie normalized")
            if (
                tie_best < -1e-12
                or tie_worst + 1e-12 < tie_best
                or tie_best > regret + 1e-12
                or not _close(
                    tie_normalized,
                    tie_best / loss_range if loss_range > 1e-12 else 0.0,
                )
            ):
                raise ValueError(f"reviewer LOMO endpoint line {line} has stale tie regret")
        expected_lomo_keys = {
            (omitted, task, dimension)
            for omitted in models
            for task, dimension in expected_task_dimensions
        }
        if lomo_keys != expected_lomo_keys:
            raise ValueError("reviewer LOMO endpoint key inventory is incomplete")

        lomo_summary_rows = csv_payloads["reviewer_lomo_summary"][1]
        summary_by_omission: defaultdict[str, dict[str, set[str]]] = defaultdict(
            lambda: {"effect": set(), "tie": set(), "random": set()}
        )
        observed_summary_keys: set[tuple[str, str]] = set()
        for line, row in enumerate(lomo_summary_rows, start=2):
            omitted = str(row.get("omitted_model", ""))
            dimension = str(row.get("dimension", ""))
            key = (omitted, dimension)
            if key in observed_summary_keys or omitted not in models or dimension not in FINAL_REVIEWER_DEFENSE_DIMENSIONS:
                raise ValueError(f"reviewer LOMO summary line {line} has invalid key")
            observed_summary_keys.add(key)
            expected_n = (
                22
                if dimension in {"applicability_alignment", "selective_relative_excess_aurc"}
                else len(classification_tasks)
                if dimension in {"brier", "ece", "selective_relative_excess_aurc_log_loss"}
                else len(regression_tasks)
            )
            if _strict_int(row.get("n_endpoints"), f"LOMO summary {line} n_endpoints") != expected_n:
                raise ValueError(f"reviewer LOMO summary line {line} endpoint count is stale")
            for column, name in (
                ("practical_effect_signal", "effect"),
                ("tie_robust_effect_signal", "tie"),
                ("random_baseline_retention_signal", "random"),
            ):
                if _boolean(row.get(column), f"LOMO summary {line} {column}"):
                    summary_by_omission[omitted][name].add(dimension)
        expected_summary_keys = {
            (omitted, dimension)
            for omitted in models
            for dimension in FINAL_REVIEWER_DEFENSE_DIMENSIONS
        }
        if observed_summary_keys != expected_summary_keys:
            raise ValueError("reviewer LOMO summary key inventory is incomplete")

        lomo_gate = _json_object(snapshots["reviewer_lomo_gate"], "reviewer LOMO gate")
        if lomo_gate.get("status") != "GO":
            raise ValueError("reviewer LOMO aggregate gate is not GO")
        _require_post_hoc_flags(lomo_gate, "reviewer LOMO gate")
        lomo_required_checks = {
            "canonical_reliability_rows_460",
            "canonical_validation_rows_550",
            "dimension_summary_rows_30",
            "endpoint_dimension_rows_460",
            "exact_uniform_random_reference_recomputed",
            "five_omissions",
            "formal_bootstrap_reps_5000",
            "formal_permutation_reps_20000",
            "formal_selection_bootstrap_reps_1000",
            "four_model_pool_every_endpoint",
            "no_row_level_molecular_output",
            "normalized_regret_recomputed_from_four_model_range",
            "oracle_and_pool_range_recomputed",
            "selection_rows_110",
            "six_dimensions_every_omission",
            "validation_reselected_in_each_pool",
        }
        _require_true_checks(lomo_gate, lomo_required_checks, "reviewer LOMO gate")
        omission_gates = lomo_gate.get("omission_effect_gates")
        if not isinstance(omission_gates, Mapping) or set(omission_gates) != models:
            raise ValueError("reviewer LOMO omission gate inventory is incomplete")
        expected_omission_status = {model: ("HOLD" if model == "gin" else "GO") for model in models}
        for omitted, expected_status in expected_omission_status.items():
            record = omission_gates.get(omitted)
            if not isinstance(record, Mapping) or record.get("status") != expected_status:
                raise ValueError(f"reviewer LOMO omission gate {omitted} must be {expected_status}")
            derived = summary_by_omission[omitted]
            passes = len(derived["effect"]) >= 2 and bool(derived["tie"]) and bool(derived["random"])
            if _boolean(record.get("frozen_criteria_pass"), f"LOMO omission {omitted} pass") != passes:
                raise ValueError(f"reviewer LOMO omission gate {omitted} is inconsistent with summary")
            for key, name in (
                ("dimensions_with_practical_effect_signal", "effect"),
                ("dimensions_with_tie_robust_signal", "tie"),
                ("dimensions_retaining_random_baseline_regret", "random"),
            ):
                if set(record.get(key, [])) != derived[name]:
                    raise ValueError(f"reviewer LOMO omission gate {omitted} {key} is stale")

        # Conditional reliability: frozen family assignment, endpoint aggregation,
        # 5,000 endpoint bootstrap, and Holm within family x threshold.
        conditional_rows = csv_payloads["reviewer_conditional_rows"][1]
        model_family = {
            model: family
            for family, family_models in FINAL_REVIEWER_DEFENSE_FAMILIES.items()
            for model in family_models
        }
        expected_conditional_keys = {
            (task, model, threshold)
            for task in expected_tasks
            for model in models
            for threshold in FINAL_REVIEWER_DEFENSE_THRESHOLDS
        }
        conditional_by_key: dict[tuple[str, str, float], dict[str, str]] = {}
        for line, row in enumerate(conditional_rows, start=2):
            task = normalize_task(row.get("task", ""))
            model = str(row.get("model", ""))
            threshold = _number(row.get("threshold"), f"conditional row {line} threshold")
            key = (task, model, threshold)
            if key in conditional_by_key or key not in expected_conditional_keys:
                raise ValueError(f"reviewer conditional row {line} has invalid key")
            if (
                str(row.get("model_family", "")) != model_family[model]
                or str(row.get("task_type", "")) != task_meta[task]["task_type"]
                or _strict_int(row.get("n_seeds"), f"conditional row {line} n_seeds") != 5
            ):
                raise ValueError(f"reviewer conditional row {line} family/task metadata is stale")
            _boolean(row.get("eligible"), f"conditional row {line} eligible")
            conditional_by_key[key] = row
        if set(conditional_by_key) != expected_conditional_keys:
            raise ValueError("reviewer conditional key inventory is incomplete")

        conditional_summary_rows = csv_payloads["reviewer_conditional_summary"][1]
        expected_conditional_summary_keys = {
            (family, threshold, metric)
            for family in FINAL_REVIEWER_DEFENSE_FAMILIES
            for threshold in FINAL_REVIEWER_DEFENSE_THRESHOLDS
            for metric in FINAL_REVIEWER_DEFENSE_CONDITIONAL_METRICS
        }
        conditional_summary_by_key: dict[tuple[str, float, str], dict[str, str]] = {}
        for line, row in enumerate(conditional_summary_rows, start=2):
            family = str(row.get("model_family", ""))
            threshold = _number(row.get("threshold"), f"conditional summary {line} threshold")
            metric = str(row.get("metric", ""))
            key = (family, threshold, metric)
            if key in conditional_summary_by_key or key not in expected_conditional_summary_keys:
                raise ValueError(f"reviewer conditional summary line {line} has invalid key")
            source_pairs: list[tuple[str, float]] = []
            for (task, model, source_threshold), source in conditional_by_key.items():
                if (
                    model_family[model] != family
                    or not _close(source_threshold, threshold)
                    or not _boolean(source.get("eligible"), "conditional eligibility")
                ):
                    continue
                value = _finite_optional(source.get(metric), f"conditional source {key} value")
                if value is not None:
                    source_pairs.append((task, value))
            endpoint_values = [
                float(np.mean(values))
                for task in sorted({task for task, _ in source_pairs})
                for values in [[value for source_task, value in source_pairs if source_task == task]]
            ]
            if not endpoint_values:
                raise ValueError(f"reviewer conditional summary line {line} has no eligible endpoints")
            pair_values = [value for _, value in source_pairs]
            expected_values = {
                "n_pairs": len(pair_values),
                "n_endpoints": len(endpoint_values),
                "pair_positive_count": sum(value > 0 for value in pair_values),
                "positive_count": sum(value > 0 for value in endpoint_values),
            }
            for column, expected in expected_values.items():
                if _strict_int(row.get(column), f"conditional summary {line} {column}") != expected:
                    raise ValueError(f"reviewer conditional summary line {line} has stale {column}")
            numeric_expected = {
                "pair_positive_fraction": np.mean(np.asarray(pair_values) > 0),
                "positive_fraction": np.mean(np.asarray(endpoint_values) > 0),
                "mean": np.mean(endpoint_values),
                "median": np.median(endpoint_values),
            }
            for column, expected in numeric_expected.items():
                if not _close(_number(row.get(column), f"conditional summary {line} {column}"), float(expected)):
                    raise ValueError(f"reviewer conditional summary line {line} has stale {column}")
            low = _number(row.get("mean_ci_low"), f"conditional summary {line} CI low")
            high = _number(row.get("mean_ci_high"), f"conditional summary {line} CI high")
            pvalue = _number(row.get("sign_flip_p"), f"conditional summary {line} p")
            holm = _number(row.get("holm_p_within_family_threshold"), f"conditional summary {line} Holm p")
            if low > high or not (0 <= pvalue <= 1 and 0 <= holm <= 1):
                raise ValueError(f"reviewer conditional summary line {line} interval/p is invalid")
            conditional_summary_by_key[key] = row
        if set(conditional_summary_by_key) != expected_conditional_summary_keys:
            raise ValueError("reviewer conditional summary key inventory is incomplete")
        for family in FINAL_REVIEWER_DEFENSE_FAMILIES:
            for threshold in FINAL_REVIEWER_DEFENSE_THRESHOLDS:
                group = [
                    conditional_summary_by_key[(family, threshold, metric)]
                    for metric in FINAL_REVIEWER_DEFENSE_CONDITIONAL_METRICS
                ]
                ordered = sorted(
                    enumerate(group),
                    key=lambda item: _number(item[1].get("sign_flip_p"), "Holm source p"),
                )
                expected_adjusted: dict[int, float] = {}
                running = 0.0
                for rank, (index, row) in enumerate(ordered):
                    raw = _number(row.get("sign_flip_p"), "Holm source p")
                    running = max(running, min(1.0, (len(group) - rank) * raw))
                    expected_adjusted[index] = running
                for index, row in enumerate(group):
                    observed = _number(row.get("holm_p_within_family_threshold"), "Holm adjusted p")
                    if not _close(observed, expected_adjusted[index]):
                        raise ValueError("reviewer conditional Holm adjustment is stale")

        conditional_gate = _json_object(
            snapshots["reviewer_conditional_gate"], "reviewer conditional family gate"
        )
        if conditional_gate.get("status") != "GO":
            raise ValueError("reviewer conditional family aggregate gate is not GO")
        _require_post_hoc_flags(conditional_gate, "reviewer conditional gate")
        if not _close(_number(conditional_gate.get("primary_threshold"), "conditional primary threshold"), 0.5):
            raise ValueError("reviewer conditional primary threshold is stale")
        family_policy = conditional_gate.get("family_policy")
        if not isinstance(family_policy, Mapping) or set(family_policy) != set(FINAL_REVIEWER_DEFENSE_FAMILIES):
            raise ValueError("reviewer conditional family policy is incomplete")
        for family, expected_models in FINAL_REVIEWER_DEFENSE_FAMILIES.items():
            if set(family_policy.get(family, [])) != set(expected_models):
                raise ValueError(f"reviewer conditional family policy {family} is stale")
        conditional_required_checks = {
            "canonical_source_rows_330",
            "five_models_assigned_once",
            "formal_endpoint_bootstrap_reps_5000",
            "formal_permutation_reps_20000",
            "frozen_eligibility_preserved",
            "holm_within_family_threshold",
            "no_row_level_molecular_output",
            "six_metrics_per_family_threshold",
            "summary_rows_36",
            "three_thresholds_per_family",
            "two_families",
        }
        _require_true_checks(
            conditional_gate, conditional_required_checks, "reviewer conditional gate"
        )
        family_gates = conditional_gate.get("family_effect_gates")
        if not isinstance(family_gates, Mapping) or set(family_gates) != set(FINAL_REVIEWER_DEFENSE_FAMILIES):
            raise ValueError("reviewer conditional effect gate inventory is incomplete")
        effect_metrics = (
            "log_error_ratio_outside_inside",
            "ece_gap_outside_inside",
            "brier_gap_outside_inside",
            "coverage_error_gap_outside_inside",
            "uq_correlation_advantage_inside",
        )
        for family in FINAL_REVIEWER_DEFENSE_FAMILIES:
            derived: dict[str, bool] = {}
            for metric in effect_metrics:
                row = conditional_summary_by_key[(family, 0.5, metric)]
                derived[metric] = bool(
                    _number(row.get("positive_fraction"), "conditional positive fraction") >= 0.6
                    and _number(row.get("mean_ci_low"), "conditional CI low") > 0
                    and _number(row.get("holm_p_within_family_threshold"), "conditional Holm") < 0.05
                )
            record = family_gates.get(family)
            if not isinstance(record, Mapping) or record.get("status") != "GO":
                raise ValueError(f"reviewer conditional family gate {family} is not GO")
            _strict_true(
                record.get("frozen_original_effect_gate_pass"),
                f"reviewer conditional family {family} pass",
            )
            if record.get("criteria") != derived:
                raise ValueError(f"reviewer conditional family gate {family} criteria are stale")
            strong = sum(derived.values())
            if (
                _strict_int(record.get("strong_conditional_dimensions"), f"family {family} strong dimensions") != strong
                or not derived["log_error_ratio_outside_inside"]
                or strong < 2
            ):
                raise ValueError(f"reviewer conditional family gate {family} split is stale")

        # Combined gate and manifest bind the complete post-hoc release.
        combined_gate = _json_object(snapshots["reviewer_combined_gate"], "reviewer combined gate")
        if combined_gate.get("status") != "GO" or combined_gate.get("scope") != FINAL_REVIEWER_DEFENSE_SCOPE:
            raise ValueError("reviewer combined gate is not final GO")
        _require_post_hoc_flags(combined_gate, "reviewer combined gate")
        combined_required_checks = {
            "conditional_family_gate_go",
            "cross_endpoint_overlap_gate_go",
            "leave_one_model_out_gate_go",
            "no_new_primary_claim",
            "no_row_level_molecular_output",
            "post_hoc_only",
            "source_inputs_sha256_and_size_unchanged",
            "zero_refit",
        }
        _require_true_checks(combined_gate, combined_required_checks, "reviewer combined gate")
        expected_parameters = {
            "bootstrap_reps": FINAL_REVIEWER_DEFENSE_BOOTSTRAP_REPS,
            "permutation_reps": FINAL_REVIEWER_DEFENSE_PERMUTATION_REPS,
            "selection_bootstrap_reps": FINAL_REVIEWER_DEFENSE_SELECTION_BOOTSTRAP_REPS,
            "random_seed": FINAL_REVIEWER_DEFENSE_RANDOM_SEED,
        }
        parameters = combined_gate.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("reviewer combined parameters are missing")
        for key, expected in expected_parameters.items():
            if _strict_int(parameters.get(key), f"reviewer combined {key}") != expected:
                raise ValueError(f"reviewer combined {key} is stale")

        manifest = _json_object(snapshots["reviewer_manifest"], "reviewer manifest")
        if manifest.get("schema") != "reviewer_defense_sensitivities_v1":
            raise ValueError("reviewer manifest schema is stale")
        _require_post_hoc_flags(manifest, "reviewer manifest")
        _strict_true(manifest.get("zero_refit"), "reviewer manifest zero_refit")
        if manifest.get("script") != "experiments/analyze_reviewer_defense_sensitivities.py":
            raise ValueError("reviewer manifest script path is stale")
        if manifest.get("script_sha256") != snapshots["reviewer_defense_script"].sha256:
            raise ValueError("reviewer manifest script hash is stale")
        manifest_parameters = manifest.get("parameters")
        if not isinstance(manifest_parameters, Mapping):
            raise ValueError("reviewer manifest parameters are missing")
        for key, expected in expected_parameters.items():
            if _strict_int(manifest_parameters.get(key), f"reviewer manifest {key}") != expected:
                raise ValueError(f"reviewer manifest {key} is stale")
        input_files = manifest.get("input_files")
        if not isinstance(input_files, list) or len(input_files) != 25:
            raise ValueError("reviewer manifest must bind exactly 25 source inputs")
        input_names: set[str] = set()
        for index, record in enumerate(input_files):
            if not isinstance(record, Mapping):
                raise ValueError("reviewer manifest input binding is malformed")
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"reviewer manifest input {index} path is unsafe")
            if relative.as_posix() in input_names:
                raise ValueError("reviewer manifest input paths are duplicated")
            input_names.add(relative.as_posix())
            if not _valid_sha256(str(record.get("sha256", ""))):
                raise ValueError(f"reviewer manifest input {index} hash is invalid")
            _strict_int(record.get("bytes"), f"reviewer manifest input {index} bytes", minimum=0)
        output_files = manifest.get("output_files")
        if not isinstance(output_files, list):
            raise ValueError("reviewer manifest output bindings are missing")
        expected_output_names = {
            filename
            for logical_name, (filename, _) in FINAL_REVIEWER_DEFENSE_OUTPUTS.items()
            if logical_name != "reviewer_manifest"
        }
        output_by_name: dict[str, Mapping[str, Any]] = {}
        for record in output_files:
            if not isinstance(record, Mapping):
                raise ValueError("reviewer manifest output binding is malformed")
            name = str(record.get("path", ""))
            if Path(name).name != name or name in output_by_name:
                raise ValueError("reviewer manifest output path is unsafe or duplicated")
            output_by_name[name] = record
        if set(output_by_name) != expected_output_names:
            raise ValueError("reviewer manifest output inventory is incomplete")
        logical_by_filename = {
            filename: logical_name
            for logical_name, (filename, _) in FINAL_REVIEWER_DEFENSE_OUTPUTS.items()
        }
        for filename, record in output_by_name.items():
            snapshot = snapshots[logical_by_filename[filename]]
            if (
                record.get("sha256") != snapshot.sha256
                or _strict_int(record.get("bytes"), f"reviewer manifest {filename} bytes") != snapshot.bytes
            ):
                raise ValueError(f"reviewer manifest output binding {filename} is stale")

        for logical_name in (
            "reviewer_conditional_table",
            "reviewer_overlap_table",
            "reviewer_lomo_table",
        ):
            text = snapshots[logical_name].data.decode("utf-8-sig")
            if not text.strip() or "\\begin{tabular}" not in text:
                raise ValueError(f"reviewer-defense table {logical_name} is empty or malformed")
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        failures.append(str(exc))

    complete = len(failures) == failure_start
    return {
        "status": "PASS" if complete else "HOLD",
        "required_outputs": len(FINAL_REVIEWER_DEFENSE_OUTPUTS),
        "analysis_script_present": "reviewer_defense_script" in snapshots,
        "observed_rows": observed_rows,
        "expected_rows": dict(FINAL_REVIEWER_DEFENSE_EXPECTED_ROWS),
        "overlap": {
            "endpoints": 22,
            "pairs": 231,
            "pairs_with_overlap": 209,
            "connected_components": 1,
        },
        "leave_one_model_out": {
            "four_model_pool_recomputed": complete,
            "passing_omissions": [
                "chemprop_dmpnn",
                "lightgbm",
                "rf",
                "xgboost",
            ],
            "held_omission": "gin",
        },
        "conditional_families": sorted(FINAL_REVIEWER_DEFENSE_FAMILIES),
        "zero_refit": True,
        "post_hoc_not_primary_claim": True,
        "no_row_level_molecular_output": True,
        "directly_releasable": complete,
    }


def audit_final_supplements(root: Path) -> FinalSupplementAudit:
    """Fail-closed audit of the locked supplementary release evidence.

    Raw prediction exports are indexed but explicitly excluded from direct
    redistribution because they contain structures and source labels. Public,
    de-identified views are generated separately by the release builder.
    """

    root = root.resolve()
    failures: list[str] = []
    catalog = SupplementSnapshotCatalog(root)
    public_extra_payloads: list[tuple[str, bytes]] = []
    tables = root / "results" / "tables"
    cleaned_dir = tables / "cleaned_sensitivity_locked_cache"
    crossfit_dir = tables / "crossfit_oracle_sensitivity_final"
    crossfit_influence_dir = tables / "crossfit_influence_sensitivity_final"
    reviewer_defense_dir = tables / "reviewer_defense_sensitivities_final"
    predictions_dir = root / "results" / "predictions" / "cleaned_sensitivity_locked_cache"

    fixed_files: dict[str, tuple[Path, str]] = {
        "sensitivity_config": (
            root / "experiments" / "config" / "sensitivity.yaml",
            "cleaned_sensitivity_config",
        ),
        "sensitivity_runner": (
            root / "experiments" / "run_cleaned_sensitivity_cpu.py",
            "cleaned_sensitivity_code",
        ),
        "sensitivity_summarizer": (
            root / "experiments" / "summarize_cleaned_sensitivity.py",
            "cleaned_sensitivity_code",
        ),
        "sensitivity_prediction_auditor": (
            root / "experiments" / "audit_cleaned_sensitivity_predictions.py",
            "cleaned_sensitivity_code",
        ),
        "sensitivity_library": (
            root / "src" / "admet_benchmark" / "sensitivity.py",
            "cleaned_sensitivity_code",
        ),
        "cleaned_runs": (
            tables / "cleaned_sensitivity_locked_cache_runs.csv",
            "cleaned_sensitivity_evidence",
        ),
        "cleaned_raw_audit": (
            tables / "cleaned_sensitivity_locked_cache_audit.jsonl",
            "cleaned_sensitivity_evidence",
        ),
        "cleaned_manifest": (
            tables / "cleaned_sensitivity_locked_cache_manifest.json",
            "cleaned_sensitivity_evidence",
        ),
        "cleaned_upgrade_receipt": (
            tables
            / "cleaned_sensitivity_locked_cache_manifest_provenance_upgrade_receipt.json",
            "cleaned_sensitivity_evidence",
        ),
        "cleaned_summary_gate": (
            cleaned_dir / "cleaned_sensitivity_gate.json",
            "cleaned_sensitivity_gate",
        ),
        "cleaned_by_task": (
            cleaned_dir / "cleaned_sensitivity_by_task.csv",
            "cleaned_sensitivity_summary",
        ),
        "cleaned_by_task_model": (
            cleaned_dir / "cleaned_sensitivity_by_task_model.csv",
            "cleaned_sensitivity_summary",
        ),
        "cleaned_flat_audit": (
            cleaned_dir / "cleaned_sensitivity_audit.csv",
            "cleaned_sensitivity_summary",
        ),
        "cleaned_prediction_gate": (
            cleaned_dir / "cleaned_sensitivity_prediction_gate.json",
            "cleaned_prediction_gate",
        ),
        "cleaned_prediction_audit": (
            cleaned_dir / "cleaned_sensitivity_prediction_audit.csv",
            "cleaned_prediction_audit",
        ),
        "cpu_defense_gate": (
            tables / "locked_cache_cpu_defense_gate.json",
            "cpu_defense_gate",
        ),
        "y_random_summary": (
            tables / "y_randomization_all_tasks_locked_cache.csv",
            "cpu_defense_evidence",
        ),
        "y_random_scores": (
            tables / "y_randomization_all_tasks_scores_locked_cache.csv",
            "cpu_defense_evidence",
        ),
        "shap_selected": (
            tables / "shap_selected_locked_cache.csv",
            "cpu_defense_evidence",
        ),
        "shap_mapping": (
            tables / "shap_morgan_bit_substructures_locked_cache.csv",
            "cpu_defense_evidence",
        ),
        "traditional_selection": (
            tables / "traditional_grid_locked_cache_best_by_task.csv",
            "cpu_defense_evidence",
        ),
        "final_neural_precheck": (
            tables
            / "final_neural_artifact_precheck_final"
            / "final_neural_precheck_gate.json",
            "final_neural_completion_evidence",
        ),
        "final_neural_checkpoint_integrity": (
            tables / "final_neural_checkpoint_integrity.json",
            "final_neural_checkpoint_integrity_evidence",
        ),
        "paper_main_tex": (
            root / "paper" / "main.tex",
            "final_manuscript_source",
        ),
        "paper_supplementary_tex": (
            root / "paper" / "supplementary.tex",
            "final_manuscript_source",
        ),
        "paper_references": (
            root / "paper" / "references.bib",
            "final_reference_inventory",
        ),
        "paper_reference_audit_final": (
            root / "paper" / "reference_audit_final_20260715.md",
            "final_reference_inventory",
        ),
        "paper_numeric_audit": (
            root / "paper" / "final_numeric_source_audit_20260714.md",
            "final_numeric_audit",
        ),
        "paper_supplementary_numeric_audit": (
            root / "paper" / "supplementary_numeric_source_audit_20260714.md",
            "final_numeric_audit",
        ),
        "paper_main_pdf": (
            root / "paper" / "build_final" / "main.pdf",
            "final_compiled_manuscript",
        ),
        "paper_supplementary_pdf": (
            root / "paper" / "build_supplementary_final" / "supplementary.pdf",
            "final_compiled_manuscript",
        ),
    }
    code_files = (
        "run_y_randomization_all.py",
        "run_shap_selected.py",
        "map_shap_morgan_bits.py",
        "summarize_locked_cache_cpu_defenses.py",
        "analyze_posthoc_ablations.py",
        "make_manuscript_tables.py",
    )
    for name in code_files:
        fixed_files[f"code_{name}"] = (
            root / "experiments" / name,
            "supplementary_analysis_code",
        )
    for logical_name, (filename, role) in FINAL_CROSSFIT_OUTPUTS.items():
        fixed_files[logical_name] = (crossfit_dir / filename, role)
    for logical_name, (relative, role) in FINAL_CROSSFIT_DERIVED_OUTPUTS.items():
        fixed_files[logical_name] = (root / relative, role)
    fixed_files["crossfit_influence_script"] = (
        root / "experiments" / "analyze_crossfit_influence_sensitivity.py",
        "crossfit_influence_analysis_code",
    )
    for logical_name, (filename, role) in FINAL_CROSSFIT_INFLUENCE_OUTPUTS.items():
        fixed_files[logical_name] = (crossfit_influence_dir / filename, role)
    fixed_files["reviewer_defense_script"] = (
        root / "experiments" / "analyze_reviewer_defense_sensitivities.py",
        "post_hoc_reviewer_defense_analysis_code",
    )
    for logical_name, (filename, role) in FINAL_REVIEWER_DEFENSE_OUTPUTS.items():
        fixed_files[logical_name] = (reviewer_defense_dir / filename, role)
    for logical_name, (relative, _) in FINAL_MANUSCRIPT_FIGURES.items():
        fixed_files[logical_name] = (
            root / relative,
            "final_manuscript_figure",
        )
    for logical_name, (relative, role) in FINAL_PUBLIC_RESULT_OUTPUTS.items():
        fixed_files[logical_name] = (root / relative, role)

    snapshots: dict[str, FileSnapshot] = {}
    for name, (path, role) in fixed_files.items():
        snapshot = catalog.require(path, role, failures)
        if snapshot is not None:
            snapshots[name] = snapshot

    for name in ("cleaned_manifest", "cleaned_upgrade_receipt"):
        if name in snapshots:
            catalog.require(
                snapshots[name].path,
                fixed_files[name][1],
                failures,
                package_action="exclude_nonportable",
            )
    if "shap_mapping" in snapshots:
        catalog.require(
            snapshots["shap_mapping"].path,
            fixed_files["shap_mapping"][1],
            failures,
            package_action="exclude_raw_structure_and_labels",
        )
        try:
            shap_columns, shap_rows = _csv_rows(
                snapshots["shap_mapping"], "SHAP structure mapping"
            )
            if len(shap_rows) != 80:
                raise ValueError(f"SHAP structure mapping has {len(shap_rows)} rows, expected 80")
            if "example_smiles" not in shap_columns:
                raise ValueError("SHAP structure mapping lacks example_smiles disclosure")
            safe_columns = [
                column
                for column in shap_columns
                if "smiles" not in column.lower()
                and "label" not in column.lower()
                and column not in {"bit_atom_index"}
            ]
            if not safe_columns:
                raise ValueError("SHAP mapping has no safe public columns")
            safe_rows = [
                {column: row.get(column, "") for column in safe_columns}
                for row in shap_rows
            ]
            public_extra_payloads.append(
                (
                    "shap_bit_importance_without_structures.csv",
                    _serialize_public_csv(safe_columns, safe_rows),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))
    if (
        "cleaned_prediction_audit" in snapshots
        and str(root).encode("utf-8") in snapshots["cleaned_prediction_audit"].data
    ):
        snapshot = snapshots["cleaned_prediction_audit"]
        catalog.require(
            snapshot.path,
            fixed_files["cleaned_prediction_audit"][1],
            failures,
            package_action="exclude_nonportable",
        )
        try:
            columns, rows = _csv_rows(snapshot, "nonportable cleaned prediction audit")
            prefix = str(root) + os.sep
            sanitized_rows = [
                {
                    column: (
                        value.replace(prefix, "")
                        if isinstance(value, str)
                        else value
                    )
                    for column, value in row.items()
                }
                for row in rows
            ]
            public_extra_payloads.append(
                (
                    "cleaned_sensitivity_prediction_audit.csv",
                    _serialize_public_csv(columns, sanitized_rows),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    if "final_neural_precheck" in snapshots:
        snapshot = snapshots["final_neural_precheck"]
        catalog.require(
            snapshot.path,
            fixed_files["final_neural_precheck"][1],
            failures,
            package_action="exclude_raw_structure_and_labels",
        )
        try:
            value = _json_object(snapshot, "final neural precheck")
            complete_by_model = value.get("complete_by_model")
            sample_alignment = value.get("sample_alignment")
            representation = value.get("representation_compatibility")
            if not isinstance(complete_by_model, Mapping):
                raise ValueError("final neural precheck completion counts are missing")
            if not isinstance(sample_alignment, Mapping):
                raise ValueError("final neural precheck sample alignment is missing")
            if not isinstance(representation, Mapping):
                raise ValueError("final neural precheck representation audit is missing")
            safe_precheck = {
                "schema_version": 1,
                "status": value.get("status"),
                "expected_total_formal_neural_runs": value.get(
                    "expected_total_formal_neural_runs"
                ),
                "expected_per_model": value.get("expected_per_model"),
                "required_models": value.get("required_models"),
                "required_tasks": value.get("required_tasks"),
                "required_seeds": value.get("required_seeds"),
                "complete_by_model": dict(complete_by_model),
                "legacy_exclusion_ok": value.get("legacy_exclusion_ok"),
                "sample_alignment": {
                    "official_split_verification_enabled": sample_alignment.get(
                        "official_split_verification_enabled"
                    ),
                    "all_expected_runs_match_official_samples": sample_alignment.get(
                        "all_expected_runs_match_official_samples"
                    ),
                },
                "representation_compatibility": {
                    "policy": representation.get("policy"),
                    "gin_global_config_ok": representation.get("gin_global_config_ok"),
                    "chemprop_global_config_ok": representation.get(
                        "chemprop_global_config_ok"
                    ),
                    "chemprop_all_included_runs_valid": representation.get(
                        "chemprop_all_included_runs_valid"
                    ),
                },
                "failure_count": len(value.get("failures", [])),
            }
            public_extra_payloads.append(
                (
                    "final_neural_completion_gate_sanitized.json",
                    (
                        json.dumps(
                            safe_precheck,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    if "final_neural_checkpoint_integrity" in snapshots:
        snapshot = snapshots["final_neural_checkpoint_integrity"]
        catalog.require(
            snapshot.path,
            fixed_files["final_neural_checkpoint_integrity"][1],
            failures,
            package_action="exclude_nonportable",
        )
        try:
            value = _json_object(snapshot, "final neural checkpoint integrity")
            models: dict[str, dict[str, Any]] = {}
            for model in SUBMISSION_NEURAL_MODELS:
                record = value.get(model)
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"final checkpoint integrity omits model {model}"
                    )
                models[model] = {
                    "files": record.get("files"),
                    "zip_integrity_ok": record.get("zip_integrity_ok"),
                }
            failures_value = value.get("failures", [])
            safe_integrity = {
                "schema_version": 1,
                "status": "PASS" if not failures_value else "HOLD",
                "models": models,
                "failure_count": len(failures_value),
            }
            public_extra_payloads.append(
                (
                    "final_neural_checkpoint_integrity_summary.json",
                    (
                        json.dumps(
                            safe_integrity,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    # Submission documents are a separate, author-approved deliverable. Keeping
    # drafts with a pending DOI, authorship, CRediT, or competing-interest text
    # out of the public data/software archive prevents a circular DOI deposit
    # and accidental publication of unapproved declarations.
    for name in (
        "paper_main_tex",
        "paper_supplementary_tex",
        "paper_references",
        "paper_reference_audit_final",
        "paper_numeric_audit",
        "paper_supplementary_numeric_audit",
        "paper_main_pdf",
        "paper_supplementary_pdf",
    ):
        if name in snapshots:
            catalog.require(
                snapshots[name].path,
                "author_controlled_publication_document",
                failures,
                package_action="exclude_nonportable",
            )

    crossfit_details, crossfit_public_payloads = _audit_crossfit_release_evidence(
        root=root,
        catalog=catalog,
        fixed_files=fixed_files,
        snapshots=snapshots,
        failures=failures,
    )
    public_extra_payloads.extend(crossfit_public_payloads)
    crossfit_influence_details = _audit_crossfit_influence_release_evidence(
        snapshots=snapshots,
        failures=failures,
    )
    reviewer_defense_details = _audit_reviewer_defense_release_evidence(
        snapshots=snapshots,
        failures=failures,
    )

    config: dict[str, Any] = {}
    if "sensitivity_config" in snapshots:
        try:
            value = yaml.safe_load(snapshots["sensitivity_config"].data.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("sensitivity config root is not a mapping")
            config = value
            tasks = tuple(str(item) for item in value.get("selected_tasks_for_cleaned_rerun", []))
            models = tuple(str(item) for item in value.get("models", []))
            seeds = tuple(
                _strict_int(item, f"sensitivity config seed {index}", minimum=1)
                for index, item in enumerate(value.get("seeds", []))
            )
            if tasks != FINAL_CLEANED_TASKS:
                failures.append("sensitivity config does not lock the exact eight-task order")
            if models != FINAL_CLEANED_MODELS:
                failures.append("sensitivity config does not lock rf/xgboost/lightgbm")
            if seeds != FINAL_CLEANED_SEEDS:
                failures.append("sensitivity config does not lock seeds 1--5")
            if value.get("protocol") != "paired-common-test-v1":
                failures.append("sensitivity config paired protocol is not locked")
        except (UnicodeDecodeError, yaml.YAMLError, ValueError, TypeError) as exc:
            failures.append(f"invalid sensitivity config: {exc}")

    manifest: dict[str, Any] = {}
    cleaned_source_data_sha256 = ""
    if "cleaned_manifest" in snapshots:
        try:
            manifest = _json_object(snapshots["cleaned_manifest"], "cleaned manifest")
            input_hashes = manifest.get("input_sha256")
            if not isinstance(input_hashes, dict) or set(input_hashes) != set(FINAL_CLEANED_TASKS):
                raise ValueError("cleaned manifest source hash task inventory is incomplete")
            for task in FINAL_CLEANED_TASKS:
                task_hashes = input_hashes.get(task)
                if not isinstance(task_hashes, dict) or set(task_hashes) != {
                    "train_val.csv",
                    "test.csv",
                }:
                    raise ValueError(f"cleaned manifest source files are incomplete for {task}")
                if not all(_valid_sha256(value) for value in task_hashes.values()):
                    raise ValueError(f"cleaned manifest source hash is invalid for {task}")
            cleaned_source_data_sha256 = stable_json_hash(input_hashes)
        except ValueError as exc:
            failures.append(str(exc))

    prediction_gate: dict[str, Any] = {}
    audit_columns: list[str] = []
    audit_rows: list[dict[str, str]] = []
    if "cleaned_prediction_gate" in snapshots:
        try:
            prediction_gate = _json_object(
                snapshots["cleaned_prediction_gate"], "cleaned prediction gate"
            )
        except ValueError as exc:
            failures.append(str(exc))
    if "cleaned_prediction_audit" in snapshots:
        try:
            audit_columns, audit_rows = _csv_rows(
                snapshots["cleaned_prediction_audit"], "cleaned prediction audit"
            )
        except ValueError as exc:
            failures.append(str(exc))

    cleaned_predictions: list[CleanedPredictionSource] = []
    inventory_records: list[dict[str, Any]] = []
    expected_cleaned_keys = {
        (task.lower(), model, seed)
        for task in FINAL_CLEANED_TASKS
        for model in FINAL_CLEANED_MODELS
        for seed in FINAL_CLEANED_SEEDS
    }
    observed_cleaned_keys: set[tuple[str, str, int]] = set()
    expected_prediction_names: set[str] = set()
    if audit_rows:
        missing_columns = sorted(
            CLEANED_PREDICTION_AUDIT_REQUIRED_COLUMNS.difference(audit_columns)
        )
        if missing_columns:
            failures.append(
                "cleaned prediction audit missing columns: " + ",".join(missing_columns)
            )
        for row_number, row in enumerate(audit_rows, start=2):
            try:
                task = str(row.get("task", "")).strip().lower()
                model = str(row.get("model", "")).strip().lower()
                seed = _strict_int(row.get("seed"), f"prediction audit line {row_number} seed", minimum=1)
                key = (task, model, seed)
                if key not in expected_cleaned_keys:
                    raise ValueError(f"prediction audit line {row_number} has unexpected key {key}")
                if key in observed_cleaned_keys:
                    raise ValueError(f"prediction audit line {row_number} duplicates key {key}")
                observed_cleaned_keys.add(key)
                expected_name = f"{task}_{model}_seed{seed}_paired.csv"
                recorded_name = Path(str(row.get("prediction_path", ""))).name
                if recorded_name != expected_name:
                    raise ValueError(
                        f"prediction audit line {row_number} filename {recorded_name!r} "
                        f"!= {expected_name!r}"
                    )
                expected_prediction_names.add(expected_name)
                snapshot = catalog.require(
                    predictions_dir / expected_name,
                    "cleaned_prediction_raw_index",
                    failures,
                    package_action="exclude_raw_structure_and_labels",
                )
                if snapshot is None:
                    continue
                columns, prediction_rows = _csv_rows(
                    snapshot, f"cleaned prediction {expected_name}"
                )
                required_prediction_columns = {
                    "task",
                    "model",
                    "seed",
                    "evaluation_set",
                    "canonical_smiles",
                    "label",
                    "official_prediction",
                    "cleaned_prediction",
                }
                if not required_prediction_columns.issubset(columns):
                    raise ValueError(
                        f"cleaned prediction {expected_name} lacks the locked raw schema"
                    )
                rows = _strict_int(
                    row.get("rows"), f"prediction audit line {row_number} rows", minimum=1
                )
                recorded_bytes = _strict_int(
                    row.get("prediction_bytes"),
                    f"prediction audit line {row_number} prediction_bytes",
                    minimum=1,
                )
                if rows != len(prediction_rows):
                    raise ValueError(
                        f"cleaned prediction {expected_name} row count is stale"
                    )
                if recorded_bytes != snapshot.bytes:
                    raise ValueError(
                        f"cleaned prediction {expected_name} byte count is stale"
                    )
                if row.get("prediction_sha256") != snapshot.sha256:
                    raise ValueError(f"cleaned prediction {expected_name} hash is stale")
                for column in CLEANED_PREDICTION_AUDIT_TRUE_COLUMNS:
                    _strict_true(
                        row.get(column),
                        f"prediction audit line {row_number} {column}",
                    )
                for index, prediction_row in enumerate(prediction_rows, start=2):
                    for column in ("official_prediction", "cleaned_prediction"):
                        if not _finite_number(prediction_row.get(column)):
                            raise ValueError(
                                f"cleaned prediction {expected_name} line {index} "
                                f"has non-finite {column}"
                            )
                inventory_records.append(
                    {
                        "filename": expected_name,
                        "task": task,
                        "model": model,
                        "seed": seed,
                        "sha256": snapshot.sha256,
                        "bytes": snapshot.bytes,
                        "rows": rows,
                    }
                )
                cleaned_predictions.append(
                    CleanedPredictionSource(task, model, seed, rows, snapshot)
                )
            except ValueError as exc:
                failures.append(str(exc))

    if observed_cleaned_keys != expected_cleaned_keys:
        failures.append(
            "cleaned prediction audit key matrix is not exactly 8 tasks x 3 models x 5 seeds"
        )
    if len(audit_rows) != FINAL_CLEANED_PREDICTION_COUNT:
        failures.append(
            f"cleaned prediction audit has {len(audit_rows)} rows, expected "
            f"{FINAL_CLEANED_PREDICTION_COUNT}"
        )
    if predictions_dir.is_dir():
        actual_names = {
            path.name
            for path in predictions_dir.iterdir()
            if path.is_file() and not _is_appledouble(path)
        }
        if actual_names != expected_prediction_names:
            failures.append(
                "cleaned prediction directory does not contain the exact audited file set"
            )
    else:
        failures.append("cleaned prediction directory is missing")

    prediction_inventory_sha256 = _prediction_inventory_digest(inventory_records)
    if prediction_gate:
        if prediction_gate.get("status") != "GO":
            failures.append("cleaned prediction gate status is not GO")
        for key in (
            "complete_expected_matrix",
            "sample_label_identity_aligned_within_task",
            "source_common_test_reconstructed_exactly",
            "prediction_metrics_recomputed_match_runs",
        ):
            _expect_true(prediction_gate, key, "cleaned prediction gate", failures)
        for key in (
            "expected_prediction_files",
            "observed_prediction_files",
            "audited_prediction_files",
        ):
            _expect_int(
                prediction_gate,
                key,
                FINAL_CLEANED_PREDICTION_COUNT,
                "cleaned prediction gate",
                failures,
            )
        if prediction_gate.get("prediction_inventory_sha256") != prediction_inventory_sha256:
            failures.append("cleaned prediction inventory aggregate digest is stale")
        inventory = prediction_gate.get("prediction_inventory")
        if not isinstance(inventory, Mapping):
            failures.append("cleaned prediction gate inventory is missing")
        else:
            _expect_int(
                inventory,
                "files",
                FINAL_CLEANED_PREDICTION_COUNT,
                "cleaned prediction inventory",
                failures,
            )
            if inventory.get("sha256") != prediction_inventory_sha256:
                failures.append("cleaned prediction inventory SHA-256 binding is stale")
            expected_total_bytes = sum(int(row["bytes"]) for row in inventory_records)
            expected_total_rows = sum(int(row["rows"]) for row in inventory_records)
            _expect_int(
                inventory,
                "bytes",
                expected_total_bytes,
                "cleaned prediction inventory",
                failures,
            )
            _expect_int(
                inventory,
                "rows",
                expected_total_rows,
                "cleaned prediction inventory",
                failures,
            )
            entries = inventory.get("entries")
            expected_entries = {
                str(row["filename"]): {
                    "sha256": row["sha256"],
                    "bytes": row["bytes"],
                    "rows": row["rows"],
                }
                for row in inventory_records
            }
            if entries != expected_entries:
                failures.append("cleaned prediction gate per-file inventory is stale")
        if "cleaned_prediction_audit" in snapshots:
            _validate_snapshot_binding(
                prediction_gate.get("audit_csv"),
                snapshots["cleaned_prediction_audit"],
                "cleaned prediction audit CSV",
                failures,
                row_count=len(audit_rows),
            )

    if "cleaned_upgrade_receipt" in snapshots:
        try:
            receipt = _json_object(
                snapshots["cleaned_upgrade_receipt"], "cleaned provenance upgrade receipt"
            )
            if (
                _strict_int(receipt.get("schema_version"), "upgrade receipt schema") != 1
                or receipt.get("operation")
                != "cleaned_sensitivity_manifest_bound_provenance_upgrade"
            ):
                raise ValueError("cleaned provenance upgrade receipt contract mismatch")
            entries = receipt.get("files")
            if not isinstance(entries, list) or len(entries) != 122:
                raise ValueError("cleaned provenance upgrade receipt must bind 122 files")
            expected_current: dict[tuple[str, str], FileSnapshot] = {}
            if "cleaned_manifest" in snapshots:
                expected_current[("manifest", snapshots["cleaned_manifest"].path.name)] = snapshots[
                    "cleaned_manifest"
                ]
            if "cleaned_runs" in snapshots:
                expected_current[("run_table", snapshots["cleaned_runs"].path.name)] = snapshots[
                    "cleaned_runs"
                ]
            expected_current.update(
                {
                    ("prediction", source.snapshot.path.name): source.snapshot
                    for source in cleaned_predictions
                }
            )
            observed_current: set[tuple[str, str]] = set()
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise ValueError(f"upgrade receipt file {index} is malformed")
                kind = str(entry.get("kind", ""))
                name = Path(str(entry.get("path", ""))).name
                key = (kind, name)
                if key in observed_current:
                    raise ValueError(f"upgrade receipt duplicates {key}")
                observed_current.add(key)
                snapshot = expected_current.get(key)
                if snapshot is None:
                    raise ValueError(f"upgrade receipt contains unexpected file {key}")
                new_state = entry.get("new")
                _validate_snapshot_binding(
                    new_state,
                    snapshot,
                    f"upgrade receipt current state {key}",
                    failures,
                )
                if kind != "manifest":
                    _strict_true(
                        entry.get("semantic_payload_unchanged"),
                        f"upgrade receipt semantic preservation {key}",
                    )
            if observed_current != set(expected_current):
                raise ValueError("upgrade receipt current file set is incomplete")
            aggregate = receipt.get("aggregate_inventory")
            if not isinstance(aggregate, Mapping):
                raise ValueError("upgrade receipt aggregate inventory is missing")
            if _strict_int(aggregate.get("file_count"), "upgrade receipt file_count") != 122:
                raise ValueError("upgrade receipt aggregate file count is not 122")
            _strict_true(
                aggregate.get("csv_semantic_payloads_unchanged"),
                "upgrade receipt aggregate semantic preservation",
            )
            config_binding = receipt.get("config_binding")
            if not isinstance(config_binding, Mapping) or "sensitivity_config" not in snapshots:
                raise ValueError("upgrade receipt config binding is missing")
            if config_binding.get("current_config_sha256") != snapshots[
                "sensitivity_config"
            ].sha256:
                raise ValueError("upgrade receipt config binding is stale")
        except ValueError as exc:
            failures.append(str(exc))

    summary_gate: dict[str, Any] = {}
    if "cleaned_summary_gate" in snapshots:
        try:
            summary_gate = _json_object(
                snapshots["cleaned_summary_gate"], "cleaned sensitivity summary gate"
            )
        except ValueError as exc:
            failures.append(str(exc))
    if summary_gate:
        if summary_gate.get("status") not in SENSITIVITY_COMPLETE_STATUSES:
            failures.append("cleaned sensitivity summary status is incomplete")
        for key in ("complete_expected_matrix", "split_source_gate"):
            _expect_true(summary_gate, key, "cleaned sensitivity summary gate", failures)
        for key, expected in (
            ("expected_rows", FINAL_CLEANED_PREDICTION_COUNT),
            ("observed_rows", FINAL_CLEANED_PREDICTION_COUNT),
            ("missing_keys_count", 0),
            ("unexpected_keys_count", 0),
            ("n_tasks", len(FINAL_CLEANED_TASKS)),
        ):
            _expect_int(
                summary_gate,
                key,
                expected,
                "cleaned sensitivity summary gate",
                failures,
            )
        if summary_gate.get("split_source") != "locked_cache":
            failures.append("cleaned sensitivity summary does not declare locked_cache")
        run_integrity = summary_gate.get("run_integrity_gate")
        if not isinstance(run_integrity, Mapping):
            failures.append("cleaned sensitivity run-integrity gate is missing")
        else:
            for key in (
                "finite_core_values",
                "strict_integer_fields",
                "fingerprint_contract_matches",
                "single_thread_cpu_contract_matches",
                "task_metadata_matches_tdc",
                "primary_metric_columns_match",
                "paired_delta_algebra_matches",
                "sample_counts_consistent_across_models",
            ):
                _expect_true(run_integrity, key, "cleaned run integrity", failures)
            _expect_int(
                run_integrity,
                "validated_rows",
                FINAL_CLEANED_PREDICTION_COUNT,
                "cleaned run integrity",
                failures,
            )
        manifest_gate = summary_gate.get("manifest_provenance_gate")
        if not isinstance(manifest_gate, Mapping):
            failures.append("cleaned manifest provenance gate is missing")
        else:
            checks = manifest_gate.get("checks")
            if not isinstance(checks, Mapping) or not checks or not all(
                value is True for value in checks.values()
            ):
                failures.append("cleaned manifest provenance checks are not all true")
        audit_gate = summary_gate.get("audit_provenance_gate")
        if not isinstance(audit_gate, Mapping):
            failures.append("cleaned raw-audit provenance gate is missing")
        else:
            for key in (
                "exact_task_seed_matrix",
                "schema_v2",
                "post_precedence_overlap_zero",
                "parser_repair_counts_match_sources",
                "split_output_counts_match_runs",
            ):
                _expect_true(audit_gate, key, "cleaned raw-audit provenance", failures)
            _expect_int(
                audit_gate,
                "expected_payloads",
                len(FINAL_CLEANED_TASKS) * len(FINAL_CLEANED_SEEDS),
                "cleaned raw-audit provenance",
                failures,
            )
        input_artifacts = summary_gate.get("input_artifacts")
        if not isinstance(input_artifacts, Mapping):
            failures.append("cleaned sensitivity summary input bindings are missing")
        else:
            summary_bindings = {
                "runs": ("cleaned_runs", "rows", FINAL_CLEANED_PREDICTION_COUNT),
                "audit": (
                    "cleaned_raw_audit",
                    "payloads",
                    len(FINAL_CLEANED_TASKS) * len(FINAL_CLEANED_SEEDS),
                ),
                "manifest": ("cleaned_manifest", None, None),
                "config": ("sensitivity_config", None, None),
            }
            for binding_name, (snapshot_name, row_key, expected_rows) in summary_bindings.items():
                if snapshot_name not in snapshots:
                    continue
                binding = input_artifacts.get(binding_name)
                _validate_snapshot_binding(
                    binding,
                    snapshots[snapshot_name],
                    f"cleaned summary {binding_name}",
                    failures,
                    row_count=expected_rows,
                    row_key=row_key or "rows",
                )

    cpu_gate: dict[str, Any] = {}
    if "cpu_defense_gate" in snapshots:
        try:
            cpu_gate = _json_object(snapshots["cpu_defense_gate"], "CPU defense gate")
        except ValueError as exc:
            failures.append(str(exc))
    if cpu_gate:
        if (
            cpu_gate.get("status") != "GO"
            or cpu_gate.get("artifact_gate") != "PASS"
            or cpu_gate.get("negative_control_signal") != "PASS"
        ):
            failures.append("CPU defense gate is not GO/PASS/PASS")
        coverage = cpu_gate.get("coverage")
        if not isinstance(coverage, Mapping):
            failures.append("CPU defense coverage is missing")
        else:
            for key, expected in (
                ("y_summary_rows", 22),
                ("y_score_rows", 220),
                ("shap_rows", 160),
                ("shap_mapping_rows", 80),
                ("mapped_substructures", 80),
            ):
                _expect_int(coverage, key, expected, "CPU defense coverage", failures)
        y_random = cpu_gate.get("y_randomization")
        if not isinstance(y_random, Mapping):
            failures.append("CPU defense y-randomization evidence is missing")
        else:
            _expect_int(
                y_random,
                "positive_margin_count",
                22,
                "CPU defense y-randomization",
                failures,
            )
        cpu_inputs = cpu_gate.get("inputs")
        if not isinstance(cpu_inputs, Mapping):
            failures.append("CPU defense input bindings are missing")
        else:
            cpu_binding_names = {
                "y_randomization_summary": "y_random_summary",
                "y_randomization_scores": "y_random_scores",
                "shap": "shap_selected",
                "shap_mapping": "shap_mapping",
                "traditional_selection": "traditional_selection",
            }
            cpu_expected_rows = {
                "y_random_summary": 22,
                "y_random_scores": 220,
                "shap_selected": 160,
                "shap_mapping": 80,
                "traditional_selection": 22,
            }
            for binding_name, snapshot_name in cpu_binding_names.items():
                if snapshot_name in snapshots:
                    try:
                        _, rows = _csv_rows(
                            snapshots[snapshot_name], f"CPU defense {binding_name}"
                        )
                        if len(rows) != cpu_expected_rows[snapshot_name]:
                            failures.append(
                                f"CPU defense {binding_name} has {len(rows)} rows, "
                                f"expected {cpu_expected_rows[snapshot_name]}"
                            )
                    except ValueError as exc:
                        failures.append(str(exc))
                    _validate_snapshot_binding(
                        cpu_inputs.get(binding_name),
                        snapshots[snapshot_name],
                        f"CPU defense {binding_name}",
                        failures,
                    )

    ablation_dir = tables / "posthoc_ablations_final"
    ablation_names = (
        "ablation_gate.json",
        "calibration_binning_sensitivity.csv",
        "ensemble_size_by_seed_subset.csv",
        "ensemble_size_summary.csv",
        "input_prediction_audit.csv",
        "sample_alignment_audit.csv",
    )
    ablation_snapshots: dict[str, FileSnapshot] = {}
    for name in ablation_names:
        snapshot = catalog.require(
            ablation_dir / name, "posthoc_ablation_evidence", failures
        )
        if snapshot is not None:
            ablation_snapshots[name] = snapshot
    if "ablation_gate.json" in ablation_snapshots:
        try:
            ablation_gate = _json_object(
                ablation_snapshots["ablation_gate.json"], "posthoc ablation gate"
            )
            if ablation_gate.get("status") != "PASS":
                failures.append("posthoc ablation gate status is not PASS")
            for key in ("all_task_seed_model_samples_aligned", "zero_model_fits"):
                _expect_true(ablation_gate, key, "posthoc ablation gate", failures)
            for key, expected in (
                ("expected_runs", 550),
                ("observed_runs", 550),
                ("ensemble_rows", 2310),
                ("calibration_rows", 7800),
            ):
                _expect_int(ablation_gate, key, expected, "posthoc ablation gate", failures)
        except ValueError as exc:
            failures.append(str(exc))

    figure_source_paths = (
        root
        / "results"
        / "figures"
        / "reliability_regret_final"
        / "figure_manifest.json",
        root
        / "results"
        / "figures"
        / "reliability_regret_final"
        / "figure_source_statistics.csv",
        root
        / "results"
        / "figures"
        / "reliability_regret_final"
        / "figure_dimension_statistics.csv",
        root
        / "results"
        / "figures"
        / "trustworthiness_submission_final"
        / "figure_manifest.json",
        root
        / "results"
        / "figures"
        / "trustworthiness_submission_final"
        / "fig02_source_data.csv",
        root
        / "results"
        / "figures"
        / "trustworthiness_submission_final"
        / "fig03_source_data.csv",
        root
        / "results"
        / "figures"
        / "trustworthiness_submission_final"
        / "figure_source_statistics.csv",
    )
    figure_snapshots: dict[str, FileSnapshot] = {}
    for path in figure_source_paths:
        snapshot = catalog.require(path, "final_figure_source", failures)
        if snapshot is not None:
            figure_snapshots[path.parent.name + "__" + path.name] = snapshot

    figure_output_failures: list[str] = []
    missing_final_figure_names = [
        Path(relative).name
        for logical_name, (relative, _) in FINAL_MANUSCRIPT_FIGURES.items()
        if logical_name not in snapshots
    ]
    if missing_final_figure_names:
        figure_output_failures.append(
            "final manuscript figures are missing: "
            + ",".join(missing_final_figure_names)
        )
    reliability_manifest_snapshot = figure_snapshots.get(
        "reliability_regret_final__figure_manifest.json"
    )
    if reliability_manifest_snapshot is None:
        figure_output_failures.append(
            "final reliability-regret figure manifest is missing"
        )
    else:
        try:
            reliability_manifest = _json_object(
                reliability_manifest_snapshot,
                "final reliability-regret figure manifest",
            )
            reliability_outputs = reliability_manifest.get("outputs")
            if not isinstance(reliability_outputs, Mapping):
                raise ValueError(
                    "final reliability-regret figure manifest outputs are missing"
                )
            for logical_name, (relative, manifest_group) in (
                FINAL_MANUSCRIPT_FIGURES.items()
            ):
                if manifest_group != "reliability_regret_final":
                    continue
                snapshot = snapshots.get(logical_name)
                if snapshot is None:
                    continue
                filename = Path(relative).name
                binding = reliability_outputs.get(filename)
                if not isinstance(binding, Mapping):
                    raise ValueError(
                        f"final figure manifest omits output binding for {filename}"
                    )
                if binding.get("sha256") != snapshot.sha256:
                    raise ValueError(f"final figure {filename} hash is stale")
                if (
                    _strict_int(
                        binding.get("bytes"),
                        f"final figure {filename} bytes",
                        minimum=1,
                    )
                    != len(snapshot.data)
                ):
                    raise ValueError(f"final figure {filename} byte count is stale")
        except ValueError as exc:
            figure_output_failures.append(str(exc))

    trust_manifest_snapshot = figure_snapshots.get(
        "trustworthiness_submission_final__figure_manifest.json"
    )
    if trust_manifest_snapshot is None:
        figure_output_failures.append(
            "final trustworthiness figure manifest is missing"
        )
    else:
        try:
            trust_manifest = _json_object(
                trust_manifest_snapshot,
                "final trustworthiness figure manifest",
            )
            trust_output_hashes = trust_manifest.get("output_sha256")
            if not isinstance(trust_output_hashes, Mapping):
                raise ValueError(
                    "final trustworthiness figure manifest output hashes are missing"
                )
            required_names = {
                Path(relative).name
                for relative, manifest_group in FINAL_MANUSCRIPT_FIGURES.values()
                if manifest_group == "trustworthiness_submission_final"
            }
            hashes_by_name: dict[str, object] = {}
            for raw_path, digest in trust_output_hashes.items():
                filename = Path(str(raw_path).replace("\\", "/")).name
                if filename not in required_names:
                    continue
                if filename in hashes_by_name:
                    raise ValueError(
                        "final trustworthiness figure manifest duplicates output "
                        f"binding for {filename}"
                    )
                hashes_by_name[filename] = digest
            for logical_name, (relative, manifest_group) in (
                FINAL_MANUSCRIPT_FIGURES.items()
            ):
                if manifest_group != "trustworthiness_submission_final":
                    continue
                snapshot = snapshots.get(logical_name)
                if snapshot is None:
                    continue
                filename = Path(relative).name
                if hashes_by_name.get(filename) != snapshot.sha256:
                    raise ValueError(f"final figure {filename} hash is stale")
        except ValueError as exc:
            figure_output_failures.append(str(exc))
    failures.extend(figure_output_failures)

    manuscript_dir = tables / "manuscript_final"
    manuscript_snapshots: dict[str, FileSnapshot] = {}
    if not manuscript_dir.is_dir():
        failures.append("final manuscript table inventory directory is missing")
    else:
        for path in sorted(manuscript_dir.iterdir()):
            if not path.is_file() or _is_appledouble(path):
                continue
            snapshot = catalog.require(path, "manuscript_final_inventory", failures)
            if snapshot is not None:
                manuscript_snapshots[path.name] = snapshot
                if path.name == "table_supp_shap_bit_substructures.csv":
                    catalog.require(
                        path,
                        "raw_structure_and_label_hash_index",
                        failures,
                        package_action="exclude_raw_structure_and_labels",
                    )
                if str(root).encode("utf-8") in snapshot.data:
                    catalog.require(
                        path,
                        "manuscript_nonportable_hash_index",
                        failures,
                        package_action="exclude_nonportable",
                    )
                    try:
                        if path.suffix == ".json":
                            sanitized_payload = (
                                json.dumps(
                                    _sanitize_public_metadata_tree(
                                        _json_object(
                                            snapshot,
                                            f"nonportable manuscript metadata {path.name}",
                                        ),
                                        root,
                                    ),
                                    indent=2,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            ).encode("utf-8")
                        elif path.suffix == ".csv":
                            columns, rows = _csv_rows(
                                snapshot, f"nonportable manuscript CSV {path.name}"
                            )
                            prefix = str(root) + os.sep
                            sanitized_payload = _serialize_public_csv(
                                columns,
                                [
                                    {
                                        column: (
                                            value.replace(prefix, "")
                                            if isinstance(value, str)
                                            else value
                                        )
                                        for column, value in row.items()
                                    }
                                    for row in rows
                                ],
                            )
                        else:
                            raise ValueError(
                                f"cannot sanitize nonportable manuscript file {path.name}"
                            )
                        public_extra_payloads.append(
                            (f"manuscript_{path.name}", sanitized_payload)
                        )
                    except ValueError as exc:
                        failures.append(str(exc))
    required_manuscript_files = {"table_inventory.csv", "source_audit.csv", "source_audit.json"}
    if not required_manuscript_files.issubset(manuscript_snapshots):
        failures.append("manuscript final inventory is missing required audit files")
    if "table_inventory.csv" in manuscript_snapshots:
        try:
            inventory_columns, inventory_rows = _csv_rows(
                manuscript_snapshots["table_inventory.csv"],
                "manuscript final table inventory",
            )
            required_inventory_columns = {"filename", "rows", "columns", "sha256"}
            if not required_inventory_columns.issubset(inventory_columns):
                raise ValueError("manuscript final table inventory schema is incomplete")
            managed_crossfit_names = {
                Path(relative).name
                for relative, _ in FINAL_CROSSFIT_DERIVED_OUTPUTS.values()
            }
            expected_table_names = {
                name
                for name in manuscript_snapshots
                if name.startswith("table_")
                and name != "table_inventory.csv"
                and name not in managed_crossfit_names
            }
            observed_table_names: set[str] = set()
            seen_inventory_names: set[str] = set()
            for row_number, row in enumerate(inventory_rows, start=2):
                filename = str(row.get("filename", ""))
                if Path(filename).name != filename or not filename.startswith("table_"):
                    raise ValueError(
                        f"manuscript inventory line {row_number} has unsafe filename"
                    )
                if filename in seen_inventory_names:
                    raise ValueError(
                        f"manuscript inventory line {row_number} duplicates {filename}"
                    )
                seen_inventory_names.add(filename)
                if filename in managed_crossfit_names:
                    if filename not in manuscript_snapshots:
                        raise ValueError(
                            f"manuscript inventory line {row_number} references missing {filename}"
                        )
                    continue
                observed_table_names.add(filename)
                snapshot = manuscript_snapshots.get(filename)
                if snapshot is None:
                    raise ValueError(
                        f"manuscript inventory line {row_number} references missing {filename}"
                    )
                actual_rows, actual_columns = _manuscript_table_shape(
                    snapshot, f"manuscript table {filename}"
                )
                if row.get("sha256") != snapshot.sha256:
                    raise ValueError(f"manuscript table {filename} hash is stale")
                if _strict_int(row.get("rows"), f"{filename} rows", minimum=0) != actual_rows:
                    raise ValueError(f"manuscript table {filename} row count is stale")
                if _strict_int(row.get("columns"), f"{filename} columns", minimum=1) != actual_columns:
                    raise ValueError(f"manuscript table {filename} column count is stale")
            if observed_table_names != expected_table_names or not observed_table_names:
                raise ValueError("manuscript final table inventory/file set is not exact")
        except ValueError as exc:
            failures.append(str(exc))
    if "source_audit.json" in manuscript_snapshots:
        try:
            source_audit = _json_object(
                manuscript_snapshots["source_audit.json"], "manuscript source audit"
            )
            if (
                source_audit.get("status") != "PASS"
                or source_audit.get("mode") != "final_locked"
                or source_audit.get("protocol_scope") != "final_locked_five_model"
            ):
                failures.append("manuscript source audit is not final_locked PASS")
            statuses = source_audit.get("scientific_gate_statuses")
            if not isinstance(statuses, Mapping):
                failures.append("manuscript source audit scientific statuses are missing")
            else:
                expected_statuses = {
                    "ablation": "PASS",
                    "cpu_defense": "GO",
                    "cleaned_prediction_exports": "GO",
                }
                for key, expected in expected_statuses.items():
                    if statuses.get(key) != expected:
                        failures.append(
                            f"manuscript source audit {key} status is not {expected}"
                        )
                if summary_gate and statuses.get("cleaned_sensitivity") != summary_gate.get("status"):
                    failures.append("manuscript cleaned sensitivity status is stale")
        except ValueError as exc:
            failures.append(str(exc))

    reference_key_count = 0
    paper_names = (
        "paper_main_tex",
        "paper_supplementary_tex",
        "paper_references",
        "paper_reference_audit_final",
        "paper_numeric_audit",
        "paper_supplementary_numeric_audit",
        "paper_main_pdf",
        "paper_supplementary_pdf",
    )
    if all(name in snapshots for name in paper_names):
        try:
            references_text = snapshots["paper_references"].data.decode("utf-8")
            main_text = snapshots["paper_main_tex"].data.decode("utf-8")
            supplementary_text = snapshots["paper_supplementary_tex"].data.decode(
                "utf-8"
            )
            reference_audit_text = snapshots[
                "paper_reference_audit_final"
            ].data.decode("utf-8")
            numeric_audit_text = snapshots["paper_numeric_audit"].data.decode("utf-8")
            supplementary_numeric_text = snapshots[
                "paper_supplementary_numeric_audit"
            ].data.decode("utf-8")
            bib_keys = re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", references_text)
            unique_bib_keys = set(bib_keys)
            reference_key_count = len(unique_bib_keys)
            if not bib_keys:
                raise ValueError("final bibliography contains no entries")
            if len(unique_bib_keys) != len(bib_keys):
                raise ValueError(
                    "final bibliography contains duplicate BibTeX keys: "
                    f"{len(bib_keys) - len(unique_bib_keys)}"
                )
            cited: set[str] = set()
            for group in re.findall(r"\\cite[A-Za-z*]*\s*\{([^}]+)\}", main_text):
                cited.update(value.strip() for value in group.split(",") if value.strip())
            if cited != unique_bib_keys:
                raise ValueError(
                    "main manuscript citation coverage is "
                    f"{len(cited & unique_bib_keys)}/{len(unique_bib_keys)}; "
                    f"missing={len(unique_bib_keys - cited)}, "
                    f"unexpected={len(cited - unique_bib_keys)}"
                )
            missing_from_audit = [key for key in bib_keys if key not in reference_audit_text]
            if missing_from_audit:
                raise ValueError(
                    f"final reference audit omits {len(missing_from_audit)} bibliography keys"
                )
            forbidden_markers = (
                "CLEANED_SENSITIVITY_PENDING",
                "REFERENCE_AUDIT_PENDING",
            )
            source_documents = {
                "main.tex": main_text,
                "supplementary.tex": supplementary_text,
                "numeric audit": numeric_audit_text,
                "supplementary numeric audit": supplementary_numeric_text,
            }
            for label, text_value in source_documents.items():
                present = [marker for marker in forbidden_markers if marker in text_value]
                if present:
                    raise ValueError(f"{label} retains final-release placeholders: {present}")
            if REFERENCE_AUDIT_SELF_SUPERSEDED_RE.search(reference_audit_text):
                raise ValueError("final canonical reference audit is marked superseded")
        except (UnicodeDecodeError, ValueError) as exc:
            failures.append(str(exc))

    public_metadata_payloads: list[tuple[str, bytes]] = list(public_extra_payloads)
    if manifest:
        try:
            public_manifest = _sanitize_public_metadata_tree(
                json.loads(json.dumps(manifest)), root
            )
            if not str(public_manifest.get("config_logical_path", "")).strip():
                failures.append(
                    "cleaned manifest lacks config_logical_path for public metadata"
                )
            public_metadata_payloads.append(
                (
                    "cleaned_sensitivity_manifest.json",
                    (
                        json.dumps(
                            public_manifest,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))
    if "cleaned_upgrade_receipt" in snapshots:
        try:
            receipt = _json_object(
                snapshots["cleaned_upgrade_receipt"], "cleaned provenance upgrade receipt"
            )
            public_receipt = _sanitize_public_metadata_tree(
                json.loads(json.dumps(receipt)), root
            )
            public_metadata_payloads.append(
                (
                    "cleaned_sensitivity_provenance_upgrade_receipt.json",
                    (
                        json.dumps(
                            public_receipt,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    generic_json_metadata: list[tuple[str, FileSnapshot]] = []
    for snapshot_name, public_name in (
        ("cleaned_summary_gate", "cleaned_sensitivity_gate.json"),
        ("cleaned_prediction_gate", "cleaned_prediction_gate.json"),
        ("cpu_defense_gate", "cpu_defense_gate.json"),
    ):
        snapshot = snapshots.get(snapshot_name)
        if snapshot is not None and str(root).encode("utf-8") in snapshot.data:
            generic_json_metadata.append((public_name, snapshot))
    for key, snapshot in figure_snapshots.items():
        if snapshot.path.suffix == ".json" and str(root).encode("utf-8") in snapshot.data:
            generic_json_metadata.append((key.replace("__", "_"), snapshot))
    for public_name, snapshot in generic_json_metadata:
        catalog.require(
            snapshot.path,
            "nonportable_json_hash_index",
            failures,
            package_action="exclude_nonportable",
        )
        try:
            payload = _sanitize_public_metadata_tree(
                _json_object(snapshot, f"nonportable metadata {snapshot.relative_path}"),
                root,
            )
            public_metadata_payloads.append(
                (
                    public_name,
                    (
                        json.dumps(
                            payload,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            )
        except ValueError as exc:
            failures.append(str(exc))

    unique_failures = tuple(dict.fromkeys(failures))
    files = catalog.files()
    details = {
        "schema_version": 1,
        "status": "PASS" if not unique_failures else "HOLD",
        "not_primary_model_records": True,
        "primary_submission_record_count_unchanged": True,
        "cleaned_prediction_artifacts_available": (
            len(cleaned_predictions) == FINAL_CLEANED_PREDICTION_COUNT
        ),
        "cleaned_prediction_files_expected": FINAL_CLEANED_PREDICTION_COUNT,
        "cleaned_prediction_files_rehashed": len(cleaned_predictions),
        "cleaned_prediction_inventory_sha256": prediction_inventory_sha256,
        "raw_structure_and_label_exports_directly_releasable": False,
        "raw_structure_and_label_package_action": "exclude_raw_structure_and_labels",
        "final_reference_entries": reference_key_count,
        "final_manuscript_figures": {
            "status": "PASS" if not figure_output_failures else "HOLD",
            "expected_files": len(FINAL_MANUSCRIPT_FIGURES),
            "observed_files": sum(
                logical_name in snapshots
                for logical_name in FINAL_MANUSCRIPT_FIGURES
            ),
            "manifest_hashes_match": not figure_output_failures,
            "files": [
                {
                    "path": snapshots[logical_name].relative_path,
                    "sha256": snapshots[logical_name].sha256,
                    "bytes": len(snapshots[logical_name].data),
                }
                for logical_name in FINAL_MANUSCRIPT_FIGURES
                if logical_name in snapshots
            ],
        },
        "crossfit_oracle_sensitivity": crossfit_details,
        "crossfit_influence_sensitivity": crossfit_influence_details,
        "reviewer_defense_sensitivities": reviewer_defense_details,
        "superseded_reference_audits_excluded": [
            "paper/reference_authenticity_audit_20260714.md",
            "paper/reference_audit.md",
        ],
        "supplement_file_count": len(files),
        "supplement_inventory_sha256": stable_json_hash(
            [
                {
                    "path": item.info.path,
                    "sha256": item.info.sha256,
                    "bytes": item.info.bytes,
                    "role": item.role,
                    "package_action": item.package_action,
                }
                for item in files
            ]
        ),
        "failures": list(unique_failures),
    }
    return FinalSupplementAudit(
        complete=not unique_failures,
        failures=unique_failures,
        files=files,
        details=details,
        cleaned_predictions=tuple(cleaned_predictions),
        cleaned_source_data_sha256=cleaned_source_data_sha256,
        public_metadata_payloads=tuple(public_metadata_payloads),
    )


def _serialize_public_csv(
    columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _public_bundle_readme_text() -> str:
    """Describe the public bundle without overstating metric replayability."""

    return """# Public reproducibility bundle

This allowlisted bundle contains the active training and locked post-processing
code, environment specifications, sanitized run configurations, deterministic
prediction views, seed/hash metadata, final tables and figures, and manuscript
artifacts selected by the release builder. It does not contain the raw TDC
datasets, source structures, ground-truth targets, source row indices, model
checkpoints, credentials, private machine paths, or historical operator guides.

## Prediction-view boundary

The files under `public/predictions/` intentionally omit `y_true`, `label`,
SMILES strings, canonical structures, and source sample indices. Opaque sample
identifiers preserve deterministic row matching without revealing those source
fields. Consequently, the public prediction views alone cannot recompute
label-dependent metrics, calibration quantities, regret values, or the complete
figures and tables.

Exact metric replay requires the user to obtain the matching TDC ADMET source
data under its original terms, verify the recorded source and split hashes in
`public/metadata/run_seeds_and_hashes.csv`, and locally reconstruct the internal
prediction schema expected by the analysis programs. No TDC source data are
redistributed here. The `scripts/run_final_posthoc_locked.sh` entrypoint consumes
that internal schema and locally obtained source data; it is not a direct runner
for the de-identified public prediction views.

## What can be audited directly

The public configurations, seeds, hashes, prediction values, source-statistic
tables, figures, and code allow independent inspection of the released claims
and a clean-room retraining attempt. Exact numerical retraining can still vary
with hardware and dependency behavior; use the recorded configurations and
environment specifications and report any deviation.

The cross-fitted oracle-sensitivity fold table, endpoint summary, dimension
summary, locked scaffold-overlap audit, fixed selections, and source hash
manifest are included as scientific evidence. The fold table is kept out of the
manuscript PDF because of its size, but it is included in the allowlisted bundle
so the endpoint and dimension summaries can be recomputed without model fitting.

## Licensing and citation

Source code is licensed under Apache-2.0 in the repository-root `LICENSE`.
De-identified prediction views, generated metrics and tables, and original
result figures are licensed under CC BY 4.0 as described in
`LICENSE-DATA.md`. Raw TDC data and model checkpoints are not released.
Citation metadata are provided in `CITATION.cff` and `.zenodo.json`; the DOI
must not be asserted until the public Zenodo record resolves.

This directory is generated. Assemble an upload only from rows marked `include`
in the release package plan, and omit AppleDouble `._*` sidecars. Do not publish
the repository or release directory wholesale.
"""


def _opaque_sample_id(
    schema: str, task: str, split: str, ordinal: int, source_sha256: str
) -> str:
    material = f"{schema}|{task}|{split}|{ordinal}|{source_sha256}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _audit_public_payload_map(
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Scan every logical public-tree payload for path and secret leakage."""

    failures: list[str] = []
    absolute_path_matches = 0
    secret_assignment_matches = 0
    raw_structure_payload_key_matches = 0
    utf8_decode_failures = 0
    for relative, data in sorted(payloads.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"public subtree has unsafe logical filename: {relative}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            utf8_decode_failures += 1
            failures.append(f"public subtree payload is not UTF-8: {relative}")
            continue
        path_count = len(PUBLIC_POSIX_ABSOLUTE_PATH_RE.findall(text)) + len(
            PUBLIC_WINDOWS_ABSOLUTE_PATH_RE.findall(text)
        )
        secret_count = len(PUBLIC_SECRET_ASSIGNMENT_RE.findall(text))
        raw_payload_count = len(PUBLIC_RAW_PAYLOAD_JSON_KEY_RE.findall(text))
        if relative_path.suffix.lower() == ".csv":
            try:
                header = next(csv.reader(io.StringIO(text, newline="")), [])
            except csv.Error:
                header = []
            raw_payload_count += sum(
                str(column).strip().lower().replace("-", "_")
                in PUBLIC_CONFIG_RAW_PAYLOAD_KEYS
                for column in header
            )
        absolute_path_matches += path_count
        secret_assignment_matches += secret_count
        raw_structure_payload_key_matches += raw_payload_count
        if path_count:
            failures.append(
                f"public subtree payload contains {path_count} absolute path token(s): {relative}"
            )
        if secret_count:
            failures.append(
                f"public subtree payload contains {secret_count} secret assignment token(s): {relative}"
            )
        if raw_payload_count:
            failures.append(
                "public subtree payload contains "
                f"{raw_payload_count} raw structure/label/index key(s): {relative}"
            )
    unique_failures = tuple(dict.fromkeys(failures))
    details = {
        "schema_version": 1,
        "status": "PASS" if not unique_failures else "HOLD",
        "logical_files_scanned": len(payloads),
        "absolute_path_matches": absolute_path_matches,
        "secret_assignment_matches": secret_assignment_matches,
        "raw_structure_payload_key_matches": raw_structure_payload_key_matches,
        "utf8_decode_failures": utf8_decode_failures,
        "appledouble_sidecars_excluded_from_logical_tree": True,
        "failures": list(unique_failures),
    }
    return details, unique_failures


def _remove_public_tree(path: Path) -> None:
    """Remove a generated tree while tolerating disappearing AppleDouble peers."""

    if not path.exists():
        return

    def missing_ok(_: Any, __: str, error_info: tuple[type[BaseException], BaseException, Any]) -> None:
        error = error_info[1]
        if isinstance(error, FileNotFoundError):
            return
        raise error

    try:
        shutil.rmtree(path, onerror=missing_ok)
    except FileNotFoundError:
        pass
    if path.exists():
        raise OSError(f"generated public tree cleanup is incomplete: {path.name}")


def _commit_public_payloads(
    *, root: Path, output_dir: Path, payloads: Mapping[str, bytes]
) -> tuple[SupplementalFile, ...]:
    """Atomically replace the deterministic de-identified public-view tree."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "final-supplement public views require output_dir inside project root"
        ) from exc
    public_dir = output_dir / "public"
    staging = output_dir / f".public.tmp.{os.getpid()}"
    backup = output_dir / f".public.backup.{os.getpid()}"
    if staging.exists():
        _remove_public_tree(staging)
    if backup.exists():
        _remove_public_tree(backup)
    staging.mkdir(parents=True)
    try:
        _, scan_failures = _audit_public_payload_map(payloads)
        if scan_failures:
            raise ValueError(scan_failures[0])
        for relative, data in sorted(payloads.items()):
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe public-view path: {relative}")
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        actual_payloads = {
            path.relative_to(staging).as_posix(): path.read_bytes()
            for path in staging.rglob("*")
            if path.is_file() and not _is_appledouble(path)
        }
        if set(actual_payloads) != set(payloads):
            raise ValueError("staged public subtree file inventory is not exact")
        for relative, expected in payloads.items():
            if actual_payloads[relative] != expected:
                raise ValueError(f"staged public subtree bytes changed: {relative}")
        _, staged_scan_failures = _audit_public_payload_map(actual_payloads)
        if staged_scan_failures:
            raise ValueError(staged_scan_failures[0])
        if public_dir.exists():
            public_dir.rename(backup)
        try:
            staging.rename(public_dir)
        except Exception:
            if backup.exists() and not public_dir.exists():
                backup.rename(public_dir)
            raise
        if backup.exists():
            _remove_public_tree(backup)
    finally:
        if staging.exists():
            _remove_public_tree(staging)

    files: list[SupplementalFile] = []
    for relative, data in sorted(payloads.items()):
        path = public_dir / relative
        project_relative = path.resolve().relative_to(root).as_posix()
        files.append(
            SupplementalFile(
                FileInfo(
                    path=project_relative,
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                    exists=True,
                ),
                role=(
                    "public_bundle_readme"
                    if relative == PUBLIC_BUNDLE_README_NAME
                    else "public_seed_and_hash_ledger"
                    if relative == f"metadata/{PUBLIC_SEED_LEDGER_NAME}"
                    else "public_prediction_view"
                    if relative.startswith("predictions/")
                    else "public_sanitized_config"
                    if relative.startswith("configs/")
                    else "public_sanitized_metadata"
                ),
                package_action="include",
            )
        )
    return tuple(files)


def build_public_prediction_views(
    *,
    root: Path,
    output_dir: Path,
    records: Sequence[dict[str, Any]],
    supplements: FinalSupplementAudit,
) -> PublicPredictionViewAudit:
    """Generate structure/label-free formal and cleaned prediction views."""

    failures: list[str] = []
    public_readme_payload = _public_bundle_readme_text().encode("utf-8")
    payloads: dict[str, bytes] = {
        PUBLIC_BUNDLE_README_NAME: public_readme_payload,
    }
    inventory: list[dict[str, Any]] = []
    config_source_cache: dict[str, bytes] = {}
    formal_records = [record for record in records if record.get("included_default")]
    formal_keys: set[tuple[str, str, int]] = set()
    for record in formal_records:
        model = str(record.get("model", ""))
        task = str(record.get("task", ""))
        try:
            seed = _strict_int(record.get("seed"), "formal public-view seed", minimum=1)
            key = (model, task, seed)
            if key in formal_keys:
                raise ValueError(f"duplicate formal public-view key: {key}")
            formal_keys.add(key)
            if record.get("run_status") != "complete":
                raise ValueError(f"formal public-view source is incomplete: {key}")
            raw_relative = str(record.get("prediction_path", ""))
            raw_path = (root / raw_relative).resolve()
            try:
                raw_path.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"formal prediction path escapes project root: {key}") from exc
            raw = raw_path.read_bytes()
            raw_sha = hashlib.sha256(raw).hexdigest()
            if raw_sha != record.get("prediction_sha256"):
                raise ValueError(f"formal prediction changed after audit: {key}")
            if len(raw) != _strict_int(
                record.get("prediction_bytes"), f"formal prediction bytes {key}", minimum=1
            ):
                raise ValueError(f"formal prediction byte count changed after audit: {key}")
            snapshot = FileSnapshot(
                raw_path,
                raw_relative,
                raw,
                raw_sha,
            )
            columns, raw_rows = _csv_rows(snapshot, f"formal prediction {key}")
            prediction_column = "y_pred" if model in TRADITIONAL_MODELS else "prediction"
            if prediction_column not in columns or "split" not in columns:
                raise ValueError(f"formal prediction schema cannot produce public view: {key}")
            source_sha = str(record.get("source_data_sha256", ""))
            if not _valid_sha256(source_sha):
                raise ValueError(f"formal prediction lacks source-data hash: {key}")
            split_ordinals: Counter[str] = Counter()
            public_rows: list[dict[str, Any]] = []
            for line_number, row in enumerate(raw_rows, start=2):
                split = str(row.get("split", "")).strip().lower()
                if split not in {"valid", "test"}:
                    raise ValueError(
                        f"formal prediction {key} line {line_number} has invalid split"
                    )
                prediction = row.get(prediction_column)
                if not _finite_number(prediction):
                    raise ValueError(
                        f"formal prediction {key} line {line_number} is non-finite"
                    )
                numeric_prediction = float(prediction)
                if (
                    record.get("task_type") == "classification"
                    and not 0.0 <= numeric_prediction <= 1.0
                ):
                    raise ValueError(
                        f"formal prediction {key} line {line_number} is outside [0,1]"
                    )
                ordinal = split_ordinals[split]
                split_ordinals[split] += 1
                public_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "seed": seed,
                        "split": split,
                        "opaque_sample_id": _opaque_sample_id(
                            "admet-public-prediction-v1",
                            task,
                            split,
                            ordinal,
                            source_sha,
                        ),
                        "prediction": format(numeric_prediction, ".17g"),
                    }
                )
            if len(public_rows) != _strict_int(
                record.get("prediction_rows"), f"formal prediction rows {key}", minimum=1
            ):
                raise ValueError(f"formal prediction public-view row count mismatch: {key}")
            filename = f"{task.lower()}_{model}_seed{seed}.csv"
            relative = f"predictions/formal/{filename}"
            payload = _serialize_public_csv(
                ("task", "model", "seed", "split", "opaque_sample_id", "prediction"),
                public_rows,
            )
            payloads[relative] = payload
            inventory.append(
                {
                    "path": f"public/{relative}",
                    "kind": "formal",
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "rows": len(public_rows),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )

            config_relative = str(record.get("config_path", ""))
            if not config_relative:
                raise ValueError(f"formal run lacks a config source: {key}")
            config_raw = config_source_cache.get(config_relative)
            if config_raw is None:
                config_path = (root / config_relative).resolve()
                try:
                    config_path.relative_to(root.resolve())
                except ValueError as exc:
                    raise ValueError(f"formal config path escapes project root: {key}") from exc
                config_raw = config_path.read_bytes()
                config_source_cache[config_relative] = config_raw
            config_sha = hashlib.sha256(config_raw).hexdigest()
            if config_sha != record.get("config_sha256"):
                raise ValueError(f"formal config changed after audit: {key}")
            if len(config_raw) != _strict_int(
                record.get("config_bytes"), f"formal config bytes {key}", minimum=1
            ):
                raise ValueError(f"formal config byte count changed after audit: {key}")
            try:
                if Path(config_relative).suffix.lower() == ".json":
                    config_payload = json.loads(config_raw.decode("utf-8"))
                else:
                    config_payload = yaml.safe_load(config_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
                raise ValueError(f"formal config is not parseable for {key}: {exc}") from exc
            if not isinstance(config_payload, dict):
                raise ValueError(f"formal config root is not a mapping for {key}")
            sanitized_config = _sanitize_public_config(config_payload, root)
            if not isinstance(sanitized_config, dict):
                raise ValueError(f"formal config cannot be sanitized for {key}")
            public_config = {
                "schema_version": 1,
                "task": task,
                "model": model,
                "seed": seed,
                "source_config_sha256": config_sha,
                "config": sanitized_config,
            }
            if not _public_config_is_clean(public_config):
                raise ValueError(f"formal sanitized config still contains unsafe content: {key}")
            config_payload_bytes = (
                json.dumps(
                    public_config,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            serialized_config = config_payload_bytes.decode("utf-8")
            if (
                EMBEDDED_ABSOLUTE_PATH_RE.search(serialized_config)
                or re.search(
                    r'(?i)"(?:command|commands|cmd|argv|password|passwd|token|secret|api_?key)"\s*:',
                    serialized_config,
                )
            ):
                raise ValueError(f"formal sanitized config scan failed: {key}")
            config_public_relative = f"configs/{task.lower()}_{model}_seed{seed}.json"
            payloads[config_public_relative] = config_payload_bytes
            inventory.append(
                {
                    "path": f"public/{config_public_relative}",
                    "kind": "config",
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "rows": 1,
                    "sha256": hashlib.sha256(config_payload_bytes).hexdigest(),
                    "bytes": len(config_payload_bytes),
                }
            )
        except (OSError, ValueError) as exc:
            failures.append(str(exc))

    seed_ledger_columns = (
        "task",
        "model",
        "seed",
        "task_type",
        "official_metric",
        "run_status",
        "split_sha256",
        "source_data_sha256",
        "execution_spec_sha256",
        "software_sha256",
        "config_sha256",
        "public_prediction_view",
    )
    seed_ledger_rows: list[dict[str, Any]] = []
    for record in sorted(
        formal_records,
        key=lambda value: (
            str(value.get("task", "")),
            str(value.get("model", "")),
            int(value.get("seed", 0)),
        ),
    ):
        task = str(record.get("task", ""))
        model = str(record.get("model", ""))
        try:
            seed = _strict_int(record.get("seed"), "public seed ledger seed", minimum=1)
            for field in (
                "split_sha256",
                "source_data_sha256",
                "execution_spec_sha256",
                "software_sha256",
                "config_sha256",
            ):
                if not _valid_sha256(str(record.get(field, ""))):
                    raise ValueError(
                        f"public seed ledger has invalid {field}: {(model, task, seed)}"
                    )
            seed_ledger_rows.append(
                {
                    "task": task,
                    "model": model,
                    "seed": seed,
                    "task_type": record.get("task_type", ""),
                    "official_metric": record.get("official_metric", ""),
                    "run_status": record.get("run_status", ""),
                    "split_sha256": record.get("split_sha256", ""),
                    "source_data_sha256": record.get("source_data_sha256", ""),
                    "execution_spec_sha256": record.get("execution_spec_sha256", ""),
                    "software_sha256": record.get("software_sha256", ""),
                    "config_sha256": record.get("config_sha256", ""),
                    "public_prediction_view": (
                        "public/predictions/formal/"
                        f"{task.lower()}_{model}_seed{seed}.csv"
                    ),
                }
            )
        except ValueError as exc:
            failures.append(str(exc))
    seed_ledger_payload = _serialize_public_csv(
        seed_ledger_columns, seed_ledger_rows
    )
    payloads[f"metadata/{PUBLIC_SEED_LEDGER_NAME}"] = seed_ledger_payload

    cleaned_keys: set[tuple[str, str, int]] = set()
    for source in supplements.cleaned_predictions:
        key = (source.task, source.model, source.seed)
        try:
            if key in cleaned_keys:
                raise ValueError(f"duplicate cleaned public-view key: {key}")
            cleaned_keys.add(key)
            if not _valid_sha256(supplements.cleaned_source_data_sha256):
                raise ValueError("cleaned public views lack a source-data aggregate hash")
            columns, raw_rows = _csv_rows(
                source.snapshot, f"cleaned public-view source {key}"
            )
            required = {
                "evaluation_set",
                "official_prediction",
                "cleaned_prediction",
            }
            if not required.issubset(columns):
                raise ValueError(f"cleaned prediction schema cannot produce public view: {key}")
            public_rows: list[dict[str, Any]] = []
            for ordinal, row in enumerate(raw_rows):
                evaluation_set = str(row.get("evaluation_set", "")).strip()
                if evaluation_set != "common_cleaned_test":
                    raise ValueError(f"cleaned prediction {key} evaluation set is not locked")
                official = row.get("official_prediction")
                cleaned = row.get("cleaned_prediction")
                if not _finite_number(official) or not _finite_number(cleaned):
                    raise ValueError(f"cleaned prediction {key} has non-finite values")
                public_rows.append(
                    {
                        "task": next(
                            task for task in FINAL_CLEANED_TASKS if task.lower() == source.task
                        ),
                        "model": source.model,
                        "seed": source.seed,
                        "evaluation_set": evaluation_set,
                        "opaque_sample_id": _opaque_sample_id(
                            "admet-cleaned-public-prediction-v1",
                            source.task,
                            evaluation_set,
                            ordinal,
                            supplements.cleaned_source_data_sha256,
                        ),
                        "official_prediction": format(float(official), ".17g"),
                        "cleaned_prediction": format(float(cleaned), ".17g"),
                    }
                )
            if len(public_rows) != source.rows:
                raise ValueError(f"cleaned public-view row count mismatch: {key}")
            filename = f"{source.task}_{source.model}_seed{source.seed}_paired_public.csv"
            relative = f"predictions/cleaned_sensitivity/{filename}"
            payload = _serialize_public_csv(
                (
                    "task",
                    "model",
                    "seed",
                    "evaluation_set",
                    "opaque_sample_id",
                    "official_prediction",
                    "cleaned_prediction",
                ),
                public_rows,
            )
            payloads[relative] = payload
            inventory.append(
                {
                    "path": f"public/{relative}",
                    "kind": "cleaned_sensitivity",
                    "task": source.task,
                    "model": source.model,
                    "seed": source.seed,
                    "rows": len(public_rows),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
        except (StopIteration, ValueError) as exc:
            failures.append(str(exc))

    expected_cleaned_keys = {
        (task.lower(), model, seed)
        for task in FINAL_CLEANED_TASKS
        for model in FINAL_CLEANED_MODELS
        for seed in FINAL_CLEANED_SEEDS
    }
    if cleaned_keys != expected_cleaned_keys:
        failures.append("cleaned public-view key set is not exactly 120 files")
    if len(formal_keys) != len(formal_records):
        failures.append("formal public-view key set is incomplete")

    metadata_payload_count = 0
    for name, payload in supplements.public_metadata_payloads:
        if Path(name).name != name:
            failures.append(f"unsafe public metadata filename: {name}")
        else:
            relative = f"metadata/{name}"
            if relative in payloads:
                failures.append(f"duplicate public metadata filename: {name}")
                continue
            try:
                payloads[relative] = _sanitize_public_metadata_payload(
                    name, payload, root
                )
                metadata_payload_count += 1
            except ValueError as exc:
                failures.append(str(exc))

    formal_count = sum(row["kind"] == "formal" for row in inventory)
    cleaned_count = sum(row["kind"] == "cleaned_sensitivity" for row in inventory)
    config_count = sum(row["kind"] == "config" for row in inventory)
    if config_count != len(formal_records):
        failures.append(f"sanitized config views {config_count}/{len(formal_records)}")
    if len(seed_ledger_rows) != len(formal_records):
        failures.append(
            f"public seed ledger rows {len(seed_ledger_rows)}/{len(formal_records)}"
        )
    public_subtree_scan, public_subtree_failures = _audit_public_payload_map(payloads)
    failures.extend(public_subtree_failures)
    unique_failures = tuple(dict.fromkeys(failures))
    files: tuple[SupplementalFile, ...] = ()
    if not unique_failures:
        try:
            files = _commit_public_payloads(
                root=root, output_dir=output_dir, payloads=payloads
            )
        except (OSError, ValueError) as exc:
            unique_failures = (str(exc),)
    details = {
        "schema_version": 1,
        "status": "PASS" if not unique_failures else "HOLD",
        "formal_views_expected": len(formal_records),
        "formal_views_generated": formal_count,
        "cleaned_views_expected": FINAL_CLEANED_PREDICTION_COUNT,
        "cleaned_views_generated": cleaned_count,
        "total_prediction_views_expected": len(formal_records)
        + FINAL_CLEANED_PREDICTION_COUNT,
        "total_prediction_views_generated": formal_count + cleaned_count,
        "sanitized_config_views_expected": len(formal_records),
        "sanitized_config_views_generated": config_count,
        "public_readme_path": f"public/{PUBLIC_BUNDLE_README_NAME}",
        "public_readme_present": PUBLIC_BUNDLE_README_NAME in payloads,
        "public_readme_sha256": hashlib.sha256(public_readme_payload).hexdigest(),
        "public_seed_ledger_path": f"public/metadata/{PUBLIC_SEED_LEDGER_NAME}",
        "public_seed_ledger_rows_expected": len(formal_records),
        "public_seed_ledger_rows_generated": len(seed_ledger_rows),
        "public_seed_ledger_sha256": hashlib.sha256(seed_ledger_payload).hexdigest(),
        "public_metadata_payloads_sanitized": metadata_payload_count,
        "public_subtree_scan": public_subtree_scan,
        "prediction_inventory_sha256": stable_json_hash(
            sorted(
                [row for row in inventory if row["kind"] != "config"],
                key=lambda row: row["path"],
            )
        ),
        "sanitized_config_inventory_sha256": stable_json_hash(
            sorted(
                [row for row in inventory if row["kind"] == "config"],
                key=lambda row: row["path"],
            )
        ),
        "raw_structure_and_labels_present_in_public_views": bool(
            public_subtree_scan["raw_structure_payload_key_matches"]
        ),
        "opaque_id_schema": (
            "sha256(schema|task|split_or_evaluation_set|row_ordinal|source_data_sha256)"
        ),
        "columns_forbidden": [
            "smiles",
            "canonical_smiles",
            "label",
            "y_true",
            "sample_index",
        ],
        "config_content_forbidden": [
            "absolute paths",
            "command/argv/cmd",
            "password/passwd/token/secret/api_key",
            "raw structure/label/source-index payload keys",
            "model_input_repairs row-level mapping",
        ],
        "failures": list(unique_failures),
    }
    return PublicPredictionViewAudit(
        complete=not unique_failures,
        failures=unique_failures,
        files=files,
        details=details,
    )


def _readme_text(
    gate: Mapping[str, Any], checkpoint_policy: Mapping[str, Any]
) -> str:
    counts = gate["complete_by_model"]
    expected = gate["expected_by_model"]
    count_lines = "\n".join(
        f"- `{model}`: {counts[model]}/{expected[model]} complete"
        for model in expected
    )
    checkpoint_excluded = (
        checkpoint_policy.get("package_action") == "exclude_by_release_policy"
    )
    checkpoint_plan_line = (
        "Checkpoint publication is explicitly disabled by the resolved release "
        "policy; formal checkpoint paths are marked `exclude_by_release_policy`."
        if checkpoint_excluded
        else "Checkpoints marked `publish_via_lfs_or_object_store` require a "
        "resolved checkpoint policy and later owner-managed upload."
    )
    package_plan_scope = (
        "small-file inclusion, policy-excluded checkpoints, and excluded legacy artifacts"
        if checkpoint_excluded
        else "small-file inclusion, excluded legacy artifacts, and checkpoint upload via LFS/object storage"
    )
    supplement_line = (
        "Final supplements are required for this build. Raw prediction exports are "
        "hash-indexed with `exclude_raw_structure_and_labels`; deterministic files "
        "under `public/predictions/` expose only opaque sample IDs and predictions."
        if gate.get("final_supplements_required")
        else "Final supplements are not required in this compatibility build."
    )
    return f"""# Reproducibility release index

Current gate: **{gate['status']}**.

This directory is a publishability index, not a copy of the experiment store.
It records repository-relative paths, SHA-256 checksums, byte sizes, task/model/
seed identities, prediction split/sample-identity fingerprints, label-alignment
diagnostics, software/config fingerprints,
and packaging actions. No raw TDC dataset or large checkpoint is copied here.

## Submission coverage

{count_lines}

When discovered, the historical artifact model id `dmpnn` is indexed as
`legacy_node_mpnn`. It is a node-message MPNN, not a directed bond-message
D-MPNN, and is excluded from the submission by default. Only
`chemprop_dmpnn` satisfies the D-MPNN requirement.

## Files

- `manifest.csv` / `manifest.json`: one record per expected run plus any
  discovered excluded legacy runs; paths, hashes, sizes, validation state, and
  packaging actions.
- `seeds.csv`: internal full seed and split-hash audit ledger.
- `package_plan.csv`: deduplicated paths and actions for {package_plan_scope}.
- `gate.json`: conservative machine-readable GO/HOLD decision.
- `release_policy.json`: user-controlled license decisions. The builder creates
  it once and never overwrites it.
- `LICENSE_AND_REDISTRIBUTION_TODO.md`: human-readable license/redistribution
  checklist derived from the policy.
- `public/README.md`: publication-safe scope and replay-boundary statement.
- `public/metadata/run_seeds_and_hashes.csv`: allowlisted public seed/hash
  ledger without structures, labels, or source indices.

{checkpoint_plan_line} This builder deliberately does not copy checkpoints.
Paths marked `include` are the small artifacts intended for the public bundle.
{supplement_line}

## Publication boundary

Do **not** upload this entire `release/` directory. The root-level
`manifest.json`, `gate.json`, and related ledgers are internal publishability
indexes and may retain nonredistributable audit detail. Assemble a fresh upload
staging directory from only `package_plan.csv` rows whose action is `include`;
`release/public/` is the generated de-identified prediction/config/metadata
subtree within that allowlist. Never upload rows marked `exclude_*` or
`publish_via_lfs_or_object_store` without the separately resolved policy. On
macOS, also exclude AppleDouble `._*` sidecars: do not archive
`release/public/` directly; construct the clean staging directory from the
allowlist.

## Rebuild and audit

From the repository root:

```bash
python experiments/build_reproducibility_release.py \\
  --output-dir release \\
  --traditional-metrics results/tables/traditional_grid_locked_cache.csv \\
  --traditional-predictions results/predictions/traditional_locked_cache \\
  --neural-root results/neural \\
  --chemprop-dir results/neural_chemprop \\
  --release-policy release/release_policy.json
```

For the formal submission bundle, add `--require-final-supplements`. This is a
release-owner audit against the locked internal evidence tree; the raw inputs
are intentionally absent from the public archive.

The gate remains HOLD if any expected run is absent, a completion marker is
invalid, prediction schema/count/finite-value checks fail, seeds differ from
1--5, cross-model split/index/SMILES identity hashes disagree, labels fail the
locked pairwise `allclose(rtol=1e-6, atol=1e-6)` check, or a blocking license
item remains unresolved. A formal build also remains HOLD if an explicitly
required training/post-processing code path is absent from the allowlist, its
local Python import closure is incomplete, the public README is absent, or an
AppleDouble sidecar enters the include plan. The maximum task/seed label
difference is recorded; a partial result is never presented as complete.
"""


def _license_text(policy: Mapping[str, Any], policy_errors: Sequence[str]) -> str:
    items = policy.get("items", {}) if isinstance(policy, dict) else {}
    checkpoint_metadata = checkpoint_release_metadata(policy)
    lines = [
        "# License and redistribution checklist",
        "",
        "This checklist is release metadata, not legal advice. No raw TDC data or",
        "large checkpoint is copied into this directory by the builder.",
        "",
    ]
    for name, item in items.items():
        status = str(item.get("status", "TODO")) if isinstance(item, dict) else "INVALID"
        blocking = bool(item.get("blocking")) if isinstance(item, dict) else True
        note = str(item.get("note", "")) if isinstance(item, dict) else "Malformed policy item."
        decision = str(item.get("decision", "TODO")) if isinstance(item, dict) else "INVALID"
        mark = "x" if status.upper() in RESOLVED_POLICY_STATUSES else " "
        lines.extend(
            [
                f"- [{mark}] **{name}** — `{status}` ({'blocking' if blocking else 'non-blocking'})",
                f"  Decision: `{decision}`.",
                f"  {note}",
            ]
        )
    lines.extend(
        [
            "",
            "## Release-owner actions",
            "",
            "- Confirm the selected code and generated-artifact licenses in `release_policy.json`.",
            (
                "- Keep checkpoints out of the public bundle as required by the resolved NOT_RELEASED decision."
                if checkpoint_metadata["checkpoints_excluded_by_policy"]
                else "- Confirm checkpoint redistribution terms and third-party notices before upload, or explicitly choose NOT_RELEASED."
            ),
            "- Keep raw TDC files out of the package unless explicit redistribution permission is documented.",
            "- Re-run the builder; all blocking policy items must be resolved for GO.",
            "",
            f"Unresolved blocking items: {len(policy_errors)}.",
            "",
        ]
    )
    return "\n".join(lines)


def build_release(
    *,
    root: Path = ROOT,
    output_dir: Path | None = None,
    tasks: Sequence[str] = tuple(TDC_TASKS),
    seeds: Sequence[int] = SEEDS,
    traditional_models: Sequence[str] = TRADITIONAL_MODELS,
    traditional_metrics_path: Path | None = None,
    traditional_prediction_dir: Path | None = None,
    neural_roots: Sequence[Path] | None = None,
    chemprop_dir: Path | None = None,
    release_policy_path: Path | None = None,
    require_final_supplements: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = (output_dir or root / "release").resolve()
    traditional_metrics_path = (
        traditional_metrics_path
        or root / "results" / "tables" / "traditional_grid_local_prelim.csv"
    ).resolve()
    traditional_prediction_dir = (
        traditional_prediction_dir
        or root / "results" / "predictions" / "traditional_local_prelim"
    ).resolve()
    neural_roots = tuple(
        path.resolve()
        for path in (
            neural_roots
            or (root / "results" / "neural", root / "results" / "neural_server_backup")
        )
    )
    chemprop_dir = (chemprop_dir or root / "results" / "neural_chemprop").resolve()
    release_policy_path = (
        release_policy_path or output_dir / "release_policy.json"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hasher = ArtifactHasher(root)

    task_list = list(tasks)
    unknown = [task for task in task_list if task not in TDC_TASK_META]
    if unknown:
        raise ValueError(f"Unknown TDC tasks: {unknown}")
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) != len(set(seed_list)):
        raise ValueError("Seeds must be unique")

    support_paths = [root / relative for relative in REQUIRED_PUBLIC_SOFTWARE_PATHS]
    support_files = [hasher.inspect(path) for path in support_paths]
    software_sha256 = aggregate_file_hash(
        [root / "requirements.txt", root / "environment.yml", root / "pyproject.toml"],
        hasher,
    )
    traditional_spec_sha256 = aggregate_file_hash(
        [
            root / "experiments" / "config" / "tdc_official.yaml",
            root / "experiments" / "run_traditional_grid.py",
            root / "experiments" / "run_traditional_predictions.py",
        ],
        hasher,
    )
    gin_spec_sha256 = aggregate_file_hash(
        [
            root / "experiments" / "config" / "tdc_official.yaml",
            root / "experiments" / "config" / "gin.yaml",
            root / "experiments" / "run_neural_a40.py",
            root / "src" / "admet_benchmark" / "smiles_compat.py",
            root / "src" / "admet_benchmark" / "neural_models.py",
        ],
        hasher,
    )
    common_config = root / "experiments" / "config" / "tdc_official.yaml"
    gin_config = root / "experiments" / "config" / "gin.yaml"
    chemprop_global_config = (
        root / "experiments" / "config" / "chemprop_dmpnn.yaml"
    )
    chemprop_global_config_errors = validate_chemprop_global_compatibility_config(
        chemprop_global_config
    )
    metrics_rows, metrics_errors = load_traditional_metrics(traditional_metrics_path)
    data_by_task = {
        task: source_data_info(root, task, hasher) for task in task_list
    }

    records: list[dict[str, Any]] = []
    for task in task_list:
        for seed in seed_list:
            for model in traditional_models:
                records.append(
                    build_traditional_record(
                        root=root,
                        task=task,
                        model=model,
                        seed=seed,
                        metrics_path=traditional_metrics_path,
                        metrics_rows=metrics_rows,
                        prediction_dir=traditional_prediction_dir,
                        common_config=common_config,
                        execution_spec_sha256=traditional_spec_sha256,
                        software_sha256=software_sha256,
                        data_info=data_by_task[task],
                        hasher=hasher,
                    )
                )
            records.append(
                build_gin_record(
                    root=root,
                    task=task,
                    seed=seed,
                    neural_roots=neural_roots,
                    gin_config=gin_config,
                    execution_spec_sha256=gin_spec_sha256,
                    software_sha256=software_sha256,
                    data_info=data_by_task[task],
                    hasher=hasher,
                )
            )
            records.append(
                build_chemprop_record(
                    root=root,
                    task=task,
                    seed=seed,
                    chemprop_dir=chemprop_dir,
                    global_config_errors=chemprop_global_config_errors,
                    software_sha256=software_sha256,
                    data_info=data_by_task[task],
                    hasher=hasher,
                )
            )
            legacy = build_legacy_record(
                task=task,
                seed=seed,
                neural_roots=neural_roots,
                common_config=common_config,
                execution_spec_sha256=gin_spec_sha256,
                software_sha256=software_sha256,
                data_info=data_by_task[task],
                hasher=hasher,
            )
            if legacy is not None:
                records.append(legacy)

    policy, policy_errors = load_or_create_policy(release_policy_path)
    checkpoint_policy = apply_checkpoint_release_policy(records, policy)
    supplement_audit = FinalSupplementAudit(
        complete=False,
        failures=(),
        files=(),
        details={
            "schema_version": 1,
            "status": "NOT_REQUIRED",
            "not_primary_model_records": True,
            "primary_submission_record_count_unchanged": True,
        },
    )
    public_view_audit = PublicPredictionViewAudit(
        complete=False,
        failures=(),
        files=(),
        details={"schema_version": 1, "status": "NOT_REQUIRED"},
    )
    if require_final_supplements:
        supplement_audit = audit_final_supplements(root)
        # Raw formal and cleaned prediction exports contain source structures
        # and labels. They remain hash-indexed evidence but are never direct
        # release payloads. Structure/label-free views are built below.
        for record in records:
            if record.get("included_default") and record.get("prediction_sha256"):
                record["prediction_package_action"] = (
                    "exclude_raw_structure_and_labels"
                )
            if record.get("included_default") and record.get("config_sha256"):
                record["config_package_action"] = "exclude_nonportable"
        if supplement_audit.complete:
            public_view_audit = build_public_prediction_views(
                root=root,
                output_dir=output_dir,
                records=records,
                supplements=supplement_audit,
            )
        else:
            public_view_audit = PublicPredictionViewAudit(
                complete=False,
                failures=(
                    "public prediction views were not generated because final "
                    "supplement evidence is incomplete",
                ),
                files=(),
                details={
                    "schema_version": 1,
                    "status": "HOLD",
                    "failures": [
                        "public prediction views were not generated because final "
                        "supplement evidence is incomplete"
                    ],
                },
            )
    gate = _build_gate(
        records,
        tasks=task_list,
        seeds=seed_list,
        traditional_models=traditional_models,
        traditional_metrics_errors=metrics_errors,
        policy_errors=policy_errors,
    )
    gate["final_supplements_required"] = bool(require_final_supplements)
    gate["final_supplements_complete"] = (
        supplement_audit.complete if require_final_supplements else None
    )
    gate["public_prediction_views_complete"] = (
        public_view_audit.complete if require_final_supplements else None
    )
    gate["checks"]["final_supplements_complete"] = bool(
        not require_final_supplements or supplement_audit.complete
    )
    gate["checks"]["public_prediction_views_complete"] = bool(
        not require_final_supplements or public_view_audit.complete
    )
    gate["final_supplements"] = supplement_audit.details
    gate["public_prediction_views"] = public_view_audit.details
    if require_final_supplements:
        added_failures = [
            *(f"final supplement: {value}" for value in supplement_audit.failures),
            *(f"public prediction view: {value}" for value in public_view_audit.failures),
        ]
        gate["failures"] = list(dict.fromkeys([*gate["failures"], *added_failures]))
        gate["failure_count"] = len(gate["failures"])
        gate["status"] = "GO" if not gate["failures"] else "HOLD"
        gate["release_ready"] = gate["status"] == "GO"
    # Raw label vectors are transient gate inputs, not release payloads. The
    # manifest retains the per-task/seed pass flag and maximum absolute
    # difference without duplicating potentially large label arrays.
    for record in records:
        record.pop("_label_alignment_rows", None)
    package_plan = build_package_plan(
        records,
        [*support_files, hasher.inspect(release_policy_path)],
        [*supplement_audit.files, *public_view_audit.files],
    )
    public_readme_path = hasher.relative(
        output_dir / "public" / PUBLIC_BUNDLE_README_NAME
    )
    public_code_release, public_code_failures = audit_public_code_package(
        root=root,
        package_plan=package_plan,
        public_readme_path=public_readme_path,
        required=require_final_supplements,
    )
    public_package_audit, public_package_failures = audit_included_public_package(
        root=root,
        package_plan=package_plan,
        required=require_final_supplements,
    )
    gate["public_code_release_complete"] = (
        public_code_release["status"] == "PASS"
        if require_final_supplements
        else None
    )
    gate["public_code_release"] = public_code_release
    gate["public_package_content_audit"] = public_package_audit
    gate["checks"]["required_public_code_paths_present"] = bool(
        public_code_release["required_code_paths_present"]
    )
    gate["checks"]["required_public_test_paths_present"] = bool(
        public_code_release["required_test_paths_present"]
    )
    gate["checks"]["public_python_local_import_closure_complete"] = bool(
        public_code_release["local_import_closure_complete"]
    )
    gate["checks"]["public_bundle_readme_present"] = bool(
        public_code_release["public_readme_present"]
    )
    gate["checks"]["appledouble_excluded_from_include_plan"] = not bool(
        public_code_release["appledouble_paths_included"]
    )
    gate["checks"]["included_public_package_content_safe"] = (
        public_package_audit["status"] == "PASS"
    )
    if require_final_supplements:
        gate["failures"] = list(
            dict.fromkeys(
                [
                    *gate["failures"],
                    *(f"public code release: {value}" for value in public_code_failures),
                    *(
                        f"public package content: {value}"
                        for value in public_package_failures
                    ),
                ]
            )
        )
        gate["failure_count"] = len(gate["failures"])
        gate["status"] = "GO" if not gate["failures"] else "HOLD"
        gate["release_ready"] = gate["status"] == "GO"
    metadata = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "project_root": ".",
        "git": git_metadata(root),
        "software_sha256": software_sha256,
        "software_and_code_files": [
            {
                "path": info.path,
                "sha256": info.sha256,
                "bytes": info.bytes,
            }
            for info in support_files
            if info.exists
        ],
        "submission_models": list(traditional_models) + list(SUBMISSION_NEURAL_MODELS),
        "legacy_model_mapping": {
            "artifact_model_id": LEGACY_SOURCE_MODEL_ID,
            "release_model_id": LEGACY_MODEL,
            "included_default": False,
            "reason": "Historical implementation has node messages and no directed bond hidden states.",
        },
        "tasks": task_list,
        "seeds": seed_list,
        "raw_data_copied": False,
        "checkpoints_copied": False,
        "checkpoint_release_policy": checkpoint_policy,
        "public_code_release": {
            "required": bool(require_final_supplements),
            "complete": (
                public_code_release["status"] == "PASS"
                if require_final_supplements
                else None
            ),
            "required_code_path_count": len(REQUIRED_PUBLIC_CODE_PATHS),
            "required_test_path_count": len(REQUIRED_PUBLIC_TEST_PATHS),
            "public_readme_path": public_readme_path,
        },
        "public_package_content_audit": {
            "required": bool(require_final_supplements),
            "complete": (
                public_package_audit["status"] == "PASS"
                if require_final_supplements
                else None
            ),
            "included_files_scanned": public_package_audit.get(
                "included_files_scanned", 0
            ),
        },
        "final_supplements": {
            "required": bool(require_final_supplements),
            "complete": (
                supplement_audit.complete if require_final_supplements else None
            ),
            "not_primary_model_records": True,
            "primary_submission_records_unchanged": True,
            "primary_submission_record_count": sum(
                bool(record.get("included_default")) for record in records
            ),
            "cleaned_prediction_files": (
                FINAL_CLEANED_PREDICTION_COUNT if require_final_supplements else 0
            ),
            "raw_prediction_exports_directly_redistributed": False,
            "raw_prediction_package_action": (
                "exclude_raw_structure_and_labels"
                if require_final_supplements
                else "not_applicable"
            ),
            "public_prediction_views_complete": (
                public_view_audit.complete if require_final_supplements else None
            ),
            "inventory": [
                {
                    "path": item.info.path,
                    "sha256": item.info.sha256,
                    "bytes": item.info.bytes,
                    "role": item.role,
                    "package_action": item.package_action,
                }
                for item in [*supplement_audit.files, *public_view_audit.files]
            ],
        },
    }
    manifest = {"metadata": metadata, "gate": gate, "records": records}

    columns = manifest_columns(records)
    write_csv(output_dir / "manifest.csv", records, columns)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    seed_rows = [
        {
            "model": record["model"],
            "task": record["task"],
            "seed": record["seed"],
            "included_default": record["included_default"],
            "run_status": record["run_status"],
            "split_sha256": record["split_sha256"],
            "label_alignment_within_tolerance": record[
                "label_alignment_within_tolerance"
            ],
            "max_abs_label_difference_within_task_seed": record[
                "max_abs_label_difference_within_task_seed"
            ],
            "config_split_sha256": record["config_split_sha256"],
            "model_input_split_sha256": record["model_input_split_sha256"],
            "source_split_sha256": record["source_split_sha256"],
            "representation_compatibility_valid": record[
                "representation_compatibility_valid"
            ],
            "representation_compatibility_policy": record[
                "representation_compatibility_policy"
            ],
            "compatibility_manifest_schema": record[
                "compatibility_manifest_schema"
            ],
            "model_input_repairs_n": record["model_input_repairs_n"],
            "source_data_sha256": record["source_data_sha256"],
            "execution_spec_sha256": record["execution_spec_sha256"],
        }
        for record in records
    ]
    write_csv(
        output_dir / "seeds.csv",
        seed_rows,
        [
            "model",
            "task",
            "seed",
            "included_default",
            "run_status",
            "split_sha256",
            "label_alignment_within_tolerance",
            "max_abs_label_difference_within_task_seed",
            "config_split_sha256",
            "model_input_split_sha256",
            "source_split_sha256",
            "representation_compatibility_valid",
            "representation_compatibility_policy",
            "compatibility_manifest_schema",
            "model_input_repairs_n",
            "source_data_sha256",
            "execution_spec_sha256",
        ],
    )
    write_csv(
        output_dir / "package_plan.csv",
        package_plan,
        [
            "relative_path",
            "sha256",
            "bytes",
            "artifact_roles",
            "models",
            "referencing_runs",
            "included_referencing_runs",
            "copied_by_builder",
            "package_action",
        ],
    )
    (output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _readme_text(gate, checkpoint_policy), encoding="utf-8"
    )
    (output_dir / "LICENSE_AND_REDISTRIBUTION_TODO.md").write_text(
        _license_text(policy, policy_errors), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the non-copying reproducibility release index and GO/HOLD gate."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--traditional-metrics", type=Path)
    parser.add_argument("--traditional-predictions", type=Path)
    parser.add_argument("--neural-root", action="append", type=Path)
    parser.add_argument("--chemprop-dir", type=Path)
    parser.add_argument("--release-policy", type=Path)
    parser.add_argument(
        "--require-final-supplements",
        action="store_true",
        help=(
            "Require, re-hash, and index the final cleaned-sensitivity, CPU-defense, "
            "ablation, figure-source, and manuscript-table supplements; also build "
            "de-identified public prediction views."
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def rooted(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    manifest = build_release(
        root=root,
        output_dir=rooted(args.output_dir),
        traditional_metrics_path=rooted(args.traditional_metrics),
        traditional_prediction_dir=rooted(args.traditional_predictions),
        neural_roots=[rooted(path) for path in args.neural_root]
        if args.neural_root
        else None,
        chemprop_dir=rooted(args.chemprop_dir),
        release_policy_path=rooted(args.release_policy),
        require_final_supplements=args.require_final_supplements,
    )
    gate = manifest["gate"]
    print(f"release_gate: {gate['status']}")
    print(f"expected_runs: {gate['expected_total_runs']}")
    for model, expected in gate["expected_by_model"].items():
        print(f"{model}: {gate['complete_by_model'][model]}/{expected}")
    print(f"failures: {gate['failure_count']}")


if __name__ == "__main__":
    main()
