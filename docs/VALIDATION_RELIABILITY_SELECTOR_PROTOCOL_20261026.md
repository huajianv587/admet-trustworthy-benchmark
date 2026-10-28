# Validation Reliability Selector Protocol

## Scope

This document defines a retrospective, validation-only model-family selector
for the ExpansionRx replication. It is an analysis developed after the
replication protocol and must not be described as preregistered. It is not a
new molecular model and it does not establish deployment utility.

## Candidate pool and inputs

The candidate pool is the five locked families: RF, XGBoost, LightGBM, GIN,
and Chemprop D-MPNN. Every endpoint must have five seeds per family. The
selector reads only validation rows from prediction artifacts. Validation
identities, targets, and predictions must agree exactly across families within
each endpoint and seed. Test-target files are not read until the selection
table and `selection_freeze.json` have been written.

## Metrics

For regression endpoints, each family is summarized by the mean across seeds
of validation MAE and the Spearman correlation between maximum Morgan
similarity to the seed-specific training identities and negative absolute
validation error. Higher support/error correlation is interpreted as better
support-risk ordering.

For binary endpoints, the corresponding metrics are AUROC, Brier score, ten-bin
ECE, and the same support/error Spearman correlation. Lower is better for MAE,
Brier, and ECE; higher is better for AUROC and support/error correlation.

Each metric is converted to a within-endpoint family rank score in [0, 1],
with one as the best score. The rank scores receive equal weight. The VRRS
choice is the largest composite score; exact composite ties are resolved by
alphabetical model-family order. A validation-MAE-only choice is retained as a
conventional comparator.

## Stability check

For every endpoint, one seed is held out in turn. The selector is recomputed
from the other four validation seeds and compared with the full five-seed VRRS
choice. The held-out validation result is reported as a stability diagnostic,
not as an independent test estimate.

## Files and gates

`experiments/analyze_validation_reliability_selector.py` writes:

- `selection_before_test.csv`;
- `crossfit_validation_selection.csv`;
- `selection_freeze.json`;
- `external_test_selector_comparison.csv`;
- `analysis_manifest.json`.

The selector is considered frozen only when `selection_freeze.json` records
`test_targets_read_before_freeze: false` and binds the prediction-file hashes.
If test labels are absent, the output remains a validation-only result and no
external test comparison may be claimed.

## Interpretation boundary

This protocol can show whether a multi-dimensional validation diagnostic changes
or stabilizes family selection. It cannot show that the selector improves an
unseen cohort until a separately audited post-freeze evaluator reads the sealed
test labels. A mixed or negative result must be retained as evidence against
method promotion rather than converted into a positive algorithmic claim.
