# ExpansionRx validation-only replication audit

Checked: 2026-10-27 (Singapore time)

## Decision

`PASS_WITH_SCOPE_LIMIT`: the validation-only replication is complete and
internally auditable. It supports a source-specific validation-stage stability
diagnostic. It does not establish independent external validation, test-set
performance, temporal validity, deployment validity, or a new model-selection
algorithm.

## Execution coverage

- Endpoints: 8 regression tasks: `logd`, `ksol`, `hlm_clint`, `mlm_clint`,
  `caco2_papp`, `caco2_efflux`, `mppb`, and `mbpb`.
- Model families: RF, XGBoost, LightGBM, GIN, and Chemprop D-MPNN.
- Seeds: 1-5 for every family and endpoint.
- Traditional slots: 120/120 completed; failures: 0.
- Neural slots: 80/80 completed and hash-verified.
- Total selector input slots: 200/200.
- Validation stability rows: 40 (one held-out seed per endpoint).

## VRRS result

VRRS combines equal within-endpoint rank scores for validation MAE and the
Spearman association between maximum Morgan similarity to the seed-specific
training set and negative absolute validation error. It is a post-hoc rule,
written after the replication execution; it was not preregistered.

| Endpoint | VRRS choice | Validation-MAE-only choice |
|---|---|---|
| `caco2_efflux` | Chemprop D-MPNN | Chemprop D-MPNN |
| `caco2_papp` | Chemprop D-MPNN | RF |
| `hlm_clint` | RF | Chemprop D-MPNN |
| `ksol` | GIN | Chemprop D-MPNN |
| `logd` | Chemprop D-MPNN | Chemprop D-MPNN |
| `mbpb` | LightGBM | Chemprop D-MPNN |
| `mlm_clint` | RF | Chemprop D-MPNN |
| `mppb` | LightGBM | Chemprop D-MPNN |

The family choice changed on 6/8 endpoints. Leave-one-seed selection agreed
with the full five-seed VRRS choice in 33/40 rows, for a stability rate of
82.5%.

## Test-label boundary

`selection_freeze.json` records `test_targets_read_before_freeze: false` and
`test_outcomes_used_for_selection: false`. The test comparison file is
`NOT_EVALUATED_TEST_TARGETS_NOT_PROVIDED`. No test-target directory was passed
to the selector, and no unseen-test performance claim is supported.

## Source artifacts

- Prediction directories: `results/external_replication_20261026/traditional`
  and `results/external_replication_20261026/neural_v2`.
- Prepared data: `results/external_replication_20261026/prepared`.
- Protocol: `docs/VALIDATION_RELIABILITY_SELECTOR_PROTOCOL_20261026.md`.
- Selector script: `experiments/analyze_validation_reliability_selector.py`.
- Frozen outputs: `results/external_replication_20261026/validation_selector/`.

The frozen manifest binds the prediction-file hashes, selection-table hash,
cross-fit-table hash, preparation-manifest hash, and selector-script hash.

