# SCIE Q2 Submission Status: 2026-10-28

Status: `PREUPLOAD_HOLD; CANDIDATE_READY_FOR_AUTHOR_REVIEW_ONLY`

This public status note describes the evidence boundary of the current source
release. It is not an acceptance guarantee or a substitute for the
authenticated journal portal record.

## Study scope

- 22 fixed TDC ADMET endpoints, five model families and five seeds: 550 fit
  slots.
- The grouped nested-refit secondary-loss audit is conditional retrospective
  evidence; it does not establish independent external validation.
- The original four-endpoint confirmatory matrix remains `0/150
  POWER_FAIL_HOLD` because its outcome-blind power gate failed before fitting.

## Calibration estimands

Figure 4 is the candidate-model-averaged calibration summary. Isotonic
calibration reduced ECE on 13/13 endpoint means and Brier score on 6/13; the
legacy raw-probability logistic transform improved ECE on 10/13 and Brier score
on 10/13. The separate validation-selected-model sensitivity reports isotonic
ECE/Brier improvement on 11/13 and 8/13 endpoints, respectively. These are
different post-hoc estimands and must not be combined.

## External-source boundary

ExpansionRx `150/150` is an execution and metric-completeness count for a
separate retrospective regression matrix. It is not the original confirmatory
matrix and is not independent prospective confirmation. ExpansionRx, Octant,
AID777 and focused CYP2D6 results are source-specific or retrospective
diagnostics. The outcome-blind feasibility scan passed `0/6` strict cohort
gates.

## Reproducibility and release

The public release contains code, sanitized configurations, de-identified
prediction views, result tables, figures and audit outputs. Raw TDC records,
source structures and labels, model checkpoints and credentials are not
redistributed. Exact label-dependent replay requires an authorised matching
TDC snapshot and the documented split protocol.

Local document packaging and regression tests establish artifact consistency;
they do not establish independent human review, external assay validity,
prospective deployment performance or SCIE Q2 acceptance.
