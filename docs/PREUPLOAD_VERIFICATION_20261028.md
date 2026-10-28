# Pre-upload verification receipt: 2026-10-28

## Repository

- Git commit: `7b8afc6839dd97902e958310098cc27caf37e023`
- Remote branch: `origin/main` contains the same commit.
- The repository update contains the nested secondary-loss audit and the
  reproducibility/provenance hardening changes. It does not constitute an
  independent scientific validation.

## Local verification

- Python environment: project `.venv`, Python 3.11.15.
- RDKit runtime checked and repaired to `2026.09.1`; the prior environment had
  stale `rdkit_pypi 2022.09.5` files despite newer package metadata.
- Regression suite: `224 passed, 6 skipped`.
- Remaining test output: 22 Pandas compatibility warnings; no test failures.
- Submission ZIP: `submission/SAR_QSAR_submission_candidate_package_refreshed_20261028_v2.zip`; `unzip -tq` passed.
- Submission ZIP SHA-256 is recorded in the sidecar file
  `submission/SAR_QSAR_submission_candidate_package_refreshed_20261028_v2.zip.sha256`.
- Named manuscript PDF: 19 pages. Supplementary PDF: 3 pages.
- Project tests: `224 passed, 6 skipped`.

## Scientific boundary

- Formal internal matrix: `550/550_ARTIFACT_COMPLETE`.
- ExpansionRx: `200/200_VALIDATION_ONLY_POST_HOC`.
- Broad confirmatory external matrix: `0/150_POWER_FAIL_HOLD`.
- Strict outcome-blind cohorts: `0/6_PASSED`.
- Focused CYP2D6: exploratory only because test-role outcomes were inspected
  before the focused protocol freeze.

No seed repetition, GPU concurrency, threshold change, duplicate record,
cohort pooling or narrative edit can convert the failed power gate into a
valid `150/150` confirmatory result.

## Venue evidence

The publisher's public journal page describes SAR and QSAR in Environmental
Research as a hybrid Open Select journal and states that an article published
on the subscription route has no APC. This is not a substitute for the final
authenticated portal fee screen or evidence about any optional page/colour
charges.

Public secondary JCR summaries disagree by category: some list Q2 for
Mathematical & Computational Biology while listing Q3 for Environmental
Sciences or other categories. The category-level institutional JCR record is
therefore still required before claiming “SCIE Q2” without qualification.

## Submission gate

Current package status remains:
`PREUPLOAD_HOLD; CANDIDATE_READY_FOR_AUTHOR_REVIEW_ONLY`.

Remaining author-controlled gates are final CRediT selection, exact AI
tool/version/date disclosure, the public release route for later diagnostics,
authenticated article type and subscription route, final fee evidence,
category-level JCR evidence, portal preview inspection and the author's final
submission decision.
