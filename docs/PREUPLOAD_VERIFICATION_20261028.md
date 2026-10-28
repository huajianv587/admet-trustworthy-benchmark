# Pre-upload verification receipt: 2026-10-28

## Repository

- Git commit: `29daa2b379cba5145fe6c3c9265fea8974d4dace`
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
- Submission ZIP: `unzip -tq` passed.
- Submission ZIP SHA-256:
  `debd879a2fcf34830126970d53282e861933486ab24f4711605c5680c7f80011`.
- Named manuscript PDF: 13 pages. Supplementary PDF: 3 pages.

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
authenticated portal fee screen.

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
