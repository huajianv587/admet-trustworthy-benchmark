# Trustworthy ADMET Benchmark

This repository supports the study:

> **A Retrospective Reliability-Regret Audit for QSAR-Oriented ADMET Model Selection:
> A 22-Endpoint TDC Benchmark Study**

The study asks a practical model-selection question: after an ADMET model is
chosen using one predictive random-resplit validation metric, what
candidate-pool opportunity gaps remain in calibration, selective prediction,
validation-residual coverage, and Morgan support/error rank association? The
association measures whether fingerprint support ranks lower-error molecules;
its dimension-specific candidate reference is neither an accuracy oracle nor a
generally reliability-optimal model. This is a model-selection and evaluation
study, not a new molecular architecture or a state-of-the-art leaderboard
claim.

## Study matrix

- 22 Therapeutics Data Commons ADMET endpoints
- five model families: random forest, XGBoost, LightGBM, GIN, and Chemprop
  directed message-passing neural network
- five seeds per endpoint and model
- 550/550 completed primary runs
- 120/120 paired cleaned-data sensitivity runs
- fixed cached test partitions with explicit sample-identity and split hashes; scaffold-disjoint provenance is not established for this run

The primary benchmark fixes each endpoint's model using mean designated
validation performance before auditing held-out reliability. The test-set
minimum is retained as a retrospective candidate-pool reference and is not a
deployable selector. The current analysis also retains separate post-hoc
selected-model and cleaning sensitivities; these do not establish external or
temporal validity.

The grouped nested-refit workflow also includes a secondary-loss audit on
untouched outer folds. It evaluates Brier/log loss for classification and
MAE/RMSE for regression after inner-fold family selection; the audit is
conditional retrospective evidence and does not establish independent
external validation or change the original `0/150 POWER_FAIL_HOLD` boundary.

## Selected-model zero-refit sensitivities

For the validation-selected models, ranking test molecules from high to low
Morgan support reduced risk--coverage AURC relative to random ordering by a
mean normalized 0.1456 (95% endpoint-bootstrap interval 0.0936--0.2024), with
a favorable direction on 20/22 endpoints. Ensemble uncertainty gave a larger
mean gain of 0.3278 (0.2464--0.4145), and Morgan support had the lower paired
AURC on only 6/22 endpoints. Morgan support was therefore informative in this
benchmark but different from and generally weaker than ensemble uncertainty;
their combination was not tested.

For the 13 selected classification models, validation-fitted isotonic mapping
improved ten-bin ECE on 11/13 endpoint means but Brier score on 8/13; the mean
Brier-gain interval crossed zero. The all-candidate-model-averaged Figure 4
summary is a separate 13/13 ECE and 6/13 Brier estimand. These mappings reused validation labels that
had informed checkpointing and model selection, so they do not represent an
independent calibration cohort. Both analyses are post-hoc descriptive
sensitivities and establish neither external nor prospective validity.

## Reproducibility boundary

The public release contains active training and analysis code, environment
specifications, sanitized configurations, de-identified prediction views,
seeds and hashes, source statistics, result tables, figures, and audit outputs.
It deliberately excludes:

- raw TDC records, source SMILES, labels, and source row indices;
- model checkpoints;
- credentials, machine-specific paths, and cluster operator materials; and
- internal prediction exports that contain nonredistributable source fields.

The released prediction views therefore support sample-identity alignment and
output audits but cannot alone reproduce label-dependent metrics. Exact replay
requires obtaining the matching TDC ADMET data under its original terms and
following [`docs/MAIN_DATA_PROTOCOL.md`](docs/MAIN_DATA_PROTOCOL.md). The
public package is traceable and conditionally replayable; it is not a
standalone exact-reproduction bundle because raw TDC data, checkpoints and the
complete formal remote traditional-CPU environment are not released.

## Environment

Conda is recommended for RDKit compatibility:

```bash
conda env create -f environment.yml
conda activate admet-trustworthy
pip install -e '.[chem,models,dev]'
```

This base environment covers the traditional models, GIN, data rebuilding,
analysis, figures, and tests. In particular, both `requirements.txt` and the
`models` extra include PyTorch Geometric; a plain core-only `pip install -e .`
is not sufficient for GIN training.

The formal Chemprop D-MPNN used an intentionally separate CLI environment.
All 110 sanitized run configurations record the same historical runtime:
Chemprop 2.2.1, Python 3.12.10, PyTorch 2.7.1+cu128, and CUDA runtime 12.8.
The runner enforces `chemprop==2.2.1`; create a platform-appropriate separate
environment and pass its executable explicitly:

```bash
python3.12 -m venv .venv-chemprop
.venv-chemprop/bin/pip install 'chemprop==2.2.1'
python experiments/run_chemprop_dmpnn.py --smoke \
  --chemprop-bin .venv-chemprop/bin/chemprop \
  --chemprop-python .venv-chemprop/bin/python
```

The exact CUDA-enabled PyTorch build is recorded for provenance, not imposed on
CPU-only readers. Select the matching PyTorch/CUDA wheel for the intended
hardware before installing Chemprop when exact GPU reruns are required.

Run the test suite:

```bash
pytest -q
```

The public source clone includes the allowlisted test suite. Tests that bind to
locally obtained label-bearing TDC data or non-source result artifacts are
explicitly skipped when those prerequisites are absent; such skips are not a
substitute for the full release-owner evidence gate below.

The locked analysis entry point is:

```bash
bash scripts/run_final_posthoc_locked.sh
```

It expects the internal label-bearing prediction schema reconstructed from
locally obtained source data. It is not a direct runner for the de-identified
public prediction views.

Build and audit the formal public bundle with:

```bash
python experiments/build_reproducibility_release.py \
  --output-dir release \
  --traditional-metrics results/tables/traditional_grid_locked_cache.csv \
  --traditional-predictions results/predictions/traditional_locked_cache \
  --neural-root results/neural \
  --chemprop-dir results/neural_chemprop \
  --release-policy release/release_policy.json \
  --require-final-supplements
```

This release-owner audit must be run against the locked internal evidence tree;
the raw prediction inputs are intentionally absent from the public archive.
The formal archive is assembled from rows marked `include` in the release
owner's internal `release/package_plan.csv`. That internal ledger also records
excluded nonpublic paths and is intentionally not bundled. Do not publish the
project directory or the complete `release/` directory wholesale.

## Author

Huajian Jiang
Nanyang Technological University, Singapore
ORCID: [0009-0001-3992-2286](https://orcid.org/0009-0001-3992-2286)

## Citation

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).
The versioned release DOI is
[10.5281/zenodo.21374237](https://doi.org/10.5281/zenodo.21374237), and the
browsable source repository is
[huajianv587/admet-trustworthy-benchmark](https://github.com/huajianv587/admet-trustworthy-benchmark).

## Licences

- Source code: Apache License 2.0; see [`LICENSE`](LICENSE).
- De-identified prediction views, generated metrics and tables, and original
  result figures: Creative Commons Attribution 4.0 International; see
  [`LICENSE-DATA.md`](LICENSE-DATA.md).
- Raw TDC data: not redistributed; obtain them under their original terms.
- Model checkpoints: not released.
- Manuscript prose: subject to its applicable manuscript or publisher licence.

This software and its outputs are research artifacts. They are not clinical,
regulatory, or stand-alone compound-safety tools.


## Current candidate update 2026 10 28

The canonical manuscript source is `paper/main.tex`, with supplementary source
in `paper/supplementary.tex`. Current readiness and limits are in
`docs/SCIE_Q2_STATUS_20261028.md`. All 550 formal runs are retained in the
audited ledger; the separate Mac replay is not substituted. A complete training
environment replay and independent external validation are not established.

The current locked validation-selected RRP winner agreement counts are 4/13
uncalibrated ECE, 8/13 Brier, 2/9 coverage error, 2/22 primary selective-risk
gain, and 4/22 applicability alignment. These are fixed-test candidate-pool
comparisons, not a deployable selector. The post-hoc grouped cross-fit
sensitivity is retained as a separate zero-retraining analysis and does not
establish external validity. The manuscript remains on scientific, disclosure
and venue hold; local tests and file packaging do not establish Q2 readiness.
Published v1.0.0 metadata describe the historical core release. The Git
repository update dated 2026-10-27 contains current source corrections and
audit code, but it does not by itself update the Zenodo v1.0.0 DOI or certify a
new archived release.
