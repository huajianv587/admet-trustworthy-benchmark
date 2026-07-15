# Trustworthy ADMET Benchmark

This repository supports the study:

> **Reliability Regret after Single-Metric Random-Resplit Validation Selection:
> A 22-Endpoint Audit on Scaffold-Disjoint TDC ADMET Tests**

The study asks a practical model-selection question: after an ADMET model is
chosen using one predictive random-resplit validation metric, what reliability
is lost in calibration, selective prediction, validation-residual coverage, or chemical-support
alignment? It is a model-selection and evaluation study, not a new molecular
architecture or a state-of-the-art leaderboard claim.

## Study matrix

- 22 Therapeutics Data Commons ADMET endpoints
- five model families: random forest, XGBoost, LightGBM, GIN, and Chemprop
  directed message-passing neural network
- five seeds per endpoint and model
- 550/550 completed primary runs
- 120/120 paired cleaned-data sensitivity runs
- scaffold-disjoint test evaluation with explicit source and split hashes

The primary benchmark fixes each endpoint's model using mean designated
validation performance before auditing held-out reliability. A zero-refit,
500-repeat scaffold-group cross-fit then probes the sensitivity of the
candidate-pool oracle comparison without changing any trained prediction.

## Reproducibility boundary

The public release contains active training and analysis code, environment
specifications, sanitized configurations, de-identified prediction views,
seeds and hashes, source statistics, result tables, figures, and audit outputs.
It deliberately excludes:

- raw TDC records, source SMILES, labels, and source row indices;
- model checkpoints;
- credentials, machine-specific paths, and cluster operator materials; and
- internal prediction exports that contain nonredistributable source fields.

The released prediction views therefore support alignment and output audits but
cannot alone reproduce label-dependent metrics. Exact replay requires obtaining
the matching TDC ADMET data under its original terms and following
[`docs/MAIN_DATA_PROTOCOL.md`](docs/MAIN_DATA_PROTOCOL.md).

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
Only paths marked `include` in `release/package_plan.csv` belong in the public
archive. Do not publish the project directory or the complete `release/`
directory wholesale.

## Author

Huajian Jiang  
College of Computing and Data Science (CCDS)  
Nanyang Technological University  
50 Nanyang Avenue, Singapore 639798, Singapore  
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
