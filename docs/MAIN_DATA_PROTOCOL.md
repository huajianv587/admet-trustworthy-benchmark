# Formal main-analysis data protocol

This document is the normative data-identity contract for the formal five-model
analysis. Historical notes that use the shorthand "official split" should be
read as preserving the cached official **test** membership, not as claiming
that the study reproduced a particular PyTDC train/validation routine.

## Locked train/validation split

For every endpoint and seed, all formal model families load these cached source
files:

- `data/tdc_official/admet_group/<task-slug>/train_val.csv`
- `data/tdc_official/admet_group/<task-slug>/test.csv`

All five formal model families use the explicit `locked_cache` protocol. The
non-test frame is split with scikit-learn
`train_test_split(test_size=0.125, shuffle=True, random_state=task_seed)`.
Labels are used for stratification on classification endpoints only. The
versioned algorithm identifier is
`sklearn_train_test_split_test_size_0.125_random_state_task_seed_classification_stratified_v1`.
This deterministic nominal 7:1 train/validation split is the portable protocol
locked to match the completed formal artifacts. It is **not** described as a
reproduction of PyTDC's version-dependent task-specific default or as an
official scaffold validation split.

The fixed cached `test.csv` is never repartitioned and is not used for training,
early stopping, calibration fitting, or model selection. Formal cross-model
analysis requires matching ordered validation/test source identities and labels
for the same task and seed.

## Enumerated parser-compatibility translations

Newer RDKit versions reject two legacy pyridine N-oxide charge/valence strings
in the fixed `Solubility_AqSolDB` test CSV. Both records are retained. Only the
private string supplied to a model parser is translated; the official source
record and every published prediction identity remain unchanged.

| Source SMILES retained in outputs | Private parser-compatible model input | Standard InChI | Standard InChIKey |
|---|---|---|---|
| `CC1=CC=C[NH+2]([O-])[CH-]1` | `CC1=C[NH+]([O])[CH-]C=C1` | `InChI=1S/C6H8NO/c1-6-3-2-4-7(8)5-6/h2-5,7H,1H3` | `FHJYMRFPLWFYPD-UHFFFAOYSA-N` |
| `O=C(O)C1=C[NH+2]([O-])[CH-]C=C1` | `[O][NH+]1C=C(C(=O)O)C=C[CH-]1` | `InChI=1S/C6H6NO3/c8-6(9)5-2-1-3-7(10)4-5/h1-4,7H,(H,8,9)` | `GDILJONVBZVAFH-UHFFFAOYSA-N` |

The source and translated forms were checked with an RDKit release that could
parse the legacy notation and were identical in both Standard InChI and the
derived Standard InChIKey. The translated forms are also explicitly tested for
parsing in the current runtime. This is a finite compatibility mapping, not a
general canonicalization, normalization, or cleaning rule.

## Identity invariants

For each of the two translated model inputs:

- source split membership is unchanged;
- split-local sample index and row order are unchanged;
- label is unchanged;
- prediction CSVs retain the original source SMILES;
- run manifests record the source string, private model-input string, sample
  index, policy identifier, and Standard InChIKey; and
- no row is dropped, duplicated, relabelled, or reassigned.

If any other source SMILES is unparseable after this enumerated policy, the run
fails before training instead of silently filtering a row. The separate
cleaned-data sensitivity analysis has its own explicit removal and
canonicalization rules and must not be conflated with this main-analysis
compatibility mapping.
