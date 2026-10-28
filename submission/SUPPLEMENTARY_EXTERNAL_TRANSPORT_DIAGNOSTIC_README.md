# Supplementary External Transport Diagnostic

`SUPPLEMENTARY_EXTERNAL_TRANSPORT_DIAGNOSTIC.csv` contains the source-by-model
summary for the descriptive external transport diagnostic reported in the
manuscript.

- Protocol: `EXTERNAL_DESCRIPTIVE_PROTOCOL_20261021`
- Cohorts: 8 source-specific combinations
- Models: RF, XGBoost, and LightGBM
- Seeds: 5 per model and cohort
- Rows: 24 cohort-model summaries
- Summary unit: mean and sample standard deviation across the five seeds
- Claim boundary: descriptive transport diagnostic only; not confirmatory
  external validation

The 120 underlying prediction files and their SHA-256 checks are retained in
`results/external_descriptive_v3_20261021/`. The planned 150-fit confirmatory
external matrix was not started because its outcome-blind power gate failed.
