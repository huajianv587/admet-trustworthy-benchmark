# Cross-fit oracle sensitivity audit - 2026-10-28

## Scope

This audit records a zero-retraining sensitivity analysis of the fixed TDC test
partition. It is not an independent external validation and does not replace
the primary reliability-regret analysis.

## Locked computation

- 500 repeated grouped two-fold splits per endpoint.
- Primary grouping: Bemis--Murcko scaffold, with canonical-identity fallback
  for acyclic molecules.
- Secondary grouping: canonical molecular identity.
- Candidate models and validation-selected family were frozen from the formal
  reliability-regret ledger.
- A dimension-specific candidate oracle was selected on one fold and evaluated
  on the opposite fold, then the direction was reversed and averaged.
- Negative signed differences were retained; endpoint-level bootstrap intervals
  used 5,000 replicates.
- Exact sign-flip/Holm values are descriptive and do not support confirmatory
  inference across independent assays.

## Primary Bemis--Murcko results

| Dimension | Endpoints | Mean signed D | 95% endpoint interval | Descriptive Holm p |
|---|---:|---:|---:|---:|
| Brier | 13 | 0.0035 | -0.0029 to 0.0113 | 0.8164 |
| ECE, 10 bins | 13 | 0.0198 | 0.0026 to 0.0405 | 0.2646 |
| Primary relative excess AURC | 22 | 0.0272 | -0.0075 to 0.0666 | 0.5506 |
| Applicability alignment | 22 | 0.0797 | 0.0402 to 0.1234 | 0.0051 |

The full seven-dimension table, canonical-identity sensitivity, fold-level
records, input alignment audits, and gate are retained under
`results/tables/crossfit_oracle_sensitivity_final/`. The analysis is consistent
with a dimension-specific reliability gap, but it still reuses previously
examined TDC test labels and does not create a deployable selection rule.

## External-matrix boundary

The completed descriptive package contains 120 traditional-model fits, 20 GIN
prediction files, and 10 Chemprop prediction files, totaling 150 external
artifacts. This does not equal a completed 150-slot confirmatory matrix. The
historical confirmatory plan remains `0/150 POWER_FAIL_HOLD` because the
outcome-blind power gate failed before confirmatory fitting.
