# Near-Neighbor Literature Comparison (Full-Text Audit)

Date: 2026-10-21

Status: `FULL_TEXT_AUDIT_FOR_CURRENT_CLAIMS`

This table is a claim-control document. It is not a systematic review and does not authorize a novelty or "first" claim. It covers the three recent works cited in the current manuscript.

| Work | Overlap with this manuscript | Difference that can be stated cautiously | Boundary |
|---|---|---|---|
| Zhao et al. (2026), *Journal of Cheminformatics* 18:95, DOI `10.1186/s13321-026-01217-2` | The full text evaluates data scarcity/OOD, class imbalance, beyond-Rule-of-5 chemistry, and activity cliffs. It uses eight representative ADMET datasets for the random-versus-perimeter comparison, a broad model pool (molecular/tabular foundation models, GNNs, AutoML, and classical models), and an ensemble/roughness diagnostic. | The present draft freezes one audited PyTDC 1.1.15 snapshot and asks whether the designated validation winner agrees with five held-out reliability dimensions. It does not reproduce perimeter/OOD, bRo5, or activity-cliff challenge splits, and it does not claim superiority. | Cite as the closest broad reliability/challenge comparator. State that the present work is complementary and narrower in estimand. |
| Kamuntavičius et al. (2025), *Journal of Cheminformatics* 17:108, DOI `10.1186/s13321-025-01041-0` | The full text asks which representations are suitable, whether cross-validation hypothesis testing is more robust than a hold-out test, how optimization changes results, and how external same-property data changes performance. It sequentially applies cleaning, feature combination, dataset-specific tuning, cross-validation tests, test evaluation, and cross-source experiments; its Table 1 lists 22 TDC ADMET tasks across regression and binary endpoints. | The present analysis fixes the candidate configurations and split protocol, then estimates validation-selection versus test-reliability regret. It does not optimize representations, tune hyperparameters per dataset, or treat cross-source transfer as the primary estimand. | Cite as a methodological comparator for evaluation design and cross-source transfer, while avoiding any claim that the current study supersedes it. |
| Zheng et al. (2026), arXiv `2607.10729v4` (6 October 2026) | The full text benchmarks six public ADMET tasks using a label-free 70/10/20 structural-frontier split against a ratio- and group-matched scaffold control, with 468 primary runs and a 46-run graph-network control. It reports frontier error inflation and robustness penalties. | The present manuscript does not introduce either split, does not claim scaffold-disjoint evaluation, and uses the fixed cached test membership only for retrospective reliability summaries. | Keep the citation as a versioned preprint. Do not describe it as a peer-reviewed journal article or imply that the current benchmark reproduces its split. |

## Safe contribution wording for the current draft

> We provide a retrospective, fixed-benchmark comparison of validation-metric model-family selection with multiple held-out reliability summaries across 22 TDC ADMET endpoints. The study is complementary to recent ADMET reliability and structural-evaluation work; it does not claim a new architecture, a new chemical split, or superiority over those studies.

## Prohibited wording until the audit is complete

- first or state-of-the-art ADMET reliability audit;
- comprehensive or systematic literature review;
- external, temporal, prospective, or scaffold-disjoint validation for the current run;
- superiority over Zhao et al., Kamuntavičius et al., or Zheng et al.

## Full-text locators and sources

- Zhao et al.: Springer full text, Methods section “Data collection and preparation” and “Algorithm selection”; the results discussion describes eight representative ADMET datasets, random versus perimeter splits, and ensemble/roughness analysis. DOI `10.1186/s13321-026-01217-2`; PubMed record PMID `42152045`.
- Kamuntavičius et al.: PDF pp. 1--3 for research questions and sequential workflow; PDF p. 3, Table 1 for the TDC task inventory; DOI `10.1186/s13321-025-01041-0`.
- Zheng et al.: arXiv v4 pp. 1--2 for the six-task, 70/10/20 frontier design and run counts; p. 4 for the primary frontier results; arXiv `2607.10729v4`.

