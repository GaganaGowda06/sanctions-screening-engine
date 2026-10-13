# Sanctions Screening Engine

Banks check customer names against sanctions lists. Set the alert threshold too low and analysts drown in false alerts; set it too high and a listed person gets through. This project is a two-stage screening engine built on OpenSanctions data: candidate retrieval, then scoring. Around it sits a layer for evaluation, fairness, audit and monitoring, and the evaluation layer is the main deliverable. It measures misses and false alerts, split by the script a name is written in, and says where the engine fails.

## Headline numbers

| Figure | Value | Measured on |
|---|---|---|
| F1 | 0.80 (0.8008) | The shipped engine on the third frozen hold-out: 2,000 constructed positive and 2,000 negative queries. The Stage 2 engine scores 0.7628 on the same set. |
| F1, hard pairs | 0.818 (0.8184) | The 186,946 of 439,813 analyst-labelled OpenSanctions pairs where the two entities share no identical name. Over all pairs F1 is 0.9692, but most pairs are easy. |
| Threshold | 0.86 is within one step of the best | On the same labelled pairs a sweep from 0.50 to 0.99 peaks at 0.85: F1 0.9712 against 0.9692 over all pairs, and 0.8199 against 0.8184 for the hard pairs. |
| Arabic false-alert burden | 32.6% to 22.6% | Share of clean Arabic-script names that raise an alert, Stage 2 engine against shipped engine, third hold-out. |

The first row is measured on constructed queries, not on customers. The hold-outs are spent: they have been scored more than once for analysis, but nothing was tuned on them.

## The stages

**Data.** The OpenSanctions entities file is downloaded, checked against its published SHA-1 and profiled. Normalisation, transliteration tables and a candidate index feed the matcher. Detail: [project report](docs/project_report.md).

**Matcher.** Retrieval uses an inverted index over folded name tokens, vowel-stripped skeletons and joined words, so spelling variants and Arabic-derived forms land on the same candidates. A hand-tuned scorer then combines name similarity with birth date and country, and alerts at 0.86. A logistic regression trained on OpenSanctions Pairs beat it at deduplicating pairs but not at screening, so it is kept for deduplication only. Typo-tolerant retrieval (deletion keys) was tried and rejected on dev, against criteria written down in advance.

**Arabic fairness.** A customer with an Arabic name should not collect more false alerts than one with a Latin or Cyrillic name. A rarity-weighted skeleton feature cut the Arabic false-alert burden from 32.6% to 22.6%. The gap to other scripts is still large, and the report says so.

**Audit and monitoring.** Every screening decision is written to a hash-chained log before it is shown, and can be replayed exactly under the same code, list and configuration. On top of that sit a retention purge with legal holds, batch monitoring, drift checks against a frozen baseline and a regression gate for list updates. Detail: [docs/stage4_audit_and_monitoring.md](docs/stage4_audit_and_monitoring.md).

**Analysis and validation.** Error analysis, label provenance, prevalence, tuning transfer, robustness to one-letter changes, and a check of the scoring against 439,813 analyst judgements. Detail: [docs/limitations_evidence.md](docs/limitations_evidence.md) and [docs/label_provenance.md](docs/label_provenance.md).

## Limits

- Monitoring is demonstrated on constructed batches, not production traffic. The baseline and the stable, mix-shift and degraded batches are samples of a spent hold-out, so how often a live deployment would warn or alert is unknown.
- There are no analyst dispositions, so live precision and recall cannot be measured, and neither can the false-alert burden (the 22.6% Arabic result of Stage 3): live traffic has no labels saying which alerts were wrong. A change in an alert rate is a reason to look, not a measurement of harm.
- The deduplication model is outside the decision path and is not monitored.

- **Shortened names can clear.** "Viktor Bout" with a matching birth date scores 0.7953 against the 0.86 threshold and clears, because the listed name has a patronymic the query lacks, while the full name scores 1.0. This came from one hand-typed query. The rate is not measured and the problem is not fixed.
- **Precision at real prevalence is the weakest point.** At a prevalence of 1 in 1,000, only about 1 alert in 100 would be real. If the hold-out false-alert rate carried over, about 7% to 9% of all customers would be alerted. The false-alert rate on real customers is unknown.
- **Vessels, securities and crypto wallets are out of scope.** The engine was built for people and organisations. On other entity types F1 is 0.7775, vessels without an identical name are mostly missed, and 99.75% of crypto wallet pairs that analysts judged different still raise an alert.

## How to run

Python 3.14.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/download_data.py                  # about 350 MB, verified
pytest                                       # default suite, runs without the download
pytest -m slow                               # full-size checks, needs data/
venv/bin/python scripts/demo.py              # end-to-end terminal demo on a throwaway audit store
venv/bin/streamlit run app/streamlit_app.py  # local page for screening one name, writes no audit record
```

The other scripts, the full list of commands and the project layout are in the [project report](docs/project_report.md).

## Licence and attribution

The code is MIT licensed (see [LICENSE](LICENSE)). That licence covers the code only, not the data and not any file derived from the data. The data comes from OpenSanctions (https://www.opensanctions.org). The entities file and the OpenSanctions Pairs dataset are used under CC BY-NC 4.0, so use is non-commercial and attribution to OpenSanctions is required; this project is non-commercial. The hold-outs in `holdout/` and the analysis reports in `reports/` (`analysis_*`) contain OpenSanctions-derived names, birth dates and IDs. They are shared here under CC BY-NC 4.0, with attribution to OpenSanctions, and stay under that licence. The raw entities file and the Pairs dataset are not in the repository, and the reports on the Pairs data hold aggregate counts only. The Pairs analysis follows arXiv:2603.11051 (CC BY 4.0). yente, the OpenSanctions matching API, was studied for reference and is not part of this repository.
