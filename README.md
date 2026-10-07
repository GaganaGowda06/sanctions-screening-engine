# Sanctions Screening Engine

A two-stage sanctions screening engine built on OpenSanctions data: a candidate retrieval step, then a scoring step, wrapped in an evaluation, monitoring and fairness layer. The evaluation layer is the main deliverable. It measures recall and precision and slices them by the script a name is written in.

This is a portfolio project for a graduate data science application in an economic crime team. yente, the open-source OpenSanctions matching API, is the engine I study and aim to improve on. It is kept for reference and is not part of this repository.

Status: Stage 1 (setup and data) is complete. Stage 2 (baseline matcher: normalisation, candidate generation, scoring, and a callable screening interface) is built, tuned and finally evaluated on a frozen hold-out. Training the matcher on the OpenSanctions Pairs dataset has started: the data and the splits are in place. The fuller evaluation harness, monitoring and audit log come in later stages.

## The data

The data is the OpenSanctions consolidated sanctions collection, downloaded as FollowTheMoney entities in JSON-lines format (one entity per line).

Licence: the data is licensed under CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0). It may be used for non-commercial purposes only, with attribution to OpenSanctions. This project is non-commercial. The data is not stored in this repository, and anyone running it downloads their own copy. See https://www.opensanctions.org/licensing/ for the full terms.

## What the data looks like

Profile of dataset version 20261006074704-hib (full output in `reports/stage1_profile.json`):

- 301,585 entities in total, of which 72,728 are sanctions targets.
- 41,820 Person entities, of which 41,766 have a name. About 75.5% of those named people have a birth date and about 76.6% have a country.
- About 95,500 names contain a non-Latin letter. Cyrillic leads (about 71,000 names), then Arabic (about 12,600), Japanese kana (about 7,400) and Han (about 4,000). Hangul and Georgian have fewer than 150 names each, so per-script results for them will be based on very small samples.
- 56% of targets with a name (39,330 of 69,637) have at least one non-Latin name. Only 1,317 have no purely Latin name at all.
- FollowTheMoney types `programId` as an identifier, but it holds a sanctions programme, not an ID for the entity. The report gives identifier coverage both with and without it. For people, own-identifier coverage is 79.6% with it and 49.7% without.

## Screening a name

One function does the whole job. The list it screens against is the named sanctions targets in the downloaded file, built on first use (a few seconds).

```
from screening import screen

for match in screen("Vladimir Putin", birth_date="1952-10-07", country="ru"):
    print(match.score, match.entity_id, match.explanation)
```

`screen()` returns the entities whose score reaches the alert threshold for the script of the query, best first. Birth date is `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, and country is a two-letter code. Each match explains itself:

- `name_score` is the name similarity before the skeleton and before birth date and country, and `skeleton_used` says whether the vowel-stripped skeleton raised it.
- `birth_date` is `agree`, `conflict`, `close` (one year apart, no effect) or `missing`, and `country` is `agree`, `conflict` or `missing`, each with the adjustment it made.
- `matched_name` is the entity's own spelling that the query matched best.

The alert threshold is one global value, 0.86, set in `src/screening_config.json` together with the other scoring settings. Per-script thresholds are kept in the same file and in the code, and `per_script` switches them on, but they are off: see "A negative result: per-script thresholds".

The same function is available from the command line:

```
python src/screen_cli.py "Vladimir Putin" --birth-date 1952-10-07 --country ru
python src/screen_cli.py "محمد علي" --json --limit 5
```

## Final results

The configuration was frozen before the final run: one global alert threshold of 0.86, a blend of character ratio and rarity-weighted score of 0.5, skeleton lift 0.2 and floor 0.6. All of it was tuned on a dev sample. It was then scored once, with no changes, on a hold-out set of 2,000 positive and 2,000 negative queries that had been frozen earlier (SHA-256 `cb3cb8f4...`) and never used for tuning. The checksum was verified before scoring. Full numbers are in `reports/final_evaluation.json`. Intervals are 95%: Wilson for proportions, bootstrap for F1.

A positive query is a real alias of a listed entity with that name removed from the list, so the engine has to find the entity under a different spelling or script. A negative query is either a synthetic name built from real first and last name tokens that matches no listed entity (75%), or a real listed name given a birth date and country that conflict with the entity's (25%).

| Figure | Hold-out | 95% interval | Test set |
|---|---|---|---|
| Recall | 74.7% | 72.7 to 76.5 | 73.0% |
| Precision | 80.9% | 79.0 to 82.6 | 80.3% |
| F1 | 0.776 | 0.761 to 0.792 | 0.764 |
| Top-1 accuracy | 79.0% | 77.1 to 80.7 | 77.9% |
| False positive rate | 11.3% | 10.0 to 12.8 | 10.8% |

False positives as alerts per 1,000 clean names screened: 113 overall (100 to 128), 145 for synthetic names (128 to 163) and 18 for namesakes with conflicting details (10 to 34). This describes the negative mix above, not a production rate: real customer names differ from recombined tokens. At this level the baseline would be too noisy to run as it is.

By direction:

| Direction | Queries | Recall | 95% interval |
|---|---|---|---|
| Latin spelling variants | 800 | 77.9% | 74.9 to 80.6 |
| Cyrillic query, Latin entry | 300 | 89.7% | 85.7 to 92.6 |
| Latin query, Cyrillic entry | 300 | 69.7% | 64.2 to 74.6 |
| Arabic query, Latin entry | 300 | 69.3% | 63.9 to 74.3 |
| Latin query, Arabic entry | 300 | 61.3% | 55.7 to 66.7 |
| Han, kana and other scripts against Latin | 200 | 0.0% | 0.0 to 1.9 |

By script of the query:

| Script of the query | Recall | 95% interval | False positive rate | 95% interval | Alerts per 1,000 clean names |
|---|---|---|---|---|---|
| Latin | 72.6% | 70.2 to 74.8 | 7.0% | 5.6 to 8.6 | 70 (56 to 86) |
| Cyrillic | 89.7% | 85.7 to 92.6 | 5.3% | 3.6 to 7.6 | 53 (36 to 76) |
| Arabic | 69.3% | 63.9 to 74.3 | 33.6% | 28.9 to 38.7 | 336 (289 to 387) |

Test set against hold-out. Recall, precision and top-1 accuracy are higher on the hold-out than on the test set (by 1.7, 0.6 and 1.1 points), the false positive rate is 0.5 points higher, and the test value sits inside the hold-out interval for all five headline figures (recall, precision, F1, top-1 accuracy and false positive rate). F1 was 0.769 on dev, 0.764 on test and 0.776 on the hold-out. I read that as no sign of overfitting. Of the 18 test-versus-hold-out comparisons in the report, two have a test value outside the hold-out interval: the false positive rate for Latin queries (7.0% on the hold-out against 4.7% on test, p = 0.02) and for namesakes with conflicting details (1.8% against 0.8%, p = 0.16). With 18 comparisons, one result under 0.05 is expected by chance, but the Latin figure is worth watching.

## A negative result: per-script thresholds

The Arabic false positive rate is far above the others, so I tried alert thresholds per query script (Latin, Cyrillic, Arabic, other). They are still in the code and the config, but switched off (`per_script` is false).

- First attempt, tuned on dev to make the false positive rate equal across scripts, with no condition on recall (commit `e3ed230`). Arabic ended up at a threshold of 1.0, so Arabic-script queries alerted only on a perfect score. On test the Arabic false positive rate fell from 24% to 9.5%, but an Arabic-script query then found only 28% of the entities listed under the same name in Latin spelling, down from 54%.
- Second attempt, requiring each script to keep its recall and then taking the lowest false positive rate (commit `64bd188`). The thresholds came out at 0.86, 0.88, 0.86 and 0.86, almost the global value. Overall false positive rate moved from 10.8% to 10.6% on test, and Arabic stayed at 37.6%.

A threshold cannot lower the Arabic false positive rate without losing Arabic recall, because in this data the scores of true and false Arabic matches overlap. The cause is in the scoring: for any pair involving Arabic the vowel-stripped skeleton can stand in for the name score, and Arabic names are made of a small set of very common consonant skeletons.

## Training data: OpenSanctions Pairs

The matcher so far is hand tuned. The next phase trains a logistic regression over explicit features, the approach nomenklatura's RegressionV1 takes, on the OpenSanctions Pairs dataset: 755,540 pairs of entities that analysts labelled as the same entity (positive) or different entities (negative). Nothing has been trained yet; this section records the data and how it was split.

Licence: non-commercial use only (CC-BY-NC-4.0 on the Hugging Face dataset card, "non-commercial use" on the OpenSanctions page). The paper is licensed separately and that licence does not apply to the data. Cite: Smith, Sesodia, Lindenberg and Schroeder de Witt (2026), OpenSanctions Pairs: Large-Scale Entity Matching with LLMs, arXiv:2603.11051, and OpenSanctions, Pairs: Cross-Referencing Training Data, snapshot 2025-12-09 (https://www.opensanctions.org/docs/opensource/pairs/). The file (390 MiB compressed) is downloaded with `src/download_pairs.py`, which checks the size, the MD5 published by OpenSanctions and the SHA-256 published by Hugging Face. It is not stored in this repository.

What is in it (full output in `reports/pairs_profile.json`):

- 581,149 positive and 174,391 negative pairs (76.9% positive). The file is ordered by source, not shuffled: the share of positives runs from 1.9% in the last tenth of the file to 100% in the middle. Never take a slice of it as a sample.
- 280,590 pairs (37.1%) label relationship and role records (Occupancy, Succession, Position, Ownership, Family and others), not named things. The 474,950 pairs of name-bearing entities (persons, companies, organisations, legal entities, vessels and a few others) are 63.8% positive, and those are the ones that matter for name matching.
- Among those entity pairs, 44.9% have Latin names only, 30.9% have non-Latin names on both sides and 24.2% on one side. Cyrillic appears on either side in 183,661 pairs, Arabic in 91,796 and other scripts in 104,180.
- Both sides of the entity pairs carry a birth date in 207,589 (43.7%), a country in 375,334 (79.0%) and an identifier in 225,042 (47.4%, or 28.8% leaving out programId). All three together: 16,891 (3.6%). The attributes are strongly tied to the labels: when both sides have a birth year and they share one, 97.1% of the pairs are positive, and 2.5% when they conflict.
- 8,280 pairs appear more than once and 14 of those have conflicting labels. One connected group of entities holds 35.2% of all pairs, and one entity appears in 376 pairs.

### Splits and the new final hold-out

The earlier hold-out belonged to the hand-tuned matcher and is spent. A fresh split was made by `src/build_pair_splits.py`: 70% train, 10% dev, 10% test and 10% final hold-out. The rule uses a hash of a group key and a fixed seed, so file order plays no part. Pairs in a small connected group (up to 100 pairs) are kept together with their whole group, so those entities appear in one split only. Pairs in the giant group are assigned one pair at a time, which keeps duplicates together but cannot keep entities apart.

| Split | Pairs | Positive | Name-bearing entity pairs | Positive among those | Pairs sharing an entity with a train pair |
|---|---|---|---|---|---|
| train | 528,358 | 77.0% | 331,614 | 63.9% | not applicable |
| dev | 75,548 | 76.9% | 47,623 | 63.8% | 35.8% |
| test | 75,439 | 76.9% | 47,472 | 63.8% | 35.9% |
| final hold-out | 76,195 | 76.7% | 48,241 | 63.6% | 35.6% |

Cautions. About a third of the dev, test and hold-out pairs share an entity with a train pair, all of them from the large connected groups (mostly the giant one), so scores on them are somewhat optimistic compared with entirely unseen entities. The hold-out pairs that share no entity with any other split (48,936 pairs, 21,084 of them entity pairs) are marked, but they are 93.0% positive, so they can check for leakage and cannot measure false positives. Use the whole hold-out for headline figures.

The assignment is frozen in `splits/pairs/` with the dataset checksum, a checksum of the pair ids in file order, and a checksum of the hold-out membership (`5439a1090adc...`). `pairs.iter_split()` refuses the hold-out unless it is asked for explicitly, and only the final evaluation may do that.

## Known limitations

These come from the measurements in `reports/stage2_candidates.json`, `reports/stage2_screening.json` and `reports/final_evaluation.json`.

- Arabic false positives are the main open problem. On the hold-out, 33.6% of synthetic and namesake names in Arabic script raise an alert (336 per 1,000), against 7.0% for Latin and 5.3% for Cyrillic. Arabic recall is also among the weakest (69.3% for an Arabic-script query, 61.3% for a Latin query against Arabic entries). Thresholds do not fix it (see above). The next thing to try is a stricter rule for the skeleton on Arabic pairs, for example requiring a second token to match as well.
- Organisation names that are translated, not transliterated, are often missed. An Arabic company name such as the Arabic for "Wafa Humanitarian Organization" shares no spelling with its English name, so no transliteration key can connect them. Looking up a Latin name against Arabic-script entries finds 94.8% of people but only 63.5% of organisations.
- Han, kana and other scripts without a transliteration table (Hebrew, Greek, Thai, Hangul, Georgian and a few more) only match themselves. A Latin query cannot find an entity listed only in those scripts, and the reverse is also true. On the hold-out, 0.0% of such queries found their entity. Same-script matching is weak too: about 14% on the test set.
- Names are scored with rarity weighting, so a shared rare surname counts for far more than a shared first name such as Mohammed. A query that matches only a common token scores low, and so does a true alias that shares only a common token.
- Precision and the false positive rate depend on the mix of negatives. The hold-out mix is synthetic recombined names and namesakes, and it has no negatives in scripts other than Latin, Cyrillic and Arabic. There is no labelled set of real customer names.

## How to run

Developed and tested on Python 3.14. The data pipeline and candidate index use only the standard library, and scoring uses rapidfuzz. Do not install followthemoney: version 4.x depends on PyICU, which needs system ICU libraries to build.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # rapidfuzz for scoring, pytest and ruff for development
python src/download_data.py         # about 350 MB, verifies SHA1, skips if already current
python src/profile_data.py          # writes reports/stage1_profile.json
python src/profile_normaliser.py    # normalisation report, reports/stage2_normaliser.json
python src/evaluate_candidates.py   # candidate recall, reports/stage2_candidates.json
python src/evaluate_scoring.py      # precision, recall, false positives, reports/stage2_scoring.json
python src/evaluate_screening.py    # tuning on dev, test report, reports/stage2_screening.json
python src/download_pairs.py        # the Pairs training data, about 390 MiB, verified
python src/profile_pairs.py         # reports/pairs_profile.json
python src/build_pair_splits.py     # frozen train/dev/test/hold-out split in splits/pairs/
pytest                              # tests run on a small fixture, no download needed
ruff check .
```

`src/evaluate_final.py` scored the frozen hold-out once and writes `reports/final_evaluation.json`. It checks the hold-out checksum and the frozen config first and will not run a second time. The other scripts can be re-run.

`download_data.py --force` downloads again even if the saved version is current. It also writes `data/sanctions/VERSION` with the dataset version and the SHA1 of the verified file. The profile report records the SHA1 of the file it read, so a later run can confirm it used the same data.

## Project structure

```
src/
  download_data.py       download and verify the OpenSanctions entities file
  profile_data.py        stream the file and write the profile report
  ftm_props.json         FollowTheMoney identifier and country property map
  normalise.py           transliterate Cyrillic and Arabic, strip accents and punctuation
  candidates.py          inverted index over folded tokens, consonant skeletons and joined tokens
  scoring.py             rarity-weighted fuzzy name score adjusted by birth date and country
  screening.py           screen(): the callable interface, per-script thresholds, explanations
  screen_cli.py          command line wrapper over screen()
  screening_config.json  frozen scoring settings, global threshold, per-script thresholds (off)
  entities.py            load named entities from the OpenSanctions file
  profile_normaliser.py, evaluate_candidates.py, evaluate_scoring.py, evaluate_screening.py   measurement scripts
  evaluate_final.py      the one-time final evaluation on the frozen hold-out
  download_pairs.py, pairs.py, profile_pairs.py, build_pair_splits.py   the Pairs training data and its splits
tests/
  test_*.py              unit tests on small hand-built data, one file per module
reports/
  stage1_profile.json    profile of the downloaded data
  stage2_*.json          candidate, scoring and screening evaluations used to build it
  final_evaluation.json  the final hold-out evaluation
  pairs_profile.json, pairs_splits.json   the Pairs data profile and split statistics
holdout/
  final_holdout.json     frozen final hold-out queries of the hand-tuned matcher, scored once
splits/pairs/            frozen split of the Pairs data: assignment, manifest with checksums
data/                    downloaded data (not committed)
reference/yente/         yente source for study (not committed)
.github/workflows/ci.yml lint and tests on every push and pull request
```

`src/ftm_props.json` was generated from the followthemoney 4.11.0 schema files, so the profiler does not need that library installed.
