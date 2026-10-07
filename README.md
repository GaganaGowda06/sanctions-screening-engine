# Sanctions Screening Engine

A two-stage sanctions screening engine built on OpenSanctions data: a candidate retrieval step, then a scoring step, wrapped in an evaluation, monitoring and fairness layer. The evaluation layer is the main deliverable. It measures recall and precision and slices them by the script a name is written in.

This is a portfolio project for a graduate data science application in an economic crime team. yente, the open-source OpenSanctions matching API, is the engine I study and aim to improve on. It is kept for reference and is not part of this repository.

Status: Stage 1 (setup and data) is complete. Stage 2 (baseline matcher: normalisation, candidate generation, scoring, and a callable screening interface) is built, tuned and finally evaluated on a frozen hold-out. A logistic regression matcher has been trained on the OpenSanctions Pairs dataset. It beats the hand-tuned engine at deduplicating pairs but not at screening, so screening still uses the hand-tuned engine. The fuller evaluation harness, monitoring and audit log come in later stages.

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

The matcher so far is hand tuned. The next phase trains a logistic regression over explicit features, the approach nomenklatura's RegressionV1 takes, on the OpenSanctions Pairs dataset: 755,540 pairs of entities that analysts labelled as the same entity (positive) or different entities (negative). Nothing has been trained yet; this section records the data, how it was cleaned and split, and the features built from it.

Licence: non-commercial use only (CC-BY-NC-4.0 on the Hugging Face dataset card, "non-commercial use" on the OpenSanctions page). The paper is licensed separately and that licence does not apply to the data. Cite: Smith, Sesodia, Lindenberg and Schroeder de Witt (2026), OpenSanctions Pairs: Large-Scale Entity Matching with LLMs, arXiv:2603.11051, and OpenSanctions, Pairs: Cross-Referencing Training Data, snapshot 2025-12-09 (https://www.opensanctions.org/docs/opensource/pairs/). The file (390 MiB compressed) is downloaded with `src/download_pairs.py`, which checks the size, the MD5 published by OpenSanctions and the SHA-256 published by Hugging Face. It is not stored in this repository.

What is in it (full output in `reports/pairs_profile.json`):

- 581,149 positive and 174,391 negative pairs (76.9% positive). The file is ordered by source, not shuffled: the share of positives runs from 1.9% in the last tenth of the file to 100% in the middle. Never take a slice of it as a sample.
- 280,590 pairs (37.1%) label relationship and role records (Occupancy, Succession, Position, Ownership, Family and others), not named things. The 474,950 pairs of name-bearing entities (persons, companies, organisations, legal entities, vessels and a few others) are 63.8% positive, and those are the ones that matter for name matching.
- Among those entity pairs, 36.4% have Latin names only, 42.1% have non-Latin names on both sides and 21.4% on one side. Cyrillic appears on either side in 217,497 pairs, Arabic in 113,008 and other scripts in 159,930.
- Both sides of the entity pairs carry a birth date in 216,281 (45.5%), a country in 412,246 (86.8%) and an identifier in 296,430 (62.4%, or 38.1% leaving out programId). All three together: 36,610 (7.7%). The attributes are strongly tied to the labels: when both sides have a birth year and they share one, 96.6% of the pairs are positive, and 1.2% when they conflict.
- An entity id does not identify a fixed record: of the ids that appear in more than one pair, 74% carry different properties in different pairs. Everything here is therefore computed from each pair's own copy of the entities, never cached by id.
- 8,280 pairs appear more than once and 14 of those have conflicting labels. One connected group of entities holds 35.2% of all pairs, and one entity appears in 376 pairs.

### Cleaning and splits

Decisions: train and evaluate on entity pairs only, drop the 14 pairs with conflicting labels (all 228 of their lines), and keep the first line of each duplicate. `src/clean_pairs.py` applies this as a frozen mask over the split, which it does not change: 280,590 relationship records, 29,378 duplicate lines and 228 conflicting lines are dropped, leaving 445,344 pairs.

The earlier hold-out belonged to the hand-tuned matcher and is spent. A fresh split was made by `src/build_pair_splits.py`: 70% train, 10% dev, 10% test and 10% final hold-out, by a hash of a group key and a fixed seed, so file order plays no part. Pairs in a small connected group (up to 100 pairs) are kept together with their whole group. Pairs in the giant group are assigned one at a time, which keeps duplicates together but cannot keep entities apart.

| Split | Pairs kept | Positive | Share an entity with a train pair |
|---|---|---|---|
| train | 310,762 | 61.6% | not applicable |
| dev | 44,719 | 61.5% | 55.3% |
| test | 44,482 | 61.5% | 55.2% |
| final hold-out | 45,381 | 61.4% | not computed (not touched) |

About 55% of the dev and test pairs share an entity (by id) with a train pair, all of them from the large connected groups. After cleaning that is more than the third measured before it, because the relationship records and duplicates that were removed were mostly entity-disjoint. Every metric on this data is reported separately for the two kinds. They differ a lot: among dev pairs that share an entity with train 35% are matches, and among those that do not 94% are, with only 1,286 non-matches, so the second kind says little about false positives.

The assignment and the cleaning mask are frozen in `splits/pairs/` with the dataset checksum, a checksum of the pair ids in file order, and a checksum of the hold-out membership (`5439a1090adc...`). `pairs.iter_split()` refuses the hold-out unless it is asked for explicitly, and only the final evaluation may do that.

### Features

`src/build_pair_features.py` builds one row per cleaned pair for train, dev and test (the hold-out is not featurised): five name scores in [0, 1] and three attribute states. The name scores take the best-matching pair of names of the two entities (at most 30 names each): the current engine's blended name score, the consonant-skeleton score, the rarity-weighted token score (weights from the train entities only), a Double Metaphone score, and a score after Arabic folding (articles, abd- compounds, bin/ibn, Mohammed and other name variants). Birth date is agree, close (a year apart), conflict or missing; country and identifier are agree, conflict or missing, with identifiers compared property by property. Tables are written to `data/pairs/features/` (not committed) and are identical from run to run.

How well each feature separates matches from non-matches on dev, as the area under the ROC curve (the attribute states are scored by their match rate in train):

| Feature | All dev pairs | Share an entity with train | No entity in train |
|---|---|---|---|
| Name score (current engine) | 0.953 | 0.977 | 0.820 |
| Skeleton score | 0.948 | 0.973 | 0.818 |
| Rarity-weighted token score | 0.954 | 0.978 | 0.821 |
| Phonetic (Double Metaphone) score | 0.948 | 0.973 | 0.817 |
| Arabic-normalised score | 0.950 | 0.974 | 0.818 |
| Birth date state | 0.815 | 0.806 | 0.754 |
| Country state | 0.679 | 0.729 | 0.525 |
| Identifier state | 0.764 | 0.780 | 0.695 |

The five name scores are close to one another and highly correlated (0.93 to 0.99 on dev), so the phonetic and Arabic scores are unlikely to add much on their own. Birth date is the strongest attribute: a conflict means 0.9% are matches. Missing values carry signal too (73% of pairs with no comparable identifier are matches, against 8% when identifiers conflict), which reflects which sources the pairs come from.

### The screening hold-out

The Pairs data measures whether two records are the same entity. Screening is a different question: given a customer name, does the engine alert on the right listed entity, and how often on innocent names. A second screening hold-out was built from our own list for the trained matcher, with the same design as the first (cross-script and variant positives, synthetic and near-miss negatives, plus negatives and same-script positives in other scripts): 4,800 queries on 2,900 entities used in no earlier set, frozen in `holdout/screening_holdout_v2.json` (SHA-256 `e5a9b7af6b94...`). It has not been scored.

## The trained matcher

A logistic regression over the eight features, trained on the Pairs train split (`src/train_matcher.py`), as an alternative to the hand-tuned blend. Rarity weights come from one table built from the 69,637 named targets of the deployment list (131,898 name tokens, `models/idf_manifest.json`), the same table for training and serving. Training uses scikit-learn; serving needs only the coefficients in `models/pairs_lr_v1.json` and `src/matcher_model.py`.

Model. The regularisation strength was chosen on dev: dev log loss is flat from C = 0.1 up, so the rule is the most regularised C within 0.0005 of the best, which is C = 1. AUC is 0.985 on train, 0.986 on dev and 0.986 on test, so it is not overfitting. Each name feature's AUC on dev, overall and by script slice:

| Feature | All dev pairs | Latin | Cyrillic | Arabic | other | cross-script |
|---|---|---|---|---|---|---|
| Name score (current engine) | 0.953 | 0.934 | 0.979 | 0.989 | 0.981 | 0.940 |
| Skeleton | 0.948 | 0.929 | 0.976 | 0.983 | 0.979 | 0.936 |
| Rarity-weighted tokens | 0.954 | 0.934 | 0.980 | 0.989 | 0.981 | 0.943 |
| Phonetic (Double Metaphone) | 0.948 | 0.929 | 0.976 | 0.984 | 0.979 | 0.937 |
| Arabic-normalised | 0.950 | 0.931 | 0.977 | 0.987 | 0.982 | 0.937 |
| Pairs in the slice | 44,719 | 16,876 | 11,805 | 2,989 | 2,213 | 10,836 |

Ablation. Each feature dropped in turn, the model refitted, change in dev AUC:

| Feature dropped | All | Latin | Cyrillic | Arabic | other | cross-script |
|---|---|---|---|---|---|---|
| name score | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0001 | -0.0001 |
| skeleton | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| rarity-weighted | +0.0001 | +0.0002 | +0.0001 | +0.0001 | +0.0000 | -0.0002 |
| phonetic | +0.0000 | +0.0001 | +0.0001 | +0.0002 | +0.0007 | -0.0003 |
| Arabic-normalised | +0.0000 | +0.0001 | +0.0000 | +0.0001 | +0.0000 | -0.0001 |
| birth date | -0.0113 | -0.0108 | -0.0077 | -0.0062 | -0.0167 | -0.0129 |
| country | -0.0027 | -0.0014 | +0.0030 | +0.0007 | +0.0014 | -0.0113 |
| identifier | -0.0094 | -0.0227 | -0.0025 | -0.0007 | -0.0014 | -0.0036 |

Dropping any one name feature moves dev AUC by no more than 0.0007 in any script slice (the largest loss is 0.0003, for the phonetic score in the cross-script slice, and in the other slice dropping the phonetic score even raises AUC by 0.0007): the five name scores are highly correlated (0.93 to 0.99) and the Arabic-normalised and phonetic scores add nothing the others do not. Birth date, identifier and country each add more than any name feature.

Coefficient stability, from 100 bootstrap resamples of the train pairs. The attribute coefficients are tight (standard deviation 0.02 to 0.09). The name coefficients are not: they are large and offset one another (8.8 on the name score, -5.5 on the Arabic score, -3.2 on the phonetic score) and the skeleton coefficient changes sign in 38% of resamples. Their sum is stable (4.99, 95% interval 4.91 to 5.12), so what the model learned about names is stable and how it splits that weight is not. The bootstrap resamples pairs, and pairs that share entities are not independent, so these spreads are somewhat too narrow.

Leakage check. On dev the model's AUC is 0.993 for pairs that share an entity with train and 0.920 for pairs that do not (95% interval 0.912 to 0.928); on test 0.993 and 0.924. The gap is 0.073 for the model and 0.157 for the raw name score, so the model gains nothing extra on pairs whose entities it has seen: the gap comes from what the two kinds of pair are (the unseen kind is 94% matches with few hard non-matches), not from memorising entities.

On the Pairs test set the trained model is better than the hand-tuned engine applied to two entities (AUC 0.986 against 0.956). Recall / precision / false positive rate:

| Pairs test | Hand-tuned (threshold 0.86) | Trained (threshold 0.76) |
|---|---|---|
| All pairs | 95.6% / 95.3% / 7.5% | 94.6% / 97.2% / 4.4% |
| Latin | 94.7% / 94.6% / 10.8% | 93.3% / 97.0% / 5.8% |
| Cyrillic | 98.5% / 93.7% / 7.5% | 96.2% / 96.8% / 3.6% |
| Arabic | 99.7% / 94.7% / 4.7% | 99.5% / 95.7% / 3.8% |
| other | 99.1% / 96.2% / 5.9% | 98.8% / 96.9% / 4.7% |
| cross-script | 93.0% / 98.2% / 3.7% | 93.6% / 98.3% / 3.4% |
| Share an entity with train | 97.1% / 90.3% / 5.8% | 97.4% / 95.0% / 2.8% |
| No entity in train | 94.9% / 97.9% / 28.4% | 93.3% / 98.3% / 23.0% |

On screening it is much worse. The same queries as the hand-tuned engine's test, the same candidates, the trained model scoring every candidate:

| Screening test, all scripts | Recall | Precision | False positive rate | Top-1 |
|---|---|---|---|---|
| Hand-tuned engine (threshold 0.86) | 73.0% | 80.2% | 10.8% | 77.9% |
| Trained, threshold 0.96 (same dev false positive rate) | 32.9% | 58.8% | 9.8% | 56.6% |
| Trained, threshold 0.02 (best dev F1) | 91.8% | 46.3% | 98.7% | 56.6% |

| By script of the query | Hand-tuned recall / precision / FPR | Trained (same dev FPR) recall / precision / FPR |
|---|---|---|
| Latin | 71.4% / 86.5% / 4.7% | 35.3% / 68.0% / 7.7% |
| Cyrillic | 88.0% / 88.9% / 5.2% | 44.3% / 79.6% / 5.9% |
| Arabic | 65.0% / 53.4% / 37.6% | 10.0% / 13.4% / 22.3% |
| other | n/a / n/a / 0.2% | n/a / n/a / 2.0% |

Its best F1 on the screening dev queries is at a threshold of 0.02, which alerts on nearly everything, so that row is not a usable operating point.

The cause is in the attribute weights, not the names. In the Pairs data a missing attribute goes with matches (pairs with no country on either side are 85% matches, pairs whose countries agree 66%), because of which sources the pairs come from. The model learned that, and gives an agreeing country a negative weight (-1.76). A customer query is not like that. Swapping weights in the same model, on 600 dev positives and 600 dev negatives (an exploratory comparison: the threshold is chosen on that sample):

| Variant of the trained model | Recall at false positive rate up to 10% | Top-1 |
|---|---|---|
| As trained | 44.7% | 76.7% |
| Country agreement weight set to zero | 67.3% | 79.5% |
| Names only (all attribute weights zero) | 35.8% | 72.5% |
| Attribute weights set by hand | 72.2% | 78.7% |

So the name side of the model is as good as the hand-tuned engine's, and the attribute weights learned from this dataset do not transfer to screening. The trained model is therefore not used for screening. Fixing it means training the attribute weights on something that has not got this artefact, for example restricting them to the pairs where both sides carry the attribute, or constraining agreement to help and conflict to hurt.

## Known limitations

These come from the measurements in `reports/stage2_candidates.json`, `reports/stage2_screening.json` and `reports/final_evaluation.json`.

- Arabic false positives are the main open problem. On the hold-out, 33.6% of synthetic and namesake names in Arabic script raise an alert (336 per 1,000), against 7.0% for Latin and 5.3% for Cyrillic. Arabic recall is also among the weakest (69.3% for an Arabic-script query, 61.3% for a Latin query against Arabic entries). Thresholds do not fix it (see above). The next thing to try is a stricter rule for the skeleton on Arabic pairs, for example requiring a second token to match as well.
- Organisation names that are translated, not transliterated, are often missed. An Arabic company name such as the Arabic for "Wafa Humanitarian Organization" shares no spelling with its English name, so no transliteration key can connect them. Looking up a Latin name against Arabic-script entries finds 94.8% of people but only 63.5% of organisations.
- Han, kana and other scripts without a transliteration table (Hebrew, Greek, Thai, Hangul, Georgian and a few more) only match themselves. A Latin query cannot find an entity listed only in those scripts, and the reverse is also true. On the hold-out, 0.0% of such queries found their entity. Same-script matching is weak too: about 14% on the test set.
- Names are scored with rarity weighting, so a shared rare surname counts for far more than a shared first name such as Mohammed. A query that matches only a common token scores low, and so does a true alias that shares only a common token.
- Precision and the false positive rate depend on the mix of negatives. The hold-out mix is synthetic recombined names and namesakes, and it has no negatives in scripts other than Latin, Cyrillic and Arabic. There is no labelled set of real customer names.

## How to run

Developed and tested on Python 3.14. The data pipeline and candidate index use only the standard library, scoring uses rapidfuzz, the phonetic feature uses Metaphone, and training uses scikit-learn. Do not install followthemoney: version 4.x depends on PyICU, which needs system ICU libraries to build.

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
python src/clean_pairs.py           # frozen cleaning mask: entity pairs, no conflicts, no duplicates
python src/build_pair_features.py   # feature tables in data/pairs/features/, reports/pairs_features.json
python src/build_screening_holdout.py   # the second screening hold-out, frozen once
python src/build_idf.py             # the deployment rarity table, models/idf_manifest.json
python src/train_matcher.py         # train, ablate, bootstrap; models/pairs_lr_v1.json
python src/compare_matchers.py      # trained against hand-tuned: reports/pairs_comparison.json
python src/diagnose_screening.py    # why the trained model does worse at screening
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
  download_pairs.py, pairs.py, profile_pairs.py, build_pair_splits.py, clean_pairs.py   the Pairs data, splits, cleaning
  pair_features.py, arabic_names.py, build_pair_features.py   the eight pairwise features and the table builder
  build_screening_holdout.py   the second screening hold-out
  idf_table.py, build_idf.py   the deployment rarity table used for training and serving
  matcher_model.py, train_matcher.py   the trained matcher: serving (standard library) and training (scikit-learn)
  compare_matchers.py, diagnose_screening.py   trained against hand-tuned, and why it falls short at screening
tests/
  test_*.py              unit tests on small hand-built data, one file per module
reports/
  stage1_profile.json    profile of the downloaded data
  stage2_*.json          candidate, scoring and screening evaluations used to build it
  final_evaluation.json  the final hold-out evaluation
  pairs_profile.json, pairs_splits.json, pairs_features.json   the Pairs data profile, splits and features
holdout/
  final_holdout.json     frozen final hold-out queries of the hand-tuned matcher, scored once
  screening_holdout_v2.json   the second screening hold-out, for the trained matcher, not yet scored
models/                  the trained coefficients and the rarity table manifest
splits/pairs/            frozen split of the Pairs data: assignment, cleaning mask, manifests with checksums
data/                    downloaded data (not committed)
reference/yente/         yente source for study (not committed)
.github/workflows/ci.yml lint and tests on every push and pull request
```

`src/ftm_props.json` was generated from the followthemoney 4.11.0 schema files, so the profiler does not need that library installed.
