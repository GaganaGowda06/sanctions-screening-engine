# Sanctions Screening Engine: full project report

This is the long form of the project README, kept whole. The short README at the repository root links here for detail. Paths in backticks are relative to the repository root.

A two-stage sanctions screening engine built on OpenSanctions data: a candidate retrieval step, then a scoring step, wrapped in an evaluation, monitoring and fairness layer. The evaluation layer is the main deliverable. It measures recall and precision and slices them by the script a name is written in.

This is a portfolio project for a graduate data science application in an economic crime team. yente, the open-source OpenSanctions matching API, is the engine I study and aim to improve on. It is kept for reference and is not part of this repository.

Status: Stage 1 (setup and data) is complete. Stage 2 (baseline matcher: normalisation, candidate generation, scoring, and a callable screening interface) is built, tuned and finally evaluated on a frozen hold-out. A logistic regression matcher has been trained on the OpenSanctions Pairs dataset. It beats the hand-tuned engine at deduplicating pairs but not at screening, even with the attribute weights constrained, so screening stays on the hand-tuned engine and the trained model is kept for deduplication only. All three hold-outs (the first screening hold-out, the second, and the Pairs hold-out) are now spent. The fuller evaluation harness, monitoring and audit log come in later stages.

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
python src/screen_cli.py init --audit-dir /path/to/audit
python src/screen_cli.py screen "Vladimir Putin" --birth-date 1952-10-07 --country ru --audit-dir /path/to/audit
python src/screen_cli.py screen "محمد علي" --json --limit 5 --customer-ref cust-123 --audit-dir /path/to/audit
python src/screen_cli.py screen --batch names.jsonl --audit-dir /path/to/audit
python src/screen_cli.py replay REQUEST_ID --audit-dir /path/to/audit
```

Every mode is an explicit subcommand (`init`, `screen`, `replay`); a bare name is a usage error (exit 2), so a customer called `init` or `replay` is screened like anyone else with `screen init`. The command line is the production entry point, so it is audited: each name's record is written before its decision is printed, and if a record cannot be written the command prints no decision and exits non-zero. The audit directory comes from `--audit-dir` or `$SCREENING_AUDIT_DIR`; there is no default, and the command refuses to start without one. `init` creates the store once and refuses a directory that is not empty; screening never creates it, and a missing or wiped store stops screening with `log_missing` rather than starting a new one. `replay REQUEST_ID` rebuilds a recorded decision from its header, payload and run context and exits 0 only if the result is identical; it refuses, with a field-level diff, if the engine's run context differs from the record's. `screen()` itself is unchanged and is what the evaluation scripts use, so evaluation runs write no audit records.

### A local web page

```
venv/bin/streamlit run app/streamlit_app.py
```

`app/streamlit_app.py` is a single local page (no deployment): type a name, and optionally a birth date and a country, and it shows the decision, the top score against the threshold of 0.86, the five best candidates (listed name, entity type, birth date, country, source list) and, in plain words, the score components behind the top match. When the birth date or the country is missing it says recall is weaker without them, with the measured figures from `docs/limitations_evidence.md`. The engine is built once and cached. The page writes **no audit record** and imports nothing from the audit modules, so it cannot touch any audit store; it is for looking at how the matcher behaves, not for decisions that must be kept and replayed, which go through `src/screen_cli.py`.

## Stage 4: audit log and monitoring

The full write-up is in [docs/stage4_audit_and_monitoring.md](stage4_audit_and_monitoring.md). Stage 4 adds no matching logic; scores and decisions are unchanged. It answers the question an auditor would ask: why was this name flagged or cleared on this date, under which list, code, configuration and threshold. It also notices when the matcher's behaviour moves, and checks that a new list snapshot does not make it worse.

- Header and payload are split. The header holds scores, the decision and a salted digest and nothing taken from the name. The payload holds the raw input and can be deleted alone.
- Headers form a hash chain with a checkpoint. This is tamper-evident, not tamper-proof.
- Replay reruns a decision from its payload under the same run context, and refuses if the context differs.
- Recovery repairs a torn last line or one orphan payload after a crash, and refuses anything that looks like tampering.
- Purge deletes payloads five years after the relationship ended, with a legal hold that overrides it.
- A store is only ever created by an explicit `init`; screening never makes one.

Monitoring groups decisions by run context and suppresses small cells. Drift compares a window with a frozen baseline using PSI limits taken from a pooled two-sample null, with a Bonferroni correction on the overall status. On 50 seeded stable batches the overall status was warn or worse in 1 (2%) and alert in none, against limits of 10% and 2%. The batches overlap, so that rate is less certain than it looks.

The list-update gate reruns the spent third hold-out on a new snapshot. In what follows, positives are all 2,400 positive queries of the hold-out and headline positives the 2,000 used for the headline F1. The forward run (20261006074704-hib to 20261008075007-crk) kept 2,398 of the 2,400 positives, left F1 at 0.8010 and the Arabic false-alert burden at 0.2257, and passed. The REVERSE-TIME CHURN TEST against the 2026-07-01 snapshot kept 2,255 and remapped 53 of the 2,400 positives, dropped 92, moved F1 from 0.8077 to 0.8070 (95% interval -0.0051 to +0.0038), left the burden at 0.2257, and passed. The 0.8010 is the current engine on the third hold-out; it is not comparable with the 0.777 of the second hold-out, which is the Stage 2 engine.

### What this does not show

- Monitoring is demonstrated on constructed batches, not production traffic. The baseline and the stable, mix-shift and degraded batches are samples of a spent hold-out, so how often a live deployment would warn or alert is unknown.
- There are no analyst dispositions, so live precision and recall cannot be measured, and neither can the false-alert burden (the 22.6% Arabic result of Stage 3): live traffic has no labels saying which alerts were wrong. A change in an alert rate is a reason to look, not a measurement of harm.
- The deduplication model is outside the decision path and is not monitored.

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

The assignment and the cleaning mask are frozen in `splits/pairs/` with the dataset checksum, a checksum of the pair ids in file order, and a checksum of the hold-out membership (`5439a1090adc...`). `pairs.iter_split()` refuses the hold-out unless it is asked for explicitly, and only the final evaluation (`src/evaluate_pairs_final.py`) may do that. It did, once; the hold-out is spent.

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

The Pairs data measures whether two records are the same entity. Screening is a different question: given a customer name, does the engine alert on the right listed entity, and how often on innocent names. A second screening hold-out was built from our own list for the trained matcher, with the same design as the first (cross-script and variant positives, synthetic and near-miss negatives, plus negatives and same-script positives in other scripts): 4,800 queries on 2,900 entities used in no earlier set, frozen in `holdout/screening_holdout_v2.json` (SHA-256 `e5a9b7af6b94...`). It was scored once, on the hand-tuned engine, and is spent (see the final results of the second hold-outs below).

## The trained matcher

Outcome. For screening this is a negative result, and screening stays on the hand-tuned engine. For deduplication it is a win, and the constrained model (`models/pairs_lr_v2.json`) is kept as the deduplication matcher only. There is no hybrid. The model fails at screening for two reasons, in order. First, the Pairs data contains a missingness artefact: a pair with an attribute missing on either side is far more likely to be a match, because of which sources the pairs come from, so an unconstrained model learned that an agreeing country lowers the match probability. Constraining the attribute weights removed that and recovered most of the gap. Second, what remains is Arabic. The hand-tuned engine gets its Arabic recall from a screening-specific rule, the vowel-stripped skeleton standing in for the name score, and the Pairs data gives a model no reason to learn it (keeping the skeleton, phonetic and Arabic-folded scores as features made screening worse, not better). The sections below give the evidence in the order it was found.

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

So the name side of the model is as good as the hand-tuned engine's, and the attribute weights learned from this dataset do not transfer to screening. The second attempt below follows the obvious fix.

### Second attempt: two name features, sign-constrained attributes

The attribute weights were the problem, so the model was refitted with only the current name score and the rarity-weighted score (the other three were redundant on the Pairs data) and with the attribute weights constrained: every agree weight at least zero, every conflict weight at most zero, a missing attribute fixed at zero effect, and the one-year-apart birth state ("close") held at most zero because in the data it behaves like a conflict. The fit is L2 regularised with the same objective as scikit-learn, minimised with bounded L-BFGS (`src/train_constrained.py`), C = 1 tuned on Pairs dev. The alert threshold is chosen on the screening dev queries, not on Pairs.

The constraint that binds is country agreement: unconstrained it is -1.83, constrained it is 0. Pairs test AUC is 0.9848, against 0.9858 for the same features unconstrained and 0.9559 for the hand-tuned engine, so the constraints cost 0.001 of AUC. The two name weights offset each other (-6.4 on the name score, 11.6 on the rarity-weighted score), which amounts to a negative weight on the plain character similarity; they were left unconstrained.

On the screening test queries, recall / precision / false positive rate / top-1:

| Screening test | Hand-tuned engine (threshold 0.86) | Constrained model (threshold 0.81, best F1 on screening dev) |
|---|---|---|
| All scripts | 73.0% / 80.2% / 10.8% / 77.9% | 61.8% / 69.5% / 14.1% / 63.3% |
| Latin queries | 71.4% / 86.5% / 4.7% / 76.6% | 65.2% / 76.2% / 12.3% / 66.3% |
| Cyrillic queries | 88.0% / 88.9% / 5.2% / 91.3% | 85.3% / 80.5% / 10.9% / 85.3% |
| Arabic queries | 65.0% / 53.4% / 37.6% / 70.3% | 22.0% / 25.4% / 24.2% / 27.7% |

It is far better than the first model (at the hand-tuned engine's dev false positive rate, overall recall rises from 33% to 60%), better than the hand-tuned engine on Latin queries against Cyrillic entries (77.3% against 68.0% recall), but still behind overall, and it fails on Arabic: recall 22.0% against 65.0%. The hand-tuned engine gets its Arabic recall from a screening-specific rule, the skeleton standing in for the name score on Arabic pairs, and the Pairs data gives a model no reason to learn it. Keeping all five name scores does not help: with the same constraints the five-score model is worse at the same dev false positive rate (overall recall 49.1% against 60.5%, Arabic 7.3% against 21.7%), and its best-F1 threshold on screening dev is 0.02, which alerts on nearly everything (`reports/pairs_comparison_v2_five_names.json`).

The trained model is not used for screening. It is better than the hand-tuned engine at telling whether two records are the same entity (see the second hold-outs below), and not at screening a name against the list.

## Final results of the second hold-outs

Both hold-outs of the trained-matcher work were scored once, after the code that scores them was committed, and are now spent. Neither was used for tuning.

Screening, on the second screening hold-out (4,800 queries on 2,900 entities used in no earlier set, SHA-256 `e5a9b7af...`), with the frozen hand-tuned engine: one global threshold of 0.86, blend 0.5, skeleton lift 0.2, floor 0.6, unchanged. The headline figures use the same design as the first hold-out, 2,000 positive and 2,000 negative queries; `reports/screening_holdout_v2_evaluation.json` has the rest.

| Figure | Hold-out 2 | 95% interval | Earlier hold-out | Test set |
|---|---|---|---|---|
| Recall | 73.7% | 71.7 to 75.6 | 74.7% | 73.0% |
| Precision | 82.2% | 80.4 to 83.9 | 80.9% | 80.3% |
| F1 | 0.777 | 0.761 to 0.793 | 0.776 | 0.764 |
| Top-1 accuracy | 79.0% | 77.2 to 80.8 | 79.0% | 77.9% |
| False positive rate | 9.3% | 8.2 to 10.7 | 11.3% | 10.8% |

F1 is 0.777 against 0.776 on the earlier hold-out, and the intervals overlap for all five figures, so the hand-tuned engine's performance reproduced on fresh queries and entities. The false positive rate is lower than on the test set (9.3% against 10.8%), so there is no sign of overfitting to the test set. By script of the query:

| Script of the query | Recall | 95% interval | False positive rate | 95% interval | Alerts per 1,000 clean names |
|---|---|---|---|---|---|
| Latin | 72.2% | 69.8 to 74.5 | 6.3% | 5.1 to 7.9 | 64 |
| Cyrillic | 89.3% | 85.3 to 92.3 | 4.2% | 2.8 to 6.4 | 42 |
| Arabic | 65.0% | 59.4 to 70.2 | 26.3% | 22.0 to 31.1 | 263 |

Arabic remains the weak spot (recall 65%, 263 alerts per 1,000 clean Arabic-script names), as in the first hold-out. Han, kana and other scripts against Latin again found 0% of their entities.

Deduplication, on the Pairs final hold-out (45,381 cleaned entity pairs, 27,842 of them matches, membership checksum `5439a109...`, checked before scoring). The constrained model is at the threshold chosen on Pairs dev, and the hand-tuned engine's rule, applied to two entities, at its frozen threshold:

| Pairs hold-out | AUC (95% interval) | Recall | Precision | False positive rate |
|---|---|---|---|---|
| **All pairs** | | | | |
| Constrained model (threshold 0.74) | 0.984 (0.983 to 0.985) | 94.1% | 97.5% | 3.9% |
| Hand-tuned engine (threshold 0.86) | 0.957 (0.955 to 0.959) | 95.4% | 95.4% | 7.3% |
| **Pairs sharing an entity with train** | | | | |
| Constrained model (threshold 0.74) | 0.993 (0.992 to 0.994) | 96.8% | 95.3% | 2.6% |
| Hand-tuned engine (threshold 0.86) | 0.979 (0.978 to 0.981) | 97.0% | 90.4% | 5.6% |
| **Pairs with no entity in train** | | | | |
| Constrained model (threshold 0.74) | 0.942 (0.936 to 0.947) | 92.9% | 98.5% | 17.8% |
| Hand-tuned engine (threshold 0.86) | 0.875 (0.864 to 0.885) | 94.6% | 98.0% | 25.4% |

The constrained model separates matches from non-matches much better than the hand-tuned rule (AUC 0.984 against 0.957) with a lower false positive rate (3.9% against 7.3%), at slightly lower recall (94.1% against 95.4%). Its test-set AUC was 0.985, so it generalised. The two kinds of pair differ as before: pairs that share an entity with train are easier (0.993 AUC) than those that do not (0.942), and the model's lead over the hand-tuned rule holds in both.

## Stage 3: Arabic alert disparity

### The fairness metric

A customer with an Arabic name should not carry more false alerts than one with a Latin or Cyrillic name. The metric is the false-alert rate by script: the share of clean names (names that match no listed entity) that raise an alert, split by the script of the query. Every alert on a clean name lands on a real customer, so this is the burden each group carries. It is reported two ways, because they can disagree: the ratio of Arabic to Latin and Cyrillic (how many times more often), and the absolute gap in percentage points (how many more customers per hundred). It is measured on synthetic recombined names, which is the like-for-like comparison, because the near-miss namesakes in the hold-out are almost all Latin or Cyrillic. Intervals are 95%: Wilson for each rate, bootstrap for ratios and gaps.

A third screening hold-out was built for this stage (`holdout/screening_holdout_v3.json`, SHA-256 `963b278b...`): same design as the second, 2,900 entities used in no earlier set. All diagnosis and tuning used dev, then test. The hold-out was read twice, once for the baseline and once for the final engine, and is now spent.

### Baseline and final on the third hold-out

| Figure | Baseline | Final |
|---|---|---|
| Arabic alert rate on clean names | 32.6% (27.9 to 37.7) | 22.6% (18.5 to 27.2) |
| Latin alert rate | 6.5% (5.0 to 8.4) | 5.1% (3.8 to 6.9) |
| Cyrillic alert rate | 6.0% (4.0 to 9.0) | 3.7% (2.2 to 6.3) |
| Gap, Arabic against Latin and Cyrillic pooled | 26.2 points (21.1 to 31.2) | 17.9 points (13.5 to 22.3) |
| Ratio, same comparison | 5.1 (4.0 to 6.7) | 4.8 (3.5 to 6.7) |
| Arabic recall | 67.7% | 72.7% |

Baseline is the Stage 2 engine (`reports/fairness_baseline.json`); final is the same engine with the rarity-weighted skeleton (`reports/fairness_current.json`). Arabic names now carry fewer false alerts and the gap is 8 points smaller, with higher recall. The ratio barely moved, and its intervals overlap. The reason is that the fix helps every script: Latin and Cyrillic rates fell as well, because their skeleton-driven false alerts also rested on common tokens. A ratio only improves if Arabic falls faster than the others, and it did not. The absolute gap is the better guide to the burden, and it did fall. The Arabic rate is still more than four times the others, so the disparity is reduced, not closed.

### The four fixes tried

Each was tried on dev, judged on test, and held to one bar: Arabic false positive rate clearly down, Arabic recall at least 65%, Latin and Cyrillic unchanged within noise. McNemar tests compare each fix with the engine before it, on the same queries.

1. Rarity-weighted skeleton (kept, on by default): the skeleton similarity is a rarity-weighted token overlap instead of a whole-string ratio. Arabic false positives fell (dev 32.6% to 27.6%, test 38.6% to 32.3%) and Arabic recall rose, but it did not clearly pass the bar and did not move the ratio, so it is kept for the gain it gives and no more (`reports/stage3_skeleton_idf.json`).
2. Particle stripping (rejected, `skeleton_particles`): leaving al, el, bin, ibn, bint, abu, umm and one-consonant tokens out of the skeleton raised Arabic false positives (test 32.3% to 43.7%) and Latin ones, because names shrank to one token that then matched in full (`reports/stage3_skeleton_particles.json`).
3. Common-token tier (rejected, `common_df`): holding skeleton matches on common tokens back from a full alert removed 16 of 186 Arabic false alerts and cut Arabic recall to 54% on dev, below the floor (`reports/stage3_common_tier.json`).
4. Orthographic skeleton (rejected, `orthographic`): keeping long vowels and the initial alif cut Arabic false positives to about 10% on dev and test and brought the ratio down (test 5.9 to 2.9), but Arabic recall fell to 33% on dev and 30% on test, because spellings of one name (Khaled and Khalid, Yusuf and Youssef) stopped meeting (`reports/stage3_orthographic.json`).

The rejected fixes are in the code behind flags that are off, so the results can be reproduced.

### Diagnosis

On dev, 186 of 677 clean Arabic-script names alert. 163 of those needed the skeleton to reach the threshold, 156 are full-name collisions (every skeleton token of the query has a partner in the listed name), and 139 rest only on common skeleton tokens such as `b` (abu, bin), `mhmd`, `hmd` and `l` (al, ali). Arabic skeletons are short and common, so unrelated names collide at a similarity of 1.0 whatever the token weights.

Birth date and country do little to separate them. Of the 186 false alerts, the birth date is missing on one side for 143 (66 on the query side, 77 on the entity side) and the country for 142 (86 and 56). Agreeing details pushed only 9 over the threshold, and 177 were over it with neither adjustment. A secondary identifier cannot rescue a screening decision when it is absent from most pairs, which is why the common-token tier cost recall and gained little (`reports/stage3_arabic_diagnosis.json`, `reports/stage3_arabic_secondary.json`).

### Future work

- A spelling-variant table for Arabic names (Khaled and Khalid, Yusuf and Youssef, Mohammed and Muhammad), so that a stricter orthographic skeleton can keep recall. The orthographic result shows that vowel information separates names, and that the cost is in variants that a rule cannot join.
- The parked embedding fallback for Han, kana and organisation translations, which the current transliteration cannot reach (0% found on the hold-out).

## Known limitations

These come from the measurements in `reports/stage2_candidates.json`, `reports/stage2_screening.json` and `reports/final_evaluation.json`.

A later analysis of the matcher's errors, how the hold-out labels were made, precision at realistic prevalence, tuning transfer and robustness to spelling changes is in [docs/limitations_evidence.md](limitations_evidence.md) and [docs/label_provenance.md](label_provenance.md). Its main findings are less favourable than the figures below: about 8% of listed parties tested are close matches that slip under the threshold, the F1 falls to 0.587 when birth date and country are withheld, and at one listed customer in 1,000 only about one alert in 100 would be real. Part 6 of the same document scores the engine against analyst-made labels from OpenSanctions Pairs (CC BY-NC 4.0, attribution to OpenSanctions, non-commercial use only; this project is non-commercial; the data is not in this repository): the external validation figure is F1 0.818 on the pairs with no shared name (interval 0.814 to 0.822, the pairs built most like the hold-outs). The overall F1 of 0.969 over 439,813 pairs is not the figure to quote, because 91% of its positives share an identical name. Vessels, securities and crypto wallets are out of scope: the engine fails on them (vessel pairs with no shared name: recall 0.088) and this validation does not support using it for them.

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
pip install -r requirements.txt     # rapidfuzz for scoring, streamlit for the local page, pytest and ruff for development
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
python src/train_constrained.py     # two name features, sign-constrained attributes; models/pairs_lr_v2.json
python src/compare_matchers.py --model models/pairs_lr_v2.json --report reports/pairs_comparison_v2.json
pytest                              # tests run on a small fixture, no download needed
ruff check .
```

`src/evaluate_final.py` scored the frozen hold-out once and writes `reports/final_evaluation.json`. It checks the hold-out checksum and the frozen config first and will not run a second time. The other scripts can be re-run.

`download_data.py --force` downloads again even if the saved version is current. It also writes `data/sanctions/VERSION` with the dataset version and the SHA1 of the verified file. The profile report records the SHA1 of the file it read, so a later run can confirm it used the same data.

## Project structure

```
app/
  streamlit_app.py       the local web page (no audit, no deployment)
scripts/
  demo.py                end-to-end terminal demo against a throwaway audit store
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
  evaluate_final.py      the one-time final evaluation of a screening hold-out
  evaluate_pairs_final.py   the one-time final evaluation on the Pairs hold-out
  download_pairs.py, pairs.py, profile_pairs.py, build_pair_splits.py, clean_pairs.py   the Pairs data, splits, cleaning
  pair_features.py, arabic_names.py, build_pair_features.py   the eight pairwise features and the table builder
  build_screening_holdout.py   the second screening hold-out
  idf_table.py, build_idf.py   the deployment rarity table used for training and serving
  matcher_model.py, train_matcher.py   the trained matcher: serving (standard library) and training (scikit-learn)
  compare_matchers.py, diagnose_screening.py   trained against hand-tuned, and why it falls short at screening
  train_constrained.py   the second attempt: two name features, sign-constrained attribute weights
tests/
  test_*.py              unit tests on small hand-built data, one file per module
reports/
  stage1_profile.json    profile of the downloaded data
  stage2_*.json          candidate, scoring and screening evaluations used to build it
  final_evaluation.json  the first screening hold-out evaluation
  screening_holdout_v2_evaluation.json, pairs_final_evaluation.json   the second hold-outs, scored once
  pairs_profile.json, pairs_splits.json, pairs_features.json   the Pairs data profile, splits and features
holdout/
  final_holdout.json     the first screening hold-out, scored once, spent
  screening_holdout_v2.json   the second screening hold-out, scored once on the hand-tuned engine, spent
models/                  the trained coefficients and the rarity table manifest
splits/pairs/            frozen split of the Pairs data: assignment, cleaning mask, manifests with checksums
data/                    downloaded data (not committed)
reference/yente/         yente source for study (not committed)
.github/workflows/ci.yml lint and tests on every push and pull request
```

`src/ftm_props.json` was generated from the followthemoney 4.11.0 schema files, so the profiler does not need that library installed.
