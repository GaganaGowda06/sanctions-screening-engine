# Sanctions Screening Engine

A two-stage sanctions screening engine built on OpenSanctions data: a candidate retrieval step, then a scoring step, wrapped in an evaluation, monitoring and fairness layer. The evaluation layer is the main deliverable. It measures recall and precision and slices them by the script a name is written in.

This is a portfolio project for a graduate data science application in an economic crime team. yente, the open-source OpenSanctions matching API, is the engine I study and aim to improve on. It is kept for reference and is not part of this repository.

Status: Stage 1 (setup and data) is complete. Stage 2 (baseline matcher: normalisation, candidate generation, scoring, and a callable screening interface) is built and evaluated. The evaluation harness, monitoring and audit log come in later stages.

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

The thresholds are per query script (Latin, Cyrillic, Arabic, other) and live in `src/screening_config.json`, which `src/evaluate_screening.py` writes. They were tuned on a dev sample to bring the false positive rate of the scripts close together, and reported on a separate test sample.

The same function is available from the command line:

```
python src/screen_cli.py "Vladimir Putin" --birth-date 1952-10-07 --country ru
python src/screen_cli.py "محمد علي" --json --limit 5
```

## Known limitations of the baseline matcher

These come from the measurements in `reports/stage2_candidates.json` and `reports/stage2_scoring.json`.

- Organisation names that are translated, not transliterated, are often missed. An Arabic company name such as the Arabic for "Wafa Humanitarian Organization" shares no spelling with its English name, so no transliteration key can connect them. Looking up a Latin name against Arabic-script entries finds 94.8% of people but only 63.5% of organisations.
- Han, kana and other scripts without a transliteration table (Hebrew, Greek, Thai, Hangul, Georgian and a few more) only match themselves. A Latin query cannot find an entity listed only in those scripts, and the reverse is also true. Cross-script candidate recall for these scripts is close to zero.
- Arabic is written without short vowels, so for any pair involving Arabic the consonant skeleton may stand in for the name score in full. That makes Arabic names produce more false alerts than other scripts, and the per-script threshold for Arabic-script queries ends up at 1.0 (a perfect score) to bring the Arabic false positive rate down to about 10%. The price is recall: an Arabic-script query finds only about 28% of the entities that list the same name in Latin spelling. Elsewhere the skeleton only confirms a name that already matches moderately.
- Names are scored with rarity weighting, so a shared rare surname counts for far more than a shared first name such as Mohammed. A query that matches only a common token scores low, and so does a true alias that shares only a common token.

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
python src/evaluate_screening.py    # per-script thresholds, reports/stage2_screening.json
pytest                              # tests run on a small fixture, no download needed
ruff check .
```

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
  screening_config.json  tuned scoring parameters and per-script thresholds
  entities.py            load named entities from the OpenSanctions file
  profile_normaliser.py, evaluate_candidates.py, evaluate_scoring.py, evaluate_screening.py   measurement scripts
tests/
  test_*.py              unit tests on small hand-built data, one file per module
reports/
  stage1_profile.json    profile of the downloaded data
holdout/
  final_holdout.json     frozen final hold-out queries, not scored until the final audit
data/                    downloaded data (not committed)
reference/yente/         yente source for study (not committed)
.github/workflows/ci.yml lint and tests on every push and pull request
```

`src/ftm_props.json` was generated from the followthemoney 4.11.0 schema files, so the profiler does not need that library installed.
