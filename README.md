# Sanctions Screening Engine

A two-stage sanctions screening engine built on OpenSanctions data: a candidate retrieval step, then a scoring step, wrapped in an evaluation, monitoring and fairness layer. The evaluation layer is the main deliverable. It measures recall and precision and slices them by the script a name is written in.

This is a portfolio project for a graduate data science application in an economic crime team. yente, the open-source OpenSanctions matching API, is the engine I study and aim to improve on. It is kept for reference and is not part of this repository.

Status: Stage 1 (setup and data) is complete. The matcher, evaluation harness, monitoring and audit log come in later stages.

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

## How to run

Developed and tested on Python 3.14. The data pipeline uses only the standard library. Do not install followthemoney: version 4.x depends on PyICU, which needs system ICU libraries to build.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # pytest and ruff only
python src/download_data.py         # about 350 MB, verifies SHA1, skips if already current
python src/profile_data.py          # writes reports/stage1_profile.json
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
tests/
  test_profile_data.py   profiler tests against a hand-built fixture
reports/
  stage1_profile.json    profile of the downloaded data
data/                    downloaded data (not committed)
reference/yente/         yente source for study (not committed)
.github/workflows/ci.yml lint and tests on every push and pull request
```

`src/ftm_props.json` was generated from the followthemoney 4.11.0 schema files, so the profiler does not need that library installed.
