# How the hold-out labels were made, and where they lean on the matcher

This document says exactly how the positives and negatives of the screening hold-outs were labelled, what that assumes, and every place the label depends on the matcher's own definitions. It is about the second hold-out (`holdout/screening_holdout_v2.json`, source of the 0.777 F1) and the third (`holdout/screening_holdout_v3.json`), which were built by the same code. The construction code is `src/evaluate_scoring.py` (`build_positives`, `build_near_misses`, `build_synthetic`, `build_final_holdout`) driven by `src/build_screening_holdout.py` and `src/build_screening_holdout_v3.py`. Numbers marked with a report name come from `reports/analysis_label_dependence.json` and `reports/analysis_query_provenance.json`, produced by `analysis/label_dependence.py` and `analysis/query_provenance.py` on the second hold-out with the Stage 2 engine, which reproduces the reported figures exactly (1,474 true alerts, 132 wrong-entity alerts, 187 false alerts, F1 0.7772).

The short version: no person decided any label. A positive is a name taken from a listed entity's own alias list. A negative is a name built by a rule that is supposed to match no one. The F1 therefore measures whether the matcher finds an alias of an entity from the other aliases it still holds, and stays quiet on names built by recombining listed people's name parts. It does not measure accuracy on real customers.

## Where the entities come from

- The list is the OpenSanctions entities file. `load_named_entities` keeps an entity if it has at least one name, and the hold-outs sample only entities flagged `target` (the main sanctions targets).
- An entity's names are all the values of four properties pooled into one list: `name`, `alias`, `previousName` and `weakAlias`. The pooling loses which property a name came from.
- Birth dates are the entity's `birthDate` values as strings, and may be a bare year. Countries are the values of `country`, `nationality`, `citizenship`, `jurisdiction`, `birthCountry`, `mainCountry` and `flag`.
- Each entity is used for at most one positive or near-miss query across dev, test and the hold-outs, so no listed entity is the subject of two such queries.

## What counts as a true match (a positive)

A positive is a query whose text is one of an entity's own names. The label is that entity's ID. A positive is "found" if the engine alerts on that ID. Nothing else decides it: there is no human review, no check that the alias is a good one, and no check that a real customer would write it.

What that bakes in:

- Identity is whatever OpenSanctions says it is. The entity's alias list is the ground truth. If OpenSanctions lists a nickname, a former name, a translation, a company acronym or an unrelated-looking alias as the same entity, the positive is valid by construction.
- The query text appears only as a primary name for 1,565 of the 2,400 positives, only as an ordinary alias for 701, as both for 98, only as a weak alias for 29, only as a previous name for 3, and in a few other combinations (`reports/analysis_query_provenance.json`, second hold-out). So weak aliases are not what drags recall down. Ordinary aliases are the ones that look unrelated to the other names, for example "Tony Montana" against "Antonio Oseguera Cervantes".
- Positives are not only people. The 2,400 positives are 1,550 Person, 720 Organization, 69 Company, 44 LegalEntity, 14 Vessel and 3 Security.

## How the positives are built

The engine is given a list in which each sampled entity keeps only part of its names, so the tested alias cannot be matched to itself. The six groups, with the counts in the second hold-out:

| Group | Count | Query | What the list keeps for that entity |
|---|---|---|---|
| variant | 800 | a random Latin-script name; the entity must have at least two Latin names that normalise differently | every name whose normalised form differs from the query's normalised form, in any script |
| cyr_to_latin | 300 | a random Cyrillic-script name | only the entity's Latin-script names |
| latin_to_cyr | 300 | a random Latin-script name | only the entity's Cyrillic-script names |
| ar_to_latin | 300 | a random Arabic-script name | only the entity's Latin-script names |
| latin_to_ar | 300 | a random Latin-script name | only the entity's Arabic-script names |
| other_to_latin | 200 | a name in another script (Han, kana, Hebrew, Thai and others) | only the entity's Latin-script names |
| other_same_script | 200 | another-script name, same script on both sides | the other names in that script |

Each positive also carries a birth date and a country, each taken at random from the entity's own values and each dropped independently with probability 0.3 (`MISSING_P`). A supplied attribute is never wrong.

## How the negatives are built

There are 2,400 negatives in the second hold-out, in two kinds.

- Synthetic (1,900): the first token of one real listed name joined to the last token of another, drawn per script from listed people's names that have two or three tokens (800 Latin, 350 Cyrillic, 350 Arabic, and 400 in Han, kana, Hebrew and Thai). A candidate is thrown away if its normalised form, or its tokens sorted, equals the normalised form of any listed name. Its birth date and country are drawn at random from the pool of listed persons' values, each dropped with probability 0.3. They belong to nobody.
- Near-miss (500: 349 Latin, 147 Cyrillic, 4 Arabic): a real name of a listed person, with a birth date that is 8 to 25 years from one of the person's dates and at least 2 years from all of them, and a country that is not one of the person's countries. Both are always supplied.

The label "negative" means: not the listed party. For synthetic names that is an assumption that a recombination matches no one. For near-miss names it is a rule: a name identical to a listed person, with a conflicting date and country, is taken to be a different person.

## How the headline F1 is counted

`metrics_at` in `src/evaluate_scoring.py` counts:

- a hit: the true entity's final score is at or over the threshold of 0.86;
- a wrong alert: the true entity is under the threshold but some other entity is at or over it, counted as a false positive;
- a false alert: a negative that alerts on anything;
- precision is hits over hits plus wrong alerts plus false alerts; recall is hits over positives.

It counts only the five headline groups (2,000 of the 2,400 positives) and only Latin, Cyrillic and Arabic negatives (2,000 of the 2,400).

## Every place the label depends on the matcher's own definitions

Each item says what the matcher decides, which way it moves the F1, and how much where that was measured.

1. **The matcher's normaliser decides which aliases are "different".** The variant group needs two names that `normalise` keeps apart, and the list keeps only names whose normalised form differs from the query. Two spellings the normaliser already treats as one are never tested against each other. Effect: removes easy cases, so this makes recall lower than it would be if those were included. Not measured.
2. **The matcher's script detection decides group membership.** `script_class` (built on `name_scripts`) decides which names are Latin, Cyrillic, Arabic or "other", which entities are eligible for which group, and which names the list keeps. A name in mixed scripts is in no group. Not measured; direction unknown.
3. **The "clean" label for synthetic negatives is checked with the matcher's normaliser.** A recombined name is clean if its normalised form is not a listed name's normalised form. A recombination that is really a different spelling of a listed person, and that `normalise` does not equate, is labelled clean. If such names exist, they would be unlabelled positives, and an alert on them counts as a false positive. Not measured, and it cannot be measured without a person judging.
4. **The near-miss label is the matcher's own attribute rule.** A name identical to a listed person is "negative" because its date and country conflict, and the matcher is built to subtract score for a conflicting date and country. So the near-miss false-alert rate (3.0%, 15 of 500) mostly tests whether the matcher applies its own penalty. With birth date and country withheld, the near-miss false-alert rate is 100% (`analysis_label_dependence.json`, part A). None of the 15 alerts was on the entity the name was copied from; every near-miss alert named another entity.
5. **Positives carry correct attributes, negatives carry unrelated ones.** The matcher adds score when a birth date or country agrees and subtracts when it conflicts. Positives get their own entity's values, never a wrong or garbled one. Negatives get random values that conflict. That separates the two groups more than real data would, where dates are mistyped, partial or missing on both sides. Measured on the second hold-out with the Stage 2 engine and the same stored scores:

   | Attributes supplied to the matcher | Recall | Precision | F1 | False-alert rate on clean names |
   |---|---|---|---|---|
   | birth date and country (the headline) | 0.737 | 0.8221 | 0.7772 | 0.0935 |
   | birth date only | 0.7225 | 0.755 | 0.7384 | 0.1645 |
   | country only | 0.6665 | 0.7121 | 0.6885 | 0.1605 |
   | neither | 0.654 | 0.5317 | 0.5865 | 0.4665 |

   F1 falls from 0.7772 to 0.5865 when the two attributes are withheld, and precision falls from 0.82 to 0.53. Real customer records often lack both. In the hold-out, 1,115 of the 2,000 headline positives have no birth date and 928 have no country, because organisations and vessels have none to give and 30% are dropped at random.
6. **The headline leaves out the groups the matcher is known to fail.** The headline uses 2,000 of the 2,400 positives and 2,000 of the 2,400 negatives. The excluded positives are other_to_latin (0 found of 200) and other_same_script (36 found of 200). Over every query of the hold-out the F1 is 0.7109 (recall 0.6292, precision 0.8171) against 0.7772. The excluded other-script negatives alert 1.0% of the time. The exclusion is stated in the code and the README as a known limitation, but the headline number does not carry it.
7. **A hit is not penalised for alerting on other entities.** A positive counts as found if its entity is among the alerts, however many other entities also alert. 414 of the 1,474 hits also alerted on at least one other entity (1,291 extra entity alerts in all), and in 103 hits another entity scored higher than the true one. Those extra alerts are not in the precision. Counted as false alerts, precision would be 0.4780 and F1 0.5799. The headline definition is a per-customer one (a customer is either alerted or not), which is a reasonable choice for a review queue; it is not the same as precision per entity returned, and the two differ by 0.34.
8. **Entity identity is the list's, duplicates included.** A wrong-entity alert on a positive is a false positive even if the other entity is the same person under another record. Of the 132 wrong-entity alerts, 18 name an entity that holds an alias identical to one of the true entity's (the same tokens in any order), and 16 more hold an alias at least 0.9 similar by a character measure that does not use the matcher. Up to 34 of 132 may be duplicates in the list rather than errors. This lowers precision slightly.
9. **The tuning data comes from the same generator.** Dev, test and both hold-outs use one construction, so agreement between them shows the tuning is stable inside this construction. It cannot show stability on a different one.
10. **The positive's alias is one the engine could be expected to read.** Because every query is an alias that OpenSanctions already lists, the set cannot contain a spelling the list has never seen, which is the case that matters most for a real customer. The adversarial test in `docs/limitations_evidence.md` adds some of those.

## What the labels do not tell

- Nothing here shows that a real customer is cleared correctly. No negative is a real customer.
- Nothing here shows what share of real alerts would be true. The two groups are the same size, 2,000 each.
- Nothing here was reviewed by a person. Labels are as right as the list, the normaliser, the script detector and the construction rules.
- The false-alert rate of 9.35% describes names built from listed people's own name parts. It could be higher or lower than the rate for a real customer book, and it is unknown which.

## Reading the F1

The 0.777 is a measure of internal consistency on a constructed set. Its direction of error is mixed: items 1 and 8 push it down, items 4, 5 and 6 push it up and item 7 depends on the definition. Items 5 and 6 are large and measured, and they push it up. A reader should take 0.777 as the matcher's score on this construction, with attributes supplied correctly 70% of the time, and not as its performance in use.
