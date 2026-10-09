# Evidence for the project's limitations

This is an analysis of the matcher, not a new feature. Nothing in scoring, matching or the audit code changed, nothing was tuned, and no hold-out file was written. Every figure comes from the real hold-outs and the real list, through scripts in `analysis/`, and is stored in a report in `reports/` named `analysis_*.json`. Where a figure here looks different from one elsewhere in the project, the reason is stated next to it.

## What was measured, and on what

- The hold-outs are the second (`screening_holdout_v2.json`, SHA-256 `e5a9b7af...`) and the third (`screening_holdout_v3.json`, SHA-256 `963b278b...`). Both are spent: they have been scored before and are now read again for analysis. Neither was used to choose any parameter, and nothing found here is a reason to change one without first confirming on dev.
- The list is the snapshot `20261006074704-hib`. The 0.777 was first measured on an earlier snapshot. The harness here reproduces it exactly on this one, which also shows the list has not moved the figure: 1,474 true alerts, 132 wrong-entity alerts and 187 false alerts, F1 0.7772.
- Two engines are used. The Stage 2 engine (no rarity-weighted skeleton) is the one behind the 0.777, so the error analysis, the label work and the prevalence table use it on the second hold-out. The shipped engine (with the rarity-weighted skeleton) is used on the third hold-out, where it reproduces F1 0.8008 (1,489 true alerts, 90 wrong-entity alerts, 140 false alerts). The adversarial test uses the shipped engine, because that is the one deployed.
- Positives means all 2,400 positive queries of a hold-out. Headline positives means the 2,000 used for the headline F1. Negatives means all 2,400 negatives; headline negatives are the 2,000 in Latin, Cyrillic and Arabic script. Every count below says which.
- Scripts and reports: `analysis/error_analysis.py`, `label_dependence.py`, `query_provenance.py`, `prevalence.py`, `tuning_transfer.py`, `adversarial.py`, `adversarial_summary.py`, `retrieval_gap.py`, `pairs_external.py` with `pairs_external_score.py`, and `stage5.py`, with `common.py`, `perturb.py` and `textsim.py` as helpers. They use the research path (the components the engine scores with), so no audit record is written.

## 1. Error analysis of the headline F1

### Method

A false negative is a headline positive whose true entity scores under the 0.86 threshold. A false positive is a headline negative that alerts, or a positive that alerts on some other entity. Each miss gets a cause and a severity. The definitions were fixed before the individual cases were read, from the score distribution of the misses (`analysis/error_analysis.py`).

Causes, first rule that applies: **non-transliterated script** (the query, or every kept alias, is in a script the transliteration does not cover); **organisation translation** (the entity is not a person and the pair crosses between Cyrillic or Arabic and Latin); **missing attribute** (the query lacked a birth date or country, and supplying the entity's own value makes it alert, tested by re-scoring); **unvowelled cross-script** (the pair crosses between Arabic script and Latin); **other**.

Severity: **dangerous** if any of: the true entity was scored and its final score is 0.80 or more (within 0.06 of the threshold); the query and the nearest listed alias are near-identical (same script, character similarity of 0.85 or more by a measure that does not use the matcher); every token of the query appears exactly in one listed alias, with a token of four letters or more. **Borderline** is a score from 0.70 to under 0.80, **weak** is under 0.70, and **no signal** means the entity was never scored at all.

### False negatives: 526 of 2,000 headline positives (26.3%)

| Cause | Misses | Share | Dangerous | Borderline | Weak | No signal |
|---|---|---|---|---|---|---|
| Organisation translation | 128 | 24.3% | 7 | 15 | 59 | 47 |
| Unvowelled cross-script (Arabic and Latin) | 114 | 21.7% | 51 | 37 | 9 | 17 |
| Missing attribute | 74 | 14.1% | 58 | 16 | 0 | 0 |
| Other | 210 | 39.9% | 42 | 32 | 78 | 58 |
| Non-transliterated script | 0 | 0.0% | 0 | 0 | 0 | 0 |
| All | 526 | 100% | 158 | 100 | 146 | 122 |

- "Other" is 163 same-script spelling variants and 47 Cyrillic and Latin pairs of people. Of the 163 variants, 126 are organisations, companies, legal entities or vessels (83 Organization, 24 LegalEntity, 7 Company, 12 Vessel), typically a long official name against a short or translated one, or an acronym against its expansion. 35 are people and 2 are securities.
- Non-transliterated script is empty only because those positives are not in the headline. Outside it, the 400 other-script positives are found 36 times in 400 (other_to_latin 0 of 200, other_same_script 36 of 200). Over all 2,400 positives recall is 62.9%, not 73.7%.
- By group, misses are: variant 189 of 800, latin_to_ar 121 of 300, ar_to_latin 105 of 300, latin_to_cyr 79 of 300, cyr_to_latin 32 of 300. By entity type: Person 269, Organization 204, LegalEntity 26, Company 13, Vessel 12, Security 2.
- 416 of the 526 misses (79.1%) had no birth date in the query, against 55.8% of all headline positives (1,115 of 2,000). 271 (51.5%) had no country, against 46.4% (928 of 2,000). 76 misses (14.4%) would have alerted had the query carried the entity's own birth date and country.
- 132 of the 526 misses (25.1%) alerted on some other entity instead. They are the 132 wrong-entity false positives below.

### Is any false negative the dangerous kind? Yes.

158 misses are dangerous, which is 30.0% of all misses and 7.9% of the 2,000 listed parties tested. These are not weak matches that happened to fail. They are names that match the listed alias closely and fall short of the threshold anyway.

- By group: variant 55, latin_to_ar 44, ar_to_latin 36, latin_to_cyr 19, cyr_to_latin 4. By entity type: Person 126, Organization 21, LegalEntity 6, Vessel 3, Company 1, Security 1.
- By criterion: 142 have a final score of 0.80 or more, 4 have near-identical text, 26 have every query token inside a listed alias. Exclusive counts: 128 on score alone, 13 on score and tokens, 1 on score and text, 3 on text alone, 13 on tokens alone. 4 of the 158 were never scored at all.
- 60 of the 158 would have alerted had the query carried the entity's own birth date and country. 149 of the 158 lack a birth date or a country; only 9 have both and still slip.
- The closest ones miss by hundredths of a point. "Vakhitov Bulat" against the listed "Vakhitov Bulat Ildarovych" scored 0.8591 against a threshold of 0.86: every token matches exactly and the only difference is a missing patronymic. The full list of 158, with entity IDs, scores and reasons, is in `reports/analysis_error_analysis_v2_stage2.json` under `dangerous_false_negatives`. The 20 closest to the threshold, and four of a different kind:

| Group | Query | Nearest alias the list kept | Final score | Why dangerous |
|---|---|---|---|---|
| ar_to_latin | عبد الله إبراهيم وهاب معروف البياتي | Abdullah Ibrahim Wahab Maruf Al-Bayati | 0.8595 | score |
| latin_to_ar | Omar Ahmed Abdullah Hamed Al-Jumaili | عمر احمد عبد الله حامد الجميلي | 0.8595 | score |
| variant | Vakhitov Bulat | Vakhitov Bulat Ildarovych | 0.8591 | score, tokens |
| ar_to_latin | جنكيز نجيب إبراهيم حمود الدوكرلي | Jankiz Najeeb Ibrahim Hamoud Al-Dukairly | 0.8585 | score |
| ar_to_latin | احمد عارف عبد الوهاب خلف | Ahmad Arif Abd al-Wahab Khalaf | 0.8571 | score |
| latin_to_ar | Global Relief Foundation | غلوبال ريليف فونديشن | 0.8571 | score |
| latin_to_ar | Abbas Kazim Mardan Ali Al-Bayati | عباس كاظم مردان علي البياتي | 0.8571 | score |
| latin_to_ar | Nawaf Zaki Hamadi Khidr Al-Ruba'i | نواف زكي حمادي خضر الربيعي | 0.8571 | score |
| latin_to_cyr | Zakharov Andrey | Захаров Андрей Георгиевич | 0.8563 | score |
| variant | Kozakov Gleb | Kozakov Hlib Oleksandrovych | 0.8548 | score |
| variant | Mendel Yuliia | Mendel Yuliia Volodymyrivna | 0.8546 | score, tokens |
| variant | Bienklian Aik | Bienklian Aik Davydovych | 0.8535 | score, tokens |
| ar_to_latin | صبحي هاشم حامد محمد المشهداني | Sobhi Hashem Hamed Mohammed Al-Mashhadani | 0.8511 | score |
| latin_to_cyr | Baisarov Leonid | Байсаров Леонід Володимирович | 0.8500 | score |
| variant | Safwan Faisal Abdullah | Abu Bakr | 0.8500 | score |
| ar_to_latin | تحسين عليوي ابراهيم خليل الحمداني | Tahseen Alawi Ibrahim Khalil Al-Hamdani | 0.8500 | score |
| ar_to_latin | نجم عبد الرزاق احمد حسين الجميلي | Najm Abd al-Razzaq Ahmed Hussein al-Jumaili | 0.8500 | score |
| ar_to_latin | احمد علاء مرعي حسن الشهواني | Ahmad Ala'a Mar'i Hasan Al-Shahwani | 0.8500 | score |
| ar_to_latin | منهل ناصر جاسم طه الجبوري | Munhal Nasser Jassim Taha Al-Jubouri | 0.8500 | score |
| latin_to_ar | Firas Hamad Jasim Muhammad Al-Hamdani | فراس حمد جاسم محمد الحمداني | 0.8500 | score |
| variant | Mehdi BAITAR | Adonis, Mehdi, Wanis BAITAR | 0.7967 | tokens |
| variant | Varochkin Anatolii | Varochkin Anatolii Hennadiiovych | 0.7840 | tokens |
| variant | Dmitrij Viktorovitj PACHANDRIN | Pakhandrin Dmitrii Viktorovich | not scored | text (0.87) |
| variant | AmavandTec | DamavandTec | not scored | text (0.95) |

The shipped engine on the third hold-out does the same thing slightly less often: 511 misses (recall 74.45%), of which 119 dangerous, 97 borderline, 195 weak and 100 no signal. By cause: organisation translation 112, unvowelled cross-script 110, missing attribute 41, other 248.

### Misses that are lost in retrieval, not in scoring

A candidate has to be retrieved before it is scored. 126 of the 526 misses had no score at all. Forcing the engine's own scoring onto those entities (`analysis/retrieval_gap.py`), 59 would have been scored and 17 would have alerted, which is 3.2% of the 526 misses. On the third hold-out with the shipped engine it is 14 of 511. In the adversarial test below retrieval matters far more.

### False positives: 319 alerts (187 on negatives, 132 wrong-entity on positives)

| Kind | Alerts | Latin | Cyrillic | Arabic |
|---|---|---|---|---|
| Synthetic negatives | 172 | 60 | 19 | 93 |
| Near-miss negatives | 15 | 13 | 2 | 0 |
| Wrong-entity alerts on positives | 132 | 98 | 7 | 27 |
| All | 319 | 171 | 28 | 120 |

False-alert rate on headline negatives: Latin 73 of 1,149 (6.4%), Cyrillic 21 of 497 (4.2%), Arabic 93 of 354 (26.3%).

- Cause of the alert: **skeleton** 229 (71.8%), meaning the consonant-skeleton similarity lifted a weaker name score over the threshold; **attribute boost** 62 (19.4%), meaning a matching birth date or country supplied the last points; **name score alone** 28 (8.8%). In Arabic script, 95 of 120 are skeleton, 20 attribute, 5 name. In Latin, 115 of 171 are skeleton, 34 attribute, 22 name. In Cyrillic, 19 of 28 are skeleton, 8 attribute, 1 name.
- 217 of the 319 alerting pairs have a name in Arabic script on one side, although Arabic-script queries make only 120 of the alerts. Latin queries alert on Arabic-script entities too.
- Of the 172 synthetic alerts, 105 (61%) share no complete token with the name they alert on, 62 share one, 5 share two. The matcher is alerting on skeleton collisions between names that look different.
- Near-miss alerts: 15 of 500 (3.0%). None alerted on the entity the name was copied from; every one named a different entity.
- The shipped engine on the third hold-out: 230 false positives (133 synthetic, 7 near-miss, 90 wrong-entity); skeleton 175, attribute boost 34, name score 21. The rarity-weighted skeleton removed some skeleton alerts and left skeleton collisions the largest cause.

### What this means for the limitations

- The README names Arabic names and organisation translation as known weaknesses. The numbers agree, and add two things it does not say. The dangerous misses are mostly people, 126 of 158, and mostly cases where the name matches but the score falls just short. A threshold lowered by 0.06 would catch most of them (142 of the 158 score 0.80 or more), but it would also raise the false-alert rate, which is already 9.35% on this construction and 26.3% for Arabic-script names. The size of that cost was not measured here.
- Missing attributes matter more than the headline suggests. 79% of the misses lacked a birth date, and the matcher relies on a birth date and country to cross the line (part 2).
- The precision problem is a skeleton problem. Seven in ten false alerts exist only because the consonant skeleton lifted a score, and 61% of the synthetic ones share no whole token with the name they hit.

## 2. Label provenance

The full account is in `docs/label_provenance.md`. In short: no person judged any label. A positive is a name taken from a listed entity's own alias list; a negative is a name built by a rule. The F1 depends on the matcher's own definitions in these measured ways:

- The headline counts 2,000 of 2,400 positives and 2,000 of 2,400 negatives, leaving out the groups the engine fails. Over every query the F1 is 0.7109, not 0.7772, and recall is 62.9%.
- Positives carry their own entity's birth date and country 70% of the time, and never a wrong one. With birth date and country withheld, recall is 0.654, precision 0.5317 and F1 0.5865, against 0.7772. With birth date only the F1 is 0.7384, with country only 0.6885.
- The near-miss negatives are cleared only because the matcher subtracts score for a conflicting date and country. With both withheld, the near-miss false-alert rate is 100%, against 3.0%.
- A hit is not penalised for alerting on other entities. 414 of the 1,474 hits also alerted on at least one other entity, 1,291 extra alerts in all, and in 103 hits another entity scored higher than the true one. Counting each extra entity alert as false, precision is 0.4780 and F1 0.5799.
- Up to 34 of the 132 wrong-entity alerts may be duplicate entities in the list (18 share an identical alias with the true entity, 16 more hold an alias at least 0.9 similar), which are counted as errors.
- Positives are mostly primary names (1,565 of 2,400 appear only as a primary name) and ordinary aliases (701). Only 33 are weak aliases.

### What this means for the limitations

The 0.777 is real as a measurement of this construction and should not be read as accuracy on real customers. It is likely too high for any setting where birth date and country are often missing or wrong, which is common, and the effect is large: about 0.19 of F1. Nothing in the project measures the false-alert rate on real customers, and the 9.35% describes names built from listed people's own name parts.

## 3. Prevalence

Recall and the false-alert rate are held at the measured values, and only the share of customers who are really listed changes (`analysis/prevalence.py`). With r the share of listed parties that alert on their own entity, w the share that alert on a different entity, and f the share of clean customers that alert:

    precision = p r / (p r + p w + (1 - p) f)
    alert rate = p (r + w) + (1 - p) f

At a prevalence of 0.5 the formula returns the measured precision of 0.8221 exactly. Stage 2 engine on the second hold-out, r = 0.737, w = 0.066, f = 0.0935:

| Prevalence | Precision | Alert rate | Alerts per 100,000 customers | True alerts per 100,000 | Alerts reviewed per true hit |
|---|---|---|---|---|---|
| 1 in 2 (the hold-out) | 0.8221 | 0.4483 | 44,825 | 36,850 | 1.2 |
| 1 in 100 | 0.0733 | 0.1006 | 10,060 | 737 | 13.6 |
| 1 in 1,000 | 0.0078 | 0.0942 | 9,421 | 73.7 | 127.8 |
| 1 in 10,000 | 0.00079 | 0.0936 | 9,357 | 7.37 | 1,270 |
| 1 in 100,000 | 0.00008 | 0.0935 | 9,351 | 0.74 | 12,688 |

Shipped engine on the third hold-out, r = 0.7445, w = 0.045, f = 0.07:

| Prevalence | Precision | Alert rate | Alerts per 100,000 customers | True alerts per 100,000 | Alerts reviewed per true hit |
|---|---|---|---|---|---|
| 1 in 2 (the hold-out) | 0.8662 | 0.4298 | 42,975 | 37,225 | 1.2 |
| 1 in 100 | 0.0964 | 0.0772 | 7,720 | 744.5 | 10.4 |
| 1 in 1,000 | 0.0105 | 0.0707 | 7,072 | 74.5 | 95.0 |
| 1 in 10,000 | 0.0011 | 0.0701 | 7,007 | 7.45 | 941 |
| 1 in 100,000 | 0.00011 | 0.0700 | 7,001 | 0.74 | 9,403 |

- At one listed customer in 1,000, about one alert in 100 is a real hit: 0.78% for the Stage 2 engine and 1.05% for the shipped one. At one in 10,000 it is between one alert in 940 and one in 1,270, and at one in 100,000 between one in 9,400 and one in 12,700.
- The alert rate does not fall with prevalence. It settles at the false-alert rate, 9.4% for the Stage 2 engine and 7.0% for the shipped one: roughly 9,400 or 7,000 alerts per 100,000 customers, whatever the true number of listed parties.
- By query script, at one in 1,000 (Stage 2 engine): precision is 0.0112 for Latin, 0.0207 for Cyrillic and 0.0025 for Arabic, which means 89, 48 and 405 alerts reviewed per true hit. For the shipped engine: 0.0175, 0.0292 and 0.0032, which is 57, 34 and 311.
- For half of all alerts to be real at one in 1,000, the false-alert rate would have to be 0.067% (one clean customer in 1,489) for the Stage 2 engine, or 0.070% (one in 1,428) for the shipped one. At one in 10,000, one in 14,902 and one in 14,294. The measured rates are 9.35% and 7.0%.

### What this means for the limitations

At any realistic prevalence the matcher cannot be run as an automatic decision. It is a candidate generator for a review queue that would hold about 7% to 9% of all customers if the hold-out false-alert rate carried over. The rate on a real book is unknown and could differ a lot, but to be useful at realistic prevalence it would have to be lower by a factor of more than 100. The F1 of 0.777 says nothing about that, because it is computed at a prevalence of 50%.

## 4. Tuning transfer

Tuned on dev: floor, lift, blend and the global threshold (grid search, step 4), per-script thresholds, the rarity-weighted skeleton (Stage 3, judged on dev and test), and the index's `max_df` of 1000. The slice used is the second hold-out, built from entities that no dev or test query uses and never used to choose any parameter (`analysis/tuning_transfer.py`). It was read once before for reporting and is read again here. I did not build a new hold-out, because the two existing ones were never used to tune and a new one would have been drawn by the same generator.

### F1, recall and alert rate by script

| Figure | Dev (tuned) | Test | First hold-out | Second hold-out | Third hold-out (shipped engine) |
|---|---|---|---|---|---|
| F1 at the frozen configuration | 0.7695 | 0.7645 | 0.7764 | 0.7772 | 0.8008 |
| Recall | n/a | 0.7295 | 0.7465 | 0.737 | 0.7445 |
| Latin alert rate on clean names | 0.0567 | 0.0473 | n/a | 0.0635 | 0.0407 |
| Cyrillic alert rate on clean names | 0.0465 | 0.0522 | n/a | 0.0423 | 0.0284 |
| Arabic alert rate on clean names | 0.325 | 0.376 | n/a | 0.2627 | 0.2251 |
| Latin recall | 0.7186 | 0.7143 | n/a | 0.7221 | 0.725 |
| Cyrillic recall | 0.88 | 0.88 | n/a | 0.8933 | 0.8533 |
| Arabic recall | 0.6067 | 0.65 | n/a | 0.65 | 0.7267 |

F1 on the untouched slice is not lower than on the data it was tuned on. The first three columns use the Stage 2 engine; the last uses the shipped engine, so it is not a like-for-like column.

### The grid dev searched, run on the slice

3,750 configurations (3 blends, 5 floors, 5 lifts, 50 thresholds). The frozen choice scores 0.7772 and ranks 5th. The best the slice would have picked is 0.7775 (blend 0.5, floor 0.2, lift 0.2, threshold 0.86), so the regret is 0.0003; 18 configurations are within 0.005 of the best. Dev's preferences carry over:

| Parameter | Dev, best F1 by value | Second hold-out, best F1 by value |
|---|---|---|
| Blend 0.0 / 0.5 / 1.0 | 0.7428 / 0.7695 / 0.7615 | 0.7461 / 0.7775 / 0.7713 |
| Floor 0.2 to 0.6 | 0.7695 at every value | 0.7775 for 0.2 to 0.5, 0.7772 for 0.6 |
| Lift 0.1 / 0.2 / 0.3 / 0.4 / 0.5 | 0.7686 / 0.7695 / 0.7627 / 0.7627 / 0.7627 | 0.7726 / 0.7775 / 0.7744 / 0.7744 / 0.7744 |

F1 by threshold at the frozen parameters on the slice: 0.78 gives 0.7613, 0.82 gives 0.7588, 0.84 gives 0.768, 0.86 gives 0.7772, 0.88 gives 0.7664, 0.90 gives 0.7569, 0.92 gives 0.7399. The peak is at 0.86, as on dev. The floor does nothing on either, so one of the five tuned dimensions never mattered.

### Per-script thresholds

Chosen on dev: Latin 0.86, Cyrillic 0.88, Arabic 0.86. On the slice, F1 goes from 0.7772 to 0.778, and the Cyrillic alert rate from 4.23% to 3.62% with Cyrillic recall unchanged at 89.33%. On dev the Cyrillic rate went from 4.65% to 3.26%. The effect transfers in direction and is tiny, which is why it was switched off.

### The rarity-weighted skeleton

On the second hold-out the shipped engine against the Stage 2 engine, both at `max_df` 1000: F1 0.7772 to 0.8085, recall 0.737 to 0.7515, precision 0.8221 to 0.8749, false-alert rate 0.0935 to 0.066. Alert rate on clean names: Latin 0.0635 to 0.039, Cyrillic 0.0423 to 0.026, Arabic 0.2627 to 0.209. Arabic recall 0.65 to 0.707. The gain on dev was F1 0.7008 to 0.74 and the Arabic rate 0.325 to 0.2747; on the third hold-out it was F1 0.7628 to 0.8008 and the Arabic rate 0.326 to 0.226. It transfers, and the size of the gain on the fresh slice (+0.031) is in line with dev (+0.039) and the third hold-out (+0.038).

### max_df

| max_df | Stage 2 engine F1 | Shipped engine F1 | Shipped: false-alert rate | Shipped: recall |
|---|---|---|---|---|
| 250 | 0.7930 | 0.8166 | 0.045 | 0.746 |
| 500 | 0.7852 | 0.8133 | 0.0525 | 0.748 |
| 1000 (set) | 0.7772 | 0.8085 | 0.066 | 0.7515 |
| 2000 | 0.7682 | 0.7964 | 0.0945 | 0.753 |
| 5000 | 0.7600 | 0.7906 | 0.114 | 0.756 |

### Does it transfer or overfit?

The tuned F1 choices transfer. The F1 on an untouched slice (0.7772) is above the dev figure the choices were tuned to (0.7695), the frozen configuration is 0.0003 from the best the slice would choose, and dev's ordering of blend, lift and threshold holds. There is no sign of overfitting in the grid, the per-script thresholds or the rarity skeleton. Two cautions. First, the slice was made by the same generator as dev, so this shows stability inside the construction, not on different data. Second, `max_df` is not tuned in the sense that matters: it was set from the candidate-retrieval work (`reports/stage2_candidates.json`, index size and candidate recall), not for F1, and F1 improves steadily as it falls. 250 gives +0.0158 (Stage 2 engine) and +0.0081 (shipped) over 1000, with a lower false-alert rate and 0.5 to 0.7 points less recall. I have not changed it: this slice has been read twice, and a change would need confirming on dev first.

### What this means for the limitations

The tuning is not the reason the matcher is weak. The F1 surface is flat in most of its parameters (the floor has no effect, many settings are within 0.005), so tuning moved the result little in either direction, and what limits the matcher lies elsewhere, in retrieval, skeleton collisions and missing attributes. The remaining tunable item with real headroom is `max_df`.

## 5. Adversarial robustness

### Method

Base set: the 1,489 headline positives of the third hold-out that the shipped engine already catches. A perturbation can only lose a catch the engine had. For each base and each family, up to three distinct single-edit variants were generated, with the same birth date and country as the base, and screened against the same list with the entity's tested alias held out. A variant is robust if the true entity still alerts and a slip-through if it does not.

The rules (`analysis/perturb.py`) are standard-library only. Their source: single-character edits of the kind classic spelling-error studies and Damerau-Levenshtein distance describe; dropped, kept and reordered tokens; Cyrillic romanisation under GOST 7.79-2000 system B and the German (Duden) convention; Latin respelling under common alternative conventions for Russian and Arabic names (ya, ia, ja; kh, h, ch; Mohammed, Muhammad, Mohamed; Abdul, Abd al; ou, u). The tables were typed from my knowledge of those schemes and have not been checked against the standards' texts. I confirmed that no matcher code is reused: the module imports nothing from `src/` (test `test_the_perturbation_and_similarity_modules_import_nothing_from_the_matcher`). One family is a control and not an alternative: the matcher's own Cyrillic table is documented as close to BGN/PCGN and is identical to the BGN table written here on all 33 Russian letters (checked by test), so that family should lose nothing, and does not. Romanisation rules for Russian or Arabic names are applied only to names whose entity has an alias in that script.

### Results by family

| Family | Positives tested | Variants | Robust | Slip-through | Slip-through rate | Positives with at least one slip |
|---|---|---|---|---|---|---|
| consonant_delete | 1,014 | 2,617 | 1,990 | 627 | 24.0% | 308 (30.4%) |
| consonant_double | 1,012 | 2,967 | 2,681 | 286 | 9.6% | 152 (15.0%) |
| vowel_delete | 1,010 | 3,005 | 2,708 | 297 | 9.9% | 142 (14.1%) |
| vowel_swap | 1,010 | 3,005 | 2,666 | 339 | 11.3% | 161 (15.9%) |
| drop_middle_token | 1,141 | 2,271 | 1,603 | 668 | 29.4% | 491 (43.0%) |
| drop_one_token | 1,475 | 2,950 | 1,469 | 1,481 | 50.2% | 1,103 (74.8%) |
| reorder_tokens | 1,475 | 3,756 | 3,741 | 15 | 0.4% | 13 (0.9%) |
| translit_cyrillic_german | 256 | 256 | 107 | 149 | 58.2% | 149 (58.2%) |
| translit_cyrillic_gost | 256 | 256 | 182 | 74 | 28.9% | 74 (28.9%) |
| respell_cyrillic_conventions | 519 | 1,281 | 1,029 | 252 | 19.7% | 145 (27.9%) |
| respell_arabic_conventions | 260 | 745 | 736 | 9 | 1.2% | 7 (2.7%) |
| control: uppercase | 1,206 | 1,206 | 1,206 | 0 | 0.0% | 0 |
| control: translit_cyrillic_bgn (the matcher's own scheme) | 256 | 256 | 256 | 0 | 0.0% | 0 |

- **Most slip-throughs:** leaving out a token (drop_one_token, 1,481), then leaving out a middle token (668), then deleting a consonant (627). Among single-letter edits, deleting a consonant is the worst family (24.0%), then vowel swap (11.3%).
- **By kind of edit:** single-letter edits 11,594 variants, 1,549 slips (13.4%); 351 of the 1,489 caught positives (23.6%) have at least one single-letter edit that loses them. Token edits 8,977 variants, 2,164 slips (24.1%); 1,229 positives (82.5%) are lost by at least one. Reordering alone is safe (15 of 3,756). Romanisation 2,538 variants, 484 slips (19.1%); 304 positives (20.4%).
- Across every family except the controls, 1,260 of the 1,489 caught positives (84.6%) are lost to at least one single edit. Leaving out a token is a large edit, so the single-letter and romanisation figures are the fairer measure of fragility.
- **By hold-out group:** single-letter edits slip 16.4% of the time for names whose entity is listed in Cyrillic (441 of 2,683), 15.0% for variants (1,008 of 6,737) and 4.6% for names listed in Arabic script (100 of 2,174). Cyrillic queries written out in Latin letters slip 43.6% of the time (223 of 512 across GOST and German; 28.9% and 58.2% separately).
- **Who slips:** of the 1,549 single-letter slips, 1,195 (77%) are organisations or other non-persons (Organization 1,094, Company 59, LegalEntity 40, Security 2) and 354 are people.
- **The mechanism:** 178 of the 1,549 single-letter slips left the true entity with no score at all, because it was not retrieved. Forcing the engine's scoring onto those 178, 128 would have alerted and 24 more would have scored between 0.80 and the threshold (`analysis/retrieval_gap.py`). A single deleted letter in a name that is itself a spelling variant of the listed alias is enough, for example "Sukhob Oymakhmadov": the entity is never retrieved, although scored directly its name score is 0.917. Another 92 slips alerted a different entity instead.

### Worst cases

Catches with a score of 0.95 or more that one realistic single-letter edit takes under the threshold: 1,205 of the 1,549 slips. Examples of people with long names, base score then variant score:

| Family | Name tested | Variant | Base score | Variant score |
|---|---|---|---|---|
| consonant_delete | Sukhrob Oymakhmadov | Sukhob Oymakhmadov | 1.00 | entity not retrieved |
| consonant_delete | Aleksandr Kozlovsky | Aleksandr Kozovsky | 1.00 | entity not retrieved |
| consonant_delete | Sofokleus Kharula | Sookleus Kharula | 1.00 | entity not retrieved |
| consonant_delete | BILANISHVILI AKAKI | BILANISHVII AKAKI | 1.00 | entity not retrieved |
| consonant_delete | Borshchevskii Evgenii | Boshchevskii Evgenii | 0.97 | entity not retrieved |
| consonant_delete | Streltsov Evgenii | Strelsov Evgenii | 0.99 | entity not retrieved |
| consonant_delete | Balyberdin Aleksei | Balybedin Aleksei | 0.97 | entity not retrieved |
| consonant_delete | Abu Bakr | Abu Bak | 1.00 | entity not retrieved |
| consonant_delete | 2Rivers (Organization) | 2River | 1.00 | entity not retrieved |
| consonant_delete | AO EPK (Organization) | AO EK | 1.00 | entity not retrieved |

The full list of every slip, with scores, is in `reports/analysis_adversarial_v3_current.json`, and the 40 worst single-letter edits are in `reports/analysis_adversarial_summary_v3_current.json`.

### What this means for the limitations

The matcher is not robust to ordinary typing and spelling variation. About one catch in four is lost to at least one one-letter slip in a name, and one in five to a different romanisation convention. A name written under the German convention is lost more often than not. Names of organisations are hit hardest. Part of this is a retrieval limit: the candidate index finds an entity from name keys, so a one-letter change can remove the entity from the candidate list even though scoring it directly would have alerted. That is a different cause from the score-threshold causes in part 1, and a different fix. It also means the recall figures in the project, which test aliases the list already contains, overstate recall against a customer who spells a listed name slightly differently.

## 6. External human-labelled validation (OpenSanctions Pairs)

Everything above is measured on queries this project built from the list. This part scores the engine against labels that people made. Nothing was changed or tuned: the engine is the shipped one, with its one threshold of 0.86.

### The data, its licence and its attribution

The data is OpenSanctions Pairs, the snapshot of 2025-12-09 (OpenSanctions, "Pairs: Cross-Referencing Training Data", https://www.opensanctions.org/docs/opensource/pairs/), read from the Hugging Face mirror `sanctions-er-anon/opensanctions_pairs`. Each pair is two entities and an analyst's judgement of whether they are the same real-world entity. The paper that describes it is Smith, Sesodia, Lindenberg and Schroeder de Witt (2026), "OpenSanctions Pairs: Large-Scale Entity Matching with LLMs", arXiv:2603.11051. The data is licensed CC BY-NC 4.0: it may be used for non-commercial purposes only, with attribution to OpenSanctions. This project is non-commercial. The data is not stored in the repository and is not redistributed; it is read from a local cache that is git-ignored (`data/external/`), and the reports hold aggregate counts only. The paper's own licence (CC BY 4.0) is separate and does not apply to the data.

File: `pairs.json.gz`, 409,088,814 bytes, SHA-256 `4044cf5f6edb0d2cdccf11326063f905408224c3b2fe261baa0d779d80a0edd1`. It is byte-identical to the copy the project had already downloaded and checked against the hash Hugging Face publishes. Each line holds `judgement`, `left` and `right`; a side has `id`, `caption`, `schema` and `properties`, with `datasets`, `referents` and timestamps besides.

### Alignment with what the engine consumes

| | Count |
|---|---|
| Pairs in the file | 755,540 |
| Positive (same entity) | 581,149 (76.9%) |
| Negative (different entities) | 174,391 (23.1%) |
| Relationship and role records, not scoreable by a name matcher | 280,590 (37.1% of the file) |
| Entity pairs | 474,950 |
| Duplicate lines of an entity pair, scored once | 29,592 |
| Pairs whose duplicates disagree on the label, left out | 14 |
| Pairs with a side that has no name property (a caption only) | 5,531 |
| Pairs scored | 439,813 (268,490 positive, 171,323 negative) |

The dataset card describes persons and organisations. The file also holds Occupancy 213,448, Succession 33,607, Position 26,330, Ownership 2,392, Family 2,301, Directorship 2,099, UnknownLink 395 and a few smaller record types. Those are 37% of the lines and were counted, not scored.

The pairs scored are 268,027 person to person, 157,218 organisation to organisation (Organization, Company, LegalEntity, PublicBody), 4,704 person to organisation, and 9,864 of other types (Vessel 7,336, Security 1,644, CryptoWallet 800, Airplane 84).

What the engine reads: names (`name`, `alias`, `previousName`, `weakAlias`), the first birth date, and country (`country`, `nationality`, `citizenship`, `jurisdiction`, `birthCountry`, `mainCountry`, `flag`). It ignores the entity type, the caption, and 89 other property keys, among them `address` (on either side of 261,552 pairs), `gender` (218,929), `idNumber` (93,511), `passportNumber` (41,711), `innCode` (87,672), `incorporationDate` (97,977), `imoNumber` (14,517), `leiCode` (13,368), and first, middle and last names. Birth dates are on both sides of 201,189 scored pairs (45.7%) and country on both sides of 383,211 (87.1%); 35,103 pairs have neither on both sides.

Script coverage of the scored pairs, by the engine's own order (Arabic over Cyrillic over other): Latin 166,896, Cyrillic 144,174, Arabic 102,369 (80,949 of them person pairs) and other non-Latin 26,374. Only 1,736 pairs have no script in common between the two sides, because entities carry aliases in several scripts.

Pairs the engine cannot score: the 280,590 relationship records; the 5,531 pairs where a side has only a caption; and 9,210 scored pairs (2.1%) for which none of the left names reached the engine's minimum name score against the right entity, so no score exists. They are counted as predicted non-matches.

### How a pair is scored

Fixed before scoring (`analysis/pairs_external.py`). Each of the left entity's first 12 names is used as a query, with the left entity's first birth date and first country, against the right entity as it would stand on the list (all its names, birth dates and countries). The pair's score is the highest final score. A pair is predicted a match at 0.86 or above. It is the engine's own scoring: the deployment list's rarity statistics, with retrieval bypassed for the one right-hand entity, checked equal to `screen()` on 300 list entities (no difference). 118,311 scored pairs had more than 12 names on a side and used only the first 12. The same was run with the right entity as the query.

A first scoring run was discarded: a mistake in the record order meant every left entity was scored against itself, and the output (recall 1.0, AUC 0.5) showed it. It was fixed and a test now pins the order. The pairs were then scored; two further runs that only added breakdown columns gave identical headline numbers.

### Result against the human labels

Left names as queries, threshold 0.86, 439,813 pairs:

| | Pairs | Positive share | Precision | Recall | F1 (95% interval) | False-alert rate | AUC |
|---|---|---|---|---|---|---|---|
| All | 439,813 | 61.0% | 0.9635 | 0.9750 | 0.9692 (0.9688 to 0.9697) | 5.79% | 0.9747 |
| Person to person | 268,027 | 69.8% | 0.9707 | 0.9888 | 0.9797 | 6.89% | 0.9694 |
| Organisation to organisation | 157,218 | 46.0% | 0.9604 | 0.9570 | 0.9587 | 3.35% | 0.9838 |
| Person to organisation | 4,704 | 87.8% | 0.9758 | 0.8696 | 0.9196 | 15.56% | 0.8977 |
| Other entity types | 9,864 | 51.7% | 0.7466 | 0.8111 | 0.7775 | 29.43% | 0.8149 |
| Latin | 166,896 | 66.2% | 0.9512 | 0.9647 | 0.9579 | 9.71% | 0.9555 |
| Cyrillic | 144,174 | 56.2% | 0.9680 | 0.9755 | 0.9717 | 4.13% | 0.9817 |
| Arabic | 102,369 | 57.7% | 0.9777 | 0.9915 | 0.9845 | 3.09% | 0.9875 |
| Other non-Latin | 26,374 | 67.9% | 0.9722 | 0.9830 | 0.9776 | 5.95% | 0.9794 |
| Person to person, Arabic | 80,949 | 66.9% | 0.9819 | 0.9926 | 0.9872 | 3.68% | 0.9833 |
| Person to person, Latin | 105,466 | 71.5% | 0.9561 | 0.9850 | 0.9703 | 11.33% | 0.9518 |
| No script in common (cross-script) | 1,736 | 98.2% | 0.9915 | 0.6860 | 0.8110 | 31.25% | 0.7677 |

With the right entity as the query the overall figures are precision 0.9637, recall 0.9644, F1 0.9641, false-alert rate 5.69%.

### Most of this is easy, and the hard part is a different number

57.5% of the pairs (252,867), and 91.4% of the positives (245,356 of 268,490), have an identical name on both sides, the same tokens in any order ignoring case and accents. Any name matcher gets those. They also contain 7,511 negatives, namesakes the analysts judged to be different entities, and the engine alerts on 86.4% of them.

| Subset | Pairs | Positives | Negatives | Precision | Recall | F1 (95% interval) | False-alert rate | AUC |
|---|---|---|---|---|---|---|---|---|
| An identical name on both sides | 252,867 | 245,356 | 7,511 | 0.9740 | 0.9920 | 0.9829 | 86.42% | 0.5637 |
| No identical name | 186,946 | 23,134 | 163,812 | 0.8428 | 0.7953 | 0.8184 (0.8144 to 0.8222) | 2.09% | 0.9596 |

The hold-out positives are built so that the tested alias is removed from the list, so no identical name is left. The "no identical name" row is the one built most like them. Its positive share is 12.4%, so its precision is not comparable with a 50% balance; at the hold-outs' balance of equal numbers its precision would be 0.9743 and its F1 0.8758. With the right entity as the query it is recall 0.7744, F1 0.8070.

Namesakes, by whether the birth date is on both sides: with an identical name and a birth date on both sides there are 750 negatives and the engine alerts on 27.5% of them; with an identical name and no birth date on both sides there are 6,761 negatives and the engine alerts on 93.0%. With no identical name and no birth date on both sides (125,130 pairs) precision is 0.7724, recall 0.7189 and F1 0.7447; with a birth date on both sides (61,816 pairs) F1 is 0.9680 and the false-alert rate 0.23%.

### Against the hold-out numbers

| | Recall | Precision | F1 | False-alert rate |
|---|---|---|---|---|
| Second hold-out, Stage 2 engine | 0.737 | 0.8221 | 0.7772 | 9.35% |
| Third hold-out, shipped engine | 0.7445 | 0.8662 | 0.8008 | 7.0% |
| Human-labelled pairs, all | 0.9750 | 0.9635 | 0.9692 | 5.79% |
| Human-labelled pairs, no identical name | 0.7953 | 0.8428 | 0.8184 | 2.09% |

Where the human-labelled result is worse than the hold-out band (F1 0.777 to 0.801), and by how much:

- **Vessels, securities and crypto wallets.** The 9,864 pairs of other entity types score F1 0.7775 (0.023 under the third hold-out) with a 29.4% false-alert rate. Without an identical name, vessel to vessel pairs score recall 0.0881, F1 0.1083 and AUC 0.292, which is below chance (318 positives, 2,067 negatives): the engine finds almost none of the vessels that analysts matched, and ranks them backwards. All 800 crypto wallet pairs are negatives and the engine alerts on 99.75% of them; 21.4% of 1,644 security pairs, all negatives, alert.
- **Pairs with no birth date on both sides and no identical name.** F1 0.7447, which is 0.056 under the third hold-out and 0.033 under the second.
- **Organisation pairs with no identical name.** F1 0.7832 (recall 0.7631), 0.018 under the third hold-out. Organization to Organization alone is F1 0.6394 (pos 963, neg 18,661). Person to organisation pairs without an identical name have recall 0.6665.
- **Person pairs written in Latin script.** False-alert rate 11.33%, against 6.35% for Latin queries on the second hold-out and 4.07% on the third.
- **Cross-script pairs.** Recall 0.6860, which is 0.051 under the second hold-out's overall recall and sits with its weak directions (Arabic to Latin 0.65, Latin to Arabic 0.5967).
- **Namesakes.** The engine alerts on 86.4% of different-entity pairs that share a name, and on 93.0% of them when no birth date is on both sides.

Where it is better: overall F1 is 0.969, which is 0.17 to 0.19 above the hold-outs, but 91% of the positives are identical names. On pairs with no identical name, F1 is 0.8184, 0.041 above the second hold-out and 0.018 above the third, and recall is 0.7953 against 0.737 and 0.7445. Arabic-script pairs have a false-alert rate of 3.09% and person pairs in Arabic script 3.68%, against 26.3% and 22.5% for Arabic-script clean names on the second and third hold-outs.

### Verdict

**On the question these labels answer, the engine holds up for people and for organisations, with named exceptions.** Scoring two given entities against each other, it gets 0.969 F1 and 0.975 AUC against analyst labels, and on the harder pairs without a shared name it gets 0.818 F1 (interval 0.814 to 0.822), inside the hold-out band and slightly above it. So the constructed hold-outs did not overstate the F1 of the pairwise scoring. It does not hold up for vessels, securities or crypto wallets, for pairs with no birth date and no identical name (F1 0.745), for organisation pairs without a shared name, for cross-script pairs, or for namesakes: when two different people share a name and a birth date is missing, it alerts about nine times in ten.

**What this does not validate.** The pairs are candidate pairs chosen by OpenSanctions' own pipeline, and each is judged one pair at a time. A false-alert rate of 3.1% on Arabic-script pairs is the share of analyst-rejected candidate pairs the engine scores over the threshold; it is not the rate at which a clean Arabic-script customer alerts against a list of 69,637 entities, which the hold-out measured at 22.6% to 26.3%. Retrieval is bypassed, so the retrieval limits in part 5 and the typo fragility are untouched by this test. Prevalence is a property of the file (77% positive before scoring), not of a customer book, so part 3 stands. The pairs also fix their spellings, so they say nothing about a customer who types a name differently.

The earlier limitation, that the labels were never human-checked, is partly answered. The pairwise scoring is now checked against 439,813 analyst judgements. The list-screening figures (the 26.3% Arabic false-alert rate, the recall of the hold-out groups, the precision at realistic prevalence) are still measured only on constructed queries.

### Would a different threshold do better on these pairs? (analysis only, nothing changed)

The first scoring run saved only aggregates at 0.86, so the pairs were scored a second time with the same code and engine and the per-pair table kept (label, entity-type pair, shared-name flag, forward and reverse score; no names or pair content) in `data/external/pairs_scores_table.npz`, which is git-ignored with the rest of the data. The table was checked against the stored report before use: at 0.86 it gives the same true positives, false positives, false negatives and F1 in all three slices (all pairs 261,790 / 9,922 / 6,700, F1 0.9692). `analysis/pairs_external_sweep.py` then swept the threshold from 0.50 to 0.99 in steps of 0.01 (left names as queries; a pair matches when its score is at or above the threshold) and wrote `reports/pairs_external_threshold_sweep.json`.

| Slice | Pairs | At 0.86: precision / recall / F1 | Best threshold | At best: precision / recall / F1 | F1 gain |
|---|---|---|---|---|---|
| All scored pairs | 439,813 | 0.9635 / 0.9750 / 0.9692 | 0.85 | 0.9608 / 0.9819 / 0.9712 | +0.0020 |
| No shared name | 186,946 | 0.8428 / 0.7953 / 0.8184 | 0.85 | 0.8283 / 0.8117 / 0.8199 | +0.0015 |
| Person to person | 268,027 | 0.9707 / 0.9888 / 0.9797 | 0.85 | 0.9690 / 0.9908 / 0.9798 | +0.0001 |

The gap is small. On these labels 0.86 is within one step of the F1 optimum in every slice, and the curve is flat around it: F1 stays within 0.005 of the best from 0.78 to 0.92 for all pairs, from 0.71 to 0.95 for person to person, and from 0.83 to 0.87 for pairs with no shared name, where it is steepest (F1 0.8047 at 0.80, 0.7998 at 0.90, 0.7318 at 0.95). The threshold is not what limits the pairwise scoring.

Limits of this. The best threshold is picked on the same pairs it is reported on, so the gain is an in-sample figure, and a gain of 0.0001 to 0.002 is below any difference worth acting on. F1 depends on the share of positives in the file (61.0% overall, 12.4% for no shared name), so the optimum would move at a customer book's prevalence, and these pairs are not a customer book. Retrieval is bypassed, so this says nothing about the threshold for list screening. The threshold in the engine stays at 0.86; no decision follows from this table.

## 7. Stage 5: deletion-neighbourhood keys in the candidate index (rejected on dev)

The change adds a key for every single-letter deletion of each indexed token and of each query token, so one wrong, missing or extra letter still retrieves the entity (`src/candidates.py`, behind `deletion_keys`, default off; scoring unchanged). The criteria were written into the project notes before any code changed. Dev is the dev split of the existing generator: 1,000 headline positives and 1,900 headline negatives. v4, the fresh hold-out, was frozen and is **not scored**, because the rule is that dev must pass first. Before is the shipped engine; after is the same engine with the new keys.

| Criterion | Before | After | Limit | Verdict |
|---|---|---|---|---|
| 1. Single-letter slip rate (5,920 variants, 750 positives caught by the before engine) | 13.90% (823) | 12.92% (765) | after at most 0.5 x before, no family rising | **fail** (ratio 0.93) |
| 2. F1 | 0.7400 | 0.7355 | fall of at most 0.005 | pass (fall 0.0045) |
| 3. False-alert rate overall | 12.21% (232 of 1,900) | 13.16% (250 of 1,900) | rise of at most 0.5 points | **fail** (+0.95) |
| 3. False-alert rate, Arabic script | 27.47% | 29.84% | rise of at most 1 point | **fail** (+2.36) |
| 4. Median candidates per query | 488 | 656.5 | at most 2x | pass (1.35x) |
| 4. p95 latency of Screener.components | 84.3 ms | 120.9 ms | at most 2x | pass (1.43x) |

**The fix is rejected**, as max_df 10000 was. Per family (slips of variants, before to after): consonant_delete 24.68% to 20.69% (1,329 variants), consonant_double 11.29% to 11.10% (1,523), vowel_delete 10.24% to 10.18% (1,533), vowel_swap 10.81% to 10.75% (1,535). No family rose, and only consonant deletion moved by more than a fraction of a point. Token edits 24.83% to 24.74%, romanisation 19.90% to 19.15%, controls 0.

Other measurements. Recall 0.750 to 0.755; precision 0.7303 to 0.7170; positives whose true entity was not scored 69 to 60; mean candidates 630.8 to 818.2, p95 1,750 to 2,226. Index over the full list: keys 1,137,909 to 2,013,081, postings 2,565,225 to 6,828,805 (2.7x), memory 210 MB to 392 MB, index build 4.9 s to 8.3 s on an idle machine. Per-script false-alert rate: Latin 4.16% to 4.29%, Cyrillic 3.02% to 3.26%, Arabic 27.47% to 29.84%.

max_df 250 against 1000 with the new index, on dev: F1 0.7355 to 0.7501 (+0.0146), recall 0.755 to 0.743 (-1.2 points), false-alert rate 13.16% to 10.11%, no script's rate rises, median candidates 201, p95 latency 41.4 ms. The adoption rule needed recall to fall by no more than a point, so it is **not adopted**.

Why it failed. The retrieval analysis (`analysis/retrieval_gap.py`) had already shown that only 178 of 1,549 single-letter slips on the third hold-out (11.5%) were retrieval misses; the rest were retrieved and scored under the threshold. A fix that only improves retrieval has a ceiling near that share. That figure existed when I set the bar of halving the slip rate, and I did not apply it; I did not change the bar after seeing the result. The extra false alerts are concentrated in Arabic script (+2.36 points, against +0.13 for Latin and +0.24 for Cyrillic). I have not diagnosed why.

What it leaves. The fragility to a one-letter change is mostly a scoring problem, and a retrieval fix is not the lever for it. If it is pursued, a change to scoring is the next candidate, not another index key.

## 8. A shortened name can clear even with a matching birth date (found in the web page, not fixed)

Found on 2026-10-09 while trying the local web page on a well-known listed person. The list holds "BOUT, Viktor Anatolijevitch" (us_trade_csl, us_ofac_sdn; birth dates 1967-01-13 and 1970-01-13; no country listed). Screening "Viktor Bout" with birth date 1967-01-13 (and country ru, which is not used because the listed entity has no country) gives a top score of **0.7953 against the threshold of 0.86, and the decision is clear**. The same person with the full name "Viktor Anatolijevitch Bout" scores 1.0000 and alerts.

The components for the short name: name similarity 0.674, consonant-skeleton similarity 0.695 (higher, so it is used), birth date agrees (+0.10), country nothing. 0.695 + 0.10 gives 0.795, which is 0.065 under the threshold. The listed name has a patronymic that the query lacks, and that missing token pulls the name similarity under what a matching birth date can make up.

This is a **scoring limitation, and it is not fixed**. The right entity was retrieved and ranked first; it was scored under the threshold, so a retrieval change would not help (consistent with part 7). It was found by one hand-typed query, so this section gives no rate: how often a query that omits a middle name or patronymic clears has not been measured, and the hold-outs do not isolate that case. Nothing was tuned or changed in response.

## What it adds up to

Stated without softening:

1. The headline F1 of 0.777 is a measurement on a constructed set, in which the matcher is given correct birth dates and countries 70% of the time, and the groups it fails are left out. Without the attributes it is 0.587. Over every query it is 0.711. It is not an estimate of accuracy on customers.
2. About 8% of the listed parties tested (158 of 2,000) are close matches that slip under the threshold, mostly people, mostly missing a birth date or country. This is the dangerous kind of false negative and it is present.
3. Precision at realistic prevalence is the weakest point. With the measured false-alert rate, only about 1 alert in 100 is real at a prevalence of 1 in 1,000, and about 7% to 9% of all customers would be alerted. The project does not measure the false-alert rate on real customers.
4. The matcher is fragile to a one-letter change in a name, in 13% of single-letter variants overall and 24% for consonant deletion. Part of the cause is retrieval, not scoring.
5. The tuning is not overfit, but it does not explain the weakness either, and `max_df` is set at a value that is worse for F1 than a smaller one on this slice.
6. Everything above comes from one generator of queries. None of it was tested on names from real customers or real screening traffic.
7. Against analyst labels the pairwise scoring holds up for people and organisations (F1 0.969 overall, 0.818 on pairs with no shared name), and fails for vessels, securities and crypto wallets, for pairs with no birth date and no shared name, and for namesakes (93% alert when no birth date is on both sides). Those labels do not validate the list-screening false-alert rate.
8. A retrieval fix for typos (deletion keys) did not pass dev: the single-letter slip rate fell only from 13.9% to 12.9%, and the false-alert rate rose, 2.4 points for Arabic script. It is rejected.
9. A query that omits a middle name or patronymic can clear even when the birth date matches: "Viktor Bout" with a matching birth date scores 0.7953 against 0.86 and clears, while the full listed name scores 1.0. Found on one hand-typed query; the rate is not measured and the limitation is not fixed (part 8).
