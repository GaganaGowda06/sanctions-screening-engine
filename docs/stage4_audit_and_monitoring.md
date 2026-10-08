## Stage 4: audit log and monitoring

Stage 4 adds no matching logic. The matcher's scores and decisions are unchanged, and a test compares the audited path with `screen()` on every query of the three hold-outs. What it adds is a way to answer, later, the question an auditor or a regulator would ask: why was this name flagged or cleared on this date, under which list, code, configuration and threshold. It also adds a way to notice that the matcher's behaviour has moved, and a check that a new list snapshot does not make it worse.

Two terms are used throughout. The third hold-out has 2,400 positives (all of its positive queries, each a real alias of a listed entity) and 2,400 negatives. The headline F1 uses only 2,000 of the positives, called the headline positives (the Stage 2 headline groups; the other 400 are other-script groups reported apart), and 2,000 headline negatives (the Latin, Cyrillic and Arabic script negatives). Every count below says which it is.

### The regulatory basis

The design follows two parts of the Money Laundering, Terrorist Financing and Transfer of Funds (Information on the Payer) Regulations 2017, as I read them from the text on legislation.gov.uk (SI 2017/692). This is not legal advice and has not been reviewed by a compliance team. Whether a screening decision record is one of the records regulation 40 lists is my reading, not something I have checked with anyone.

- Regulation 40 requires a firm to keep certain records (customer due diligence documents and information, and supporting records of transactions) for five years from the date it knows, or has reasonable grounds to believe, that an occasional transaction is complete or that the business relationship has ended. For records of transactions within a relationship it need not keep them for more than ten years. Once the period has expired it must delete the personal data obtained for the Regulations, unless an exception applies: it is required to keep the data by law or for court proceedings, the data subject consented, or it has reasonable grounds to believe the data is needed for legal proceedings. So the store must keep enough to explain a decision for the period, and must be able to delete the personal data on its own without destroying the evidence that a decision was made. The purge command implements the five-year minimum from an end date the caller supplies. It does not enforce the ten-year cap, and it models the exceptions only through the legal hold below.
- Regulation 87 is the offence of prejudicing an investigation. It applies where a person knows or suspects that an appropriate officer (for example of the FCA or HMRC) is acting, or proposing to act, in connection with an investigation into a potential contravention of the Regulations. Falsifying, concealing, destroying or disposing of documents relevant to that investigation is an offence, with defences where the person does not know or suspect the documents are relevant or does not intend to conceal what they disclose. So deletion must stop for relevant records as soon as someone knows or suspects an investigation, and the store cannot know that: the hold list belongs to the caller.

### The audit design

Every screening decision is written to an audit store before the answer is returned. If the record cannot be written, the call fails and no decision is produced: a name is never screened unlogged.

A record has a header and, for decisions, a payload.

- The header holds the request ID, the UTC time, the hash of the run context, the candidates with their numeric scores, the final score, the threshold, the decision, the wall time of the evaluation in whole milliseconds, and a digest of the payload. It holds nothing taken from the customer's name, birth date or country, not even the script of the name. Headers go into one append-only JSON lines file.
- The payload holds the raw input, the normalised forms, the name tokens, the retrieval keys, an optional opaque `customer_ref` supplied by the caller, and a random 32-byte salt. Payloads live in a separate store, one file per request, so one can be deleted without touching the chain.
- The digest in the header is SHA-256 over the salt and the canonical JSON of the payload. Because the salt is in the payload, deleting the payload also removes the means of guessing the name back from the digest.
- The run context is an immutable record of what the engine ran under: the git commit and whether the tree was dirty, the configuration and threshold, the engine settings the configuration does not hold (such as the candidate cap), the list snapshot ID and SHA-256, the rarity table hash, the SHA-256 of every score-time source file, library versions, and the Python and Unicode database versions. Its hash is computed once, when the engine is built, and every record takes the hash from there. The context is also stored as a file in the audit directory, so a later reader can see what changed between two hashes.
- The chain. Each header carries `seq`, the hash of the previous record and its own hash over the canonical JSON of the record. A checkpoint file holds the record count and the last hash, so deleting the last lines of the log is visible. Appends take an exclusive file lock from reading the head to writing the checkpoint, and the payload is written inside the same lock, so concurrent writers cannot fork the chain and verification alongside writers never sees half a record. The log refuses to append when its head is not trustworthy.
- `verify_chain` checks the whole store: the chain, the checkpoint, every payload against its digest, every record against the allowed fields for its kind, and payload files that no header points to. It separates tampering (edited or missing records, broken links, a missing or mismatched checkpoint, a missing payload nothing explains) from crash damage (a torn final line, a checkpoint one record behind, one orphan payload, an interrupted purge).

Every record has a kind:

- `decision`: a name was screened. The only kind with a payload and the only kind that can be replayed.
- `engine_start`: an engine started under a run context different from the last one in the chain. It holds the old and new context hashes and the names of the fields that changed, never their values. A change of threshold, configuration, code, list snapshot or library version is therefore recorded automatically the first time an engine runs under it.
- `recovery`: an operator ran `recover` after a crash during a write. See below.
- `purge`: payloads were deleted at the end of their retention period. See below.

Replay. `python src/screen_cli.py replay REQUEST_ID` finds the header, checks its hash and the payload digest, and runs the decision again from the payload. It requires the live engine to have the same run context as the record, and otherwise refuses with a field-level diff naming the commit and snapshot needed. With the same context it requires identical candidate IDs, per-feature scores (exact float equality), final score, threshold and decision. Replay writes nothing. Replaying a record made under an older context is a manual procedure: check out the recorded commit, restore the recorded snapshot, rebuild, then replay. If the tree was dirty when the record was made, the commit alone may not be enough.

Recovery. A crash during a write can leave a partial last line, a payload whose header was never written, or both. `recover` moves the partial line into `quarantine/`, cuts the log back to its last complete line, and deletes the orphan payload (no decision was made, so there is no reason to keep the personal data in it). It appends one `recovery` record holding the SHA-256 and size of the torn bytes and the request ID and file hash of the orphan, never any content. It refuses a clean log, more than one orphan, and any sign of tampering.

Retention and legal hold. Regulation 40 counts from the end of the relationship, or from the completion of an occasional transaction, and the matcher knows neither, so a timestamp-based expiry would delete the records of customers who are still customers. `purge --mapping FILE` takes the caller's mapping of `customer_ref` to relationship end date (JSON lines; for an occasional transaction, supply the date it was completed). A payload is deleted only if it has a `customer_ref` in the mapping, its record is dated on or before that end date, and the end date plus five years is in the past. Records with no `customer_ref`, a reference not in the mapping, or a date after their customer's end date (a later relationship) are never deleted. The `purge` record is chained first and the files are deleted after, so a crash in between shows as `purge_incomplete` and running purge again finishes the job. `verify_chain` treats a missing payload that a purge record covers as purged, and replay of such a record says it was purged. `--dry-run` reports the counts and changes nothing. `purge --hold FILE` takes JSON lines, each with a `customer_ref` or a `request_id`; a held record is never purged, whatever its end date. Holds are checked before every eligibility rule, and the purge record names no holds and no customer references. A hold on a request that was already purged cannot bring it back, and is reported. Dates are UTC.

Purge deletes files only. It is not a secure erase: copies in backups, filesystem snapshots, journals or swap are outside its reach, and production retention has to cover them, for example by encrypting payloads at rest and destroying the keys, or by expiring backups on the same schedule.

The tamper-evident limit. The chain and the checkpoint are tamper-evident, not tamper-proof. Anyone who can rewrite the log can rewrite the checkpoint, so the chain only shows tampering by someone who cannot. Deleting the whole audit directory leaves nothing to compare with, and a fresh store looks the same. Production would anchor the head outside the machine, in write-once storage or a periodic signed export of the count and last hash.

The CLI has explicit subcommands: `screen`, `init`, `replay`, `recover`, `purge`, `metrics`, `drift`, `report` and `gate`. The audit directory comes from `--audit-dir` or `$SCREENING_AUDIT_DIR` and has no default, because payloads hold personal data. Screening never creates a store: `init` does, and refuses a directory that is not empty, because a wiped trail and a new deployment look the same from inside and a person has to decide which it is.

### Monitoring

`python src/screen_cli.py metrics --from SEQ --to SEQ` reads the decision records in a window of the chain (by `seq`) and writes an aggregate report to `monitoring/<from>-<to>.json`, created once and never overwritten. Before measuring it verifies the whole chain and refuses on any sign of tampering; crash damage is allowed and listed by type and count.

- Records are grouped by run context hash and never mixed across contexts, so a change of threshold, code or list starts a new group.
- Per group: the count and alert rate, the final score histogram over 20 fixed equal bins on [0, 1] (fixed so a later drift comparison uses the same bins), the count and rate of decisions within a band of the threshold (0.05), per-script counts and alert rates, and p50, p95 and p99 of the duration.
- A record's script comes from the raw name in its payload, with the same script detection as the Stage 3 fairness work. A purged or missing payload counts as `unavailable`.
- Suppression. `min_cell` (default 10, in `src/monitoring_config.json`). A group with fewer decisions shows only its count as `<10`. A per-script cell below `min_cell` shows its count as `<10` and no rate. If exactly one cell in a group is hidden, the smallest visible cell is withheld too, so the hidden count cannot be worked out as the total minus the rest; when a whole group is hidden the window's decision total is withheld for the same reason.
- The report holds no request ID, name or customer reference, and records the SHA-256 of the config file it used.

### Drift

`python src/make_baseline.py --audit-dir DIR` screens a seeded sample of the spent third hold-out through the audited path into a dedicated audit directory, measures it, and saves `baseline/<decision_context_hash>.json`. A baseline applies to the decision context of the engine that made it: the configuration and threshold, list snapshot, scoring and normalisation code, rarity table, library versions, and Python and Unicode database versions. It leaves out the git commit, the dirty flag and the audit schema, so a new commit with the same decisions keeps matching, while replay still needs the full context hash. A report from a different decision context is refused.

`python src/screen_cli.py drift --report FILE` compares a metrics report with its baseline:

- PSI over the 20 score bins, and PSI over each script's share of the decisions (reported separately);
- PSI of the score distribution within each script that has enough decisions in both;
- each script's alert rate against the baseline, as an absolute difference and a ratio (a status needs both past their limits);
- the near-threshold rate against the baseline.

Durations are shown beside the baseline's as information only, never compared. A comparison with fewer than 200 decisions on either side gets `insufficient_data`.

Limits. The conventional PSI limits (0.1 to warn, 0.25 to alert) come from credit scorecard monitoring, not from this matcher or this data, and they assume large samples. At the sizes here sampling noise alone reaches the warn limit: two random halves of the 796 Latin baseline decisions (about 400 each) have a median PSI of 0.10. So every PSI comparison builds a two-sample null. The baseline and live histograms are pooled into one set of proportions, and 1,000 times with a fixed seed `n_baseline` and `n_live` decisions are drawn multinomially from it and their PSI is taken. The limit is the larger of the convention and the null: warn above max(0.1, null 95th percentile), alert above max(0.25, null 99th percentile). A first version drew only the live sample and treated the baseline as exact; it understated the noise, and a clean batch warned.

Multiplicity. The overall status is the worst of k comparisons, so a clean batch would warn far more often than 5% of the time if each comparison used its own 95th percentile. The overall status therefore uses Bonferroni-corrected percentiles, 1 - 0.05/k for warn and 1 - 0.01/k for alert, where k is the number of PSI comparisons in that report. The report states k and the corrected percentiles, and each comparison keeps its uncorrected status beside its corrected one. The alert-rate and near-threshold limits are not PSI and keep fixed values (judgement).

Demonstrated on constructed batches. `reports/stage4_drift_batches.json` records, for each batch, the pattern written down in the code before the run and what was observed. The limits were not adjusted to fit.

| Batch | Expected | Seen | Numbers |
|---|---|---|---|
| Stable (one fresh sample of 800) | all ok | yes | overall ok; score PSI 0.069 (limits 0.1 and 0.25); script mix PSI 0.003; Latin PSI 0.122 against a null 95th percentile of 0.121, which is a warn on its own and ok after the correction for k = 3 |
| Script-mix shift (400 Arabic-script, 400 other) | input-mix PSI moves, within-script PSI stays ok | yes | input-mix PSI 0.62 (alert); Arabic 0.17 against an uncorrected warn limit of 0.24 and Latin 0.16 against 0.19, both ok (also ok against the corrected limits); near-threshold rate 0.027 to 0.068 (warn); overall alert |
| Degraded (no birth date, name cut to first token) | score PSI and near-threshold rate move | yes | score PSI 0.63 (alert), Latin 0.95 (alert); near-threshold rate 0.027 to 0.195 (alert) |

One stable batch cannot be asserted against a probabilistic limit, so after the first run the stable expectation was changed, and the change is recorded in the test: over 50 seeded stable batches the overall status is warn or worse in at most 10% and alert in at most 2%. It was warn or worse in 1 of 50 (2%) and alert in none. Without the correction the overall warn rate would have been 8% (4 of 50). Each batch had k = 3, because only Latin has 200 decisions in a batch of 800. The 50 batches are samples of one pool of 3,300 screened queries, so they overlap and are not independent draws; the 2% is less certain than 50 independent batches would make it.

### The list-update regression gate

`python src/download_snapshot.py --after-current` fetches the latest published OpenSanctions snapshot if it is newer than the current one, into `data/snapshots/` (git-ignored), and keeps it only if its SHA-1 and size match the index. `python src/screen_cli.py gate --new-snapshot PATH` then asks whether moving to it hurts the matcher on the spent third hold-out, used here as a fixed regression benchmark and never to tune anything:

1. Each positive query names the listed entity it should find. That entity is looked up in the new snapshot's screening list: present means kept. Otherwise a referent link in either direction remaps it. Otherwise it is dropped. Dropped queries are counted by script and reason. Negatives are never dropped.
2. The same configuration is run on both snapshots through `screen()` (the research path, no audit records), with the positive's tested alias held out of the list by the hold-out's own rule.
3. F1 (the Stage 2 headline definition) and the Arabic false-alert burden (the alert rate on synthetic clean Arabic-script names, the Stage 3 metric) are computed on the resolved set for both snapshots, and a paired bootstrap (1,000 draws, fixed seed) gives each delta with a 95% interval.
4. The gate fails when the F1 interval lies entirely below minus 0.01, or the Arabic burden interval lies entirely above plus 0.02. Both tolerances are judgement, in `src/monitoring_config.json`. An interval that touches a tolerance passes.
5. New alerts on hold-out negatives that appear only with the new list are counted, and those caused by entities that exist only in the new snapshot are separated out.

Exit status: 0 pass, 1 fail, 2 refused (a snapshot that is missing, unverifiable against its VERSION file, or the same as the current one). If the new snapshot is older than the current one, every output carries the label REVERSE-TIME CHURN TEST.

Which hold-out is behind which number. The gate's F1 of 0.8010 is the current engine (the hand-tuned matcher with the rarity-weighted skeleton from Stage 3) on the third hold-out (`screening_holdout_v3.json`), over the positives that survive resolution in the forward run (1,999 of the 2,000 headline positives). It is not comparable with the F1 of 0.777 in the README's Final results of the second hold-outs (`reports/screening_holdout_v2_evaluation.json`), which is the Stage 2 engine on the second hold-out. `src/f1_provenance.py` separates the two on the third hold-out (`reports/stage4_f1_provenance.json`): the Stage 2 engine scores F1 0.763 there and the current engine 0.8008 over all 2,000 headline positives (Arabic false-alert burden 32.6% and 22.6%). From 0.777 to 0.8010, a hold-out effect of about -0.014 is outweighed by an engine effect of about +0.038, the rarity-weighted skeleton. Both engines in that comparison run on the current snapshot (20261006074704-hib), not the ones the earlier hold-outs were scored against, so the hold-out effect also contains list drift. Every gate report carries a `provenance` section naming the hold-out file and its SHA-256, the engine, and the other figures it must not be confused with.

Forward run (`reports/stage4_list_update_gate_20261008075007-crk.json`, snapshot 20261006074704-hib to 20261008075007-crk, two days apart): the screening list went from 69,637 to 69,620 entities (10 added, 27 removed, 101 changed). Of the 2,400 positives, 2,398 were kept, none remapped and 2 dropped (both entities are still in the file but no longer targets), which leaves 1,999 of the 2,000 headline positives. F1 was 0.8010 on both snapshots (delta 0.0000, 95% interval 0.0000 to 0.0000) and the Arabic burden 0.2257 on both. No negative gained or lost an alert. Gate: PASS. A two-day update changes few entities, so this shows the machinery works end to end and that this update is harmless; it does not show how the gate behaves on a large update.

REVERSE-TIME CHURN TEST (`reports/stage4_list_update_gate_REVERSE_TIME_20260701194702-fjb.json`). The current snapshot was treated as the old list and the 2026-07-01 snapshot as the new one. This is not a forecast of a future update. It runs time backwards, so "dropped" means an entity listed after 1 July, not one that was delisted. There was no pass or fail expectation; what happened:

| | Reverse-time result |
|---|---|
| Screening list | 69,637 to 68,349: 3,628 added, 4,916 removed, 8,708 changed, 56,013 unchanged |
| Positives (all 2,400) | 2,255 kept, 53 remapped through referents (4 ambiguous), 92 dropped |
| By script (kept / remapped / dropped, all positives) | Arabic 288 / 7 / 5, Cyrillic 259 / 11 / 30, Latin 1,322 / 26 / 52, other 386 / 9 / 5 |
| F1 on the resolved set (1,913 of the 2,000 headline positives) | 0.8077 to 0.8070, delta -0.0007, 95% interval -0.0051 to +0.0038 |
| Arabic false-alert burden (the 350 synthetic Arabic-script negatives) | 0.2257 to 0.2257, delta 0.0000, 95% interval -0.0157 to +0.0166 |
| Negatives (all 2,400; none dropped) | 8 gain an alert (2 from entities only on the 1 July list), 9 lose one |
| Gate | PASS |

The reverse-time run showed the headline figures to be robust to about three months of list churn. The 92 dropped positives (counted over all positives) are removed from both sides, so the deltas describe the 2,308 positives (1,913 of them headline positives) that could be resolved, not the full hold-out. The gate has failed only on hand-built cases in the tests (a delisting, a fall in F1 beyond tolerance, new entities that raise the Arabic burden), never on real data, so its failing path has not been exercised on a real list update.

### The monitoring report

`python src/screen_cli.py report --from SEQ --to SEQ` writes `monitoring/<from>-<to>.md`, a plain markdown report assembled from the outputs above and calculating nothing new: the `verify_chain` status and finding counts, the window and run context groups, the metrics with suppression as in the JSON, every drift comparison with its PSI, sample sizes, null percentiles, limits and status together with the overall status, k and the corrected percentiles, the latest gate report with its provenance and its REVERSE-TIME CHURN TEST label if it has one, and a fixed section on the limits of the report. The exit status is the worst status: 0 ok, 3 insufficient data, 10 warn, 20 alert, 2 refused (tampering, or no baseline that matches). The gate result is shown but does not change the exit status.

### What this does not show

- Monitoring is demonstrated on constructed batches, not production traffic. The baseline and the stable, mix-shift and degraded batches are samples of a spent hold-out, so how often a live deployment would warn or alert is unknown.
- There are no analyst dispositions, so live precision and recall cannot be measured, and neither can the false-alert burden (the 22.6% Arabic result of Stage 3): live traffic has no labels saying which alerts were wrong. A change in an alert rate is a reason to look, not a measurement of harm.
- The deduplication model is outside the decision path and is not monitored.
