# Monitoring report

Aggregate figures only. No names and no request IDs appear in this report.

## Integrity

| Item | Value |
|---|---|
| verify_chain | clean |
| Records checked | 801 |
| Payloads checked | 800 |
| Payloads purged (covered by a purge record) | 0 |
| Tampering-class findings | 0 |

No crash-class findings.

## Window and run contexts

| Item | Value |
|---|---|
| First record (seq) | 1 |
| Last record (seq) | 800 |
| Records | 800 |
| Decisions | 800 |
| First timestamp (UTC) | 2026-10-07T10:00:00.000000+00:00 |
| Last timestamp (UTC) | 2026-10-07T10:00:00.000000+00:00 |
| Metrics generated (UTC) | 2026-10-08T12:00:00+00:00 |

Run context groups. Metrics are never mixed across run contexts.

| Run context | Decision context | Decisions |
|---|---|---|
| {RUN}... | {DEC}... | 800 |

## Metrics

Near-threshold band 0.05, min_cell 10, config file sha256 none.

### Run context {RUN}...

| Item | Value |
|---|---|
| Decisions | 800 |
| Alerts | 210 |
| Alert rate | 0.2625 |
| Near-threshold decisions | 110 |
| Near-threshold rate | 0.1375 |
| No candidates | 0 |
| Duration p50, p95, p99 in ms (information only) | 5, 5, 5 |

Final score histogram (20 fixed bins):

| Score bin | Decisions |
|---|---|
| 0.00 to 0.05 | 20 |
| 0.05 to 0.10 | 40 |
| 0.10 to 0.15 | 120 |
| 0.15 to 0.20 | 60 |
| 0.20 to 0.25 | 80 |
| 0.25 to 0.30 | 80 |
| 0.30 to 0.35 | 60 |
| 0.35 to 0.40 | 60 |
| 0.40 to 0.45 | 20 |
| 0.45 to 0.50 | 20 |
| 0.50 to 0.55 | 0 |
| 0.55 to 0.60 | 0 |
| 0.60 to 0.65 | 0 |
| 0.65 to 0.70 | 0 |
| 0.70 to 0.75 | 0 |
| 0.75 to 0.80 | 0 |
| 0.80 to 0.85 | 30 |
| 0.85 to 0.90 | 80 |
| 0.90 to 0.95 | 80 |
| 0.95 to 1.00 | 50 |

By script. Cells below min_cell are suppressed as in the JSON report.

| Script | Decisions | Alert rate |
|---|---|---|
| Latin | 400 | 0.2500 |
| Cyrillic | 200 | 0.2000 |
| Arabic | 200 | 0.3500 |

## Drift against the baseline

### Decision context {DEC}...

Overall status: ok.

The overall status is the worst of k = 5 PSI comparisons and so uses Bonferroni-corrected percentiles: 99.0000 for warn (alpha 0.05) and 99.8000 for alert (alpha 0.01). Overall status without the correction: ok.

Live decisions 800, baseline decisions 800. Limits are the larger of the convention (0.1 and 0.25) and the pooled two-sample null percentile. Each status uses the uncorrected percentiles; the last column uses the corrected ones.

| PSI comparison | PSI | n live | n baseline | Null p95 | Null p99 | Warn limit | Alert limit | Status | Corrected status |
|---|---|---|---|---|---|---|---|---|---|
| Score | 0.0000 | 800 | 800 | 0.0549 | 0.0677 | 0.1000 | 0.2500 | ok | ok |
| Input mix | 0.0000 | 800 | 800 | 0.0148 | 0.0241 | 0.1000 | 0.2500 | ok | ok |
| Latin score | 0.0000 | 400 | 400 | 0.0889 | 0.1155 | 0.1000 | 0.2500 | ok | ok |
| Cyrillic score | 0.0000 | 200 | 200 | 0.1570 | 0.2041 | 0.1570 | 0.2500 | ok | ok |
| Arabic score | 0.0000 | 200 | 200 | 0.1730 | 0.2222 | 0.1730 | 0.2500 | ok | ok |

| Rate comparison | Baseline | Live | Difference | Ratio | Status |
|---|---|---|---|---|---|
| Near-threshold rate | 0.1375 | 0.1375 | +0.0000 | n/a | ok |
| Latin alert rate | 0.2500 | 0.2500 | +0.0000 | 1.0000 | ok |
| Cyrillic alert rate | 0.2000 | 0.2000 | +0.0000 | 1.0000 | ok |
| Arabic alert rate | 0.3500 | 0.3500 | +0.0000 | 1.0000 | ok |

## Latest list-update gate report

| Item | Value |
|---|---|
| File | stage4_list_update_gate_20261008075007-crk.json |
| Label | forward list update |
| Direction | the new snapshot is newer |
| Old snapshot | 20261006074704-hib |
| New snapshot | 20261008075007-crk |
| Hold-out | screening_holdout_v3.json |
| Hold-out sha256 | cccccccccccc... |
| Engine | the current engine: the Stage 2 hand-tuned matcher plus the Stage 3 rarity-weighted skeleton |
| Positives kept, remapped, dropped | 90, 2, 1 |
| F1 old to new | 0.8000 to 0.7900 |
| F1 delta (95% interval) | -0.0100 (-0.0200 to 0.0000) |
| Arabic burden old to new | 0.2200 to 0.2300 |
| Arabic burden delta (95% interval) | 0.0100 (-0.0100 to 0.0300) |
| Gate | PASS |

Figures the headline numbers must not be confused with:

| Figure | What it is |
|---|---|
| F1 0.777 | the Stage 2 engine on the SECOND hold-out |

The gate result is shown for information and does not change this report's exit status.

## Limits of this report

Monitoring is demonstrated on constructed batches, not production traffic. The baseline and the stable, mix-shift and degraded batches are samples of a spent hold-out, so how often a live deployment would warn or alert is unknown.

There are no analyst dispositions, so live precision and recall cannot be measured, and neither can the false-alert burden (the 22.6% Arabic result of Stage 3): live traffic has no labels saying which alerts were wrong. A change in an alert rate is a reason to look, not a measurement of harm.

The deduplication model is outside the decision path and is not monitored.

Duration figures depend on the machine and are shown for information only.
