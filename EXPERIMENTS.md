# Did the harness improve the experiment process?

**Yes, in the tested situation: access to a saved failure reduced experiment attempts from four to three in both controlled comparisons, while assessing the same three correct alternatives.** This is a 25% reduction in attempts on one fixed workload with relevant prior evidence. It is not a 25% GPU speedup or a general cost-saving estimate.

The raw [Atlas comparison export](evidence/harness-ablation.json), [independently recomputed summary](evidence/experiment-summary.json), and [measurement script](scripts/measure_harness.py) make the claim inspectable.

## Two different questions

1. **Does the experiment manager waste less work?** Measure attempts needed to assess the same correct alternatives, plus repeated known-invalid attempts. Fewer is better, provided correct alternatives are still checked.
2. **Does a tested alternative make the calculation faster?** Measure its execution time against the reference calculation under the same conditions. Lower is better, provided every required answer check passes.

The first improved in this comparison. The second has no supported improvement. Keeping these outcomes separate prevents a reduction in testing work from being presented as a faster calculation.

## Experiment 1: change only access to past evidence

| Design concept | Implementation |
| --- | --- |
| Hypothesis | Access to a known previous failure reduces repeated incorrect tests without dropping the reference set of correct alternatives. |
| Comparison without memory | The same program cannot read results from previous runs. It still sees results from its own current run. |
| Comparison with memory | The same program receives a frozen copy of four real Atlas experiment records, including one observed correctness failure. |
| Factor changed | Availability of previous-run evidence. |
| Held fixed | GPU, software, input values and seed, workload, candidate order, candidate allowlist, answer checks, timing protocol, and five-attempt budget. |
| Primary outcome | Actual experiment attempts, using the without-memory count as the denominator. |
| Guardrails | All three previously validated alternatives are evaluated in both conditions; no incorrect result is accepted; no worker errors or extra attempts. |
| Repetition and order | Two blocks: without/with memory, then with/without memory. This is counterbalanced, not randomized. |
| Data isolation | Each memory-enabled arm uses only the same seed run; neither arm can learn from the other comparison arms. Both retain their own current-run results. |
| Secondary outcome | Elapsed time for the run loop, including Python worker startup and Atlas operations. Seed-history setup is outside this interval. |

The protocol was implemented before collecting these observations. The hypothesis did not force the outcome: the script checks actual counts, correct-alternative coverage, history references, errors, and attempts. `protocol_pass` and `hypothesis_supported` were both true. A separate audit recomputed these checks from raw experiment and decision documents.

## Observed results

| Block | Order | Attempts without memory | Attempts with memory | Correct alternatives in each | Known-invalid attempt repeated, without → with |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Without, then with | 4 | 3 | Same 3 | 1 → 0 |
| 2 | With, then without | 4 | 3 | Same 3 | 1 → 0 |

The reduction in each block is `(4 - 3) / 4 × 100 = 25%`. Both conditions checked stable sort, stable argsort, and packed-key top-k. Neither accepted an incorrect result. Every experiment had one execution attempt.

The without-memory condition tried direct top-k, observed its tie-order failure, then screened unstable sort using its newly learned evidence. The with-memory condition already had that failure and skipped the repeat. Therefore the appropriate comparison is **four versus three**, not the five-attempt budget versus three. Neither condition establishes that every untested alternative is invalid; unstable sort is excluded by a conservative shared rule.

| Block | Without-memory elapsed time | With-memory elapsed time | Descriptive reduction |
| --- | ---: | ---: | ---: |
| 1 | 8.492 s | 6.878 s | 19.0% |
| 2 | 8.392 s | 6.532 s | 22.2% |

Elapsed time is secondary evidence from two short comparisons, not a production speed estimate. It includes host/process/database overhead and cannot be converted into GPU-hour savings. The supplied prior history's acquisition cost is excluded: this measures reuse after evidence already exists.

Atlas provenance:

| Role | Run ID |
| --- | --- |
| Shared seed and fresh recovery demo | `28236d7c01de4472b7ef84bdbe30379e` |
| Block 1, memory off | `acb1562060ac4b82ba5b108b441fd2a3` |
| Block 1, memory on | `892d2f2d8d424d63a9b7bf4c578abc3f` |
| Block 2, memory on | `04a084f0a70a45be83a1ddf1ba13c6ce` |
| Block 2, memory off | `5565271cdc0340b7a83e1194e1acfdc4` |

All four comparison runs used evaluator version 3, the same input/contract/context hashes, and recorded source revision `d862a51`. Their `runs.experiment_design` fields identify condition, block, order, and seed. Each saved decision links to exactly the permitted seed records plus its own earlier completed experiments. The seed was not modified.

## Experiment 2: is the calculation actually faster?

The performance experiment changes the implementation while keeping the input, expected answer, GPU, and software fixed. Each correct alternative receives 30 initial timing samples after warmup, then 12 nearby reference/alternative comparisons in alternating execution order. Incorrect alternatives are excluded before timing.

A measurement audit found that evaluator version 2 warmed the alternative but not the reference before the paired recheck. Its packed-key comparisons contained approximately 15 ms first-reference outliers. **Version 3 warms both implementations and synchronizes before pairing.** The version bump prevents mixing that new protocol with old compatibility hashes. Historical reports remain available and are labeled as earlier evidence.

For each pair, the directional effect is `100 × (reference time - alternative time) / reference time`. Positive means faster; negative means slower. We summarize the median of those paired effects, rather than treating separate runs' smallest medians as proof.

Fresh version-3 results for stable argsort:

| Run | Initial separate-median reduction | Median paired reduction | Pairs faster than reference |
| --- | ---: | ---: | ---: |
| Seed | -7.1% | -6.8% | 3/12 |
| Block 1, memory off | -0.2% | -2.8% | 5/12 |
| Block 1, memory on | +10.7% | +1.1% | 6/12 |
| Block 2, memory on | -16.0% | -4.6% | 5/12 |
| Block 2, memory off | -19.0% | +0.9% | 6/12 |

The direction changes and argsort wins at most half the pairs. **These observations do not support a repeatable speedup.** Packed-key top-k was also slower in every fresh paired comparison. The same conclusion remains after correcting warmup.

The existing acceptance rule requires every contender sample to beat every reference sample within a recheck. All fresh flags were false. This is a conservative sample-separation rule, not a statistical confidence interval. Twelve nearby pairs are not twelve independent machines or workloads; the five runs also share one machine and fixed input. No population confidence interval or significance claim is reported.

The current search order is fixed. It uses correctness failures to change eligibility; it does not learn a new search strategy from latency. `best_experiment` means lowest initial median among correct results and must not be described as a confirmed faster replacement.

## Experiment 3: saved work survives an interruption

The fresh seed run used `--fault-after-commit`. It exited with code 86 after the correct stable baseline was saved. A fresh `status` process showed one completed experiment while the run was still marked running. `resume` completed the other three experiments; the baseline remained at attempt 1. Its [export](evidence/report-28236d7c01de4472b7ef84bdbe30379e.json) preserves the final state.

This tests the specific failure boundary after saving a result and before updating progress. Work interrupted before its result is saved can execute again. There is no matched restart-from-scratch timing comparison here, so recovery is presented as a verified behavior rather than a percentage efficiency result.

## Reproduce or inspect

Use a configured Atlas connection and the repository's CUDA environment. The live comparison creates four new runs and preserves the seed:

```bash
.venv/bin/python scripts/measure_harness.py \
  --seed-run-id 28236d7c01de4472b7ef84bdbe30379e \
  --output artifacts/new-harness-comparison.json
```

The seed must exist in your configured Atlas database and match the current environment. On a new compatible database, run a four-evaluation seed first and pass its ID. The script refuses to overwrite an existing output file. Recompute the published presentation metrics without Atlas or a GPU:

```bash
.venv/bin/python scripts/summarize_experiments.py
```

Five existing tests passed after the warmup change, followed by the real recovery run and four real Atlas/GPU comparison runs. The raw evidence, not the mocked tests, supports the live measurements above.

## What the evidence can support

The demonstrated improvement is **less repeated testing after a relevant failure has already been learned**. There is no evidence yet of faster kernels, customer adoption, production savings, general learning across workloads, or sustained autonomous reasoning. A larger evaluation should include many tasks, cold starts with no history, potentially stale/mismatched history, and the cost of acquiring that history before estimating broader benefit.

The design follows the rationale of controlling nuisance factors in [NIST's experimental-design guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm) and warming/synchronizing GPU measurements in [PyTorch's benchmarking guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html). Our two-block study remains a narrow demonstration.
