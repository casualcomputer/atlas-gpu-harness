# Hackathon review: what we can prove

Reviewed October 11, 2026 against the supplied Harness Engineering & Model Wrangling participant guide. This is a strategy and evidence audit, not a prediction of judging results. See [EXPERIMENTS.md](EXPERIMENTS.md) for the measured comparisons and their limits.

## The problem in ordinary language

An engineer asks a computer to find a faster way to perform a calculation. The computer tries alternatives, but some return the wrong answer and others only look faster because measurements fluctuate. If the program stops unexpectedly, it can also forget completed work and repeat earlier mistakes.

Atlas GPU Harness is an **experiment manager with a permanent lab notebook**. It checks each answer, measures correct alternatives, saves the evidence in Atlas, and reads that evidence before deciding what to try next. A restart can continue from saved work. A later run can avoid a previously disproved alternative.

The intended user is an engineer running repeated GPU optimization experiments. The measurable benefit is **less repeated or known-invalid work while preserving the same correct alternatives**. A separate question is whether any alternative actually makes the calculation faster. Those outcomes must be reported separately.

## Is this an important problem?

There is primary-source evidence for the broader engineering problem:

- Anthropic reports that long-running agents lose continuity across sessions, spend effort reconstructing earlier work, and sometimes declare success without adequate testing. That supports preserving verifiable progress outside a session. [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- KernelBench evaluates generated GPU kernels on both functional correctness and speedup. That supports requiring the right answer before accepting a performance result. [KernelBench paper](https://arxiv.org/abs/2502.10517)
- PyTorch documents how missing warmup and GPU synchronization can distort benchmarks. That supports careful measurement rather than selecting the lowest isolated timing. [PyTorch benchmarking recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

These sources establish relevant failure modes. They do not establish demand for this particular product. We have not measured customer adoption, willingness to pay, production GPU-hour savings, or return on investment. This prototype covers one operator on one desktop; any broader savings remain a hypothesis.

## Show improvement with a fair comparison

The matched experiment in [EXPERIMENTS.md](EXPERIMENTS.md) tests the contribution of saved history. The questions and comparisons are:

| Question | Comparison | Outcome and guardrail |
| --- | --- | --- |
| Does remembering failures avoid wasted work? | Same candidate list and evaluation rules, with historical failure retrieval disabled versus enabled | Count actual evaluations; require equal coverage of correct alternatives. |
| Does saved progress survive an interruption? | Stop after saving a result, then resume in a fresh process | Verify the saved experiment is not repeated and unfinished work continues. A quantified advantage needs an explicit restart-from-scratch comparator. |
| Is a candidate actually faster? | Compare baseline and candidate on the same input and hardware, with warmup, repeated measurements, and balanced execution order | Report paired timings and uncertainty; reject incorrect candidates first. |

Keep workload, seed, candidate order, correctness rules, software, GPU, and evaluation budget fixed when testing the effect of memory. Explain intentional differences between conditions. Do not equate unused budget with percentage improvement: the denominator is work the comparison actually performed.

Use counts for repeated work and known-invalid evaluations; use elapsed time for runtime; use correctness checks as guardrails. A lower count does not automatically imply lower end-to-end latency because database and process overhead still cost time. Repeated timing samples within one run are not independent replications across workloads or machines.

NIST explains why comparisons should control nuisance factors and randomize remaining variation. The project should describe which controls it implements, without claiming a full randomized study merely because measurements alternate. [NIST experimental-design guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)

Actual protocols, results, and limitations belong in [EXPERIMENTS.md](EXPERIMENTS.md); earlier observations and raw exports remain in [EVIDENCE.md](EVIDENCE.md).

## What the judging rubric rewards

The supplied guide gives these round-one weights; round two weights all four criteria equally.

| Criterion | Observable demo proof | Remaining gap or claim boundary |
| --- | --- | --- |
| Impact potential: 20% | Show the same correct alternatives assessed with fewer repeated or known-invalid evaluations in a matched comparison. | No customer validation or measured production cost savings. |
| Technical demo: 35% | Run a real GPU experiment, show its Atlas record, deliberately stop the program, restart, and verify preserved progress. Then show the memory comparison. | A slide summarizing success is weaker evidence than the working sequence and its saved records. |
| Creativity: 15% | Open the saved failure and the later decision that retrieved it; show that stored evidence changes the next action. | Persistence and experiment tracking already exist. Distinction rests on this integrated, verifiable GPU workflow. |
| Implementation difficulty: 30% | Explain independent answer checking, GPU timing, stable experiment identity, guarded writes, and recovery from a stale progress summary. | Five predefined candidates and a deterministic policy limit the search and reasoning challenge. |

Technical demonstration and implementation difficulty account for 65% of round one. Prioritize a reliable, understandable live comparison. The completed memory comparison now provides a concrete workflow result; broader impact and autonomous optimization remain open.

## Track and eligibility

**Statement 2: Long Horizon Engineering is the closest fit.** Saved results, recoverable progress, and decisions informed by hard measurements demonstrate part of that track. The current build does not demonstrate an LLM working across billions of tokens, multi-day reasoning, or a large autonomous search. Track acceptance is the judges' decision.

The fixed policy does not evolve its own rules or architecture, so Recursive Harnessing and the recursive-self-improvement prize are not demonstrated. Finalist eligibility requires the event-provided Atlas Sandbox; successful Atlas reads and writes alone do not prove Sandbox membership.

The supplied guide also requires a public repository, an accessible one-minute demo video, and clear identification of work built during the event. It prohibits projects whose main feature is a dashboard. Keep the demonstration centered on the working experiment loop and its measured effects. See [NEW_WORK.md](NEW_WORK.md) and [DEMO.md](DEMO.md).

## Guidance and public skills reviewed

This review used targeted searches for relevant hackathon skills and primary organizer guidance. It does not claim to cover every skill on the internet. Public skill files were read as reference material; their scripts were not installed or executed. The supplied event rules take precedence over generic advice.

| Source | Useful guidance for this project |
| --- | --- |
| [Devpost founder: successful hackathon demos](https://info.devpost.com/blog/how-to-present-a-successful-hackathon-demo) | Briefly explain the problem, demonstrate its working resolution, then describe potential impact. |
| [Devpost: advice from judges](https://info.devpost.com/blog/hackathon-judging-tips) | Meet requirements; make the demonstration, written story, and implementation reinforce one another. |
| [Devpost official curriculum: scope skill](https://github.com/challengepost/devpost-curriculum/blob/main/plugins/hackathon-in-a-plugin/skills/scope/SKILL.md) | Name the user and the problem; preserve one focused concept instead of scattered features. Its beginner onboarding workflow is unnecessary here. |
| [Community: Hackathon Grand Prize Strategist](https://github.com/iamaanahmad/hackathon-skills/blob/main/hackathon-grand-prize/SKILL.md) | Make the differentiator visible, use sponsor technology materially, and support claims with evidence. |
| [Community: Hackathon Kickoff](https://github.com/zghanw/hackathon-skills/blob/main/skills/hackathon-kickoff/SKILL.md) | Map each weighted criterion to a concrete proof a judge can observe quickly. |
| [Community: Colorfactory hackathon skill](https://github.com/colorfactory-studio/hackathon/blob/main/SKILL.md) | Keep the demo narrow, expose evidence gaps, and verify submission links. |
| [Kaggle official hackathon-judging skill](https://github.com/Kaggle/kaggle-skills/blob/main/hackathon-judging/SKILL.md) | Reviewed and excluded: it serves hosts grading Kaggle submissions, rather than this participant's build. |

## Recommended presentation

Use the three-minute slot to show one complete story: approximately 20 seconds for the engineer's problem, 30 for the fair comparison, 75 for interruption and recovery, 35 for measured results and correctness, and 20 for Atlas evidence and scope. Rehearse the sequence and preserve a recording of the real run.

The strongest defensible pitch is: **“In two controlled comparisons, saved evidence reduced testing from four attempts to three while checking the same three correct alternatives. The program also resumed saved work after an interruption.”** Do not claim faster kernels, industry-wide savings, or long-context reasoning without the corresponding evidence.
