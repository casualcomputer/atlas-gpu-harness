# Hackathon review: importance, originality and observable proof

Reviewed against the supplied October 11, 2026 Harness Engineering & Model Wrangling participant guide. This is an evidence audit and presentation plan, not a prediction of judging results.

## Start with the person and the problem

A developer wants to try a capable new open model on their existing desktop. The model's weights exceed GPU memory, and making it run can require unfamiliar settings, repeated crashes, and uncertain compromises in response speed. Buying another card or moving to hosted inference changes cost and deployment choices. We want a repeatable local setup path with evidence about what actually works.

**Our solution is a harness that writes and tests placement policies.** An existing engine runs the neural network. A local coding model writes a small rule deciding how many layers keep their experts in system RAM. The harness measures answers, memory and speed, returns failures/results for code revision, and preserves the record in Atlas. A candidate earns a recommendation only after the checks; the engine's normal settings can win.

Strata and Maya address large-model inference across consumer GPU/CPU/memory tiers, and llama.cpp already includes automatic fitting. Their existence demonstrates engineering work on this problem, not customer demand for our product. We have not measured willingness to pay, adoption, production GPU-hour savings, or a broad advantage over those engines. [Strata](https://github.com/Niko1221/Strata), [Maya](https://github.com/mw00/project-maya), [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Distinguish three outcomes

| Outcome | Evidence to show | What it does not prove |
| --- | --- | --- |
| The model runs here | Real model/quantization, actual GPU capacity, server success, scored outputs and observed memory. | That our harness improved the engine or invented its offloading capability. |
| The harness adapts | First generated source, measured failure/result, later revised source, and Atlas evidence IDs supplied for revision. | Arbitrary architecture self-improvement or a useful improvement merely because code changed. |
| A configuration improves | Same model/workload against engine auto-fitting, separate confirmation questions, reversed order, and the acceptance rule. | Better general intelligence, lower production cost, or support for every future model. |

**Kimi provides the larger-than-VRAM example:** its 30.06 GB Q4_K_M file ran through llama.cpp on the 16,303 MiB RTX 5070 Ti. Automatic fitting passed **12/12 checks** at **47.21 short-answer decode tokens/s**, with **15,472 MiB** sampled GPU peak. The first generated policy used 13 CPU expert layers and reached **22.77 tokens/s**; after receiving those measurements, the next proposal changed its code to choose 14 and reached **29.77 tokens/s**. Both passed 12/12. That is **30.73% recovery over the first candidate but still 36.94% below the engine baseline**. It establishes a real code revision and measured tradeoff, not an inference gain over llama.cpp. The fastest passing choice—the baseline—then passed **4/4 reserved questions** at **14,664 MiB** sampled peak and became the saved recommendation. The [completed four-trial Atlas export](evidence/inference/70be4572a9824186999857dadeb1237d.json) contains the source, feedback, measurements, and final decision.

The 35B run completed seven passing trials. Confirmation gains of **+3.5647% and +1.9202%** missed the >5% rule, so the harness **retained the engine baseline**. Its two generated policies were identical, and its 11.39 GB file fits within VRAM. Qwen 30B's 18.56 GB artifact is verified but untested. [The model manifest](evidence/inference-models.json) records publisher revisions/checksums; [INFERENCE_RESULTS.md](INFERENCE_RESULTS.md) records measured status. The local suite passes 46 tests.

The older operator memory comparison established four versus three attempts for the same three correct alternatives. Preserve it as supporting evidence of Atlas experiment memory; its 25% is not an LLM performance result. The earlier 10,000 MiB inference limit was a simulated constraint, not the card's physical capacity.

## Fairness and experimental design

Use the engine's automatic placement/fitting as the baseline. Hold weights, quantization, engine, context allocation, tasks, decoding requests and hardware fixed. Compare placement policies; among generated candidates, only the CPU expert-layer count changes. Hide four confirmation questions from the proposer. **Only the best passing tuning choice is confirmed:** a winning baseline receives one four-question confirmation; a winning generated policy receives two comparisons with the baseline in reversed order. Report memory, answer checks, and speed separately. Requested thinking controls are model-dependent; the Kimi template ignores the `enable_thinking:false` flag.

All exact-answer tasks must pass and sampled memory must stay within budget. The 12 tuning checks cover six unique questions twice; Kimi's tuning responses contain only 6–8 completion tokens each. A speedup requires greater than 5% median paired decode improvement in each of two confirmation blocks. This is a descriptive acceptance rule, not a confidence interval. These short outputs, one desktop and concurrent machine activity limit generalization. [NIST experimental-design guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)

The protocol is in [CONSUMER_GPU_PLAN.md](CONSUMER_GPU_PLAN.md). If confirmation does not support improvement, show that the harness kept the baseline. Evidence-based rejection is useful behavior, but cannot substitute for a capacity or speed gain when presenting the impact claim.

## Judging rubric

Round one uses the weights below. Round two gives all four criteria equal weight.

| Criterion | Observable demo proof | Remaining limitation |
| --- | --- | --- |
| Impact potential: 20% | A 30.06 GB model running on a 16 GB card, plus a measured configuration workflow. | Offload capability belongs to the engine. Incremental capacity/speed gains and reduced setup effort are unproven; no customer study. |
| Technical demo: 35% | Real generated code, real inference, scored answers, sampled resources, Atlas records and recommendation. | Generated code or a download alone is not a working model demonstration. |
| Creativity: 15% | Kimi placement code changes 13 → 14 after measured feedback; Atlas supplies the next proposal's evidence. | Calibration/offloading already exist; the revision remains slower than the engine baseline. |
| Implementation difficulty: 30% | Constrained code execution, process ownership, compatibility checks, persistent proposal budgets, separate confirmation and recovery. | One placement integer changes; no new GPU kernels or inference engine. |

Technical demo and implementation difficulty total 65% of round one. Show one complete feedback cycle and its actual result before broader ambitions.

## Tracks and eligibility

**Statement 1, Recursive Harnessing:** the new build generates and revises placement-policy code from trial feedback. Present this as policy evolution inside a fixed evaluator. The complete harness architecture, guardrails and tool permissions are not self-modified. A measured better result is required to call that evolution an improvement. Track and recursive-self-improvement prize acceptance are the judges' decision.

**Statement 2, Long Horizon Engineering:** Atlas results, recoverable runs, bounded proposal budgets and hard metric feedback demonstrate part of the statement. The prototype does not demonstrate billions of tokens, hours or days of coherent autonomous work, or broad model transfer. Durable experiment history differs from long-context model reasoning.

The guide requires the **event-provided MongoDB Atlas Sandbox** for finalists. On October 11, 2026, the user confirmed creating the Atlas environment through the event invitation. Record this as user-confirmed provenance; independent browser/UI verification was unavailable. A database connection alone was not treated as membership proof. Repositories must be public, the one-minute demo link accessible, and teams no larger than four. The demo must identify work created during the event. A dashboard cannot be the main feature.

Submission is due **5:00 PM Pacific / 8:00 PM Toronto on October 11, 2026**. Repo visibility, video accessibility and submitting the form remain separate tasks; documentation does not complete them. Sandbox provenance is user-confirmed as described above.

## Guidance and skills reviewed

Targeted review covered the supplied guide and public guidance below. It was not an exhaustive search of every skill. Public skill files were references; their scripts were not installed or executed. Event rules take precedence.

| Source | Application |
| --- | --- |
| [Devpost founder: successful demos](https://info.devpost.com/blog/how-to-present-a-successful-hackathon-demo) | Problem, working resolution, then potential. |
| [Devpost: advice from judges](https://info.devpost.com/blog/hackathon-judging-tips) | Keep demo, implementation and claims consistent; meet requirements. |
| [Devpost curriculum: scope](https://github.com/challengepost/devpost-curriculum/blob/main/plugins/hackathon-in-a-plugin/skills/scope/SKILL.md) | One user, one problem, one observable result. |
| [Community: Hackathon Grand Prize Strategist](https://github.com/iamaanahmad/hackathon-skills/blob/main/hackathon-grand-prize/SKILL.md) | Show sponsor use and differentiator through actual evidence. |
| [Community: Hackathon Kickoff](https://github.com/zghanw/hackathon-skills/blob/main/skills/hackathon-kickoff/SKILL.md) | Map weighted criteria to demo proof. |
| [Community: Colorfactory hackathon](https://github.com/colorfactory-studio/hackathon/blob/main/SKILL.md) | Expose evidence gaps and verify submission links. |
| [Kaggle: hackathon judging](https://github.com/Kaggle/kaggle-skills/blob/main/hackathon-judging/SKILL.md) | Reviewed and excluded: serves hosts grading Kaggle submissions. |

Karpathy guidelines informed implementation: one bounded policy, existing inference frameworks, explicit assumptions and verifiable outcomes. The strongest story identifies what changed, how it was measured, and which conclusion the measurements support.
