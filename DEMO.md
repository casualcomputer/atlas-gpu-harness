# Demo runbook

## Opening for a general audience

“Imagine asking a computer to find a faster way to run an AI calculation. It tries different methods, but some give the wrong answer and others only look faster because timing fluctuates. If the program stops, it can also repeat work it already finished. We built an experiment manager with a saved lab notebook. In two controlled comparisons, remembering previous evidence reduced testing from four attempts to three while checking the same three correct alternatives.”

Say **experiment manager**, **saved notebook**, and **reference answer** before using the technical terms controller, database, and oracle. The goal is to find faster correct GPU code. Today's build evaluates predefined PyTorch alternatives; it does not write new GPU kernels and has not established a GPU speedup.

## Three references to explain before results

| Question | Reference |
| --- | --- |
| Is the answer right? | An independently computed CPU answer with fixed rules for ties. |
| Is the calculation faster? | A trusted PyTorch stable-sort implementation measured on the same GPU and input. |
| Does the workflow waste fewer attempts? | The identical experiment manager with access to previous-run evidence disabled. |

Use actual attempts as the denominator. The five-attempt allowance is a budget, not a measured baseline. The 25% result is `(4 - 3) / 4`; it is conditional on useful prior evidence being available.

## Before recording

1. Confirm the configured cluster belongs to the **event-provided Atlas Sandbox project**. Hide credentials and `.env`.
2. Keep the real GPU preflight and [published comparison](EXPERIMENTS.md) ready. Five tests passed after the warmup correction; live Atlas/GPU evidence supports the integration claims.
3. Open Atlas Data Explorer beside the terminal. Select `atlas_gpu_harness`. For current records, use the run IDs in [EXPERIMENTS.md](EXPERIMENTS.md).
4. Existing compatible history should cause fresh runs to skip direct top-k. Show its saved failure instead of deleting history to stage another failure.

## Three-minute live demo

**0:00–0:20 — Problem.** Deliver the opening. Show the GPU identity. Explain that the sample task selects the 64 highest-scoring positions among 16,384 scores, with a consistent tie rule.

**0:20–0:50 — Fair comparison.** Show the three references above. Explain: “We kept the computer, input, allowed methods, answer checks, and budget the same. We changed only access to previous notes, then reversed the order and repeated the comparison.”

**0:50–1:35 — Save a result, then stop.** Run:

```bash
.venv/bin/python -m atlas_gpu_harness run --budget-evals 4 --fault-after-commit
```

Copy its run ID. The clearly labeled exit 86 is intentional. In Atlas, show the reference experiment as `completed` with `attempt: 1`. Say: “The answer and measurements are already saved, even though the program stopped before updating its progress summary.”

**1:35–2:05 — Continue from saved work.** Run:

```bash
.venv/bin/python -m atlas_gpu_harness status --run-id RUN_ID
.venv/bin/python -m atlas_gpu_harness resume --run-id RUN_ID
```

Show that the completed reference remains at one attempt while unfinished work continues. With compatible stored failures, the run may finish after three of its four allowed attempts.

**2:05–2:40 — Show measured improvement.** Recompute the published results:

```bash
.venv/bin/python scripts/summarize_experiments.py
```

Label this as analysis of saved real runs. It shows four versus three attempts in each block, the same three correct alternatives, and zero accepted incorrect results. Open the failed direct top-k record from seed run `28236d7c01de4472b7ef84bdbe30379e`: its tie case required `[0, 1]` but returned `[1, 0]`. Show a later decision that retrieved its ID.

**2:40–3:00 — Result and scope.** “Saved evidence prevented one repeated invalid test in each comparison—a 25% reduction in attempts. We also kept completed work after an interruption. We did not prove a faster calculation: repeated speed checks rejected that claim. The experiment manager and evidence loop are the work we built today; PyTorch supplies the GPU operations.”

If the audience asks about the science, point to [EXPERIMENTS.md](EXPERIMENTS.md): hypothesis, single changed factor, controls, reversed order, repeated comparisons, correctness safeguards, raw data, and limitations. No production cost savings or confidence interval is claimed from two short comparisons.

## One-minute video narration

“Finding faster GPU code means trying alternatives without accepting wrong answers or being fooled by timing noise. We built an experiment manager with a saved notebook in MongoDB Atlas. It checks each answer, compares speed against a fixed reference, and remembers the evidence. Here we deliberately stop the program after saving a result. Restart continues without repeating that test. We then compare the same workflow with previous notes hidden versus available. Both test the same three correct alternatives, but remembering a known failure takes three attempts instead of four. Reversing the order gives the same result: 25% fewer attempts in this test. We built the experiment and recovery loop today. We have not established faster GPU calculations yet.”

Suggested visuals: 0–10 seconds problem and GPU; 10–25 seconds saved result and restart; 25–45 seconds matched four-versus-three result and failed-answer evidence; 45–60 seconds original code, scope, and repository.

## Submission package

Public repository: https://github.com/casualcomputer/atlas-gpu-harness.

Suggested description:

“Atlas GPU Harness is an experiment manager for finding faster correct GPU code. It checks answers, measures GPU execution, stores evidence in MongoDB Atlas, and resumes saved work after interruption. In two controlled comparisons, access to prior evidence reduced attempts from four to three while evaluating the same three correct alternatives. The current build tests predefined PyTorch implementations and uses a deterministic policy; a repeatable GPU speedup was not established. The experiment manager, evaluator, memory/recovery integration, and comparison were built during the event.”

Target Statement 2, Long Horizon Engineering, as a prototype of durable progress and evidence reuse. It does not demonstrate a billion-token model or multi-day autonomous reasoning. The guide requires an accessible one-minute video, a public repository, and team members added to the submission page. Submit before **5:00 PM Pacific on October 11** (8:00 PM Toronto). Video upload and submission remain outstanding; finalist eligibility depends on the event's Atlas Sandbox.

The existing 59-second captioned backup at `artifacts/atlas-gpu-harness-demo-backup.mp4` uses earlier verified exports. It does not include the new controlled comparison. Prefer a new screen recording using this script. The generator and raw earlier evidence remain available, with the video ignored by Git.
