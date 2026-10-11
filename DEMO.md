# Demo runbook

## Opening: pain, impact, solution, proof

“A GPU tuning run finishes an experiment, then the controller crashes. Repeating that work costs compute; trusting an incorrect result costs confidence in the whole search. We built Atlas GPU Harness so every result becomes durable evidence before the next decision. Today we will kill the controller, recover its completed work from Atlas, and show a later run avoiding a failure it already learned about.”

Lead with this story, then show the system behaving as claimed. The target is **Long Horizon Engineering**. The demonstration is a small deterministic prototype of durable memory, recovery, and decisions informed by measured results.

## Before recording

1. Confirm the configured cluster belongs to the **event-provided Atlas Sandbox project**. Do not display `.env` or connection credentials.
2. Run `atlas-check`, `preflight`, and the five tests with `.venv/bin/python`. Four tests use controller/store doubles; one exercises the packed-key candidate on the actual GPU.
3. Open Atlas Data Explorer alongside the terminal. Select `atlas_gpu_harness`, with `runs`, `experiments`, and `decisions` ready to inspect. [ARCHITECTURE.md](ARCHITECTURE.md) has exact filters.
4. Keep the original exported reports available. Existing Atlas failure history means a fresh run should skip direct top-k; show its stored rejection rather than expecting to execute it again.

## Three-minute live demo

**0:00–0:25 — Establish the problem and contract.** Deliver the opening. Show the real RTX 5070 Ti preflight. Explain the task in one sentence: select 64 indices from 16,384 scores, with the lower original index winning ties. Point to the independent CPU oracle.

**0:25–1:10 — Make a result durable, then crash.** Run:

```bash
.venv/bin/python -m atlas_gpu_harness run --budget-evals 4 --fault-after-commit
```

Copy the printed run ID. Point to the committed baseline and the clearly labeled recovery test. Exit 86 is deliberate. In Atlas, filter `experiments` by this run ID and show `sort_stable`, `status: completed`, `attempt: 1`, correctness checks, and timing samples.

**1:10–1:55 — Restart from evidence.** Run:

```bash
.venv/bin/python -m atlas_gpu_harness status --run-id RUN_ID
.venv/bin/python -m atlas_gpu_harness resume --run-id RUN_ID
```

Show that the baseline remains at one attempt while the other eligible candidates continue. A decision document references retrieved experiment IDs. Explain that committed experiments rebuild progress even though the run summary was not updated before the crash. With existing failure history, completion at 3/4 is expected.

**1:55–2:25 — Show why memory changes the next run.** Open the saved failed `topk_sorted` experiment from run `2eb9f21a2e374f358c4a42b9fbe2258d`. Its first tie case expected `[0, 1]` and returned `[1, 0]`; no timing was recorded. Then show the first decision of run `21dbac5116a04dafa1ddaed17ea34030`: it retrieved four earlier records. That run evaluated three candidates from a budget of five, skipping the known failure and screening unstable sort as a related tie risk.

**2:25–2:45 — Show the engineering built today.** Point to the fixed oracle and CUDA timing adapter, the Atlas identity and owner-token completion guard, and the failure-aware policy. PyTorch supplies the underlying GPU operations; the event contribution is documented in [NEW_WORK.md](NEW_WORK.md).

**2:45–3:00 — Close on measured outcomes.** “We preserved a completed experiment across a crash, rejected an invalid result, and used stored failure evidence in the next run. Argsort had a lower initial median, but our paired recheck did not support a speedup claim. The useful result today is a search loop that keeps and acts on verified evidence.”

If asked about impact, explain that fewer lost or repeated experiments is the intended path to saving engineering time and GPU compute. This microbenchmark has not measured dollar or GPU-hour savings.

## One-minute submission video

| Time | Visual | Narration point |
| --- | --- | --- |
| 0–10 s | GPU identity and fixed contract | An experiment can be lost after a crash, and an incorrect output can invalidate an optimization claim. |
| 10–22 s | Correct baseline and Atlas experiment | Validate on the GPU and commit raw evidence before moving on. |
| 22–35 s | Deliberate exit, fresh status, resume | Completed baseline remains at attempt 1. |
| 35–48 s | Stored tie failure and later decision references | A later run retrieves the failure and avoids repeating it. |
| 48–60 s | Original controller/store/policy code and report | State proven recovery and memory outcomes; attribute PyTorch; no supported speedup claim. |

Copy-ready narration:

“A GPU optimization result must be correct, and the controller must remember it after a crash. We built Atlas GPU Harness to preserve that progress. This baseline passes an independent CPU oracle, runs on a real RTX GPU, and commits raw timings to Atlas. We deliberately crash the controller. Resume reads Atlas and skips the completed evaluation. Earlier, direct top-k failed our exact tie-order requirement, so it was never timed. A later run retrieved that failure and avoided repeating it. Each selected decision links to the evidence it retrieved. We did not establish a speedup today. We demonstrated preserved work, rejected incorrect output, and memory changing the next experiment.”

## Submission package

Public repository: https://github.com/casualcomputer/atlas-gpu-harness.

Suggested project description:

“Atlas GPU Harness is a Long Horizon Engineering prototype for GPU optimization experiments. It validates exact top-k outputs against a CPU oracle, measures real CUDA execution, stores experiments and decisions in MongoDB Atlas, and resumes after a deliberate controller crash without rerunning committed work. A later run retrieves an earlier correctness failure and avoids repeating it. We built the controller, Atlas evidence model, memory policy, recovery path, and evaluation adapter during the event, reusing PyTorch for GPU operations. The verified outcomes are durable progress and evidence-informed decisions; a repeatable performance gain was not established.”

The guide requires an accessible one-minute video, a public repository, and team members added to the submission page. Submit before **5:00 PM Pacific on October 11** (8:00 PM Toronto time). Video upload and event submission remain outstanding. Finalist eligibility depends on using the event's supplied Atlas Sandbox.

A 59-second silent captioned backup is available locally at `artifacts/atlas-gpu-harness-demo-backup.mp4`. `scripts/make_demo_video.py` can regenerate it from the tracked evidence using Pillow and `ffmpeg`. It is generated from verified exports and should be labeled as such; a real terminal and Atlas screen recording provides stronger visible proof. The video is ignored by Git.
