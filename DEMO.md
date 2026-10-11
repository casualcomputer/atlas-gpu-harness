# Demo runbook

## Before recording

1. Verify `.env` exists locally, is ignored by Git, and points to the **hackathon Atlas Sandbox**. Do not display it.
2. Run `atlas-check`, `preflight`, `baseline`, and the four local tests. Keep the terminal in this repository and use `.venv/bin/python` for every command.
3. Open Atlas Data Explorer beside the terminal, showing `atlas_gpu_harness` collections after the real run. Hide credentials and connection settings.

## Three-minute live demo

**0:00–0:25 — Problem.** “GPU optimization agents repeat failures and lose progress when they crash. Our harness persists each experiment and decision in Atlas, then resumes from verified evidence.” Show GPU name and the correct local baseline.

**0:25–1:10 — Durable experiment.** Run `run --budget-evals 4 --fault-after-commit`. Point to the printed `RUN_ID`, committed baseline, and clearly labeled recovery-test exit. Show the baseline document in Atlas Data Explorer.

**1:10–2:05 — Recovery and memory.** Run `status`, then `resume` with the same ID. Show that `sort_stable` remains at one attempt, the next decision references its experiment ID, and the invalid top-k tie candidate is rejected. Show `decisions` and `experiments` in Atlas.

**2:05–2:40 — Engineering depth.** Show the exact CPU oracle, CUDA event samples, unique identity and owner token, and failure-aware policy. These are the parts built during the event; PyTorch supplies the GPU operators.

**2:40–3:00 — Result.** Export the report, state the measured medians and whether paired recheck supports a speedup. If not, emphasize saved work, avoided repetition, and correctness rejection. Do not claim a custom kernel or an LLM policy.

## One-minute submission video storyboard

- **0–10 s:** Title and problem statement; display actual RTX 5070 Ti preflight.
- **10–22 s:** Show the correct CUDA baseline and raw timing summary.
- **22–35 s:** Run with fault injection; show Atlas committed experiment before exit.
- **35–48 s:** Resume the same run; show retrieved IDs and no baseline rerun.
- **48–60 s:** Show invalid candidate rejection, brief code tour, and measured conclusion. Call out PyTorch reuse.

The public repository is https://github.com/casualcomputer/atlas-gpu-harness. Record and upload the actual video, then submit the repository link plus the video through the event form before **5:00 PM Pacific**. Video upload and event submission still need to be completed.

A 59-second silent, captioned backup is generated at `artifacts/atlas-gpu-harness-demo-backup.mp4` by `scripts/make_demo_video.py` from the two verified Atlas exports. The video is ignored by Git; the generator can recreate it from the tracked `evidence/` copies with Pillow and `ffmpeg`. A real screen recording with Atlas Data Explorer and the terminal will make the stronger submission.
