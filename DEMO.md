# Consumer GPU harness demo runbook

## Opening for a general audience

“New AI models can be bigger than the memory in a desktop graphics card. Getting them to run can mean guessing settings, crashing, and trying again. We built a harness that asks a local coding model to write a placement rule, tests it on real hardware, and gives the measurements back for revision. MongoDB Atlas remembers every experiment. The existing inference engine does the computation; our work is the measured improvement loop around it.”

An **expert** is a part of an MoE model selected for an input. Keeping some experts in system RAM can reduce GPU memory use, while increasing CPU work or data movement. Explain that tradeoff before using the flag name `n_cpu_moe`.

## Before presenting

1. Atlas Sandbox provenance is **user-confirmed**: the user said on October 11, 2026 that this environment was created through the event invitation. Independent UI verification was unavailable. Hide `.env`, credentials and unrelated desktop windows.
2. Check the engine using [INFERENCE_SETUP.md](INFERENCE_SETUP.md). Confirm **GeForce RTX 5070 Ti, 16,303 MiB**, not an RTX PRO card.
3. Select one completed report and saved source files from [INFERENCE_RESULTS.md](INFERENCE_RESULTS.md). Distinguish tuning from separate confirmation. Use physical capacity as the main limit; label earlier 10,000 MiB trials as simulated constraints. Disclose concurrent downloads when interpreting timings.
4. Open Atlas Data Explorer, database `atlas_gpu_harness`. Show the run in `runs`, `experiments`, and `decisions`. Filter experiments/decisions with `{"run_id":"YOUR_RUN_ID"}`, and the run with `{"_id":"YOUR_RUN_ID"}`.
5. Rehearse and record the real sequence. Loading or generating policy code can exceed the live slot; a clearly labeled recording of an actual run is an appropriate backup. Do not present replayed measurements as live.

## Three-minute sequence

**0:00–0:25 — Problem.** Deliver the opening and show the actual card. State the model and quantization. If the file is larger than VRAM, show both byte counts. The cached 35B IQ2 file is smaller than VRAM; its parameter count alone is not larger-than-VRAM proof.

**0:25–0:50 — Fair starting point.** Show the engine-auto baseline. “We keep the model, precision, questions and computer fixed. The engine already tries to fit the model automatically. Our generated rule must earn its place against that behavior.”

**0:50–1:30 — New code.** Open a saved generated `choose(context)` function, its hypothesis and CPU expert-layer choice. Show the restricted syntax/output validation. Then show the actual inference record: answers, sampled memory, request time and decode rate.

**1:30–2:05 — Feedback.** Compare earlier and later generated source. Show the measured result/error in the later receipt and the Atlas decision's evidence IDs. Explain what changed. A revision alone is not an improvement; its measurement decides.

**2:05–2:40 — Confirmation.** Show the separate questions and both comparison orders if completed. Report the actual accepted result. If the candidate did not clear the rule, say so and show that the harness kept the baseline. Do not substitute tuning scores for confirmation.

**2:40–3:00 — Original work and limits.** “We built the code-generation, controlled evaluation, feedback and persistent evidence loop today. llama.cpp supplies inference; upstream projects supply models and quantization. We currently tune one placement setting for a small architecture allowlist. This result is specific to this model, machine and task suite.”

## Commands for a real run

Choose a supported local single-file GGUF. It must already be present; this command does not download it.

```bash
MODEL=/absolute/path/to/model.gguf
ENGINE="$PWD/artifacts/engines/llama-b11429/launch-llama.sh"

.venv/bin/python -m atlas_gpu_harness.inference inspect --model "$MODEL"
.venv/bin/python -m atlas_gpu_harness.inference evolve \
  --model "$MODEL" --engine "$ENGINE" --budget-evals 3
```

The default limit is physical GPU capacity. The printed `INFERENCE_RUN_ID` identifies Atlas records and `artifacts/inference/RUN_ID` outputs. The default proposer is local Ollama `gemma3:12b`. Three budget slots mean the baseline plus two proposals; confirmation can add up to four engine launches. Larger-model readiness can take up to the current 600-second timeout, so do not depend on a fresh load finishing within the live demo slot.

Resume a compatible run with the same model, engine and contract:

```bash
.venv/bin/python -m atlas_gpu_harness.inference evolve \
  --model "$MODEL" --engine "$ENGINE" --resume-run RUN_ID
```

Saved terminal trials and proposals are reused. Interrupted proposal attempts consume their reserved slot; restarting cannot silently create an unlimited generation budget. Changed source, model, engine or context causes resume to refuse.

## One-minute narration

“New language models can exceed a desktop GPU's memory. Making them run often requires trial and error. Our harness turns that into measured experiments. First, it runs the engine's automatic configuration. Then a local coding model writes a rule deciding which experts stay in system RAM. We validate the rule, run the real model, and measure answers, memory and speed. Atlas saves the evidence and supplies it to the next code revision. Promising settings face separate questions and a repeated baseline comparison. Here is our actual recommendation and its evidence. We built this policy-generation and experiment loop during the hackathon. The inference engine and pretrained models are reused; arbitrary future model support remains future work.”

Replace “here is our actual recommendation” with the measured result in the recording. Use a numeric improvement only when its displayed report supports it. The completed 35B run has seven passing trials. Its two held-out changes, +3.5647% and +1.9202%, fell below the >5% rule, so engine auto-fitting was retained. Present this as a measured rejection of an unsupported improvement. Qwen 30B and Kimi 48B downloads are complete and checksum-verified, but larger-model inference results are pending at this update.

## Submission and remaining gaps

Public repository: [casualcomputer/atlas-gpu-harness](https://github.com/casualcomputer/atlas-gpu-harness).

Suggested description:

“Atlas GPU Harness uses a local coding model to generate and revise a bounded CPU/GPU expert-placement policy for supported MoE models. It benchmarks engine auto-fitting, tests real inference for exact answers and sampled memory, feeds measurements into the next code revision, and persists evidence in Atlas. Separate confirmation questions and reversed-order comparisons guard recommendations. The prototype completed a seven-trial evaluation of a quantized 35B MoE on a 16 GB RTX 5070 Ti and retained the engine baseline after the candidate missed the held-out improvement threshold. Larger-than-VRAM capability is reported only when completed model experiments establish it.”

Target **Recursive Harnessing** as a narrow placement-policy evolution mechanism, with **Long Horizon Engineering** relevance through durable metric-driven experiments. It does not evolve the entire architecture or demonstrate billion-token memory. See [HACKATHON_REVIEW.md](HACKATHON_REVIEW.md).

The guide requires a public repo, accessible one-minute video, all team members on the submission, and the event-provided Atlas Sandbox for finalist eligibility. Submit by **5:00 PM Pacific / 8:00 PM Toronto on October 11, 2026**. A local video file is not an accessible submission link.

**The existing `artifacts/atlas-gpu-harness-demo-backup.mp4` is stale:** it shows the earlier top-k prototype. Generate a 59-second saved-evidence walkthrough for a completed inference run with `.venv/bin/python scripts/make_inference_video.py --run-id RUN_ID`, or record the real interface. The generated video labels itself as a saved-evidence walkthrough, not live capture. Upload the selected new video and verify that judges can access its link. The operator crash/recovery demonstration and 25% attempt reduction remain historical supporting evidence in [EXPERIMENTS.md](EXPERIMENTS.md).
