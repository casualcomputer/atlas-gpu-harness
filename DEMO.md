# Demo: a 30 GB model on a 16 GB card

## The story

**Audience:** developers who want capable open models on the desktop hardware they already own.

**Problem:** a new model can exceed GPU memory. Existing offload engines make execution possible, but choosing a useful configuration still requires experiments: fit the model, check its answers, compare speed, and remember what worked.

**Working result:** Kimi-Linear-48B-A3B Q4_K_M is a **30.06 GB model file**. On the **GeForce RTX 5070 Ti, 16,303 MiB**, llama.cpp automatic fitting passed **12/12 tuning checks** with a sampled GPU peak of **15,472 MiB**, then **4/4 reserved questions** at **14,664 MiB**. The four-trial experiment completed with a saved baseline recommendation.

**Our contribution:** a local coding agent writes a Python placement policy; our harness validates and measures it against that baseline, returns evidence for the next attempt, and preserves the experiment in Atlas. Reserved questions determine whether a candidate earns a saved recipe. The capability to execute the model comes from llama.cpp and the published quantization; the generated-code experiment and acceptance workflow are our work.

**Final outcome:** Kimi's first generated policy chose **13 CPU expert layers** and achieved **22.77 short-answer decode tokens/s**, about **52% below the 47.21 baseline**. With that evidence in its context, the second proposal changed the code to choose **14 layers**, reaching **29.77 tokens/s**: **30.73% above policy 1, still 36.94% below baseline**. All three tuning trials passed 12/12 checks. Policy 2 used a sampled **15,890 MiB** peak and **1.137 s** median full request, versus policy 1's **15,791 MiB** and **1.405 s**. The fastest passing choice—the baseline—passed all four reserved questions and was retained.

## One-minute narration

This narration reflects the completed Kimi report.

> Your next AI model is 30 gigabytes. Your desktop GPU has 16. Existing engines can share work with system RAM, but each model needs settings that work.
>
> Atlas GPU Harness turns that setup into a coding experiment.
>
> Here is Kimi 48B running on an RTX 5070 Ti: twelve baseline checks passed within the card's memory capacity.
>
> We start with llama.cpp's automatic configuration. A local agent writes Python choosing where expert weights live. We test real inference and measure answers, memory, and speed.
>
> Atlas preserves code and links decisions to evidence. The best passing choice faces reserved questions before we save a recipe.
>
> Feedback changed Kimi's policy from thirteen to fourteen CPU expert layers. Decode rate recovered 31% from the first candidate, but stayed 37% below the engine baseline.
>
> The baseline passed four reserved questions and was retained. Today we tune one placement setting; no speedup over the engine is established.

### One-minute visual sequence

| Time | Show | Audience takeaway |
| --- | --- | --- |
| 0:00–0:10 | Kimi file size beside the actual GPU capacity from the model manifest and baseline record. | The model is larger than the card. |
| 0:10–0:21 | Kimi `baseline.json`: `quality`, `correct`, and `memory.device_peak_mib`. | Real inference passed the fixed checks and memory gate. |
| 0:21–0:33 | First and second `choose(context)` sources; the `- 1` is removed, changing placement 13 → 14. | The agent revised executable code after receiving measurements. |
| 0:33–0:45 | Atlas decision joined to the baseline evidence; then the next proposal's feedback context. | Stored measurements enter the next coding attempt. |
| 0:45–0:53 | Kimi short decode rates: baseline 47.21; first policy 22.77; revised policy 29.77. All passed 12/12. | Recovery from a poor candidate still does not beat the baseline. |
| 0:53–1:00 | Recommendation: `engine_autofit`; reserved questions 4/4; tested scope and repository link. | The completed experiment retained the strongest passing choice. |

Caption the tuning score **“6 unique tasks × 2 = 12 checks.”** If showing Kimi's **47.21 decode tokens/s** or **0.948 s median full request**, label them **short-answer baseline measurements**. The narration's final sentence carries the scope limits; do not interrupt every scene with another disclaimer.

## Three-minute live walkthrough

### 0:00–0:30 — The user and the working model

“Developers should be able to evaluate new open models on the hardware they already own. Here is a 30.06 GB Kimi model running on this 16 GB card. The engine can share work with system RAM. Our question is which configuration deserves to be used.”

Show the verified model file size, GPU capacity, and baseline answers. The Kimi baseline passed all 12 checks, with a 15,472 MiB sampled GPU peak. Run ID: `70be4572a9824186999857dadeb1237d`.

**Preload before the judges arrive.** The observed cold baseline load took about six minutes; the first candidate took about four. Start the confirmed current-version recipe before the presentation and wait for `/health` to report ready. Show one API answer live alongside the saved experiment evidence. Run the full generation/comparison loop outside the three-minute slot. If the recipe has not been launched and checked, use the recorded results and clearly label the replay; completing the experiment alone does not prove a subsequent relaunch.

### 0:30–1:10 — Show the code we generate

Open the saved Kimi policy, its hypothesis, and receipt. The current first proposal is actual generated code:

```python
def choose(context):
  n_cpu_moe = int(context['feedback']['initial_estimate_n_cpu_moe'] - 1)
  return {'n_cpu_moe': n_cpu_moe}
```

Its evaluated result is **13**: keep expert weights from the first 13 layers on the CPU. Explain an expert as one of the model's specialist parameter groups. Show that the model file, precision, questions, and computer remain fixed across the comparison.

A hypothesis can be mistaken. The generated explanation suggests reducing CPU experts could conserve GPU memory; moving more expert weights onto the GPU generally moves that tradeoff in the opposite direction. Preserve that hypothesis as the proposal's record. Then show its measured regression and the actual revision: the second generated function removes `- 1`, choosing **14** layers. Its **29.77 tokens/s** improves on the first candidate's **22.77**, but remains below the engine's **47.21**. All three passed answer checks.

### 1:10–1:50 — Show Atlas changing the next attempt's input

Open the decision-to-evidence aggregation. It joins a decision's `retrieved_experiment_ids` to the actual experiment documents, including answer checks, memory, and speed. Then open the corresponding proposal receipt's `context.previous_trials` and `context.previous_policy` when present.

“Atlas holds the notebook. Before asking the coding model for the next proposal, the harness retrieves compatible measurements. The next attempt receives the previous code and what happened when we ran it.”

Show the generated source and measurements, not just a collection count or database connection. A decision link proves which evidence was supplied; the following measured trial determines whether the new policy helped.

### 1:50–2:30 — Show the acceptance decision

Confirm only the best passing tuning choice. Kimi's fastest choice was the baseline, which passed **one four-question confirmation** at **14,664 MiB** sampled GPU peak. The recommendation records `engine_autofit` with reason **“Engine baseline passed held-out validation.”** That confirmation's short-answer median decode rate was **52.70 tokens/s** and median full request **0.815 s**; it used different questions from tuning, so this difference is not an improvement estimate. If a generated candidate wins tuning, the protocol instead uses **baseline → candidate → candidate → baseline**, requiring both blocks to pass and exceed a 5% median paired gain before claiming speedup.

The completed 35B example has three tuning trials passing 12/12 each and four confirmation trials passing 4/4 each. Its two generated policies were identical. Candidate confirmation changes of **+3.5647% and +1.9202%** fell below the rule, so the recommendation retained automatic fitting. That is the measured outcome; it establishes the rejection path rather than an inference improvement.

For Kimi, show the committed 13 → 14 code revision beside all three tuning rates and the final baseline recommendation. The second policy recovers part of the first policy's regression. Do not describe the 30.73% recovery as a speedup over llama.cpp.

### 2:30–3:00 — The useful artifact and next step

Show `recommendation.json` and its `validated_experiment_id`, then the recipe launch command. A saved recipe is reusable when its model, engine, backend, and hardware still match the confirmed report. Explain the value: the developer gets the setting, code, measurements, and reason for accepting it together.

Close once with scope: “Today we tune one placement setting and validate short tasks on one machine. llama.cpp supplies inference. We have not yet established a speedup. Next we can use the same discipline to test deeper offload, pruning, or merging ideas.” Those algorithms are future work.

## Evidence to have open

| Artifact | Purpose |
| --- | --- |
| [Model manifest](evidence/inference-models.json) | Qwen/Kimi publisher revisions, file sizes, and verified full checksums. |
| [Completed Kimi report](evidence/inference/70be4572a9824186999857dadeb1237d.json) | All three tuning trials, generated-code revision, reserved baseline confirmation, and final recommendation. |
| `artifacts/inference/70be4572a9824186999857dadeb1237d/baseline.json` | Actual Kimi baseline answers, commands, timings, and memory samples. |
| `artifacts/inference/70be4572a9824186999857dadeb1237d/policy_1.json` and `policy_2.json` | Both actual policy sources and receipts under `result.proposal`, their feedback contexts, and measured results. |
| [Completed 35B report](evidence/inference/b88ec5bd304443a3add2b2fa767e2450.json) | Full seven-trial example, identical proposals, confirmation blocks, baseline retained. |
| [Inference results](INFERENCE_RESULTS.md) | Latest measured status; consult before recording or submitting. |
| [Atlas aggregation](queries/decision-evidence.json) | Read-only decision-to-experiment links; run against `decisions`. |

The Kimi report is exported for public inspection; local artifacts retain server logs and individual files. Label any recorded walkthrough as a replay of actual saved results. The confirmed record's ID is `70be4572a9824186999857dadeb1237d:a272641887051186ac3993aee9d204b06d44b972f2303e81775e57033e4ba4c5`.

## Atlas evidence command

From the repository root, with the local `MONGODB_URI` configuration already present:

```bash
.venv/bin/python scripts/show_atlas_evidence.py \
  --run-id 70be4572a9824186999857dadeb1237d
```

This executes the checked-in read-only aggregation. It creates no indexes, probe records, or Atlas UI saved query. The equivalent Data Explorer path is database `atlas_gpu_harness` → collection `decisions` → Aggregations. Paste [the pipeline](queries/decision-evidence.json) and replace `REPLACE_WITH_RUN_ID` in its first stage.

For direct record inspection, filter `runs` with `{"_id":"RUN_ID"}` and `experiments` / `decisions` with `{"run_id":"RUN_ID"}`. Keep credentials off screen.

## Running and reusing experiments

Prepare the local engine using [INFERENCE_SETUP.md](INFERENCE_SETUP.md), local Ollama `gemma3:12b`, and the verified model file. Do not start a duplicate GPU run during an active experiment. For a new run after the current run finishes:

```bash
MODEL="$PWD/artifacts/models/Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf"
ENGINE="$PWD/artifacts/engines/llama-b11429/launch-llama.sh"

.venv/bin/python -m atlas_gpu_harness.inference inspect --model "$MODEL"
.venv/bin/python -m atlas_gpu_harness.inference evolve \
  --model "$MODEL" --engine "$ENGINE" --budget-evals 3
```

Default memory budget is the physical card. Three slots cover one baseline plus two proposals; confirmation may add four engine launches. Loading can take minutes, with a 600-second readiness timeout. Preload the confirmed recipe before the live slot, then show one answer and the saved evidence.

Resume the printed run ID with unchanged model, engine, machine, and source:

```bash
.venv/bin/python -m atlas_gpu_harness.inference evolve \
  --model "$MODEL" --engine "$ENGINE" --resume-run RUN_ID
```

Launch a confirmed recipe from its matching software revision:

```bash
.venv/bin/python -m atlas_gpu_harness.inference_serve \
  --recipe artifacts/inference/RUN_ID/recommendation.json --port 8081
```

Keep `report.json` beside the recommendation. The API becomes available at `http://127.0.0.1:8081/v1` after loading. The older 35B report predates a backend update, so current compatibility checks reject direct reuse of that recipe until the configuration is evaluated on the current revision.

In a second terminal, check readiness before presenting, then make the short live request:

```bash
curl --fail --noproxy '*' http://127.0.0.1:8081/health
curl --fail --noproxy '*' http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Return only one JSON object with exactly one key, \"answer\". No explanation or Markdown."},{"role":"user","content":"What is 37 + 58? Return the integer answer.\n/no_think"}],"temperature":0,"seed":20261011,"max_tokens":96,"stream":false,"cache_prompt":false}'
```

The live answer demonstrates the loaded recipe; the saved controlled experiment supplies the comparison evidence. This exact request was verified after relaunch from both WSL and Windows; see [the launch receipt](evidence/inference-serve.json). Keep the prompt unchanged for the recorded-format check.

## Submission text

> Atlas GPU Harness makes consumer-GPU model setup a measured coding experiment. We ran a 30.06 GB Kimi-Linear-48B model on a 16 GB RTX 5070 Ti using llama.cpp's existing offload. A local agent generates Python placement policies; the harness checks the code, tests real inference, and returns measured evidence for revision. Atlas preserves experiments and links decisions to prior measurements. Kimi's revised policy changed 13 CPU expert layers to 14 and recovered 30.73% from its first candidate's decode rate, while remaining 36.94% below the engine baseline. The baseline passed four reserved questions and became the saved recommendation. No speedup over the engine is established.

Public repository: [casualcomputer/atlas-gpu-harness](https://github.com/casualcomputer/atlas-gpu-harness).

Describe **Recursive Harnessing** as bounded placement-policy evolution and **Long Horizon Engineering** as durable progress driven by measurements. The implemented scope and track boundaries are in [HACKATHON_REVIEW.md](HACKATHON_REVIEW.md).

The project owner confirms the Atlas cluster was created through the event invitation; this provenance has not been independently checked in the Atlas UI. The event requires a public repository, an accessible one-minute video, all team members on the submission, and the event Atlas Sandbox for finalist eligibility. The supplied deadline is **5:00 PM Pacific / 8:00 PM Toronto, October 11, 2026**.

The old `artifacts/atlas-gpu-harness-demo-backup.mp4` shows the earlier top-k prototype and is not the current submission story. For the completed Kimi run, `.venv/bin/python scripts/make_inference_video.py --run-id 70be4572a9824186999857dadeb1237d` produces a captioned 59-second saved-evidence walkthrough; it is explicitly labeled as saved evidence. The rendered copy is [demo/atlas-inference-demo.mp4](demo/atlas-inference-demo.mp4). It was visually inspected and verified as 59 seconds. Record the real interface for the spoken version above; this captioned version is a saved-evidence walkthrough.

## Supporting evidence, if asked

The original operator prototype rejected an incorrect tie ordering, declined a speed claim that failed repeated checks, and measured **four versus three attempts** with prior history enabled. Its deliberate interruption also retained committed work. These are useful supporting checks of the notebook mechanism, documented in [EXPERIMENTS.md](EXPERIMENTS.md) and [EVIDENCE.md](EVIDENCE.md). Keep them in the appendix; they are not new inference gains from the consumer MoE harness.
