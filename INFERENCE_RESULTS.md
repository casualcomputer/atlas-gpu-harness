# Consumer MoE inference: measured results

Measured on October 11, 2026 using a **GeForce RTX 5070 Ti with 16,303 MiB VRAM**, an i7-14700F, and WSL with about 48 GiB assigned RAM. The harness uses the physical card capacity by default. A model may use both GPU memory and CPU RAM.

## New 32K engine comparison — October 11, 2026

Same Qwen3.8-Flash-Next IQ2_XS files and 32,768 input token IDs, native Windows, RTX 5070 Ti 16 GB, i7-14700F and 64 GB RAM. Reasoning disabled; one request per engine.

| Metric | llama.cpp b11429 | Strata v0.1.43 |
| --- | ---: | ---: |
| Generation, tokens/s | **18.22** | **107.77** |
| Prompt processing, tokens/s | 146.13 | 2,814.73 |
| First token, seconds | 224.30 | 11.66 |
| Request time excluding loading, seconds | 244.01 | 14.90 |
| Document retrieval | 8/8 correct | 8/8 correct |

**5.92× observed decode throughput in this run.** Both engines completed inference and retrieved every final document value, but both reported 413 for a separate arithmetic total whose correct value is 503. These are fictional service values in a synthetic document, not engine settings.

Strata used MTP4 and reported route-tail expert skipping; llama.cpp had no draft head. Kernels and cache formats differ. The first measured long request after a short warmup may include startup/cache effects. This is a descriptive runtime comparison, not a repeatability claim, proof of equivalent output quality or a generated-policy improvement.

[Comparison GUI](demo-ui/index.html#comparison) · [Verified measurements](evidence/engine-comparison/summary.json) · [Atlas records and raw answers](evidence/engine-comparison/latest.json) · [Reproduction runner](scripts/quick_engine_comparison.py)

**Atlas inventory:** 61 experiment records and 61 decisions across 14 runs, including one superseded run and one failed preflight. Of these, 58 records belong to 13 runs marked completed; that status includes the preflight and does not mean every trial passed. [Read-back count audit](evidence/engine-comparison/atlas-counts.json).

## Earlier inference evidence

| Model | Weight file | Result |
| --- | ---: | --- |
| Qwen3.5-35B-A3B, Unsloth UD-IQ2_M | 11.39 GB | Completed seven real inference trials. Every scored response passed. Generated policies did not meet the speed-improvement rule; the engine baseline was retained. |
| Qwen3-30B-A3B, publisher Q4_K_M | 18.56 GB | Download complete and full SHA256 verified; inference trial pending. |
| Kimi Linear 48B-A3B Instruct, Bartowski Q4_K_M | 30.06 GB | Three real tuning trials completed, each passing 12/12 checks. The 14-layer revision improved on the 13-layer proposal but remained slower than engine automatic fitting. Engine baseline then passed all four reserved confirmation questions and was saved as the recommendation. |

The latter two files exceed this GPU's physical VRAM. Kimi has produced actual scored outputs; the downloaded Qwen3-30B has not been evaluated. Kimi also passed held-out validation and produced a confirmed launch recipe. GB here means decimal bytes; GPU measurements use MiB. [Pinned model sources and verified checksums](evidence/inference-models.json).

## Completed 35B experiment

Run: `b88ec5bd304443a3add2b2fa767e2450`. [Full Atlas export](evidence/inference/b88ec5bd304443a3add2b2fa767e2450.json) · [Machine-readable summary](evidence/inference-summary.json).

| Trial | CPU expert layers | Exact answers | Sampled GPU peak, MiB | Median decode tokens/s | Median request seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuning baseline | Engine auto-fit | 12/12 | 12,880 | 135.02 | 0.376 |
| Generated policy 1 | 0 | 12/12 | 12,875 | 143.05 | 0.245 |
| Generated policy 2 | 0 | 12/12 | 12,895 | 130.26 | 0.259 |
| Confirmation baseline 1 | Engine auto-fit | 4/4 | 12,773 | 141.11 | 0.225 |
| Confirmation candidate 1 | 0 | 4/4 | 12,730 | 144.15 | 0.245 |
| Confirmation candidate 2 | 0 | 4/4 | 12,674 | 142.09 | 0.234 |
| Confirmation baseline 2 | Engine auto-fit | 4/4 | 12,668 | 136.96 | 0.250 |

The coding model produced the same policy twice. Both read the estimated starting count, which was zero for this compressed model. This is evidence of code generation and measured feedback, **not a successful policy improvement**. The identical placement also produced varying timings, illustrating why picking the fastest tuning sample is insufficient.

Matched-question confirmation gains were **3.5647%** and **1.9202%**. Both were below the predeclared greater-than-5% rule, so `directionally_supported_speedup=false` and the saved choice was `engine_autofit`.

This run used source commit `3c40d89`. Subsequent timeout, feedback, and recommendation fixes are in `a8d8fd6`; the launcher intentionally rejects an old recipe after its recorded backend source changes. The completed Kimi run below uses the corrected implementation and has a confirmed recipe.

## Kimi: larger-than-VRAM inference and measured revision

Run: `70be4572a9824186999857dadeb1237d`. [Full Atlas export](evidence/inference/70be4572a9824186999857dadeb1237d.json). The **30.06 GB** Q4_K_M file runs through existing llama.cpp CPU/GPU offload on the **16,303 MiB** card, using system RAM. The host has 64 GB installed RAM; WSL exposes about 48 GiB. This demonstrates actual large-model operability, not a new capacity improvement over the engine.

| Trial | CPU expert layers | Exact answers | Sampled GPU peak, MiB | Median decode tokens/s | Median request seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Tuning baseline | Engine auto-fit | 12/12 | 15,472 | 47.21 | 0.948 |
| Generated policy 1 | 13 | 12/12 | 15,791 | 22.77 | 1.405 |
| Generated policy 2 | 14 | 12/12 | 15,890 | 29.77 | 1.137 |
| Confirmation baseline | Engine auto-fit | 4/4 | 14,664 | 52.70 | 0.815 |

The first generated policy's hypothesis incorrectly suggested that reducing the CPU expert-layer count could conserve GPU memory. The harness kept that prediction as evidence and measured the result: answers passed, but decode rate regressed by **51.76%** relative to automatic fitting.

The next proposal explicitly cited the slowdown and changed its Python from `initial_estimate_n_cpu_moe - 1` to `initial_estimate_n_cpu_moe`, producing **13 → 14** CPU expert layers. That revision's observed median decode rate recovered by **30.73% relative to the first proposal**, but remained **36.94% below the engine baseline**. These are descriptive tuning comparisons on short answers; the recovery is not a confirmed speedup over llama.cpp.

The baseline was fastest, so the harness selected it for one fresh confirmation run on four reserved questions. All four answers and the memory check passed; the saved recommendation is `engine_autofit`, with reason **"Engine baseline passed held-out validation."** No candidate beat the tuning baseline, so this run does not perform the two candidate comparison blocks used in the 35B experiment.

## Resume verification

After completion, the actual CLI resumed the Kimi run and returned its stored recommendation in **7.80 seconds**, without changing experiment IDs, attempt counts, decisions, or proposal slots. [Before/after Atlas receipt](evidence/inference-resume.json). This verifies completed-run reuse; the original operator prototype provides the separate deliberate-crash demonstration.

## Saved-recipe launch verification

The actual `inference_serve` command accepted the Kimi recommendation and launched the recorded engine configuration. The health endpoint passed; a request using the evaluator's unchanged prompt returned `{"answer":95}` from both **WSL and Windows**. The existing llama.cpp web interface returned its HTML successfully. [Launch and API receipt](evidence/inference-serve.json). This is a usability smoke check, separate from the controlled measurements above.

An earlier request with a shortened system instruction returned bare `95`: the arithmetic was correct, but the required object format failed. That response is retained in the receipt. The demo command now uses the exact evaluation prompt. Passing the short task suite does not imply reliable formatting or quality for arbitrary prompts.

## What these checks establish

- **Real inference:** actual quantized model weights were loaded by llama.cpp; requests produced stored, scored responses.
- **Fair reference:** the baseline uses the engine's existing automatic fitting. Candidate trials use generated Python to choose the CPU expert-layer count.
- **Fixed conditions:** weights, quantization, engine, 4,096-token context allocation, thread count, prompts, temperature, seed and scoring rules stay fixed within a run.
- **Separate confirmation:** six tuning questions are each repeated twice. Four different confirmation questions are withheld from the coding model. They are basic arithmetic, extraction and code-reading checks, not a general quality benchmark.
- **Durable evidence:** Atlas stores the run context, generated source and hypotheses, measured experiments, referenced evidence IDs, failed attempts and final recommendation. The JSON files here are exports of those records.

## Interpretation limits

The short test responses are usually only a few tokens. Decode rates are engine-reported rates for those requests, not a production serving benchmark. Request latency is full nonstream completion time, not time to first token. GPU memory is sampled every 250 ms across the whole device, including other processes; peaks between samples can be missed. CPU process RSS is retained separately and is not total machine RAM consumption.

Model downloads and checksum reads overlapped the early 35B trials, so their timing differences are confounded by storage and memory activity. No 35B speedup is claimed. The 11.39 GB compressed weights already fit this card, so that run does not prove larger-than-VRAM capacity. Earlier development runs using a 10,000 MiB artificial budget are excluded from the current physical-capacity claim.

No CUDA kernel, model training, quantization method, or generic architecture support was invented by this project. llama.cpp supplies inference and offloading; model authors and quantizers supply the weights. See [attribution](ATTRIBUTION.md).

## Verification and eligibility

All **46 code checks passed**, including the original CUDA property check, generated-policy restrictions, failure feedback, persistence, holdout promotion and saved-recipe guards. The code checks use fixtures where appropriate; they are separate from the actual Atlas/GPU runs above.

The user confirmed on October 11 that this cluster was created through the event's Atlas Sandbox invitation. Database read/write operation is verified; organization membership was not independently inspected in the browser. [Hackathon criteria and submission requirements](HACKATHON_REVIEW.md).

To publish completed local exports without model weights or credentials:

```bash
uv run python scripts/export_inference_evidence.py RUN_ID [ANOTHER_RUN_ID]
```
