# Consumer MoE inference: measured results

Measured on October 11, 2026 using a **GeForce RTX 5070 Ti with 16,303 MiB VRAM**, an i7-14700F, and WSL with about 48 GiB assigned RAM. The harness uses the physical card capacity by default. A model may use both GPU memory and CPU RAM.

## Current evidence

| Model | Weight file | Result |
| --- | ---: | --- |
| Qwen3.5-35B-A3B, Unsloth UD-IQ2_M | 11.39 GB | Completed seven real inference trials. Every scored response passed. Generated policies did not meet the speed-improvement rule; the engine baseline was retained. |
| Qwen3-30B-A3B, publisher Q4_K_M | 18.56 GB | Download complete and full SHA256 verified; inference trial pending. |
| Kimi Linear 48B-A3B Instruct, Bartowski Q4_K_M | 30.06 GB | Download complete and full SHA256 verified; inference trial in progress. |

The latter two files exceed this GPU's physical VRAM. File size alone does not establish that a model runs; actual outputs and held-out validation are required. GB here means decimal bytes; GPU measurements use MiB. [Pinned model sources and verified checksums](evidence/inference-models.json).

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

This run used source commit `3c40d89`. Subsequent timeout, feedback, and recommendation fixes are in `a8d8fd6`; the launcher intentionally rejects an old recipe after its recorded backend source changes. Use the larger-model results below for a recipe validated against the corrected implementation.

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
.venv/bin/python scripts/export_inference_evidence.py RUN_ID [ANOTHER_RUN_ID]
```
