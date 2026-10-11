# Submission draft

**Status:** Project description prepared; submission form and team-member entry remain pending.

## Project title

**Atlas GPU Harness — Big Models, Measured Desktop Inference**

## One-line description

A local coding agent writes and tests CPU/GPU placement policies for large MoE models, using MongoDB Atlas to turn each experiment into evidence for the next attempt.

## Problem and impact

Developers want capable open models on existing hardware. Models larger than GPU memory require repeated tuning. A configuration that fits can still be slow or produce unusable answers. Each new model restarts this work.

Our goal is a repeatable path from a supported model file to measured, reusable settings for the user's machine.

## Implemented solution and contribution

We built a bounded coding-and-evaluation harness around llama.cpp. It first measures the engine's automatic fitting. A local Ollama model then writes a Python placement policy; constrained validation and execution permit one output: how many early layers keep their experts in CPU RAM.

The harness runs real inference, checks exact answers, samples memory, records timings, and supplies results and failures to the next code revision. Separate questions and reversed-order comparisons decide whether to promote a candidate, keep the baseline, or return no recommendation. Saved recipes have compatibility checks for reuse.

llama.cpp already provides quantized inference, CPU/GPU offload and automatic fitting. Our original contribution is the chain **generated source → executed configuration → measured result → feedback → verified recommendation**, with persistent budgets and resumable evidence. The evaluator and acceptance rules remain fixed while policy code changes.

## Working demonstration and measured result

**Kimi-Linear-48B-A3B Q4_K_M:** a **30.06 GB** model file ran on a **16 GB GeForce RTX 5070 Ti**, using existing engine offload and system RAM.

| Trial | Answer checks | Sampled GPU peak | Median short-answer decode | Median request |
| --- | --- | --- | --- | --- |
| Engine automatic fitting | 12/12 | 15,472 MiB | 47.21 tokens/s | 0.948 s |
| First policy: 13 CPU expert layers | 12/12 | 15,791 MiB | 22.77 tokens/s | 1.405 s |
| Revised policy: 14 CPU expert layers | 12/12 | 15,890 MiB | 29.77 tokens/s | 1.137 s |

All four Kimi trials completed: three tuning trials passed 12/12, and separate baseline confirmation passed 4/4. Twelve tuning checks are six distinct questions repeated twice. The revision measured 30.73% faster than the first policy but remained 36.94% below engine auto-fitting. **The harness confirmed and retained the engine baseline.** Its confirmation measured 14,664 MiB peak, 52.6966 decode tokens/s and 0.8151-second median request latency on different questions. No engine speedup was established.

The completed 35B run also retained its baseline. Full records: [INFERENCE_RESULTS.md](INFERENCE_RESULTS.md).

## MongoDB Atlas

Atlas stores runs, generated policies, experiments, measurements, failures, decisions and recommendations. Each new coding attempt receives compatible tuning evidence; confirmation questions stay outside its feedback. Decision records identify the experiments retrieved, and a read-only aggregation makes those links inspectable. The owner confirmed creating this Atlas environment through the event invitation.

## Track fit

**Recursive Harnessing:** bounded evolution of placement-policy code from measured feedback. **Long Horizon Engineering:** durable experiment memory, metric-driven progress and resume. The current demonstration covers one placement function and supported model formats; broader architectural self-improvement and billion-token reasoning are future ambitions.

## Credits and submission links

Built during the event: harness orchestration, policy validation, evaluator, Atlas integration, confirmation and evidence tooling. Reused: llama.cpp, Ollama/Gemma, Moonshot AI's Kimi model and Bartowski's GGUF quantization. Strata and Project Maya inspired the consumer-hardware direction.

- Repository: [casualcomputer/atlas-gpu-harness](https://github.com/casualcomputer/atlas-gpu-harness)
- One-minute demo: [59-second evidence walkthrough](https://github.com/casualcomputer/atlas-gpu-harness/raw/refs/heads/main/demo/atlas-inference-demo.mp4).
- Team members: **[Add actual registered members]**
