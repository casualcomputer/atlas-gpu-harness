# New work for the October 11, 2026 hackathon

This repository began empty. The following components were built during the event.

## Consumer MoE inference harness

- **Model/hardware inspection:** `atlas_gpu_harness/model_probe.py` reads GGUF metadata, checks a limited architecture allowlist, records actual GPU/RAM/disk facts, and estimates a starting placement. Metadata identity is distinguished from a full weight checksum.
- **Generated and revised code:** `atlas_gpu_harness/policy_codegen.py` asks local Ollama to write `choose(context)`. It validates restricted syntax, evaluates the function in a bounded process, saves source/receipts, and supplies previous code, errors and measurements to later proposals. The rule selects only `n_cpu_moe`.
- **Real model evaluator:** `atlas_gpu_harness/inference_backend.py` owns the inference server, runs fixed tuning or separate confirmation tasks, checks exact JSON answers, samples GPU memory and process RSS, and records outputs, token counts, timings and failures. Engine auto-fitting is the baseline.
- **Persistent improvement loop:** `atlas_gpu_harness/inference.py` connects proposals, trials, compatible Atlas history, reserved proposal budgets, resume, confirmation and recommendations. Failed code or disappointing measurements are evidence, rather than success claims.
- **Runtime setup:** `atlas_gpu_harness/setup_engine.py` (`atlas-harness setup-engine`) pins official llama.cpp/CUDA assets by SHA256, supplies a local compatibility path for older Linux, verifies the GPU backend, and records a manifest. It does not change system drivers or download weights.
- Focused tests for inspection boundaries, generated-policy validation, evaluator behavior and run persistence; the current local suite passes **46 tests**. Larger-model readiness now allows 600 seconds, and saved selection reasons are based on held-out outcomes.
- `scripts/make_inference_video.py` renders a 59-second captioned walkthrough from saved inference evidence, with actual source excerpts, measurements, decision and attribution. It explicitly identifies saved evidence and is not a live-capture tool.

The physical-capacity Qwen3.5-35B-A3B run completed seven trials, all passing answer/resource checks. Candidate held-out changes of +3.5647% and +1.9202% did not clear the >5% rule, so the engine baseline was retained. Qwen 30B and Kimi Linear 48B files are fully downloaded with full SHA256 checksums verified in [the model manifest](evidence/inference-models.json); Kimi completed four real trials: three tuning placements passed 12/12 checks each, and the fastest engine baseline passed four reserved questions. Generated code changed from 13 to 14 CPU expert layers after feedback, recovering some performance while still trailing the engine baseline. The 30.06 GB Kimi file therefore has larger-than-VRAM inference evidence, with no claimed efficiency gain over llama.cpp. Qwen3-30B remains unevaluated. Each capacity/performance claim must point to its own completed comparison in [INFERENCE_RESULTS.md](INFERENCE_RESULTS.md).

The user confirmed on October 11, 2026 that the Atlas environment was created through the event invitation. That provenance is user-confirmed; independent UI verification was unavailable.

## Earlier operator prototype

The initial build remains a distinct experiment-memory demonstration:

- Exact top-k contract, independent CPU reference, correctness fixtures, CUDA timing and a packed-key alternative composed from PyTorch in `atlas_gpu_harness/evaluator.py`.
- Atlas runs/experiments/decisions, stable experiment identity and owner-token completion guard in `atlas_gpu_harness/store.py`.
- Fixed candidate memory policy in `atlas_gpu_harness/policy.py`; CLI/process locking, durable-result-first progression, injected crash, resume, status and export in `atlas_gpu_harness/__main__.py`.
- Controlled history comparison and metric recomputation in `scripts/measure_harness.py` and `scripts/summarize_experiments.py`. Evaluator version 3 corrected warmup on both sides of paired timing.

Its four-to-three attempt reduction concerns repeated operator evaluations. It is not an LLM speed or capacity result.

## Reused work and inspiration

**llama.cpp** supplies inference, GPU kernels, model-format support, expert offloading and automatic fitting. **Ollama** serves an existing local coding model. Upstream model authors and quantizers supply weights, tokenizer, training and quantization; these are not our original work. **PyTorch** supplies the earlier GPU operations and CUDA events. **PyMongo** connects the harness to Atlas.

**Strata and Project Maya are inspirations**, not integrated engines. Their code, calibration, model support and published performance are not claimed as our work. We created the bounded generated-policy/measurement/feedback loop and its saved evidence. We did not create a new model, quantizer, CUDA kernel or general inference architecture.
