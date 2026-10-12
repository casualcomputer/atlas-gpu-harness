# Atlas GPU validation — 2026-10-12

## Outcome

Completed real GPU runs for Qwen3-8B (llama.cpp) and Qwen3-4B (vLLM/SGLang). The 8B winner is llama.cpp Q6_K at **355.0 output tokens/s**; the 4B winner is **vllm at 358.0 output tokens/s**. Winner scores use the median of the initial measurement and two confirmation reruns. These are synthetic chat workloads, not a general model-quality benchmark.

## Hardware and runtime

- NVIDIA GeForce RTX 5070 Ti, 16,303 MiB VRAM (15.92 GiB). Windows NVIDIA driver 591.86; WSL NVIDIA-SMI 590.57, CUDA compatibility 13.1.
- WSL2 Ubuntu, Linux 6.6.87.2; Docker Desktop Linux engine 29.6.2. 47.0 GiB RAM visible to WSL.
- Docker GPU passthrough verified with an actual CUDA container. Ollama and LM Studio Windows processes stopped. Desktop/device use remained approximately 1.2–1.4 GiB; reported peaks include all device use.
- With user authorization, removed three LM Studio GGUF model files (about 39 GiB); applications and settings preserved. C: free space increased from 34 to 73 GiB before image/model preparation. Existing unrelated Docker workloads were preserved.
- Public models; no HF token, agent LLM or MongoDB mirror configured. Deterministic candidates and recovery rules were used. No cloud LLM proposed configurations.
- Repository source matched remote branch atlas-tune at 67122da before edits. The supplied folder lacked .git; restored metadata from a verified clone and preserved its pre-existing handoff and PowerShell files. All fixes pushed only to atlas-tune.

## Commands run

From `/mnt/c/Users/henry/Documents/atlas-tune` with `/home/henryl/.local/bin` and `/usr/lib/wsl/lib` on PATH:

```bash
nvidia-smi
docker info
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
uv sync
uv run atlas doctor
uv run python -m unittest discover -s tests
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
docker pull vllm/vllm-openai:latest
docker pull lmsysorg/sglang:latest
ATLAS_HOME=/home/henryl/.atlas-tune-gpu-cached uv run atlas run Qwen/Qwen3-8B --rounds 2
ATLAS_HOME=/home/henryl/.atlas-tune-gpu-fixed uv run atlas run Qwen/Qwen3-8B --rounds 2
ATLAS_HOME=/home/henryl/.atlas-tune-gpu-fixed uv run atlas run Qwen/Qwen3-4B --engines vllm,sglang
bash -n atlas-out/c8c3eb78840b/serve.sh
bash atlas-out/c8c3eb78840b/serve.sh  # API_KEY supplied through the environment
bash atlas-out/e83c52b324ec/serve.sh  # API_KEY supplied through the environment
uv run atlas history Qwen/Qwen3-8B
uv run atlas history Qwen/Qwen3-4B
```

Additional Python probes called DockerRunner.start/wait_ready for a deliberately oversized vLLM reservation, then passed its captured failure through Tuner.next_round and Tuner.trial. Generated recipe verification used an ephemeral API key, an authenticated chat-completion request and an unauthenticated request. No API key was printed or committed.

## What broke and what changed

1. **Weights-only planning overcommitted full-context KV memory** (`29de792`). Both Qwen3 configs have 36 layers, 8 KV heads, explicit head_dim 128. FP16/BF16 cache = 36 × 8 × 128 × 2 × 2 = 147,456 bytes/token. Sixteen 8192-token slots require 18 GiB before the 8.11 GiB Q8 weights. The original configuration actually ran on this driver: 8/8 quality, 196.0 tokens/s, 15.04 GiB sampled peak. It did NOT immediately OOM. Logs confirmed all 16 full-context slots; shared-memory spill/lazy allocation was not independently measured. The fix carries model KV geometry into HF/GGUF artifacts, honors explicit head_dim, budgets full context and existing GPU use, and sets bounded concurrency for all engines. Q8 slots fell to four; Q6_K used six. Shortlisting now reserves at least one full-context cache. Unknown architecture starts with one slot. Full unit and real-GPU validation passed.
2. **Failed docker run leaked Created containers** (`494793b`). Image extraction contention caused startup timeouts before a Server was returned, bypassing caller cleanup. The fix reserves identity before launch and cleans up failures/interruption without hiding the original exception. Verified with timeout regressions and a real occupied-port launch: its container was removed, existing service untouched.
3. **vLLM startup memory failure produced no recovery** (`ea8a11f`). At utilization 0.94, vLLM reported 14.66 GiB free versus 14.97 GiB requested. Atlas missed the Free memory ... less than desired wording. The fix recognizes the pre-allocation guard and lowers the reservation; generic OOM recovery no longer increases small sequence limits. The real captured failure went through Tuner.next_round: its exact 0.89/four-sequence recovery passed 8/8 quality and all requests at concurrency 1/4/16, at 80.5/314.3/311.3 tokens/s. Peak: 15.18 GiB. This is a tested startup-memory recovery, not an observed CUDA allocator crash.
4. **SGLang engine version was mislabeled as CUDA** (`0dcca97`). Its version command prints CUDA Version 13.0.3 then 0.5.21. The parser now prefers the package version and refreshes mistaken cached CUDA-banner entries. Verified against raw image output and regressions. Final unit suite: **34 tests passed**.

## Infrastructure observations

- Concurrent large-image extraction caused Docker-info/docker-run timeouts and high disk I/O pressure. Canceled the competing pull clients and completed image preparation sequentially. These infrastructure attempts are excluded from performance comparisons.
- A manual restart during the first slow model download raced with Atlas cleanup. That attempt was interrupted; it is not evidence of OOM. Separate local history directories prevent reuse of those cached failures in clean validation.
- Native Hugging Face prefetch completed Q8_0, Q6_K and the Qwen3-4B HF weights. Container-created llama.cpp cache files needed user ownership for host-side prefetch; the actual WSL cache mount was verified from both host and container.
- llama.cpp: 0.6.0-dev, build 11552, commit 23b0202a1; 14 tuning flags recognized. vLLM: 0.31.0; 12 flags. SGLang: 0.5.21; 13 flags, including successful alias mapping to cuda_graph_max_bs_decode.

### Reproducibility identifiers

- llama.cpp image: `ghcr.io/ggml-org/llama.cpp@sha256:9d835c2febd6f09a0aafb30e9c1b29a67a49923b67075993551ab4593d84491f`.
- vLLM image: `vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`.
- SGLang image: `lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`.
- Cached Qwen/Qwen3-8B-GGUF revision: `7c41481f57cb95916b40956ab2f0b139b296d974`.
- Cached Qwen/Qwen3-4B revision: `1cfa9a7208912126459214e8b04321603b3df60c`.

Generated recipes pin engine images; model references are not pinned to these cached revisions, so a future fresh download may differ.

## Measured results

Unchanged chat workload: context 8192, 256 target input/output tokens, concurrency 1/4/16, and 8/8/32 measured requests respectively. Scores are the best aggregate output-token rate across concurrency levels. All candidates retain the original quality gate, load scoring and finalist confirmation. GPU peaks are total device memory sampled every 0.5 s, including other processes.

| Model / engine | Configuration | Quality | Initial tok/s | Confirmation tok/s | Final score | Peak GiB | Status |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 8B / llama.cpp | Original Q8_0, 16 slots | 8/8 | 196.0 | Not confirmed | Comparison only | 15.04 | ok |
| 8B / llamacpp | Q8_0 `{"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on", "n_gpu_layers": 999, "parallel": 4}` | 8/8 | 293.2 | Not a finalist | 293.2 | 11.49 | ok |
| 8B / llamacpp | Q8_0 `{"flash_attn": "on", "kv_unified": true, "n_gpu_layers": 999, "parallel": 4}` | 8/8 | 209.8 | Not a finalist | 209.8 | 13.55 | ok |
| 8B / llamacpp | Q6_K `{"flash_attn": "on", "n_gpu_layers": 999, "parallel": 6}` | 8/8 | 359.5 | 355.0, 354.1 | 355.0 | 14.07 | ok |
| 8B / llamacpp | Q8_0 `{"flash_attn": "on", "n_gpu_layers": 999, "parallel": 4}` | 8/8 | 303.5 | 291.4, 288.7 | 291.4 | 13.53 | ok |
| 8B / llamacpp | Q8_0 `{"batch_size": 2048, "flash_attn": "on", "n_gpu_layers": 999, "parallel": 4, "ubatch_size": 512}` | 8/8 | 296.8 | 296.1, 293.3 | 296.1 | 13.53 | ok |
| 4B / vllm | bfloat16 `{"gpu_memory_utilization": 0.9, "max_num_seqs": 5}` | 8/8 | 350.5 | Not a finalist | 350.5 | 15.24 | ok |
| 4B / sglang | bfloat16 `{"max_running_requests": 4, "mem_fraction_static": 0.85}` | 8/8 | 343.9 | Not a finalist | 343.9 | 15.18 | ok |
| 4B / vllm | bfloat16 `{"gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8", "max_num_seqs": 5}` | 8/8 | 336.3 | Not a finalist | 336.3 | 15.31 | ok |
| 4B / vllm | bfloat16 `{"gpu_memory_utilization": 0.9, "max_num_seqs": 2}` | 8/8 | 165.3 | Not a finalist | 165.3 | 15.1 | ok |
| 4B / vllm | bfloat16 `{"gpu_memory_utilization": 0.9, "max_num_batched_tokens": 8192, "max_num_seqs": 5}` | 8/8 | 351.8 | Not a finalist | 351.8 | 15.37 | ok |
| 4B / sglang | bfloat16 `{"chunked_prefill_size": 2048, "max_running_requests": 4, "mem_fraction_static": 0.85}` | 8/8 | 352.5 | Not a finalist | 352.5 | 14.81 | ok |
| 4B / sglang | bfloat16 `{"max_running_requests": 4, "mem_fraction_static": 0.85, "schedule_policy": "lpm"}` | 8/8 | 351.8 | Not a finalist | 351.8 | 14.8 | ok |
| 4B / vllm | bfloat16 `{"async_scheduling": true, "gpu_memory_utilization": 0.9, "max_num_seqs": 5}` | 8/8 | 358.0 | 353.9, 361.0 | 358.0 | 14.85 | ok |
| 4B / sglang | bfloat16 `{"cuda_graph_max_bs_decode": 32, "max_running_requests": 4, "mem_fraction_static": 0.85}` | 8/8 | 356.0 | 352.2, 352.9 | 352.9 | 14.81 | ok |
| 4B / sglang | bfloat16 `{"kv_cache_dtype": "fp8_e4m3", "max_running_requests": 4, "mem_fraction_static": 0.85}` | 8/8 | 354.3 | 351.0, 351.2 | 351.2 | 14.79 | ok |

4B stability note: runner-up is within 5% (352.9 vs 358.0): treat as a tie

## Run outputs and deployment checks

- 8B: `atlas-out/c8c3eb78840b/`.
- 4B: `atlas-out/e83c52b324ec/`.
- Each completed winning run writes report.md, serve.sh, compose.yaml and k8s.yaml. Images are pinned by digest.
- 8B generated serve.sh actually started Q6_K on port 8000 and answered 95 for 37 + 58 with HTTP 200. Without authentication, the same endpoint returned HTTP 401. Both YAMLs parsed and bash -n passed. The server was removed after validation to free the GPU.
- 4B generated serve.sh started the winning vLLM configuration on port 8000. Authenticated requests returned HTTP 200; an unauthenticated chat request returned HTTP 401. All eight unchanged benchmark quality prompts passed through this generated deployment. Both YAMLs parsed and bash -n passed. The server was removed after validation.
- **Additional 4B quality limitation:** the user-only prompt `What is 37 + 58? Return only the integer answer.` with thinking disabled returned `65` on two separate generated-server launches. The original benchmark prompt plus its JSON-answer system instruction returned `{"answer": 95}` and all eight benchmark tasks passed. The additional prompt failure is retained as a limitation, not counted as a pass or fixed by changing the benchmark. Its model/engine-level cause was not isolated.
- Verified runs were merged into the default `/home/henryl/.atlas-tune/atlas.db` without replacing existing rows. Both ordinary `atlas history` commands now show the completed runs. The pre-merge backup is `atlas.before-gpu-validation-0e8ec796.db` alongside that database. Isolated validation history remains at `/home/henryl/.atlas-tune-gpu-fixed/atlas.db`; original comparison history is `/home/henryl/.atlas-tune-gpu-cached/atlas.db`. Databases, caches, tokens and .env files are excluded from commits.

## Karpathy autoresearch: useful patterns for Atlas

Reviewed the upstream [README](https://github.com/karpathy/autoresearch) and [experiment instructions](https://github.com/karpathy/autoresearch/blob/master/program.md). Autoresearch establishes a baseline, fixes the evaluation and training budget, records commit/metric/memory/status/description in a TSV, and retains or discards each experiment based on its measured outcome. It targets model training; Atlas should adapt the experiment discipline to inference rather than import the training harness.

Atlas already has substantial equivalents: `Store` saves configurations, hypotheses, engine digests, quality, performance and failures; `Tuner.next_round` passes past trials and current failures to rule-based fixes and the optional LLM; finalist confirmation measures stability. Without `ATLAS_LLM`, stock candidates and OOM rules run, but open-ended research proposals do not.

Recommended additions after runtime correctness:

1. Preserve an append-only record of every attempt and confirmation, linked to a candidate and parent hypothesis. Current trial rows are replaced by configuration key, so they are a cache rather than a complete attempt history.
2. Record source commit, immutable model revision, image digest, workload/scorer version, hardware, hypothesis, changed settings, elapsed time, quality, throughput, latency, peak VRAM, outcome and next-step rationale.
3. Keep quality/load/confirmation rules fixed. Choose a winner only from eligible candidates and use repeated measurements for small gains. Store failures with null measurements, not a numeric zero that could rank as a result.
4. Preserve failed experiments in history and use their signatures to avoid repeats; distinguish transient startup/download errors from invalid configurations so transient failures can be retried.
5. Display best-so-far performance alongside the full attempt history. A flat best score should trigger a new evidence-backed hypothesis, not weaker evaluation.

These are design recommendations, not implemented feedback-loop features.

## Limits and still unverified

- Native Windows Python/PowerShell execution was not tested; these runs used WSL2 with Docker Desktop and WSL cache paths.
- Compose and Kubernetes manifests were generated and parsed; neither was deployed to a Compose stack or Kubernetes cluster.
- No CUDA allocator-level OOM occurred in the oversized original llama.cpp baseline. vLLM startup memory refusal and successful automatic recovery were directly tested.
- The synthetic workload uses relatively short prompts; full 8192-token prompts in every simultaneous slot were not load-tested. Memory planning reserves that context, but this is not proof against every long-context or adversarial workload.
- The extra user-only arithmetic prompt failed on the generated 4B server, despite 8/8 benchmark prompts passing; broader prompt robustness remains unverified.
- Eight exact-answer quality questions are a smoke test. No broad quality benchmark, gated models, cloud-LLM proposal loop, or MongoDB mirror was validated.
- Autoresearch-inspired append-only history changes are recommendations, not implemented features.


## Follow-up: history, recovery and offloading fixes

This follow-up supersedes the earlier statements that append-only history and model revision pins were only recommendations. The original full-run measurements above remain historical results from their original workload/scorer version.

Implemented and regression-tested:

- Immutable trial, reuse, confirmation and interruption records; one-time legacy snapshot migration; hypotheses, sources, model revision, source commit, workload version, failures and full measurements retained. The initial interrupted download attempt predated interruption-record handling and has only its run status.
- Readable categories for memory reservation, allocator/KV OOM, storage, ports, authentication, unsupported options/hardware, timeouts, network/TLS, quality, load, latency limits and interruptions. Unknown failures remain unknown; categories are hints, not asserted root causes.
- Automatic retry of transient cached failures on a new run; explicit `--retry-failed`; cached OOMs can still produce deterministic fixes.
- Complete confirmation required even for a lone finalist. Quality and load gates run on every confirmation. Incomplete/disabled confirmation is provisional and exports no verified recipe.
- Every requested concurrency level must pass. Optional `--concurrency`, `--max-ttft-ms`, and `--quality-file` define additional requirements. Custom exact-answer prompts retain their specified system/user structure and all must pass. Latency ranking uses inverse p99 response time.
- Opt-in dense-model CPU offloading with RAM/KV estimates and one active sequence; required offload settings are not silently stripped. HF revisions, GGUF URLs and saved image commands are pinned.
- CPU model/count and total RAM now participate in cache identity. Legacy hardware records remain visible to the agent as labeled context, but are not automatically reusable measurements.
- The AI receives actual serving context, required workload and recent experiment history. Architecture remains one optional LLM agent with research/proposal tools, plus deterministic execution, scoring and storage modules.

Final suite: **61 tests passed in 48.058 seconds**. Includes fake-provider research, failure recovery, legacy migration, immutable attempts, interrupted trials, relaunch quality failures, incomplete confirmations, workload gates, offload planning, immutable launches and cache identity.

### Real partial-offload regression

Physical hardware: the same RTX 5070 Ti (16,303 MiB), Intel Core i7-14700F, WSL2/Docker Desktop. Used an **artificial 8 GiB planning budget** to force partial CPU execution of the already-cached Qwen3-8B Q8_0 model; this does not claim validation on a physical 8 GiB GPU.

The Python validation fixture supplied only the pinned Q8_0 artifact to `Tuner` so the shortlist could not select a smaller GPU-fitting quant. Options: `engines=['llamacpp']`, `workload='smoke'`, `vram_gib=8`, `allow_cpu_offload=True`, `max_trials=1`, `rounds=1`, `confirm=2`, `budget_minutes=30`. The regular planner chose `n_gpu_layers=19`, `parallel=1`, `threads=10`, FP16 KV and context 8192.

Run: `22967e409d42`; code loaded at `c317e59`; workload/scorer `v4`. New llama.cpp image: build 11561, commit e937b17a2, `ghcr.io/ggml-org/llama.cpp@sha256:20c6e722e2fcc5ca9c15401d3fd8e35408aaf8f9c54a12f32920f637c57959bb`. Model revision: `7c41481f57cb95916b40956ab2f0b139b296d974`.

| Attempt | Quality | Requests at concurrency 1 / 2 | Best output tokens/s | Sampled peak GiB |
| --- | --- | --- | ---: | ---: |
| trial | 8/8 | 8/8 / 8/8 | 14.5 | 5.99 |
| confirmation | 8/8 | 8/8 / 8/8 | 14.5 | 5.99 |
| confirmation | 8/8 | 8/8 / 8/8 | 14.5 | 5.99 |

Median: **14.5 tokens/s**. Smoke workload: 64 target input tokens, 32 output tokens, concurrency 1/2. This is functional offload validation; its score is not directly comparable with the earlier chat workload or older image. All three launches retained complete measurements in local history.

All four output files were written under `atlas-out/22967e409d42/`. The generated `serve.sh` actually started the pinned model/image, passed all eight original quality prompts and the additional user-only arithmetic prompt, returned HTTP 200 with authentication and HTTP 401 without it. Both YAMLs parsed; the server was removed afterward.

Cache observation: a copied GGUF without a matching `.etag` sidecar was removed and redownloaded by the native URL downloader. Interrupted that redundant download through normal harness cleanup, verified the cached model's SHA256 against Hugging Face metadata, and prepared the matching ETag before retrying. The subsequent trial, both confirmations and generated server loaded the persistent file successfully. This behavior matches [llama.cpp's URL cache validation](https://github.com/ggml-org/llama.cpp/blob/e937b17a2/common/download.cpp). Existing unpinned caches may need a one-time download into the new revision-specific path.

Commands/probes executed in this follow-up:

```bash
uv run python -m unittest discover -s tests
uv run atlas run --help
uv run atlas history Qwen/Qwen3-4B --details
# Temporary, locally executed validation fixtures:
.venv/bin/python /tmp/atlas-live-offload.py
.venv/bin/python /tmp/atlas-verify-offload-serve.py Qwen/Qwen3-8B
uv run atlas history --run-id 22967e409d42 --json
```

Still unverified: a live external-LLM research/recovery loop (no configured provider or key was found), vLLM/SGLang's new automatic CPU-offload baselines, oversized-model MoE offloading on this GPU, and deployed Compose/Kubernetes workloads. Original full-GPU vLLM/SGLang results remain documented above. No multiagent orchestration was introduced.
