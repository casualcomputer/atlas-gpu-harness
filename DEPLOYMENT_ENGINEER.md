# LLM deployment engineer

The deployment workflow extends the llama.cpp GGUF experiments with **vLLM, SGLang, and Ollama** serving candidates. [Runtime research](RUNTIME_RESEARCH.md) records current official documentation for those engines and Unsloth. Unsloth supplies fine-tuned or exported artifacts; vLLM, SGLang, Ollama, or llama.cpp serves them after format and model-support checks.

The workflow has three steps: plan the same workload across available model artifacts, benchmark each running local OpenAI-compatible server, and compare recorded outcomes. It does not install an engine or silently replace the existing llama.cpp result. An emitted command is a test candidate; check the installed runtime's version and help, model support, tokenizer, and chat template before launch.

Search official documentation indexes before choosing an experiment:

```bash
python -m atlas_gpu_harness.deployment_engineer docs --query "MoE offload" > docs.json
```

This searches the first-party vLLM navigation and SGLang, Ollama, and Unsloth documentation indexes by title and description. It returns matching links and any index fetch errors. Read the matched pages for the installed version before using a flag; the index search is discovery, not a claim that it has read every document.

## 1. Describe the model and hardware

Save a `spec.json` like this, replacing the example model identifiers and measured sizes. `gpu_total_gib` is the aggregate memory of the GPUs assigned to one server, and `ram_available_gib` and `ssd_free_gib` are free capacity before launch. Give each runtime its actual artifact size because an HF checkpoint and an Ollama quantization can differ greatly. Only list runtimes for which you have a compatible artifact. vLLM and SGLang normally use a compatible Hugging Face checkpoint; Ollama uses an installed model tag or imported GGUF. Keep the exact model bytes, quantization, and chat template in your experiment notes.

```json
{
  "model": {"architecture": "moe", "weights_gib": 30},
  "hardware": {"gpu_total_gib": 16, "gpu_count": 1, "ram_available_gib": 48, "ssd_free_gib": 100},
  "context_tokens": 4096,
  "targets": {
    "vllm": {"model": "your-org/compatible-hf-model", "weights_gib": 30},
    "sglang": {"model": "your-org/compatible-hf-model", "weights_gib": 30},
    "ollama": {"model": "your-installed-model:tag", "weights_gib": 18}
  }
}
```

Generate a plan:

```bash
python -m atlas_gpu_harness.deployment_engineer plan --spec spec.json > plan.json
```

The plan contains exact command argument arrays, environment settings, the cited flag documentation, prerequisites, and hypotheses. Start with each runtime's baseline. Where the artifact is larger than safe GPU headroom and RAM is available, it adds a CPU offload test. After a recorded GPU pressure or server error, it proposes a KV cache memory test. For a multi-GPU MoE it compares expert parallel placement against the same GPU count without expert parallelism. SSD experiments compare cold and warm load using verified identical bytes; the endpoint benchmark cannot measure load or disk reads, so record those separately.

The planner does not execute its command arrays. Run only a supported command after checking its arguments against the installed engine. Start and stop one server at a time. This keeps trial ownership and resource effects clear.

## 2. Benchmark a running server

When a candidate is listening on a local OpenAI-compatible endpoint, use its actual served model ID. For example:

```bash
python -m atlas_gpu_harness.deployment_engineer benchmark \
  --runtime vllm --model your-org/compatible-hf-model \
  --endpoint http://127.0.0.1:8000 \
  --plan plan.json --experiment-id vllm:baseline > baseline-vllm.json
```

The benchmark sends six exact-answer questions twice in reverse order, checks strict JSON answers, records request latency and completion tokens per full response second, and samples total GPU and available RAM around each request. A sample can miss a transient peak. It does not measure decode-only speed, time to first token, external server startup, per-process memory, SSD reads, or sustained concurrent throughput. Record server logs and runtime/model revisions with the JSON result. The endpoint must be local; the tool never launches or terminates the serving process.

After a baseline, feed results back into the planner:

```bash
python -m atlas_gpu_harness.deployment_engineer plan \
  --spec spec.json --results baseline-vllm.json > next-plan.json
```

Then test a proposed variant and save its benchmark JSON. Compare outcomes:

```bash
python -m atlas_gpu_harness.deployment_engineer compare \
  baseline-vllm.json candidate-vllm.json > comparison.json
```

`worked` means the 12 short answer checks passed. It is not a production quality or speedup claim. The comparison sorts passing trials by completion tokens per **full response** second and leaves a caution about hardware, model bytes, tokenizer, and load conditions. A fair conclusion needs a pinned model revision and chat template, identical prompts and context, repeat trials, startup and SSD measurements, and representative concurrency. The existing [llama.cpp workflow](README.md) remains the stricter baseline for its supported GGUF MoE models.

## Documentation and optimization scope

- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) and [expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) cover KV memory, CPU weight offload, expert placement, and balancing.
- [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) and [expert parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism) cover static memory, offload, KV precision, and multi-GPU experts.
- [Ollama FAQ](https://docs.ollama.com/faq) covers context, parallel requests, KV precision, Flash Attention, CPU/GPU placement, and model storage.
- [Unsloth vLLM deployment](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide.md) and [GGUF export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) explain how to prepare artifacts for the serving runtimes. Preserve [chat templates and special tokens](https://docs.unsloth.ai/basics/saving-and-using-models/troubleshooting).
