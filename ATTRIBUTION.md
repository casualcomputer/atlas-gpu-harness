# Attribution

## Consumer MoE inference

- **llama.cpp** provides model loading, CPU/GPU placement, automatic fitting, CUDA kernels, chat templates, and the inference API. We use the official pinned `b11429` build; see [INFERENCE_SETUP.md](INFERENCE_SETUP.md) and the [upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b11429). The `gguf` Python library reads model metadata.
- **Qwen** provides the Qwen3 and Qwen3.5 model weights. The larger Qwen3 trial uses the publisher's [Qwen3-30B-A3B Q4_K_M GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF). The cached Qwen3.5 trial uses [Unsloth's UD-IQ2_M quantization](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF). We did not train or quantize these models.
- **Moonshot AI** provides [Kimi Linear 48B-A3B Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct). The Kimi trial uses [Bartowski's third-party Q4_K_M GGUF](https://huggingface.co/bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF). Kimi Linear is a distinct model from Kimi K2. Model support and quantization are upstream work.
- **Ollama and Google's Gemma 3 12B** provide the local coding model used to propose Python placement policies. The harness supplies the constrained interface, experiment feedback, validation, persistence, and promotion rules. See [Ollama](https://github.com/ollama/ollama) and [Gemma](https://ai.google.dev/gemma).
- **Strata and Project Maya** inspired the consumer-hardware goal and use of CPU/GPU memory tiers. Their existing automatic calibration and model-specific engineering deserve credit. Neither codebase is embedded in this implementation. [Strata](https://github.com/Niko1221/Strata), [Maya](https://github.com/mw00/project-maya).
- **vLLM and SGLang** are possible future inference adapters; no adapter to either is implemented or measured here.

Our event work is the generated-policy experiment loop, fixed tests, measurement and acceptance protocol, Atlas records and recovery, hardware/model inspection, and validated-recipe launcher. Upstream models and engines retain their own licenses. Large model files and engine binaries are not committed to this repository.

## Original operator experiment and shared dependencies

- **PyTorch** (`torch.sort`, `torch.argsort`, `torch.topk`, `torch.cuda.Event`): used for GPU operator implementations and timing. Copyright and license belong to the PyTorch contributors. [Repository and license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [sort documentation](https://docs.pytorch.org/docs/stable/generated/torch.sort.html), [CUDA event documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).
- **PyMongo**: official MongoDB Python driver. [Documentation](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/), [license](https://github.com/mongodb/mongo-python-driver/blob/master/LICENSE).
- **filelock**: local process lock for one GPU worker and one controller per run. [Repository](https://github.com/tox-dev/filelock).
- **python-dotenv**: reads an ignored local `.env` file. [Repository](https://github.com/theskumar/python-dotenv).
- **mongomock**: used only by controller tests; it does not establish Atlas integration. [Repository](https://github.com/mongomock/mongomock).

The Strata `qsa_select` benchmark was considered in the project plan. Its CUDA implementation is large and would require build/toolchain work; no Strata code or results were copied into this repository. The verified desktop had a working CUDA PyTorch environment, so this MVP uses that path.
