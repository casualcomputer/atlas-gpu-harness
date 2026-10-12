# Reproduce the consumer GPU inference runtime

The model experiments use the official **llama.cpp b11429**, commit `d81235049`, Linux x86_64 CUDA 12.8 release. The helper installs it inside the ignored `artifacts/engines` directory. It does not install system packages, replace a GPU driver, download models, or start inference.

## Requirements

- Linux x86_64 or WSL, Python 3.11.14+ (including the standard-library tar extraction security filter), `curl`, and an NVIDIA GPU with a working driver.
- Approximately 3 GB of free space for the engine and CUDA archives plus their extracted files. Model storage is additional.
- On older distributions such as Ubuntu 22.04: `dpkg-deb` and an existing GCC 13+ C++ runtime containing `libstdc++.so.6` and `libgcc_s.so.1`. The helper defaults to `~/miniconda3/lib`; use `--cxx-dir` for another installation. Ubuntu 24.04 can normally run the release natively.

## Setup and verification

From the repository root:

```bash
uv run atlas setup-engine
# If the modern C++ runtime is elsewhere:
uv run atlas setup-engine --cxx-dir /path/to/existing/runtime/lib

# Read-only checks: pinned engine version, visible CUDA device, cached archive checksums.
uv run atlas setup-engine --check
artifacts/engines/llama-b11429/launch-llama.sh --list-devices
```

Successful setup writes `artifacts/engines/llama-b11429/setup-manifest.json`, including source URLs, archive SHA256 values, copied C++ library hashes, launcher hash, exact engine version, and detected CUDA devices. This is a runtime check; it does not establish that a model fits or produces useful answers.

The engine and CUDA archives are pinned to the [official release](https://github.com/ggml-org/llama.cpp/releases/tag/b11429), with SHA256 values matching GitHub's asset metadata. The optional [Ubuntu libc6 package](https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8.9_amd64.deb) is also pinned to its verified download hash. It is extracted locally and invoked with a separate loader. Existing system libraries remain unchanged.

## Interrupted setup

Downloads resume from `.download` files and are accepted only after the pinned checksum passes. Archive extraction uses a temporary directory before publishing its completed directory. Existing marked extractions and a working launcher are reused. A working launcher is preserved.

If an existing unmarked extraction cannot be verified, the helper stops. Use a fresh destination with `--engine-dir artifacts/engines/llama-b11429-clean`; existing experiments and runtime files remain available. A checksum mismatch also stops setup rather than replacing the unexpected file.

The older-Linux compatibility path needs a modern C++ runtime supplied through `--cxx-dir`; it does not silently install or upgrade one. Its hashes are recorded because the C++ package can differ between machines. Pinned model, engine, and hardware manifests still belong in each experiment record.
