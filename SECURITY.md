# Security and isolation

atlas is a single-user CLI that runs on the machine with the GPU. This page lists what each part can reach, what leaves the machine, and what is **not** yet in place for enterprise use.

## What can touch what

| Part | Can reach | Cannot reach |
| --- | --- | --- |
| **Agent LLM** (optional) | Five tools: the engine image's cached `--help`, GitHub issue search, public web page fetch, Hugging Face model card, `submit_configs`. Plus the provider's built-in web search when it has one. | Local files, shell commands, Docker, environment variables, the local database. There is no tool for any of these. |
| **`fetch_page` / `model_card`** | Public internet hosts over http(s). Each redirect hop is re-checked. | `localhost`, private LAN ranges (10/8, 172.16/12, 192.168/16), link-local and cloud metadata (169.254.169.254), `file://`. |
| **Agent proposals** | A config dict for a listed engine and artifact, using flags from a per-engine allowlist that also exist in the pulled image's `--help`. | Any other container argument. Every value is type- and range-checked. No image, entrypoint, volume, env or network changes. |
| **Engine containers** | The GPU; two mounted caches (`~/.cache/huggingface`, `~/.cache/llama.cpp`); the internet, to download weights; `HF_TOKEN` if set. | Your working directory, home folder, `.env` and other files: nothing else is mounted. They run with `no-new-privileges`, and trial ports bind to `127.0.0.1` only. |
| **Image help check** | Runs `--help` / `--version` inside each image with `--network none`, then removes the container. | — |
| **atlas itself** | Docker, `nvidia-smi`, the two caches, its SQLite file (`~/.atlas-tune`), `./atlas-out`. | It removes only containers carrying its own `atlas-tune` label. |

## What leaves the machine

- **To the LLM provider** (only if `ATLAS_LLM` is set):
  - model ID, GPU name, VRAM, driver, RAM, disk free and CPU count
  - candidate artifacts, and the engine images' help text
  - trial summaries: configs, throughput, latency, quality score, wrong answers, error message up to 300 characters
  - for failed trials in the current run, the last 60 lines of the engine container's log, with tokens, API keys, passwords and connection strings removed (`--no-share-logs` turns this off)
  - text the agent's tools fetched

  Your files and keys are never sent.
- **To GitHub search** (anonymous unless `GITHUB_TOKEN`): the agent's queries, and for a failed trial the engine's error line with paths and numbers removed.
- **To Hugging Face**: model lookups and weight downloads (by the engines).
- **To MongoDB Atlas** (only if `MONGODB_URI` and the `atlas` extra): runs, trials (without log tails) and recipes.

Use `--no-research` to keep the agent off the network, or leave `ATLAS_LLM` unset to run with no LLM at all.

## Trust boundaries worth knowing

- **Text from the web is untrusted.** A page or issue can contain instructions aimed at the agent. The worst it can do is make the agent propose an allowlisted config, which then has to be measured and pass the quality check like any other.
- **Container images default to `:latest`** from Docker Hub and GHCR. Each trial records the exact image digest. For controlled environments, pin by digest (`ATLAS_VLLM_IMAGE=vllm/vllm-openai@sha256:…`) or point to an internal registry mirror.
- **Model weights are third-party code paths.** Some models need `--trust-remote-code`; atlas never passes it, so those models fail rather than run unreviewed code.
- **`--ipc=host`** is set because vLLM and SGLang use shared memory for workers. It shares the host IPC namespace with the container.
- **DNS rebinding:** the public-address check happens before the request, so a hostile DNS server could still change the answer between check and fetch. Run behind an egress proxy if that matters.
- **Docker itself is root-equivalent** for whoever can use it. Rootless Docker or Podman reduces that; atlas has not been tested with either yet.

## Not ready for enterprise yet

| Need | Status |
| --- | --- |
| SSO, roles, per-team access | none (single user, local files) |
| Central audit log | trials and decisions are in SQLite or Atlas, but there's no tamper-evident log |
| Secrets management | env vars and `.env` only; no Vault or cloud secret manager integration |
| Egress control and proxy | uses system proxy settings; no built-in allowlist |
| Air-gapped mode | partial: `--no-research` plus a local registry and pre-downloaded weights, not packaged |
| Policy (approved models, images, max GPU time) | per-run flags only |
| Multi-GPU, clusters, Kubernetes | single GPU, single host |
| Support for the engines' own security advisories | none; pin images and track upstream |

Report a security issue privately to the repository owner rather than in a public issue.
