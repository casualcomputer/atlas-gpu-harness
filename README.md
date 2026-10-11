# Atlas GPU Harness

A small long-horizon GPU optimization harness for the October 11, 2026 Harness Engineering & Model Wrangling hackathon. It evaluates **one fixed exact top-k operator** on a real NVIDIA GPU, checks output indices against an independent CPU oracle, times CUDA execution, and writes every decision and result to the hackathon MongoDB Atlas Sandbox. After a deliberate controller crash, `resume` reconstructs progress from Atlas and skips the committed evaluation.

The original work is the Atlas-backed experiment controller, structured memory policy, correctness gate, and recovery flow. The GPU operators are **PyTorch implementations**, not new kernels. The policy is deterministic, not an LLM agent. No speedup is promised.

## Today's verified machine

- WSL2, NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM, compute capability 12.0.
- NVIDIA driver 591.86. `nvidia-smi` reports driver-supported CUDA 13.1; the working PyTorch wheel uses CUDA 12.8. `nvcc` was not installed.
- Python 3.11 in the project `.venv`, reusing a locally installed PyTorch 2.8.0+cu128 environment.
- The local `baseline` command passed the CPU oracle and produced 30 real CUDA event samples. The manifest and raw samples are saved under ignored `artifacts/`.

## Setup

Use a working CUDA PyTorch environment. On the verified desktop, this project reuses the installed `henry_benchmark` Conda environment without copying or reinstalling PyTorch:

```bash
cd /mnt/c/Users/henry/Documents/ChatGPT/gpu-engineer-agent
/home/henryl/miniconda3/envs/henry_benchmark/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m atlas_gpu_harness preflight
.venv/bin/python -m atlas_gpu_harness baseline
```

On another CUDA system, create a Python environment, install an appropriate CUDA PyTorch wheel from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install `requirements.txt`. The preflight command must report a successful CUDA computation before continuing.

## Atlas Sandbox setup

1. Accept the hackathon invitation to the supplied **MongoDB Atlas Sandbox** and select that project. A cluster in a personal Atlas project does not meet the event requirement.
2. Under **Security → Database & Network Access → Database Users**, create a username/password database user with `readWrite` on the `atlas_gpu_harness` database. These are database credentials, distinct from your Atlas sign-in.
3. Under **Security → Database & Network Access → IP Access List**, add this desktop's **public outbound IP**. The WSL or Tailscale private IP is not what Atlas sees. Restrict the entry to the current IP rather than opening all addresses.
4. In the cluster's **Connect → Drivers** flow, copy the Python connection string and replace its credential placeholders. URL-encode special characters in the password. Save it only in the repository's ignored `.env` file as `MONGODB_URI=...`; `.env.example` contains placeholders only. Never commit or show the real URI.
5. Run the check below. It performs a live ping and a write/read/delete probe. In Atlas Data Explorer, the demo records will appear in `atlas_gpu_harness.runs`, `.experiments`, and `.decisions`.

```bash
.venv/bin/python -m atlas_gpu_harness atlas-check
```

See [MongoDB's database user](https://www.mongodb.com/docs/atlas/security-add-mongodb-users/), [IP access list](https://www.mongodb.com/docs/atlas/security/add-ip-address-to-list/), and [cluster connection](https://www.mongodb.com/docs/atlas/connect-to-database-deployment/) documentation for the current UI.

## Demonstration

```bash
# Create a durable run, intentionally exit 86 just after the first result commit.
.venv/bin/python -m atlas_gpu_harness run --budget-evals 4 --fault-after-commit
# Copy the printed RUN_ID. The deliberate nonzero exit is expected.
.venv/bin/python -m atlas_gpu_harness status --run-id RUN_ID
.venv/bin/python -m atlas_gpu_harness resume --run-id RUN_ID
.venv/bin/python -m atlas_gpu_harness export --run-id RUN_ID
```

The first candidate is the stable sort baseline. The policy retrieves compatible successes **and failures**, then tries stable argsort, a packed-key partial top-k algorithm, and PyTorch's direct top-k. The packed-key approach is our small candidate algorithm built from PyTorch operations; on the verified GPU it was correct but slower than stable sort. An exact tie failure rejects direct top-k and blocks the related unstable sort attempt. Atlas holds completed work before the injected crash. On restart, that completed identity is skipped; interrupted running work may be retried after the GPU file lock is released. This is idempotent result commitment, **not exactly-once GPU execution**.

The report under `artifacts/` includes raw microsecond samples, median, range, correctness checks, decisions with retrieved experiment IDs, and negative results. Credential-free copies of the verified runs are in `evidence/` for public review. `paired_recheck.speedup_supported` is deliberately conservative and only true if every paired candidate sample beats every paired baseline sample. Do not claim a speedup from the first 30-sample median alone.

## Evaluation contract

One contiguous float32 vector of length 16,384; `k=64`; fixed CPU seed `20261011`. Output is the exact sequence of indices ranked by descending score and ascending index on ties. The independent oracle is Python's stable sort of CPU values. Correctness fixtures cover ties, negative values, `k=1`, `k=n`, plus the primary workload. Incorrect candidates are never timed or selected as best. Warmup is 10 calls, then 30 CUDA event samples. Compilation and input transfer are outside the timing window. The contract and hardware/software fingerprint are stored with each run and embedded into each experiment identity.

## Tests and limits

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The four local controller tests use a MongoDB test double and mocked worker output; they are **not** Atlas or GPU integration proof. `preflight` and `baseline` are real GPU checks. The complete Atlas run and actual recovery demo require a configured hackathon Sandbox. One GPU worker runs at a time through a file lock; the controller also locks each run locally. This is a single-desktop demo, not a distributed scheduler. The candidate allowlist is intentionally small, and semantic/vector retrieval is outside this MVP.

See [EVIDENCE.md](EVIDENCE.md) for verified live results, [NEW_WORK.md](NEW_WORK.md) for original contributions, [ATTRIBUTION.md](ATTRIBUTION.md) for reused software, and [DEMO.md](DEMO.md) for the live demo and video storyboard.
