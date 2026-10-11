# New work for the October 11, 2026 hackathon

This repository began empty. The work built for this event is:

- A fixed exact top-k evaluation contract, independent CPU oracle, correctness fixtures, CUDA event timing, a packed-key partial-selection candidate built from PyTorch operations, and conservative paired recheck adapter in `atlas_gpu_harness/evaluator.py`.
- A MongoDB Atlas record model for runs, experiments, and decisions, with stable experiment identity and owner-token completion guard in `atlas_gpu_harness/store.py`.
- A small structured memory policy that retrieves both success and failure records and refuses known invalid or duplicate candidates in `atlas_gpu_harness/policy.py`.
- A CLI controller, single-GPU worker lock, durable-result-first progression, explicit injected crash, restart, status, and evidence export in `atlas_gpu_harness/__main__.py`.
- Focused local controller tests and reproducible demo instructions.
- A controlled comparison that changes only access to a fixed prior Atlas history, with reversed execution order, equal correct-alternative checks, raw exports, and independent metric recomputation in `scripts/measure_harness.py` and `scripts/summarize_experiments.py`. Evaluator version 3 corrects warmup on both sides of paired GPU comparisons.

PyTorch performs every GPU operator and supplies CUDA events. PyMongo is the Atlas client. No upstream kernel, model, benchmark, or published latency is represented as original work. The benchmark numbers in our evidence files come only from this desktop's actual execution.
