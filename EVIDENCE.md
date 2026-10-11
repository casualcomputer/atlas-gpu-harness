# Verified local and Atlas evidence — October 11, 2026

These are measurements from the RTX 5070 Ti on this desktop. Credential-free raw reports and the preflight manifest are in the tracked `evidence/` folder. Local working copies in ignored `artifacts/` can be regenerated from Atlas using `export --run-id`.

## Environment and local gate

- WSL2; GPU: NVIDIA GeForce RTX 5070 Ti; VRAM: 16,303 MiB; compute capability 12.0.
- Driver 591.86, driver-supported CUDA 13.1, PyTorch 2.8.0+cu128 with CUDA 12.8. CUDA smoke computation passed. `nvcc` was absent.
- Fixed exact top-k baseline passed every CPU oracle fixture and 30 CUDA event timing samples. One local run measured 95.6 µs median. That number is not an Atlas run result.
- Five tests passed: four controller tests plus one real CUDA property test of the packed-key candidate. The controller tests use a MongoDB test double and mocked worker results.

## Atlas write/read and recovery

`atlas-check` returned `ping=ok` and `write_read=ok` against database `atlas_gpu_harness` using the supplied cluster URI. The cluster was supplied by the user; this CLI cannot independently inspect Atlas project membership in the event's Sandbox UI.

**Run `2eb9f21a2e374f358c4a42b9fbe2258d`** was started with a four-evaluation budget and the deliberate `--fault-after-commit` option. The controller exited with code 86 after committing the stable baseline. A fresh `status` read from Atlas showed that completed baseline at attempt 1 while the run remained in progress. `resume` evaluated the other three candidates and completed the run without rerunning the baseline.

| Candidate | Exact correctness | Median, µs | Paired speedup supported? |
| --- | --- | ---: | --- |
| Stable sort baseline | Pass | 104.1 | Reference |
| Stable argsort | Pass | 92.8 | No |
| Packed-key partial top-k | Pass | 222.9 | No |
| Direct PyTorch top-k | Fail on tie fixtures | Not timed | Not eligible |

Every experiment above had attempt 1. The four decision records referenced 0, 1, 2, and 3 retrieved experiment IDs respectively. Stable argsort had the lowest measured median, but its alternating recheck did **not** support a speedup claim.

**Run `21dbac5116a04dafa1ddaed17ea34030`** started with a five-evaluation budget and retrieved four records from the first run before its first decision. It committed only three correct configurations (stable sort, stable argsort, packed-key top-k), then stopped at 3/5. The stored direct top-k tie failure prevented repeating that candidate, and the policy also blocked unstable sort as a related tie-risk neighbor. This run demonstrates memory affecting future work. Medians were 121.8, 107.6, and 220.3 µs respectively; order/thermal variation is visible and is another reason not to claim a speedup.

**Run `64499b1447254259a47ba7699ac8e20a`** committed one further correct baseline after the initial Git commit. Its Atlas context records source revision `6f34b55a84161ef63d847a270ad2452a88e479b0` and evaluator version 2. It measured 105.3 µs median and retrieved seven compatible prior records before its decision.

## Limits

The repository is published at https://github.com/casualcomputer/atlas-gpu-harness. The 59-second backup video remains a local artifact until it is uploaded for submission. The real Atlas run validates connection, persistence, retrieval, and recovery, but the Atlas project identity still needs a visual check in the hackathon Sandbox UI. No LLM, custom CUDA kernel, or guaranteed performance gain is claimed.
