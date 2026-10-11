# MongoDB Atlas and demo replay audit

Read-only check on October 11, 2026 against the configured `atlas_gpu_harness` database. The check queried `runs`, `experiments`, and `decisions` directly without calling `AtlasStore.init()` or writing a probe. No connection string or credential was recorded.

## What the names mean

`sort_stable`, `argsort_stable`, `packed_topk`, and `topk_sorted` are real candidate names in the **earlier PyTorch top-k operator experiment**, defined in [`evaluator.py`](atlas_gpu_harness/evaluator.py). They call PyTorch operators or the local packed-key expression; they are not llama.cpp, Strata, or Maya MoE inference policies. The [operator controller](atlas_gpu_harness/__main__.py) evaluates them, and [`AtlasStore`](atlas_gpu_harness/store.py) writes each candidate, result, status, attempt, and timing to `experiments`. Decisions store the experiment IDs retrieved before proposing the next candidate.

The newer Kimi inference path stores `baseline`, `policy_1`, `policy_2`, and `holdout_baseline_1` under a separate run with `workflow=consumer-inference-v1`. A second live read confirmed that both proposals retain Python source, source hash, hypothesis, and chosen CPU expert-layer count in the run document; the recommendation retains a validated experiment ID. See the [full export](evidence/inference/70be4572a9824186999857dadeb1237d.json). The vLLM/SGLang deployment planner writes JSON to standard output and does **not** yet write to Atlas.

## Live read results

| Run | Stored experiments | Decision evidence links | Finding |
| --- | --- | --- | --- |
| `2eb9f21a2e374f358c4a42b9fbe2258d` | Four, all attempt 1: three completed operators and a failed `topk_sorted` tie check. | 0, 1, 2, 3 retrieved IDs; zero dangling IDs. | Live status `completed`; stored medians 104.1, 92.8, and 222.9 µs match the [export](evidence/report-2eb9f21a2e374f358c4a42b9fbe2258d.json). |
| `21dbac5116a04dafa1ddaed17ea34030` | Three completed operators, all attempt 1. | 4, 5, 6 retrieved IDs; zero dangling IDs. | Live status `completed`; stored medians 121.8, 107.6, and 220.3 µs match the [export](evidence/report-21dbac5116a04dafa1ddaed17ea34030.json). |
| `70be4572a9824186999857dadeb1237d` | Four completed Kimi inference trials, all attempt 1. | 0, 1, 2, 0 retrieved IDs; zero dangling IDs. | Live recommendation is `engine_autofit`; decode rates 47.21, 22.77, 29.77, and 52.70 tokens/s match the [export](evidence/inference/70be4572a9824186999857dadeb1237d.json). |

The UI's two older-run dot plots contain 180 displayed timing samples rounded to 0.1 µs. Every displayed rounded sample and median matches the corresponding export. The shortened experiment and decision IDs shown in the replay also match the exports. The page embeds these saved values in JavaScript; it is a **static replay, not a live Atlas dashboard**.

## Scope of the integration

The tested single-controller flow persists a result before updating run progress, retrieves compatible prior experiments, stores evidence IDs with decisions, and can resume after a deliberate post-commit exit; see [EVIDENCE.md](EVIDENCE.md). The live reads above confirm the records and links exist now. They do not establish event Sandbox membership, production concurrency behavior, or a speedup. `AtlasStore.start()` can reclaim a `running` experiment without a distributed lease; the local file lock constrains this demo to one controller on one machine. The deployment planner still needs Atlas-backed trial storage and resume before it can make the same durability claim.
