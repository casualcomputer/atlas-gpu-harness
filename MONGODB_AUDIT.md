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

## Native Strata 32K run

An independent live read on October 11, 2026 verified run `e5359ac4eb6d42358187f0530d66b1d9` after all trials completed. The [audit receipt](evidence/strata/audit.json) records:

- 18 real native inference results and 18 decisions; every experiment remains at attempt 1.
- 153 evidence references, all resolved and pointing to results committed before their decision.
- The first trial was committed, the controller exited deliberately, and resume recovered that record without executing it again.
- Every engine response reports 32,768 input tokens and zero reused prompt tokens. All saved input artifacts and the runner source match their recorded SHA-256 checksums.
- All 18 outputs retrieved every final setting correctly; none computed the requested total correctly. The run is complete with `recommendation: null`; completion does not imply a passing candidate.

These are real Strata `default`, `cpu_4`, `cpu_7`, `adaptive_160`, and `no_pcie` trials. They are separate from the earlier PyTorch `sort_stable`/`argsort_stable` experiment. The [new UI](demo-ui/strata.html) reads the [Atlas export](evidence/strata/latest.json) after each committed result and exposes its commands, answers, counters and prompt downloads. It receives no database credentials. Strata’s policy grid is predetermined; the ledger is used for resume, evidence lineage and gated selection, not an LLM-generated policy loop.
