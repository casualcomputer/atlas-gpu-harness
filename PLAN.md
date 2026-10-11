# Four-hour execution checkpoints

> Historical operator-prototype schedule. The user subsequently clarified the goal as running larger MoE language models on a consumer GPU. The implemented inference loop and current validation are described in [CONSUMER_GPU_PLAN.md](CONSUMER_GPU_PLAN.md), [INFERENCE_RESULTS.md](INFERENCE_RESULTS.md), and [DEMO.md](DEMO.md). The GPU memory budget defaults to the physical card capacity.

- **12:10–12:40 PM PT:** Verify actual GPU and CUDA path; get one independent-correctness baseline. Completed locally at about 12:14 PM.
- **12:40–1:25 PM PT:** Build Atlas records and controller; verify the Sandbox connection and first durable GPU experiment as soon as URI access is available.
- **1:25–2:10 PM PT:** Run injected-crash and same-run recovery against Atlas; inspect records in Data Explorer.
- **2:10–3:00 PM PT:** Validate memory decisions and compare candidate timing; export evidence.
- **3:00–4:00 PM PT:** Fix demo blockers only; record one-minute video and rehearse live demo.
- **By 4:45 PM PT:** Publish public repository and submit video/link, leaving deadline buffer.

Progress log: GPU preflight and local baseline completed. Atlas ping and write/read probe passed. Run `2eb9f21a2e374f358c4a42b9fbe2258d` proved commit-before-crash and same-run resume. Run `21dbac5116a04dafa1ddaed17ea34030` used prior failure evidence to avoid invalid candidates. Five tests pass, including a real CUDA property test. Next: visually confirm the cluster is in the hackathon Sandbox, rehearse, record video, publish the repository, and submit.
