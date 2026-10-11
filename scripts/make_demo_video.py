"""Render a captioned 60-second backup video from verified Atlas export files.

Requires Pillow and ffmpeg. This is a backup; a live screen recording of Atlas
Data Explorer and the CLI is preferable for the submission.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
FIRST = "2eb9f21a2e374f358c4a42b9fbe2258d"
SECOND = "21dbac5116a04dafa1ddaed17ea34030"


def font(size: int, mono: bool = False):
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def slide(number: int, title: str, kicker: str, lines: list[str], highlight: str) -> Path:
    image = Image.new("RGB", (1280, 720), "#0a1020")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 12), fill="#48d5d1")
    draw.text((80, 54), "ATLAS GPU HARNESS", font=font(24), fill="#48d5d1")
    draw.text((80, 108), title, font=font(51), fill="white")
    draw.text((80, 188), kicker, font=font(24), fill="#a9b6cd")
    draw.rounded_rectangle((70, 252, 1210, 618), radius=20, fill="#131e34", outline="#3b4e6e", width=2)
    for i, line in enumerate(lines):
        draw.text((105, 290 + i * 45), line, font=font(25, mono=True), fill="#e6edf7")
    draw.text((80, 660), highlight, font=font(21), fill="#6ce1ad")
    draw.text((1155, 660), f"{number}/6", font=font(20), fill="#a9b6cd")
    path = OUT / f"video-slide-{number:02d}.png"
    image.save(path)
    return path


def main() -> None:
    def report(run_id: str) -> dict:
        name = f"report-{run_id}.json"
        local = OUT / name
        path = local if local.exists() else ROOT / "evidence" / name
        return json.loads(path.read_text())

    first = report(FIRST)
    second = report(SECOND)
    by_name = {item["candidate"]: item for item in first["experiments"]}
    median = lambda key: by_name[key]["result"]["timing"]["median_us"]
    files = [
        slide(1, "An agent that remembers", "Long Horizon Engineering • October 11, 2026", [
            "GPU: NVIDIA RTX 5070 Ti, 16 GB, CUDA smoke test passed",
            "Task: exact top-k on 16,384 float32 scores",
            "Oracle: independent CPU sort with exact tie order",
            "Memory: MongoDB Atlas runs + decisions + experiments",
        ], "Original work: experiment controller, memory policy, recovery"),
        slide(2, "Measure real GPU work", "Fixed contract • 30 CUDA event samples", [
            f"stable sort   correct=True   median={median('sort_stable'):.1f} us",
            f"stable argsort correct=True   median={median('argsort_stable'):.1f} us",
            f"packed top-k   correct=True   median={median('packed_topk'):.1f} us",
            "direct top-k   correct=False  exact tie IDs differ",
        ], "Incorrect candidates are never timed or chosen as best"),
        slide(3, "Crash after commit", "Fault injection: exit 86 after the first Atlas write", [
            "RUN_ID=2eb9f21a2e374f358c4a42b9fbe2258d",
            "evaluating sort_stable; retrieved 0 records",
            "committed sort_stable: correct=True",
            "RECOVERY TEST: exiting before progress update",
        ], "Fresh Atlas status still showed the baseline at attempt 1"),
        slide(4, "Resume the same run", "Progress reconstructed from durable experiments", [
            "sort_stable     completed  attempt=1  (not rerun)",
            "argsort_stable  completed  attempt=1",
            "packed_topk     completed  attempt=1",
            "topk_sorted     failed     attempt=1",
        ], "Decisions referenced 0, 1, 2, then 3 prior experiment IDs"),
        slide(5, "Failures change the search", "A second Atlas run reads the first run's evidence", [
            f"Retrieved {len(first['experiments'])} prior experiments before its first decision",
            f"Budget: {second['run']['budget_evals']} evaluations",
            f"Executed: {len(second['experiments'])} correct configurations",
            "Skipped known tie failure and related unstable sort",
        ], "No repeated invalid candidate in the follow-up run"),
        slide(6, "What we built today", "Clear attribution and honest measurements", [
            "Atlas-backed durable controller + stable identities",
            "Correctness gate + CUDA event timing + raw samples",
            "Retrieved successes/failures + crash recovery",
            "PyTorch provides GPU operators; no speedup claim",
        ], "Paired recheck did not establish a reliable speedup"),
    ]
    concat = OUT / "video-frames.txt"
    concat.write_text("".join(f"file '{p.name}'\nduration 10\n" for p in files) +
                      f"file '{files[-1].name}'\n")
    output = OUT / "atlas-gpu-harness-demo-backup.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(concat), "-vf", "fps=24,format=yuv420p",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "59", str(output)],
                   check=True, cwd=OUT)
    print(output)


if __name__ == "__main__":
    main()
