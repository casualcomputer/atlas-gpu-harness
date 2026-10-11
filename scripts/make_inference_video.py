"""Render a 59-second captioned evidence walkthrough, never a claimed live capture.

Requires Pillow and ffmpeg. Read only a saved inference export; no GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1280, 720
INK, MUTED, ACCENT, PANEL = "#ecf1fa", "#a9b6cd", "#57e3c0", "#142138"


def font(size=25, mono=False):
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fit_text(draw, text, x, y, width, height, size=25, color=INK, mono=False):
    """Wrap on measured pixel width and mark clipped excerpts explicitly."""
    face = font(size, mono)
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for character in paragraph:
            if draw.textlength(line + character, font=face) > width:
                split = line.rfind(" ")
                if split > len(line) // 2:
                    lines.append(line[:split])
                    line = line[split + 1:] + character
                else:
                    lines.append(line)
                    line = character
            else:
                line += character
        lines.append(line)
    line_height = size + 10
    maximum = max(1, height // line_height)
    if len(lines) > maximum:
        lines = lines[:maximum]
        while draw.textlength(lines[-1] + " ...", font=face) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += " ..."
    for line in lines:
        draw.text((x, y), line, font=face, fill=color)
        y += line_height
    return y


def frame(number, title, subtitle, test):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#0a1020")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 10), fill=ACCENT)
    badge = "TEST / NOT SUBMISSION" if test else "SAVED EXPERIMENT EVIDENCE"
    draw.text((60, 35), "ATLAS GPU HARNESS", font=font(24), fill=ACCENT)
    draw.text((760, 40), badge, font=font(18), fill="#ffd080" if test else MUTED)
    fit_text(draw, title, 60, 90, 1160, 102, size=38)
    fit_text(draw, subtitle, 60, 190, 1160, 65, size=22, color=MUTED)
    draw.rounded_rectangle((50, 275, 1230, 625), radius=18, fill=PANEL)
    draw.text((60, 654), "EVIDENCE WALKTHROUGH | saved export, not live capture", font=font(19), fill=ACCENT)
    draw.text((1160, 654), f"{number}/6", font=font(20), fill=MUTED)
    return canvas, draw


def readable_number(value, digits=1):
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def placement(value):
    return "engine auto" if value is None else f"{value} CPU layers"


def render(report, directory, test):
    run = report["run"]
    context = run.get("context", {})
    model = context.get("model", {})
    gpu = context.get("hardware", {}).get("gpu", {})
    physical_mib = gpu.get("total_mib", 0)
    budget = context.get("fixed", {}).get("gpu_budget_mib", physical_mib)
    file_gib = model.get("file_bytes", 0) / 1024 ** 3
    gpu_gib = physical_mib / 1024
    model_name = model.get("name", "Model name unavailable")
    quant = model.get("quantization", {})
    quant = quant.get("name", "unknown") if isinstance(quant, dict) else str(quant)
    trials = [e for e in report.get("experiments", []) if e.get("result", {}).get("stage") == "tuning"]
    holdout = [e for e in report.get("experiments", []) if e.get("result", {}).get("stage") == "holdout"]
    proposals = sorted(run.get("proposals", {}).items())
    decision = run.get("recommendation") or {}
    slides = []

    title = "Weights larger than the graphics card" if file_gib > gpu_gib > 0 else "A large MoE on a desktop GPU"
    canvas, draw = frame(1, title, model_name, test)
    fit_text(draw, f"Model file: {file_gib:.2f} GiB  |  Physical GPU: {gpu_gib:.2f} GiB\n"
             f"{gpu.get('name', 'GPU unavailable')}\nQuantization: {quant}\n"
             "Problem: find settings that fit, answer correctly, and respond usefully.",
             85, 301, 1100, 220, size=27)
    note = "The CPU and system RAM can carry experts that do not fit on the GPU."
    if file_gib <= gpu_gib:
        note = "This quantized file fits below VRAM size; it is not larger-than-VRAM proof."
    if budget < physical_mib:
        note += f" This run uses a simulated {budget:,.0f} MiB limit."
    fit_text(draw, note, 85, 530, 1100, 85, size=22, color="#ffd080")
    slides.append(canvas)

    canvas, draw = frame(2, "Write a rule. Test it. Learn from the result.",
                         "Normal engine auto-fitting is the baseline; Atlas preserves the evidence.", test)
    boxes = [(82, "1  Measure", "Engine auto baseline"),
             (480, "2  Generate code", "Local model writes policy"),
             (878, "3  Test", "Score, memory, speed")]
    for x, heading, body in boxes:
        draw.rounded_rectangle((x, 314, x + 320, 437), radius=12, outline=ACCENT, width=2)
        fit_text(draw, heading, x + 18, 333, 285, 38, size=24, color=ACCENT)
        fit_text(draw, body, x + 18, 380, 285, 45, size=21)
    for x in (420, 818):
        draw.line((x, 375, x + 42, 375), fill=INK, width=3)
        draw.polygon([(x + 42, 375), (x + 32, 368), (x + 32, 382)], fill=INK)
    fit_text(draw, "Atlas saves the trial → measurements return to the next code revision\n"
             "Separate questions confirm a promising policy. Keeping the baseline is allowed.\n"
             "Only one knob changes: how many initial layers keep experts in CPU RAM.",
             85, 472, 1100, 140, size=23)
    slides.append(canvas)

    if proposals:
        first_name, first = proposals[0]
        last_name, last = proposals[-1]
        changed = first.get("n_cpu_moe") != last.get("n_cpu_moe")
        revision = (f"Saved placement choices: {placement(first.get('n_cpu_moe'))} → "
                    f"{placement(last.get('n_cpu_moe'))}" if changed else
                    "Same placement choice; code generation alone does not prove improvement.")
    else:
        first_name = last_name = "No accepted proposal in export"
        first = last = {}
        revision = "No accepted generated policy is available in this saved export."
    canvas, draw = frame(3, "Actual generated policy code", revision, test)
    for x, name, proposal in ((85, first_name, first), (665, last_name, last)):
        fit_text(draw, name + " — saved source excerpt", x, 294, 525, 40, size=20, color=ACCENT)
        code = proposal.get("code", "No generated source saved")
        fit_text(draw, code.expandtabs(2), x, 340, 525, 220, size=19, mono=True)
    fit_text(draw, "Generated Python is checked before bounded execution. It selects n_cpu_moe only.",
             85, 575, 1100, 35, size=21, color=MUTED)
    slides.append(canvas)

    canvas, draw = frame(4, "What happened on the real machine", "Tuning results · 6 unique short tasks, repeated in reversed order · not general model quality", test)
    columns = [(85, "Placement", 300), (410, "Answers", 150), (620, "GPU peak MiB", 230), (930, "Decode tok/s", 250)]
    for x, label, width in columns:
        fit_text(draw, label, x, 296, width, 40, size=23, color=ACCENT)
    for index, experiment in enumerate(trials[:5]):
        result = experiment["result"]
        quality = result.get("quality", {})
        gate = "pass" if result.get("correct") else "fail"
        values = [placement(result.get("recipe", {}).get("n_cpu_moe")) + f" [{gate}]",
                  f"{quality.get('passed', 0)}/{quality['total']}" if "total" in quality else "not scored",
                  readable_number(result.get("memory", {}).get("device_peak_mib"), 0),
                  readable_number(result.get("median_decode_tokens_per_second"))]
        for (x, _, width), value in zip(columns, values):
            fit_text(draw, value, x, 347 + index * 43, width, 40, size=24)
    if not trials:
        fit_text(draw, "No completed tuning records in this export", 85, 355, 1100, 50)
    fit_text(draw, f"Budget: {budget:,.0f} MiB. Total device memory sampled every 0.25 s; peaks can be missed.\n"
             "Short-answer decode rates are descriptive. Tuning numbers are not a confirmed speedup.",
             85, 565, 1100, 60, size=18, color=MUTED)
    slides.append(canvas)

    recipe = decision.get("recipe")
    confirmation = decision.get("confirmation") or {}
    accepted = confirmation.get("directionally_supported_speedup") is True
    heading = "Recommendation after separate checks"
    conclusion = "No accepted recipe in this export. Do not claim a validated improvement."
    if recipe:
        conclusion = "Recommended placement: " + placement(recipe.get("n_cpu_moe"))
        conclusion += ". Engine baseline retained." if recipe.get("n_cpu_moe") is None else "."
    canvas, draw = frame(5, heading, conclusion, test)
    lines = []
    for experiment in holdout:
        result = experiment["result"]
        quality = result.get("quality", {})
        lines.append(f"{experiment.get('candidate', 'holdout')}: "
                     f"{quality.get('passed', 0)}/{quality.get('total', '?')} answers; "
                     f"resource pass={bool(result.get('resource_pass'))}")
    if not lines:
        lines.append("No completed separate-question confirmation in this export.")
    for index, block in enumerate(confirmation.get("blocks", []), 1):
        gain = readable_number(block.get("median_paired_decode_change_pct"))
        lines.append(f"Comparison block {index}: {gain}% paired decode change")
    fit_text(draw, "\n".join(lines), 85, 301, 1100, 235, size=24, mono=True)
    supported = "Rule passed: >5% paired gain in both blocks, with answer/resource checks." if accepted else "No confirmed speedup claim. A passing recipe can still establish feasibility."
    fit_text(draw, supported + "\nSmall descriptive check; inspect raw outputs and concurrent-workload limitations.",
             85, 550, 1100, 70, size=20, color=ACCENT)
    slides.append(canvas)

    canvas, draw = frame(6, "Built today: the evidence-driven harness", "Original work and upstream contributions are separate.", test)
    fit_text(draw, "Our work: policy generation + validation + real evaluation + feedback\n"
             "Atlas: saved runs, experiments, decisions, proposals and recommendation\n"
             f"Run ID: {run['_id']}\n"
             "Reused: llama.cpp inference, Ollama, pretrained models and quantization\n"
             "Scope: selected MoE formats and one placement knob; no new CUDA kernels.",
             85, 300, 1100, 260, size=25)
    fit_text(draw, "github.com/casualcomputer/atlas-gpu-harness", 85, 576, 1100, 35, size=23, color=ACCENT)
    slides.append(canvas)
    paths = []
    for index, image in enumerate(slides, 1):
        path = directory / f"scene-{index:02d}.png"
        image.save(path)
        paths.append(path)
    contact = Image.new("RGB", (1280, 1080), "#0a1020")
    for index, image in enumerate(slides):
        contact.paste(image.resize((640, 360)), ((index % 2) * 640, (index // 2) * 360))
    contact.save(directory / "contact-sheet.png")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test", action="store_true", help="Watermark incomplete/historical exports; separate test output")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.run_id):
        parser.error("run-id must contain only letters, numbers, underscores and hyphens")
    locations = [ROOT / "artifacts/inference" / args.run_id / "report.json",
                 ROOT / "evidence/inference" / (args.run_id + ".json")]
    source = next((path for path in locations if path.is_file()), None)
    if source is None:
        parser.error("No saved inference report found for this run ID")
    report = json.loads(source.read_text(encoding="utf-8"))
    run = report["run"]
    if run.get("_id") != args.run_id:
        parser.error("Saved report ID does not match --run-id")
    if not args.test and (run.get("status") != "completed" or "recommendation" not in run):
        parser.error("Normal output requires a completed run with recommendation; use --test for a marked smoke test")
    prefix = "TEST-" if args.test else ""
    directory = ROOT / "artifacts/inference-video" / (prefix + args.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    frames = render(report, directory, args.test)
    concat = directory / "frames.txt"
    concat.write_text("".join(f"file '{path.name}'\nduration {10 if i < 5 else 9}\n"
                              for i, path in enumerate(frames)) + f"file '{frames[-1].name}'\n")
    output = directory / (prefix + "atlas-inference-evidence-walkthrough.mp4")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-vf", "fps=24,format=yuv420p", "-c:v", "libx264",
                    "-preset", "fast", "-threads", "2", "-t", "59", str(output)], check=True)
    receipt = {"run_id": args.run_id, "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "test_output": args.test, "duration_seconds": 59, "video": str(output),
               "description": "Captioned saved-evidence walkthrough, not live capture; no inference performed"}
    (directory / "video-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
