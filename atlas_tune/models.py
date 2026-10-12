"""Turn a Hugging Face model ID into concrete artifacts each engine can load, with sizes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

GGUF_ORGS = ("ggml-org", "unsloth", "bartowski", "lmstudio-community", "Qwen", "MaziyarPanahi")
HF_QUANT = re.compile(r"(awq|gptq|fp8|int4|int8|w4a16|w8a8|bnb|nf4)", re.IGNORECASE)
GGUF_QUANT = re.compile(r"((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_K(?:_[SMLX]+)?|Q\d_\d|BF16|F16))", re.IGNORECASE)


@dataclass
class Artifact:
    id: str             # stable short name shown to the user and the agent
    engine_format: str  # "hf" (vLLM/SGLang) or "gguf" (llama.cpp)
    repo: str
    size_gib: float
    quant: str = "bf16"
    file: str | None = None
    moe: bool = False
    layers: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _weights_gib(info, suffix: str) -> float:
    return round(sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(suffix)) / 2**30, 2)


def _config(repo: str) -> dict:
    import json

    from huggingface_hub import hf_hub_download

    try:
        return json.loads(open(hf_hub_download(repo, "config.json")).read())
    except Exception:  # noqa: BLE001 - gated or missing config is not fatal
        return {}


def _hf_artifact(info, cfg: dict) -> Artifact | None:
    size = _weights_gib(info, ".safetensors")
    if not size:
        return None
    match = HF_QUANT.search(info.id)
    text = (cfg.get("text_config") or cfg)
    moe = bool(text.get("num_experts") or text.get("num_local_experts") or text.get("n_routed_experts"))
    return Artifact(id=f"hf:{info.id}", engine_format="hf", repo=info.id, size_gib=size,
                    quant=match.group(1).lower() if match else str(text.get("torch_dtype", "bf16")),
                    moe=moe, layers=text.get("num_hidden_layers"))


def _gguf_repos(api, base: str, explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    want = base.lower() + "-gguf"
    found = [m.id for m in api.list_models(search=base, filter="gguf", sort="downloads", limit=40)
             if m.id.split("/")[1].lower() == want]
    rank = {org.lower(): i for i, org in enumerate(GGUF_ORGS)}
    return sorted(found, key=lambda rid: rank.get(rid.split("/")[0].lower(), 99))[:2]


def resolve(model: str, gguf_repo: str | None = None, extra_hf: tuple[str, ...] = ()) -> list[Artifact]:
    """All loadable artifacts: the model itself, any extra quantized HF repos, and single-file GGUFs."""
    api = _api()
    info = api.model_info(model, files_metadata=True)
    cfg = _config(model)
    out: list[Artifact] = []
    for repo in (model, *extra_hf):
        repo_info = info if repo == model else api.model_info(repo, files_metadata=True)
        artifact = _hf_artifact(repo_info, cfg if repo == model else _config(repo))
        if artifact:
            out.append(artifact)
    base = model.split("/")[-1]
    moe = any(a.moe for a in out)
    layers = next((a.layers for a in out if a.layers), None)
    for repo in _gguf_repos(api, base, gguf_repo):
        ginfo = api.model_info(repo, files_metadata=True)
        for sibling in ginfo.siblings:
            name = sibling.rfilename
            if not name.endswith(".gguf") or "/" in name or "-of-" in name or "mmproj" in name.lower():
                continue  # single-file quants only; split and vision projector files are skipped
            quant = GGUF_QUANT.search(name)
            if not quant:
                continue
            out.append(Artifact(id=f"gguf:{repo}:{quant.group(1)}", engine_format="gguf", repo=repo,
                                file=name, quant=quant.group(1), size_gib=round((sibling.size or 0) / 2**30, 2),
                                moe=moe, layers=layers))
    return out


# Common, well-tested GGUF quants from highest to lowest precision.
GGUF_PREFERENCE = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_XXS", "Q2_K")


def shortlist(artifacts: list[Artifact], vram_gib: float, ram_gib: float) -> list[Artifact]:
    """Keep what can plausibly run on this machine, highest precision first.

    HF: up to two repos whose weights fit in VRAM. GGUF (one repo): the best two common
    quants that fit in VRAM, plus for MoE models one quant (Q4_K_M preferred) that only
    fits with experts offloaded to RAM, since that is often faster than a tiny quant.
    """
    gpu_room, ram_room = vram_gib * 0.85, ram_gib * 0.7
    hf = sorted((a for a in artifacts if a.engine_format == "hf" and 0 < a.size_gib <= gpu_room),
                key=lambda a: -a.size_gib)[:2]
    gguf_all = [a for a in artifacts if a.engine_format == "gguf" and a.size_gib > 0]
    repo = gguf_all[0].repo if gguf_all else None
    by_quant = {a.quant.upper(): a for a in gguf_all if a.repo == repo}
    preferred = [by_quant[q] for q in GGUF_PREFERENCE if q in by_quant]
    fits = [a for a in preferred if a.size_gib <= gpu_room][:2]
    offload = [a for a in preferred if gpu_room < a.size_gib <= gpu_room + ram_room and a.moe]
    offload.sort(key=lambda a: (a.quant.upper() != "Q4_K_M", -a.size_gib))
    return hf + fits + offload[:1]
