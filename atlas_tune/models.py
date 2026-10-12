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
    revision: str | None = None
    kv_bytes_per_token: int | None = None  # FP16/BF16 K + V across all layers

    def to_dict(self) -> dict:
        return asdict(self)


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _weights_gib(info, suffix: str) -> float:
    return round(sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(suffix)) / 2**30, 2)


def _config(repo: str, revision: str | None = None) -> dict:
    import json

    from huggingface_hub import hf_hub_download

    try:
        return json.loads(open(hf_hub_download(repo, "config.json", revision=revision)).read())
    except Exception:  # noqa: BLE001 - gated or missing config is not fatal
        return {}


def _kv_bytes_per_token(cfg: dict) -> int | None:
    """Full-context FP16/BF16 cache estimate; weight quantization does not shrink KV.

    Explicit head_dim matters (Qwen3-4B's is not hidden_size / attention_heads).
    Unknown architectures use a conservative single-slot default in the planner.
    """
    text = cfg.get("text_config") or cfg
    layers = text.get("num_hidden_layers")
    heads = text.get("num_attention_heads")
    kv_heads = text.get("num_key_value_heads", heads)
    dim = text.get("head_dim")
    if dim is None:
        hidden = text.get("hidden_size")
        if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads == 0:
            dim = hidden // heads
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (layers, kv_heads, dim)):
        return None
    return layers * kv_heads * dim * 2 * 2


def parallel_slots(art: Artifact, vram_gib: float, context: int, requested: int,
                   used_gib: float = 0, fraction: float = 0.9) -> int:
    """Slots whose weights + full KV fit the budget, before runtime verification.

    Leave room for existing device use as well as the engine's workspace. Unknown
    KV layouts and MoE offload plans start with one slot; real trials decide fit.
    """
    if not art.kv_bytes_per_token or context <= 0:
        return 1
    room = vram_gib * min(fraction, 0.9) - used_gib - art.size_gib
    per_slot = context * art.kv_bytes_per_token / 2**30
    return max(1, min(requested, int(room // per_slot)))


def _hf_artifact(info, cfg: dict) -> Artifact | None:
    size = _weights_gib(info, ".safetensors")
    if not size:
        return None
    match = HF_QUANT.search(info.id)
    text = (cfg.get("text_config") or cfg)
    moe = bool(text.get("num_experts") or text.get("num_local_experts") or text.get("n_routed_experts"))
    return Artifact(id=f"hf:{info.id}", engine_format="hf", repo=info.id, size_gib=size,
                    quant=match.group(1).lower() if match else str(text.get("torch_dtype", "bf16")),
                    moe=moe, layers=text.get("num_hidden_layers"),
                    revision=getattr(info, "sha", None), kv_bytes_per_token=_kv_bytes_per_token(cfg))


def _gguf_repos(api, base: str, explicit: str | None, model_org: str = "") -> list[str]:
    """GGUF repos named exactly <model>-GGUF: the model author's own first, then known quantizers."""
    if explicit:
        return [explicit]
    want = base.lower() + "-gguf"
    found = [m.id for m in api.list_models(search=base, filter="gguf", sort="downloads", limit=40)
             if m.id.split("/")[1].lower() == want]
    rank = {org.lower(): i for i, org in enumerate(GGUF_ORGS)}
    org = lambda rid: rid.split("/")[0].lower()  # noqa: E731
    return sorted(found, key=lambda rid: (org(rid) != model_org.lower(), rank.get(org(rid), 99)))[:2]


def resolve(model: str, gguf_repo: str | None = None, extra_hf: tuple[str, ...] = ()) -> list[Artifact]:
    """All loadable artifacts: the model itself, any extra quantized HF repos, and single-file GGUFs."""
    api = _api()
    info = api.model_info(model, files_metadata=True)
    cfg = _config(model, getattr(info, "sha", None))
    out: list[Artifact] = []
    for repo in (model, *extra_hf):
        repo_info = info if repo == model else api.model_info(repo, files_metadata=True)
        artifact = _hf_artifact(repo_info, cfg if repo == model else _config(repo, getattr(repo_info, "sha", None)))
        if artifact:
            out.append(artifact)
    base = model.split("/")[-1]
    moe = any(a.moe for a in out)
    layers = next((a.layers for a in out if a.layers), None)
    for repo in _gguf_repos(api, base, gguf_repo, model.split("/")[0]):
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
                                moe=moe, layers=layers, revision=getattr(ginfo, "sha", None), kv_bytes_per_token=_kv_bytes_per_token(cfg)))
    return out


# Common, well-tested GGUF quants from highest to lowest precision.
GGUF_PREFERENCE = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_XXS", "Q2_K")


def shortlist(artifacts: list[Artifact], vram_gib: float, ram_gib: float,
              context: int = 8192, used_gib: float = 0, allow_cpu_offload: bool = False) -> list[Artifact]:
    """Keep what can plausibly run on this machine, highest precision first.

    HF: up to two repos whose weights and one full-context KV cache fit in VRAM.
    GGUF (one repo): the best two common
    quants that fit in VRAM, plus for MoE models one quant (Q4_K_M preferred) that only
    fits with experts offloaded to RAM, since that is often faster than a tiny quant.
    """
    gpu_room, ram_room = vram_gib * 0.85, ram_gib * 0.7

    def fits_one(a):
        kv_gib = (a.kv_bytes_per_token or 0) * context / 2**30
        return 0 < a.size_gib <= gpu_room and a.size_gib + kv_gib <= vram_gib * 0.9 - used_gib

    hf = sorted((a for a in artifacts if a.engine_format == "hf" and fits_one(a)),
                key=lambda a: -a.size_gib)[:2]
    gguf_all = [a for a in artifacts if a.engine_format == "gguf" and a.size_gib > 0]
    repo = gguf_all[0].repo if gguf_all else None
    by_quant = {a.quant.upper(): a for a in gguf_all if a.repo == repo}
    preferred = [by_quant[q] for q in GGUF_PREFERENCE if q in by_quant]
    fits = [a for a in preferred if fits_one(a)][:2]
    offload = [a for a in preferred if gpu_room < a.size_gib <= gpu_room + ram_room and a.moe]
    offload.sort(key=lambda a: (a.quant.upper() != "Q4_K_M", -a.size_gib))
    if allow_cpu_offload:
        # Require room for a full host-side weight copy, not just the GPU spill.
        def can_offload(a):
            kv = (a.kv_bytes_per_token or 0) * context / 2**30
            return (not fits_one(a) and a.layers and a.kv_bytes_per_token
                    and 0 < a.size_gib <= ram_room and vram_gib * 0.8 - used_gib - kv > 1)
        hf_extra = sorted((a for a in artifacts if a.engine_format == "hf" and can_offload(a)),
                          key=lambda a: a.size_gib)[:1]
        dense = [a for a in preferred if not a.moe and can_offload(a)]
        # One high-precision offload option makes the speed/precision tradeoff measurable.
        return hf + hf_extra + fits + offload[:1] + dense[:1]
    return hf + fits + offload[:1]
