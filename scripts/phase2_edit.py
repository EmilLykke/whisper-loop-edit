"""Phase 2: build one Weight Edit (or Control Edit) of whisper-large-v3 as an HF checkpoint.

    .venv/bin/python -I scripts/phase2_edit.py build <id>          -> models/hf/edits/<id>/
    .venv/bin/python -I scripts/phase2_edit.py verify <id> <ggml>  ggml tensors == fp16 HF tensors
    .venv/bin/python -I scripts/phase2_edit.py record <id> <ggml>  add the ggml's sha256 to edit.json
    .venv/bin/python -I scripts/phase2_edit.py list

Every edit is an explicit formula over the original weights (ADR-0001): computed in fp32 from the
fp16 originals, rounded once to fp16, tensor names and shapes unchanged. No gradient anywhere.
The edited dir gets model.safetensors, copies of the tokenizer/preprocessor/config files, and
edit.json (formula, parameters, every touched tensor with its relative Frobenius change, sha256s).
The build is deterministic: the same id always writes the same bytes (sha256 in edit.json).

Heads are (layer, head) in decoder self-attention; zeroing head h of layer L is
`model.decoder.layers.L.self_attn.out_proj.weight[:, 64h:64h+64] <- 0` (bias unchanged), i.e. ggml
`decoder.blocks.L.attn.out.weight`, same slice. The direction edits are
`W <- W - a u (u^T W)`, `b <- b - a u (u^T b)` on `self_attn.out_proj` and `fc2` of the layers given.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models/hf/whisper-large-v3"
EDITS = ROOT / "models/hf/edits"
DIRS = ROOT / "results/phase1/directions_findA.pt"
LOCALIZE = ROOT / "results/phase1/localize.json"
DH, N_HEADS, D = 64, 20, 1280
COPY = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "added_tokens.json", "merges.txt", "normalizer.json", "preprocessor_config.json",
        "special_tokens_map.json"]

HEADS_2 = [(12, 13), (25, 18)]
LATE_5 = [(21, 14), (24, 0), (25, 18), (25, 19), (26, 6)]
ALL_11 = [(12, 13), (13, 7), (13, 18), (15, 17), (2, 19), (18, 19), (25, 18), (25, 19), (24, 0), (26, 6), (21, 14)]
DIR_LAYERS = list(range(13, 25))  # 13..24 inclusive
# Calm-Whisper (Wang et al., Interspeech 2025, arXiv 2505.12969): decoder self-attention heads #1, #6, #11
# (0-based, "#0-#19"), the same index in every decoder layer. Masking all three: UrbanSound8K
# non-speech hallucination 99.97% -> 24.10%, LibriSpeech WER clean 2.12 -> 3.57, other 4.07 -> 5.98.
CALM = [(L, h) for L in range(32) for h in (1, 6, 11)]
C1_SEED = 0
C3_SEED = 2
C4_SEED = 4  # final run (ADR-0002): a second Control Edit for E3, fixed before any Held-out decode
C2_SEED = 1024  # = Phase 1's random u for layer 24 (`wproj=...:24:r`), here orthogonalised against u


def u_loop() -> torch.Tensor:
    """Phase 1's layer-24 contrast-2 direction (LOOP states that loop vs do not), find-A, fp32 unit."""
    u = torch.load(DIRS)[24][1].to(torch.float32)
    return u / u.norm()


def u_control() -> torch.Tensor:
    u = u_loop()
    r = torch.randn(D, generator=torch.Generator().manual_seed(C2_SEED), dtype=torch.float32)
    r = r - (r @ u) * u
    return r / r.norm()


def top30_self() -> set[tuple[int, int]]:
    rows = json.loads(LOCALIZE.read_text())["sweep"]["rows"]
    s = sorted([r for r in rows if r["kind"] == "self"], key=lambda r: r["d_ld"])[:30]
    return {(r["L"], r["h"]) for r in s}


def c1_heads(seed: int = C1_SEED) -> list[tuple[int, int]]:
    """One random self-attention head per E3 slot, in that slot's layer, never one of E3's 11, never
    a Phase 1 top-30 self head (sweep ranked by LOOP logit-diff drop), never picked twice.
    C4 is the same draw with seed C4_SEED."""
    rng = np.random.default_rng(seed)
    banned = set(ALL_11) | top30_self()
    out = []
    for L, _ in ALL_11:
        pool = [h for h in range(N_HEADS) if (L, h) not in banned and (L, h) not in out]
        out.append((L, int(rng.choice(pool))))
    return out


def c3_heads() -> list[tuple[int, int]]:
    """Control for E9: in each decoder layer, as many random self-attention heads as E9 zeroes there,
    never one of E9's, never a Phase 1 top-30 self head, never picked twice."""
    rng = np.random.default_rng(C3_SEED)
    e9 = list(dict.fromkeys(ALL_11 + CALM))
    banned = set(e9) | top30_self()
    out = []
    for L in range(32):
        for _ in range(sum(1 for l, _ in e9 if l == L)):
            pool = [h for h in range(N_HEADS) if (L, h) not in banned and (L, h) not in out]
            out.append((L, int(rng.choice(pool))))
    return out


def edit_spec(eid: str) -> dict:
    """heads: list of (L, h) to zero; proj: (u, alpha, layers) or None."""
    if eid == "E1":
        return dict(heads=HEADS_2, proj=None, desc="zero self-attn heads L12H13, L25H18")
    if eid == "E2":
        return dict(heads=LATE_5, proj=None, desc="zero the late-5 copy-writer heads")
    if eid == "E3":
        return dict(heads=ALL_11, proj=None, desc="zero all 11 Phase 1 induction/copy heads")
    if eid == "E4":
        return dict(heads=[], proj=("U[24][1]", u_loop(), 1.5), desc="layer-24 loop direction out of L13-24 writers, a=1.5")
    if eid == "E5":
        return dict(heads=[], proj=("U[24][1]", u_loop(), 2.0), desc="layer-24 loop direction out of L13-24 writers, a=2")
    if eid == "E6":
        return dict(heads=LATE_5, proj=("U[24][1]", u_loop(), 1.0), desc="E2 + layer-24 direction edit a=1")
    if eid == "E7":
        return dict(heads=ALL_11, proj=("U[24][1]", u_loop(), 1.0), desc="E3 + layer-24 direction edit a=1")
    if eid == "E8":
        return dict(heads=ALL_11, proj=("U[24][1]", u_loop(), 0.5), desc="E3 + layer-24 direction edit a=0.5")
    if eid == "E9":
        return dict(heads=list(dict.fromkeys(ALL_11 + CALM)), proj=None,
                    desc="E3 + zero Calm-Whisper heads #1, #6, #11 in all 32 decoder layers")
    if eid == "C3":
        return dict(heads=c3_heads(), proj=None,
                    desc=f"Control for E9: random self-attn heads, same count per layer as E9 (seed {C3_SEED})")
    if eid == "C1":
        return dict(heads=c1_heads(), proj=None,
                    desc=f"Control for E3: 11 random self-attn heads, one per E3 slot's layer (seed {C1_SEED})")
    if eid == "C4":
        return dict(heads=c1_heads(C4_SEED), proj=None,
                    desc=f"Second Control for E3 (final run): 11 random self-attn heads, one per E3 slot's layer (seed {C4_SEED})")
    if eid == "C2":
        return dict(heads=[], proj=(f"random unit, seed {C2_SEED}, orthogonal to U[24][1]", u_control(), 2.0),
                    desc="Control for E5: random direction, same sites, a=2")
    raise SystemExit(f"unknown edit id {eid}")


IDS = ["E1", "E2", "E3", "E4", "E5", "E6", "C1", "C2", "E7", "E8", "E9", "C3", "C4"]


CONV_MAP = {  # convert-h5-to-ggml.py's conv_map (CrispASR v0.8.29), copied
    "self_attn.k_proj": "attn.key", "self_attn.q_proj": "attn.query", "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "attn_ln", "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value", "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln", "fc1": "mlp.0", "fc2": "mlp.2", "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias", "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding", "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight", "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
}


def ggml_name(hf: str) -> str:
    """The converter's HF -> ggml tensor name rule."""
    nn = hf.split(".")[1:]
    if nn[1] == "layers":
        nn[1] = "blocks"
        mid = ".".join(nn[3:-1])
        if mid == "encoder_attn.k_proj":
            mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
        else:
            mapped = CONV_MAP[mid]
        return ".".join(nn[:3] + [mapped] + nn[-1:])
    n = ".".join(nn)
    return CONV_MAP.get(n, n)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def apply(eid: str, sd: dict[str, torch.Tensor]) -> list[dict]:
    """Edit `sd` in place. Returns one record per touched tensor."""
    spec = edit_spec(eid)
    new32: dict[str, torch.Tensor] = {}

    def get(name):
        if name not in new32:
            new32[name] = sd[name].to(torch.float32).clone()
        return new32[name]

    if spec["proj"] is not None:
        _, u, a = spec["proj"]
        for L in DIR_LAYERS:
            for mod in ("self_attn.out_proj", "fc2"):
                for p in ("weight", "bias"):
                    name = f"model.decoder.layers.{L}.{mod}.{p}"
                    W = get(name)
                    if W.dim() == 2:
                        W -= a * torch.outer(u, u @ W)
                    else:
                        W -= a * u * (u @ W)
    for L, h in spec["heads"]:
        get(f"model.decoder.layers.{L}.self_attn.out_proj.weight")[:, DH * h:DH * (h + 1)] = 0.0

    rows = []
    for name in sorted(new32, key=lambda n: (int(n.split(".")[3]), n)):
        W0 = sd[name].to(torch.float32)
        exact = new32[name]
        r16 = exact.to(torch.float16)
        dW = r16.to(torch.float32) - W0
        rows.append(dict(
            name=name, ggml=ggml_name(name), shape=list(W0.shape),
            rel_frobenius_change=float(dW.norm() / W0.norm()),
            max_abs_change=float(dW.abs().max()),
            fp16_rounding_max_abs=float((r16.to(torch.float32) - exact).abs().max()),
            unchanged_after_rounding=bool(torch.equal(r16, sd[name])),
        ))
        sd[name] = r16
    return rows


def build(eid: str):
    spec = edit_spec(eid)
    out = EDITS / eid
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = BASE / "model.safetensors"
    sd = load_file(src)
    n0 = {k: (tuple(v.shape), v.dtype) for k, v in sd.items()}
    rows = apply(eid, sd)
    assert {k: (tuple(v.shape), v.dtype) for k, v in sd.items()} == n0, "names/shapes/dtypes changed"
    save_file(sd, out / "model.safetensors", metadata={"format": "pt"})
    for f in COPY:
        shutil.copy2(BASE / f, out / f)
    proj = spec["proj"]
    formula = []
    if proj is not None:
        formula.append(f"for L in 13..24, M in {{self_attn.out_proj, fc2}}: M.weight <- M.weight - a u (u^T M.weight); "
                       f"M.bias <- M.bias - a u (u^T M.bias); a = {proj[2]}, u = {proj[0]}")
    if spec["heads"]:
        formula.append("for (L, h) in heads: model.decoder.layers.L.self_attn.out_proj.weight[:, 64h:64h+64] <- 0")
    rec = dict(
        id=eid, description=spec["desc"], formula=formula,
        heads=[f"L{L}H{h}" for L, h in spec["heads"]],
        direction=None if proj is None else dict(source=proj[0], alpha=proj[2], layers=DIR_LAYERS,
                                                 cos_with_U24_1=float(proj[1] @ u_loop()),
                                                 u=proj[1].tolist()),
        seeds=dict(C1=C1_SEED, C2=C2_SEED, C3=C3_SEED, C4=C4_SEED) if eid.startswith("C") else None,
        compute="fp32 from the fp16 originals, rounded once to fp16 (round-to-nearest-even)",
        base=dict(path=str(src.relative_to(ROOT)),
                  sha256="a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"),
        tensors_touched=rows,
        n_tensors_touched=len(rows),
        safetensors_sha256=sha256(out / "model.safetensors"),
        script=dict(path="scripts/phase2_edit.py", sha256=sha256(Path(__file__))),
        built=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    old = json.loads((out / "edit.json").read_text()) if (out / "edit.json").exists() else {}
    if "ggml" in old and old.get("safetensors_sha256") == rec["safetensors_sha256"]:
        rec["ggml"] = old["ggml"]
    (out / "edit.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{eid}: {len(rows)} tensors, sha256 {rec['safetensors_sha256'][:16]}, {time.time() - t0:.0f} s")
    for r in rows:
        print(f"  {r['name']}: rel {r['rel_frobenius_change']:.4f}")


def verify(eid: str, ggml: Path):
    """Every ggml tensor equals the edited HF fp16 tensor (1-D/positional ones stored as f32 of the fp16)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_ggml import parse
    from safetensors import safe_open
    _, _, _, _, tensors, _ = parse(ggml)
    rec = json.loads((EDITS / eid / "edit.json").read_text())
    touched = {r["ggml"] for r in rec["tensors_touched"]}
    f = safe_open(EDITS / eid / "model.safetensors", "pt")
    hf = {ggml_name(k): k for k in f.keys() if k != "proj_out.weight"}
    bad, checked, touched_ok = [], 0, 0
    names = set(tensors)
    for gname, (ftype, shape, data) in tensors.items():
        hname = hf.get(gname)
        if hname is None:
            continue
        ref = f.get_tensor(hname).numpy().astype(np.float16).reshape(-1)  # converter: squeeze, fp16
        got = np.asarray(data).astype(np.float16).reshape(-1)  # f32 tensors hold fp16 values
        same = ref.shape == got.shape and np.array_equal(got.view(np.uint16), ref.view(np.uint16))
        checked += 1
        if gname in touched:
            touched_ok += same
        if not same:
            bad.append(gname)
    missing = touched - names
    unmatched = sorted(set(hf) - names)
    print(f"{eid}: checked {checked} ggml tensors against HF fp16 by name; touched {len(touched)}, "
          f"equal {touched_ok}; mismatches {len(bad)} {bad[:5]}; touched not found {sorted(missing)}")
    return dict(checked=checked, touched=len(touched), touched_equal=touched_ok, mismatches=bad,
                touched_missing=sorted(missing), ggml_tensors=len(tensors), hf_unmatched=unmatched)


def record(eid: str, ggml: Path, verified: dict | None = None):
    p = EDITS / eid / "edit.json"
    rec = json.loads(p.read_text())
    g = rec.get("ggml", {})
    g.update(path=str(ggml.resolve().relative_to(ROOT)), sha256=sha256(ggml), bytes=ggml.stat().st_size)
    if verified is not None:
        g["verified_against_hf_fp16"] = verified
    rec["ggml"] = g
    p.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{eid}: ggml sha256 {g['sha256']}")


def main():
    cmd = sys.argv[1]
    torch.set_grad_enabled(False)
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "verify":
        v = verify(sys.argv[2], Path(sys.argv[3]))
        record(sys.argv[2], Path(sys.argv[3]), v)
    elif cmd == "record":
        record(sys.argv[2], Path(sys.argv[3]))
    elif cmd == "list":
        for e in IDS:
            s = edit_spec(e)
            print(e, s["desc"], [f"L{L}H{h}" for L, h in s["heads"]])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
