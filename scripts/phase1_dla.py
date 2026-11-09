"""Phase 1: direct logit attribution of every decoder head / site to the Loop logit diff.

Reads the capture file written by `phase1_localize.py capture` (activations at the decision
position) and the fp16 weights straight from safetensors (CPU, no model load). For each state,
the logit-diff direction is r = W_U[S0] - W_U[C0] (W_U = embed_tokens, tied); a component c's
direct effect is (gamma * (c - mean(c))) . r / sigma, with sigma the final LayerNorm's std of
that state's last residual. Adds key `dla` to results/phase1/localize.json.

    .venv/bin/python -I scripts/phase1_dla.py --cap-file <capture.pt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_common as pc  # noqa: E402
from phase1_common import DH, N_HEADS, N_LAYERS  # noqa: E402
from phase1_localize import save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-file", required=True)
    args = ap.parse_args()
    z = torch.load(args.cap_file)
    caps, meta = z["caps"], z["meta"]
    ds = pc.load_dataset()
    pairs = ds["pairs"]
    valid = {(i, str(k)) for i, k in ds["valid"]}
    loop_tok = [(pairs[i]["S0"], pairs[i]["C0"]) for i, p in enumerate(pairs) for k in p["loop"]]
    loop_valid = np.array([(i, k) in valid for i, p in enumerate(pairs) for k in p["loop"]])
    clean_pairs = sorted({i for i, k in valid})
    clean_tok = [(pairs[i]["S0"], pairs[i]["C0"]) for i in clean_pairs]
    assert len(loop_tok) == len(meta["loop_all"]) and len(clean_tok) == len(meta["clean"])

    f = safe_open(str(pc.ROOT / "models/hf/whisper-large-v3/model.safetensors"), "pt")
    WU = f.get_tensor("model.decoder.embed_tokens.weight").float()
    gamma = f.get_tensor("model.decoder.layer_norm.weight").float()

    def dla(cap, toks):
        r = torch.stack([WU[a] - WU[b] for a, b in toks]) * gamma  # (N, D)
        x = cap["resid"][:, N_LAYERS - 1]
        sigma = x.std(1, unbiased=False) + 1e-5
        out = {}
        for site in ("attn_out", "cross_out", "mlp_out"):
            c = cap[site]  # (N, L, D)
            c = c - c.mean(-1, keepdim=True)
            out[site] = (c * r[:, None]).sum(-1) / sigma[:, None]  # (N, L)
        for kind, key, mod in (("self", "z_self", "self_attn"), ("cross", "z_cross", "encoder_attn")):
            per = torch.zeros(len(toks), N_LAYERS, N_HEADS)
            for L in range(N_LAYERS):
                Wo = f.get_tensor(f"model.decoder.layers.{L}.{mod}.out_proj.weight").float()
                zz = cap[key][:, L]  # (N, H, dh)
                for h in range(N_HEADS):
                    c = zz[:, h] @ Wo[:, h * DH:(h + 1) * DH].T  # (N, D)
                    c = c - c.mean(-1, keepdim=True)
                    per[:, L, h] = (c * r).sum(-1) / sigma
            out[kind] = per
        return out

    dl = dla({k: v[torch.tensor(loop_valid)] for k, v in caps["loop_all"].items()},
             [t for t, v in zip(loop_tok, loop_valid) if v])
    dc = dla(caps["clean"], clean_tok)
    heads = []
    for kind in ("self", "cross"):
        for L in range(N_LAYERS):
            for h in range(N_HEADS):
                heads.append(dict(L=L, kind=kind, h=h, loop=float(dl[kind][:, L, h].mean()),
                                  clean=float(dc[kind][:, L, h].mean()),
                                  diff=float(dl[kind][:, L, h].mean() - dc[kind][:, L, h].mean())))
    sites = []
    for site in ("attn_out", "cross_out", "mlp_out"):
        for L in range(N_LAYERS):
            sites.append(dict(L=L, site=site, loop=float(dl[site][:, L].mean()), clean=float(dc[site][:, L].mean()),
                              diff=float(dl[site][:, L].mean() - dc[site][:, L].mean())))
    out = dict(note="direct effect on log p(S0) - log p(C0) at the decision position, final LN frozen; "
                    "valid LOOP states vs CLEAN states (nats)",
               heads_by_diff=sorted(heads, key=lambda r: -r["diff"])[:30],
               sites_by_diff=sorted(sites, key=lambda r: -r["diff"])[:20],
               total_diff=float(sum(r["diff"] for r in sites)))
    save("dla", out)
    for r in out["heads_by_diff"][:15]:
        print(r)
    for r in out["sites_by_diff"][:10]:
        print(r)
    print("total", out["total_diff"])


if __name__ == "__main__":
    main()
