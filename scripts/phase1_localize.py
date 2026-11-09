"""Phase 1, step 2: where in the decoder is the Loop written?

Reads results/phase1/dataset.json (valid LOOP/CLEAN pairs, split find-A / find-B by clip).
Subcommands, each adding its key to results/phase1/localize.json:

    sweep    zero-ablate every decoder self- and cross-attention head (32 x 20 x 2), one at a
             time, on find-A: change in LOOP logit diff, flips, CLEAN cost
    patch    activation patching CLEAN -> LOOP per layer and site, at the decision position and
             over the shared tail (task tokens + prefix); per-head patching for the top heads
    capture  one read-only pass over every state: self-attn induction / copy scores, residual
             directions (mean diff, AUC, Cohen's d, logistic probe on find-B), who writes the
             direction (sites, heads, MLP neurons), cross-attention entropy and audio mass
    verify   interventions on find-B (and the natural cases): greedy Loop rate and CLEAN cost

    .venv/bin/python -I scripts/phase1_localize.py sweep|patch|capture|verify

Only forward passes. No gradient is computed anywhere.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_common as pc  # noqa: E402
from phase1_common import D, DH, N_HEADS, N_LAYERS  # noqa: E402

OUT = pc.P1 / "localize.json"
CAP = Path("/private/tmp") / "phase1_capture.pt"  # default for --cap-file (1.5 GB, keep it out of the repo)
TAIL = 4  # shared tail: <|sot|> <|da|> <|transcribe|> + <|notimestamps|> or the first timestamp
AUDIO_FRAMES = 150  # cross-attn "current audio": first 3 s of the window (encoder frame = 20 ms)


def save(key, val):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[key] = val
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))


def load_out():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


# ------------------------------------------------------------------------------ items


def build_items(d: pc.Dec, ds, valid_only=True):
    """LOOP items (one per valid (pair, k) or every (pair, k)), CLEAN items (one per pair, decision
    state) and CLEAN-cont items (CLEAN + its own continuation, for the clean cost)."""
    pairs = ds["pairs"]
    valid = {(i, str(k)) for i, k in ds["valid"]}
    loop, clean = [], {}
    for i, p in enumerate(pairs):
        for k, st in p["loop"].items():
            if valid_only and (i, k) not in valid:
                continue
            loop.append(dict(pair=i, k=k, split=p["split"], nt=p["nt"], seq=st["prompt"] + p["prefix"],
                             prompt=st["prompt"], prefix=p["prefix"], S=p["S"], S0=p["S0"], C0=p["C0"],
                             win=d.encode(p["path"], p["clip"], p["seek"]), loops=st["loops"], ld=st["ld"],
                             valid=(i, k) in valid, path=p["path"], clip=p["clip"]))
    for it in loop:
        i = it["pair"]
        if i not in clean:
            p = pairs[i]
            clean[i] = dict(pair=i, split=p["split"], nt=p["nt"], seq=p["clean_prompt"] + p["prefix"],
                            prompt=p["clean_prompt"], prefix=p["prefix"], cont=p["clean_cont"], S=p["S"],
                            S0=p["S0"], C0=p["C0"], win=it["win"], path=p["path"], clip=p["clip"])
    nat_loop, nat_clean = [], []
    for j, n in enumerate(ds["natural"]):
        w = d.encode(n["path"], n["clip"], n["seek"])
        st = n["loop"]["nat"]
        nat_loop.append(dict(pair=f"nat{j}", name=n["name"], k="nat", split=n["split"], nt=n["nt"],
                             seq=st["prompt"] + n["prefix"], prompt=st["prompt"], prefix=n["prefix"], S=n["S"],
                             S0=n["S0"], C0=n["C0"], win=w, loops=st["loops"], ld=st["ld"], valid=True,
                             path=n["path"], clip=n["clip"]))
        nat_clean.append(dict(pair=f"nat{j}", name=n["name"], split=n["split"], nt=n["nt"],
                              seq=n["clean_prompt"] + n["prefix"], prompt=n["clean_prompt"], prefix=n["prefix"],
                              cont=n["clean_cont"], S=n["S"], S0=n["S0"], C0=n["C0"], win=w,
                              path=n["path"], clip=n["clip"]))
    return loop, list(clean.values()), nat_loop, nat_clean


class Set:
    """A fixed list of sequences in fixed chunks, with optional cached residuals per layer."""

    def __init__(self, d: pc.Dec, items, cont=False, bs=24):
        self.d, self.items, self.cont, self.bs = d, items, cont, bs
        self.seqs = [it["seq"] + (it["cont"] if cont else []) for it in items]
        self.wins = [it["win"] for it in items]
        self.slices = list(d.chunks(len(items), bs))
        self.resid = None
        self.nout = [len(it["cont"]) + 1 if cont else 1 for it in items]

    def cache_resid(self):
        self.resid = []
        for sl in self.slices:
            store = {}
            hooks = {(L, "resid_pre"): (lambda L: lambda x: store.__setitem__(L, x.clone()))(L)
                     for L in range(N_LAYERS)}
            r = self.d.forward(self.seqs[sl], self.wins[sl], hooks=hooks)
            self.resid.append((store, r["real"]))

    def run(self, hooks_fn=None, start=0):
        """Returns list (per item) of log-softmax rows at its last nout positions (n, V) on CPU."""
        out = []
        for c, sl in enumerate(self.slices):
            hooks = hooks_fn(sl) if hooks_fn else None
            n = max(self.nout[sl])
            if start > 0 and self.resid is not None:
                store, real = self.resid[c]
                r = self.d.forward(None, self.wins[sl], hooks=hooks, start=start, resid=store[start], real=real,
                                   last_n=n)
            else:
                r = self.d.forward(self.seqs[sl], self.wins[sl], hooks=hooks, last_n=n)
            lg = r["logits"]
            if n == 1:
                lg = lg[:, None]
            lp = torch.log_softmax(lg, -1).cpu()
            for b, i in enumerate(range(sl.start, sl.stop)):
                out.append(lp[b, n - self.nout[i]:])
        return out


def loop_metrics(items, lps):
    ld = np.array([float(lp[0, it["S0"]] - lp[0, it["C0"]]) for it, lp in zip(items, lps)])
    am = np.array([int(lp[0].argmax()) == it["S0"] for it, lp in zip(items, lps)])
    return ld, am


def clean_metrics(items, lps):
    ld = np.array([float(lp[0, it["S0"]] - lp[0, it["C0"]]) for it, lp in zip(items, lps)])
    cont = np.array([float(np.mean([lp[j, t] for j, t in enumerate(it["cont"])])) for it, lp in zip(items, lps)])
    return ld, cont


def head_zero(L, kind, heads, scale=0.0):
    hs = list(heads)

    def f(z):
        z = z.clone()
        z[:, :, hs] *= scale
        return z
    return {(L, "z_self" if kind == "self" else "z_cross"): f}


def merge_hooks(hlist):
    """Combine hooks; several on the same (layer, site) are chained."""
    out = {}
    for h in hlist:
        for key, f in h.items():
            out[key] = _chain(out[key], f) if key in out else f
    return out


def _chain(g, f):
    def h(x):
        y = g(x)
        x = x if y is None else y
        y = f(x)
        return x if y is None else y
    return h


def heads_hooks(heads, scale=0.0):
    """heads: list of (L, kind, h)."""
    by = {}
    for L, kind, h in heads:
        by.setdefault((L, kind), []).append(h)
    return merge_hooks([head_zero(L, kind, hs, scale) for (L, kind), hs in by.items()])


def pick_sweep_items(loop, clean, split="A", cap=96, seed=0):
    rng = np.random.default_rng(seed)
    L = [it for it in loop if it["split"] == split and it["valid"]]
    if len(L) > cap:
        L = [L[i] for i in sorted(rng.choice(len(L), cap, replace=False))]
    pairs = {it["pair"] for it in L}
    C = [it for it in clean if it["pair"] in pairs]
    return L, C


# ------------------------------------------------------------------------------ sweep


def cmd_sweep(args):
    d = pc.Dec()
    ds = pc.load_dataset()
    loop, clean, _, _ = build_items(d, ds)
    Li, Ci = pick_sweep_items(loop, clean, "A", cap=args.cap)
    print(f"sweep on find-A: {len(Li)} LOOP, {len(Ci)} CLEAN", flush=True)
    d.cache_kv = len({it['win'] for it in Li}) * 0.25 < 12
    SL, SC = Set(d, Li), Set(d, Ci, cont=True, bs=16)
    t0 = time.time()
    SL.cache_resid()
    SC.cache_resid()
    base_ld, base_am = loop_metrics(Li, SL.run())
    base_cld, base_cont = clean_metrics(Ci, SC.run())
    print(f"baseline LOOP LD {base_ld.mean():.3f} argmax=S {base_am.mean():.2f}; CLEAN LD {base_cld.mean():.3f} "
          f"cont lp {base_cont.mean():.3f}; {time.time() - t0:.0f}s", flush=True)
    rows = load_out().get("sweep", {}).get("rows", []) if args.resume else []
    done = {(r["L"], r["kind"], r["h"]) for r in rows}
    for L in range(N_LAYERS):
        for kind in ("self", "cross"):
            for h in range(N_HEADS):
                if (L, kind, h) in done:
                    continue
                hk = head_zero(L, kind, [h])
                ld, am = loop_metrics(Li, SL.run(lambda sl: hk, start=L))
                cld, cont = clean_metrics(Ci, SC.run(lambda sl: hk, start=L))
                rows.append(dict(L=L, kind=kind, h=h, d_ld=float((ld - base_ld).mean()),
                                 flip=float(((base_ld > 0) & (ld < 0)).mean() / max(1e-9, (base_ld > 0).mean())),
                                 argmax_S=float(am.mean()), d_clean_ld=float((cld - base_cld).mean()),
                                 d_clean_cont=float((cont - base_cont).mean())))
            print(f"L{L} {kind} done {time.time() - t0:.0f}s; best so far "
                  f"{min(rows, key=lambda r: r['d_ld'])}", flush=True)
            save("sweep", dict(n_loop=len(Li), n_clean=len(Ci), base=dict(ld=float(base_ld.mean()),
                               argmax_S=float(base_am.mean()), clean_ld=float(base_cld.mean()),
                               clean_cont=float(base_cont.mean())), rows=rows))
    print("sweep done", time.time() - t0)


# ------------------------------------------------------------------------------ patch


def capture_clean_tail(d, S, sites=("attn_out", "cross_out", "mlp_out", "resid_post", "z_self", "z_cross")):
    """Per chunk: dict[(L, site)] -> activations at the last TAIL positions."""
    caps = []
    for sl in S.slices:
        store = {}
        hooks = {}
        for L in range(N_LAYERS):
            for s in sites:
                hooks[(L, s)] = (lambda key: lambda x: store.__setitem__(key, x[:, -TAIL:].clone()))((L, s))
        d.forward(S.seqs[sl], S.wins[sl], hooks=hooks)
        caps.append(store)
    # flatten to per item
    per = []
    for c, sl in enumerate(S.slices):
        for b in range(sl.stop - sl.start):
            per.append({k: v[b] for k, v in caps[c].items()})
    return per


def cmd_patch(args):
    d = pc.Dec()
    ds = pc.load_dataset()
    loop, clean, _, _ = build_items(d, ds)
    Li, Ci = pick_sweep_items(loop, clean, "A", cap=args.cap)
    SL, SCd = Set(d, Li), Set(d, Ci)
    d.cache_kv = len({it['win'] for it in Li}) * 0.25 < 12
    t0 = time.time()
    cl_act = capture_clean_tail(d, SCd)
    cidx = {it["pair"]: j for j, it in enumerate(Ci)}
    SL.cache_resid()
    base_ld, _ = loop_metrics(Li, SL.run())
    cld = np.array([float(lp[0, it["S0"]] - lp[0, it["C0"]]) for it, lp in zip(Ci, SCd.run())])
    clean_ld = np.array([cld[cidx[it["pair"]]] for it in Li])
    gap = clean_ld - base_ld
    print(f"LOOP LD {base_ld.mean():.3f} CLEAN LD {clean_ld.mean():.3f}", flush=True)

    def patch_hook(L, site, sl, npos, head=None):
        acts = [cl_act[cidx[Li[i]["pair"]]][(L, site)] for i in range(sl.start, sl.stop)]
        A = torch.stack(acts)  # (b, TAIL, ...)

        def f(x):
            x = x.clone()
            if head is None:
                x[:, -npos:] = A[:, -npos:]
            else:
                x[:, -npos:, head] = A[:, -npos:, head]
            return x
        return {(L, site): f}

    def frac(ld):
        return dict(frac_restored=float((ld - base_ld).mean() / gap.mean()),
                    median_item_frac=float(np.median((ld - base_ld) / np.where(np.abs(gap) > 1e-6, gap, 1e-6))),
                    flipped=float((ld < 0).mean()), d_ld=float((ld - base_ld).mean()))

    rows = []
    for L in range(N_LAYERS):
        for site in ("attn_out", "cross_out", "mlp_out", "resid_post"):
            for npos, tag in ((1, "decision"), (TAIL, "tail")):
                ld, _ = loop_metrics(Li, SL.run(lambda sl: patch_hook(L, site, sl, npos), start=L))
                rows.append(dict(L=L, site=site, pos=tag) | frac(ld))
        print(f"patch L{L} {time.time() - t0:.0f}s " + " ".join(
            f"{r['site']}/{r['pos']}={r['frac_restored']:.2f}" for r in rows[-8:]), flush=True)
    out = dict(n_loop=len(Li), loop_ld=float(base_ld.mean()), clean_ld=float(clean_ld.mean()), sites=rows)
    sw = load_out().get("sweep")
    if sw:
        top = sorted(sw["rows"], key=lambda r: r["d_ld"])[:30]
        hrows = []
        for r in top:
            site = "z_self" if r["kind"] == "self" else "z_cross"
            for npos, tag in ((1, "decision"), (TAIL, "tail")):
                ld, _ = loop_metrics(Li, SL.run(lambda sl: patch_hook(r["L"], site, sl, npos, head=r["h"]),
                                                start=r["L"]))
                hrows.append(dict(L=r["L"], kind=r["kind"], h=r["h"], pos=tag) | frac(ld))
        out["heads"] = hrows
    save("patch", out)
    print("patch done", time.time() - t0)


# ------------------------------------------------------------------------------ capture


def copy_positions(seq, S):
    """Start indices of every occurrence of S in seq."""
    n = len(S)
    return [j for j in range(len(seq) - n + 1) if seq[j:j + n] == S]


def run_starts(seq):
    """Indices of the first text token of every text run (token after a timestamp / EOT / PREV)."""
    out = []
    for j in range(1, len(seq)):
        if seq[j] < pc.EOT and seq[j - 1] >= pc.EOT:
            out.append(j)
    return out


def capture_set(d, items, extra=None):
    """Read-only pass at the last position. extra: per-item token list appended (induction step)."""
    seqs = [it["seq"] + (extra[i] if extra else []) for i, it in enumerate(items)]
    wins = [it["win"] for it in items]
    res = {k: [] for k in ("resid", "resid_tail", "attn_out", "cross_out", "mlp_out", "z_self", "z_cross",
                           "mlp_act", "ind_S", "copy_S", "seg_other", "cross_ent", "cross_audio", "self_ent")}
    for sl in d.chunks(len(items), 16):
        st = {}
        hooks = {}
        T = max(len(s) for s in seqs[sl])
        for L in range(N_LAYERS):
            def mk(L):
                def last(key):
                    return lambda x: st.__setitem__((L, key), x[:, -1].float().cpu())
                hooks[(L, "resid_post")] = lambda x: (st.__setitem__((L, "resid"), x[:, -1].float().cpu()),
                                                      st.__setitem__((L, "resid_tail"),
                                                                     x[:, -TAIL:].float().mean(1).cpu()))[0]
                for k in ("attn_out", "cross_out", "mlp_out", "z_self", "z_cross"):
                    hooks[(L, k)] = last(k)
                hooks[(L, "mlp_act")] = lambda x: st.__setitem__((L, "mlp_act"), x[:, -1].half().cpu())
                hooks[(L, "self_probs")] = lambda p: st.__setitem__((L, "self_probs"), p.cpu())
                hooks[(L, "cross_probs")] = lambda p: st.__setitem__((L, "cross_probs"), p.cpu())
            mk(L)
        d.forward(seqs[sl], wins[sl], hooks=hooks)
        for b, i in enumerate(range(sl.start, sl.stop)):
            it = items[i]
            off = T - len(seqs[i])
            S = it["S"]
            cp = [j + off for j in copy_positions(seqs[i][:len(it["prompt"])], S)]
            starts = [j + off for j in run_starts(seqs[i][:len(it["prompt"])])]
            ind_t = [j + (len(extra[i]) if extra else 0) for j in cp]
            allS = sorted({j + t for j in cp for t in range(len(S))})
            other = [j for j in starts if j not in cp]
            sp = torch.stack([st[(L, "self_probs")][b] for L in range(N_LAYERS)])  # (32, H, T)
            cpr = torch.stack([st[(L, "cross_probs")][b] for L in range(N_LAYERS)])  # (32, H, 1500)
            res["ind_S"].append(sp[:, :, ind_t].sum(-1) if ind_t else torch.zeros(N_LAYERS, N_HEADS))
            res["copy_S"].append(sp[:, :, allS].sum(-1) if allS else torch.zeros(N_LAYERS, N_HEADS))
            res["seg_other"].append(sp[:, :, other].sum(-1) if other else torch.zeros(N_LAYERS, N_HEADS))
            res["self_ent"].append(-(sp.clamp_min(1e-12).log() * sp).sum(-1))
            res["cross_ent"].append(-(cpr.clamp_min(1e-12).log() * cpr).sum(-1))
            res["cross_audio"].append(cpr[:, :, :AUDIO_FRAMES].sum(-1))
            for k in ("resid", "resid_tail", "attn_out", "cross_out", "mlp_out", "z_self", "z_cross", "mlp_act"):
                res[k].append(torch.stack([st[(L, k)][b] for L in range(N_LAYERS)]))
    return {k: torch.stack(v) for k, v in res.items()}


def cmd_capture(args):
    d = pc.Dec()
    d.cache_kv = False
    ds = pc.load_dataset()
    loop_all, _, nat_loop, nat_clean = build_items(d, ds, valid_only=False)
    loop, clean, _, _ = build_items(d, ds, valid_only=True)
    t0 = time.time()
    caps = {}
    caps["loop_all"] = capture_set(d, loop_all)
    print("loop_all", time.time() - t0, flush=True)
    caps["clean"] = capture_set(d, clean)
    caps["nat_loop"] = capture_set(d, nat_loop)
    caps["nat_clean"] = capture_set(d, nat_clean)
    # induction at a later step: S[0], S[1] already written; target = S[2] in each copy
    caps["loop_step2"] = capture_set(d, loop, extra=[it["S"][:2] for it in loop])
    print("captured", time.time() - t0, flush=True)
    meta = dict(loop_all=[{k: it[k] for k in ("pair", "k", "split", "valid", "loops", "ld", "nt", "path")}
                          for it in loop_all],
                loop=[{k: it[k] for k in ("pair", "k", "split")} for it in loop],
                clean=[{k: it[k] for k in ("pair", "split", "nt", "path")} for it in clean])
    torch.save(dict(caps=caps, meta=meta), args.cap_file)
    analyze(d, caps, meta)


def analyze(d, caps, meta):
    from sklearn.linear_model import LogisticRegression

    la = meta["loop_all"]
    split_l = np.array([m["split"] for m in la])
    valid_l = np.array([m["valid"] for m in la])
    loops_l = np.array([m["loops"] for m in la])
    split_c = np.array([m["split"] for m in meta["clean"]])
    valid_pairs = {m["pair"] for m in la if m["valid"]}
    out = {}

    # ---- (c) induction / copy scores (valid LOOP vs CLEAN), per head
    L_ = caps["loop_all"]
    C_ = caps["clean"]
    vmask = torch.tensor(valid_l)
    ind = L_["ind_S"][vmask].mean(0)
    copy = L_["copy_S"][vmask].mean(0)
    seg_loop = L_["seg_other"][vmask].mean(0)
    seg_clean = C_["seg_other"].mean(0)
    ind2 = caps["loop_step2"]["ind_S"].mean(0)
    heads = []
    for L in range(N_LAYERS):
        for h in range(N_HEADS):
            heads.append(dict(L=L, h=h, ind_S=float(ind[L, h]), copy_S=float(copy[L, h]),
                              ind_step2=float(ind2[L, h]), seg_other_loop=float(seg_loop[L, h]),
                              seg_clean=float(seg_clean[L, h])))
    out["induction"] = dict(
        note="mean attention of the decision position (last token) to: ind_S = first token of each copy of S; "
             "copy_S = all tokens of the copies; seg_* = first tokens of the other text runs; ind_step2 = "
             "attention after S[0],S[1] were written to S[2] in the copies (classic induction)",
        top_ind=sorted(heads, key=lambda r: -r["ind_S"])[:30],
        top_copy=sorted(heads, key=lambda r: -r["copy_S"])[:30],
        top_step2=sorted(heads, key=lambda r: -r["ind_step2"])[:30])

    # ---- (d) directions per layer
    def dirs(key, Xl, Xc, pos_mask_A, neg_mask_A, pos_mask_B, neg_mask_B):
        rows = []
        for L in range(N_LAYERS):
            P_A, N_A = Xl[pos_mask_A, L], Xc[neg_mask_A, L]
            P_B, N_B = Xl[pos_mask_B, L], Xc[neg_mask_B, L]
            u = P_A.mean(0) - N_A.mean(0)
            un = u / u.norm()
            pb, nb = (P_B @ un).numpy(), (N_B @ un).numpy()
            XA = torch.cat([P_A, N_A]).numpy()
            yA = np.r_[np.ones(len(P_A)), np.zeros(len(N_A))]
            XB = torch.cat([P_B, N_B]).numpy()
            yB = np.r_[np.ones(len(P_B)), np.zeros(len(N_B))]
            mu, sd = XA.mean(0), XA.std(0) + 1e-6
            clf = LogisticRegression(C=0.05, max_iter=3000).fit((XA - mu) / sd, yA)
            acc = float((clf.predict((XB - mu) / sd) == yB).mean())
            # threshold from A (midpoint of class means) applied to B
            thr = float(((P_A @ un).mean() + (N_A @ un).mean()) / 2)
            acc_md = float((np.r_[pb > thr, nb <= thr]).mean())
            rows.append(dict(L=L, auc_B=pc.auc(pb, nb), d_B=pc.cohens_d(pb, nb), probe_acc_B=acc,
                             meandiff_acc_B=acc_md, norm_u=float(u.norm()),
                             resid_norm=float(torch.cat([P_A, N_A]).norm(dim=1).mean())))
        return rows

    Xl, Xc = L_["resid"], C_["resid"]
    pairA = np.array([m["split"] == "A" for m in la])
    # contrast 1: valid LOOP vs CLEAN
    c1 = dirs("resid", Xl, Xc, torch.tensor(valid_l & (split_l == "A")), torch.tensor(split_c == "A"),
              torch.tensor(valid_l & (split_l == "B")), torch.tensor(split_c == "B"))
    c1t = dirs("resid_tail", L_["resid_tail"], C_["resid_tail"], torch.tensor(valid_l & (split_l == "A")),
               torch.tensor(split_c == "A"), torch.tensor(valid_l & (split_l == "B")), torch.tensor(split_c == "B"))
    # contrast 2: LOOP states that write S vs LOOP states that do not (same kind of context, decision differs)
    c2 = dirs("resid", Xl, Xl, torch.tensor(loops_l & (split_l == "A")), torch.tensor(~loops_l & (split_l == "A")),
              torch.tensor(loops_l & (split_l == "B")), torch.tensor(~loops_l & (split_l == "B")))
    out["directions"] = dict(
        n=dict(loop_valid_A=int((valid_l & (split_l == "A")).sum()), loop_valid_B=int((valid_l & (split_l == "B")).sum()),
               clean_A=int((split_c == "A").sum()), clean_B=int((split_c == "B").sum()),
               loops_A=int((loops_l & (split_l == "A")).sum()), noloop_A=int((~loops_l & (split_l == "A")).sum()),
               loops_B=int((loops_l & (split_l == "B")).sum()), noloop_B=int((~loops_l & (split_l == "B")).sum())),
        loop_vs_clean_decision=c1, loop_vs_clean_tail_mean=c1t, loops_vs_noloop_within_loop_prompts=c2)
    # best layer: highest find-B AUC of the loop-writes vs not contrast, tie -> d
    bestL = max(c2, key=lambda r: (round(r["auc_B"], 3), r["d_B"]))["L"]
    bestL1 = max(c1, key=lambda r: (round(r["auc_B"], 3), r["d_B"]))["L"]
    out["directions"]["best_layer_loops_vs_noloop"] = bestL
    out["directions"]["best_layer_loop_vs_clean"] = bestL1

    # unit directions from find-A for later use (verify)
    U = {}
    for L in range(N_LAYERS):
        u1 = Xl[torch.tensor(valid_l & (split_l == "A")), L].mean(0) - Xc[torch.tensor(split_c == "A"), L].mean(0)
        u2 = Xl[torch.tensor(loops_l & (split_l == "A")), L].mean(0) - Xl[torch.tensor(~loops_l & (split_l == "A")), L].mean(0)
        U[L] = (u1 / u1.norm(), u2 / u2.norm())
    torch.save(U, pc.P1 / "directions_findA.pt")

    # ---- who writes the direction (contrast 1, at the best layer of contrast 1; also contrast 2)
    def writers(Lstar, which):
        u = U[Lstar][which]
        if which == 0:
            pm, nm, Pc, Nc = torch.tensor(valid_l), torch.tensor(np.ones(len(split_c), bool)), L_, C_
        else:
            pm, nm, Pc, Nc = torch.tensor(loops_l), torch.tensor(~loops_l), L_, L_
        rows = []
        for L in range(Lstar + 1):
            for s in ("attn_out", "cross_out", "mlp_out"):
                pp = (Pc[s][pm, L] @ u)
                nn_ = (Nc[s][nm, L] @ u)
                rows.append(dict(L=L, site=s, diff=float(pp.mean() - nn_.mean()), loop_mean=float(pp.mean()),
                                 clean_mean=float(nn_.mean())))
        hrows = []
        for L in range(Lstar + 1):
            for kind, z, mod in (("self", "z_self", d.layers[L].self_attn.out_proj),
                                 ("cross", "z_cross", d.layers[L].encoder_attn.out_proj)):
                Wo = mod.weight.float().cpu()  # (D, D): out = Wo @ z
                for h in range(N_HEADS):
                    Wh = Wo[:, h * DH:(h + 1) * DH]  # (D, dh)
                    uw = u @ Wh  # (dh,)
                    pp = Pc[z][pm, L, h] @ uw
                    nn_ = Nc[z][nm, L, h] @ uw
                    hrows.append(dict(L=L, kind=kind, h=h, diff=float(pp.mean() - nn_.mean())))
        # MLP neurons: activation diff x alignment of their fc2 column with u
        nrows = []
        for L in range(Lstar + 1):
            W2 = d.layers[L].fc2.weight.float().cpu()  # (D, F)
            align = u @ W2  # (F,)
            adiff = Pc["mlp_act"][pm, L].float().mean(0) - Nc["mlp_act"][nm, L].float().mean(0)
            score = adiff * align
            top = torch.topk(score.abs(), 10).indices
            for n in top.tolist():
                nrows.append(dict(L=L, n=n, score=float(score[n]), act_diff=float(adiff[n]),
                                  align=float(align[n]), col_norm=float(W2[:, n].norm())))
        nrows.sort(key=lambda r: -abs(r["score"]))
        return dict(layer=Lstar, sites=sorted(rows, key=lambda r: -abs(r["diff"]))[:25],
                    heads=sorted(hrows, key=lambda r: -abs(r["diff"]))[:30], neurons=nrows[:30])

    out["writers_loop_vs_clean"] = writers(bestL1, 0)
    out["writers_loops_vs_noloop"] = writers(bestL, 1)

    # ---- (e) cross-attention: entropy and mass on the window's first 3 s, per layer (head mean)
    cross = []
    for L in range(N_LAYERS):
        el = L_["cross_ent"][vmask, L]
        ec = C_["cross_ent"][:, L]
        al = L_["cross_audio"][vmask, L]
        ac = C_["cross_audio"][:, L]
        cross.append(dict(L=L, ent_loop=float(el.mean()), ent_clean=float(ec.mean()),
                          audio_loop=float(al.mean()), audio_clean=float(ac.mean()),
                          auc_ent=pc.auc(el.mean(1).numpy(), ec.mean(1).numpy()),
                          auc_audio=pc.auc(al.mean(1).numpy(), ac.mean(1).numpy())))
    # per head, biggest drop in audio mass
    hd = []
    for L in range(N_LAYERS):
        for h in range(N_HEADS):
            al, ac = L_["cross_audio"][vmask, L, h], C_["cross_audio"][:, L, h]
            el, ec = L_["cross_ent"][vmask, L, h], C_["cross_ent"][:, L, h]
            hd.append(dict(L=L, h=h, audio_loop=float(al.mean()), audio_clean=float(ac.mean()),
                           ent_loop=float(el.mean()), ent_clean=float(ec.mean()),
                           auc_audio=pc.auc(al.numpy(), ac.numpy())))
    out["cross_attention"] = dict(note=f"decision position; audio = mass on encoder frames [0,{AUDIO_FRAMES}) "
                                       "(first 3 s); entropy in nats over 1500 frames, head mean",
                                  layers=cross,
                                  heads_audio_drop=sorted(hd, key=lambda r: r["audio_loop"] - r["audio_clean"])[:20],
                                  L18H13=next(r for r in hd if r["L"] == 18 and r["h"] == 13))
    # natural cases: projection on best directions
    nat = []
    for j in range(caps["nat_loop"]["resid"].shape[0]):
        r = {}
        for L in (bestL1, bestL):
            for w in (0, 1):
                u = U[L][w]
                r[f"L{L}_u{w + 1}_loop"] = float(caps["nat_loop"]["resid"][j, L] @ u)
                r[f"L{L}_u{w + 1}_clean"] = float(caps["nat_clean"]["resid"][j, L] @ u)
        nat.append(r)
    out["natural_projections"] = nat
    save("capture", out)
    print(json.dumps({"best_c1": [r for r in c1 if r["L"] == bestL1], "best_c2": [r for r in c2 if r["L"] == bestL]},
                     indent=1))


# ------------------------------------------------------------------------------ verify


def evaluate(d, loopB, cleanB, hooks, n_steps, ds):
    """Greedy Loop rate + LD on LOOP items, LD + continuation log-prob + greedy-writes-S on CLEAN."""
    SL = Set(d, loopB)
    ld, am = loop_metrics(loopB, SL.run(lambda sl: hooks))
    gen = d.greedy([it["prompt"] for it in loopB], [it["win"] for it in loopB], n_steps,
                   [it["nt"] for it in loopB], prefixes=[it["prefix"] for it in loopB], hooks=hooks)
    loops = np.array([pc.writes_s(d.W, g, it["S"]) for g, it in zip(gen, loopB)])
    SC = Set(d, cleanB, cont=True, bs=16)
    cld, cont = clean_metrics(cleanB, SC.run(lambda sl: hooks))
    cgen = d.greedy([it["prompt"] for it in cleanB], [it["win"] for it in cleanB], 30,
                    [it["nt"] for it in cleanB], prefixes=[it["prefix"] for it in cleanB], hooks=hooks)
    same = np.array([g[:len(it["cont"])] == it["cont"][:len(g)] for g, it in zip(cgen, cleanB)])
    cl_loops = np.array([pc.writes_s(d.W, g, it["S"]) for g, it in zip(cgen, cleanB)])
    return dict(loop_rate=float(loops.mean()), loop_ld=float(ld.mean()), loop_argmax_S=float(am.mean()),
                clean_ld=float(cld.mean()), clean_cont_lp=float(cont.mean()),
                clean_greedy_unchanged=float(same.mean()), clean_writes_S=float(cl_loops.mean()),
                per_item_loops=loops.tolist(), gens=[d.W.decode(g) for g in gen])


def resid_project_hooks(layers, u, alpha=1.0):
    u = u.to(torch.float16)

    def f(x):
        uu = u.to(x.device)
        return x - alpha * (x @ uu)[..., None] * uu
    return {(L, "resid_post"): f for L in layers}


def writer_project_hooks(layers, u, sites=("attn_out", "cross_out", "mlp_out"), alpha=1.0):
    """Activation-level twin of W <- W - a u u^T W (+ bias) for the given writers."""
    u = u.to(torch.float16)

    def f(x):
        uu = u.to(x.device)
        return x - alpha * (x @ uu)[..., None] * uu
    return {(L, s): f for L in layers for s in sites}


def head_write_project_hooks(heads, u, alpha=1.0):
    """Remove the u-component of selected heads' writes only: z_h W_h^T -> z_h W_h^T (I - a u u^T),
    done as an extra hook on the attention output: x -= a ((z_h W_h^T) . u) u."""
    by = {}
    for L, kind, h in heads:
        by.setdefault((L, kind), []).append(h)
    hooks = {}
    store = {}
    for (L, kind), hs in by.items():
        zkey = "z_self" if kind == "self" else "z_cross"
        okey = "attn_out" if kind == "self" else "cross_out"

        def zf(z, key=(L, kind)):
            store[key] = z
            return None

        def of(x, key=(L, kind), hs=hs, L=L, kind=kind):
            mod = pc_dec.layers[L].self_attn.out_proj if kind == "self" else pc_dec.layers[L].encoder_attn.out_proj
            z = store[key]
            uu = u.to(x.device, x.dtype)
            for h in hs:
                Wh = mod.weight[:, h * DH:(h + 1) * DH]  # (D, dh)
                c = z[..., h, :] @ (Wh.T @ uu)  # (B, T)
                x = x - alpha * c[..., None] * uu
            return x
        hooks[(L, zkey)] = zf
        hooks[(L, okey)] = of
    return hooks


pc_dec = None


def neuron_hooks(neurons, scale=0.0):
    by = {}
    for L, n in neurons:
        by.setdefault(L, []).append(n)

    def mk(ns):
        def f(a):
            a = a.clone()
            a[..., ns] *= scale
            return a
        return f
    return {(L, "mlp_act"): mk(ns) for L, ns in by.items()}


def parse_cond(spec, U):
    """Condition specs (all are activation twins of Weight Edits, applied at every position):
      heads=L.kind.h,...[@scale]          scale head h's o_proj input slice (0 = zero its W_O columns)
      neurons=L.n,...[@scale]             scale MLP neuron n (= its fc2 column)
      wproj=L1-L2:sites:uL:w[@alpha]      W <- (I - a u u^T) W for the writers `sites` (a|c|m = attn
                                          out_proj, cross out_proj, fc2; bias too) of layers L1..L2,
                                          u = find-A direction of layer uL, contrast w (1 or 2, or r =
                                          random unit vector)
      hproj=L.kind.h,...:uL:w[@alpha]     same, only the o_proj column slices of the given heads
      rproj=L:uL:w                        diagnostic: project u out of the residual after layer L
    """
    kind, _, rest = spec.partition("=")
    sc = 0.0
    if "@" in rest:
        rest, a = rest.rsplit("@", 1)
        sc = float(a)

    def heads(txt):
        return [(int(a), b, int(c)) for a, b, c in (x.split(".") for x in txt.split(","))]

    def vec(uL, w):
        if w == "r":
            g = torch.Generator().manual_seed(1000 + int(uL))
            r = torch.randn(D, generator=g)
            return r / r.norm()
        return U[int(uL)][int(w) - 1]
    if kind == "heads":
        return heads_hooks(heads(rest), sc)
    if kind == "neurons":
        return neuron_hooks([tuple(map(int, x.split("."))) for x in rest.split(",")], sc)
    alpha = 1.0 if "@" not in spec else sc
    if kind == "wproj":
        lr, sites, uL, w = rest.split(":")
        a, b = map(int, lr.split("-"))
        names = {"a": "attn_out", "c": "cross_out", "m": "mlp_out"}
        return writer_project_hooks(range(a, b + 1), vec(uL, w), [names[c] for c in sites], alpha)
    if kind == "hproj":
        hs, uL, w = rest.split(":")
        return head_write_project_hooks(heads(hs), vec(uL, w), alpha)
    if kind == "rproj":
        L, uL, w = rest.split(":")
        return resid_project_hooks([int(L)], vec(uL, w), alpha)
    raise ValueError(spec)


def cmd_verify(args):
    global pc_dec
    d = pc.Dec()
    pc_dec = d
    d.cache_kv = False
    ds = pc.load_dataset()
    loop, clean, nat_loop, nat_clean = build_items(d, ds)
    LB = [it for it in loop if it["split"] == "B"]
    CB = [it for it in clean if it["split"] == "B"]
    if args.cap_b and len(LB) > args.cap_b:
        rng = np.random.default_rng(1)
        LB = [LB[i] for i in sorted(rng.choice(len(LB), args.cap_b, replace=False))]
        CB = [it for it in CB if it["pair"] in {x["pair"] for x in LB}]
    n_steps = args.steps
    res = load_out()
    sw = res["sweep"]["rows"]
    U = torch.load(pc.P1 / "directions_findA.pt")
    conds = {"baseline": {}}
    ranked = sorted(sw, key=lambda r: r["d_ld"])
    ranked_self = [r for r in ranked if r["kind"] == "self"]
    ranked_cross = [r for r in ranked if r["kind"] == "cross"]
    # cost-aware rank: LD reduction minus clean cost (both in nats)
    ranked_eff = sorted(sw, key=lambda r: r["d_ld"] - 4 * min(0.0, r["d_clean_cont"]))
    for n in (1, 3, 5, 10, 20):
        conds[f"top{n}_heads"] = heads_hooks([(r["L"], r["kind"], r["h"]) for r in ranked[:n]])
        conds[f"top{n}_self"] = heads_hooks([(r["L"], r["kind"], r["h"]) for r in ranked_self[:n]])
        conds[f"top{n}_cross"] = heads_hooks([(r["L"], r["kind"], r["h"]) for r in ranked_cross[:n]])
        conds[f"top{n}_costaware"] = heads_hooks([(r["L"], r["kind"], r["h"]) for r in ranked_eff[:n]])
        rng = np.random.default_rng(n)
        for s in range(2):
            pick = rng.choice(len(sw), n, replace=False)
            conds[f"rand{n}_heads_s{s}"] = heads_hooks([(sw[i]["L"], sw[i]["kind"], sw[i]["h"]) for i in pick])
    if args.no_auto:
        conds = {"baseline": {}}
    for spec in args.extra or []:
        conds[spec] = parse_cond(spec, U)
    cap = res.get("capture", {}).get("directions", {})
    for key, w in (("best_layer_loop_vs_clean", 0), ("best_layer_loops_vs_noloop", 1)):
        if key in cap:
            L = cap[key]
            u = U[L][w]
            conds[f"resid_proj_L{L}_u{w + 1}"] = resid_project_hooks([L], u)
            conds[f"writers_proj_L0-{L}_u{w + 1}"] = writer_project_hooks(range(L + 1), u)
            g = torch.Generator().manual_seed(L)
            r = torch.randn(D, generator=g)
            r = r / r.norm()
            conds[f"control_resid_proj_L{L}_rand"] = resid_project_hooks([L], r)
            conds[f"control_writers_proj_L0-{L}_rand"] = writer_project_hooks(range(L + 1), r)
    only = set(args.only.split(",")) if args.only else None
    out = res.get("verify", {}) if args.resume else {}
    out.setdefault("conditions", {})
    out["n_loop_B"], out["n_clean_B"] = len(LB), len(CB)
    t0 = time.time()
    for name, hk in conds.items():
        if only and name not in only and name != "baseline":
            continue
        if name in out["conditions"] and args.resume:
            continue
        r = evaluate(d, LB, CB, hk, n_steps, ds)
        rn = evaluate(d, nat_loop, nat_clean, hk, 60, ds)
        r["natural"] = dict(loop_rate=rn["loop_rate"], loop_ld=rn["loop_ld"], per_case=dict(
            zip([it["name"] for it in nat_loop], rn["per_item_loops"])), clean_cont_lp=rn["clean_cont_lp"],
            gens=rn["gens"])
        out["conditions"][name] = r
        print(f"{name}: loop {r['loop_rate']:.2f} LD {r['loop_ld']:.2f} | clean lp {r['clean_cont_lp']:.3f} "
              f"unchanged {r['clean_greedy_unchanged']:.2f} | natural loops {rn['loop_rate']:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
        save("verify", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "patch", "capture", "verify", "analyze"])
    ap.add_argument("--cap", type=int, default=96)
    ap.add_argument("--cap-file", default=str(CAP))
    ap.add_argument("--cap-b", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--extra", nargs="*")
    ap.add_argument("--no-auto", action="store_true")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    if args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "analyze":
        z = torch.load(args.cap_file)
        analyze(pc.Dec(), z["caps"], z["meta"])
    else:
        {"sweep": cmd_sweep, "patch": cmd_patch, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
