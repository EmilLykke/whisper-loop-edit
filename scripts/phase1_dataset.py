"""Phase 1, step 1: contrastive LOOP / CLEAN decoder states on the Find Set.

For every rebuilt crispasr window that has previous text (results/phase1/windows), and for
up to 3 sentences S taken from that previous text:

  CLEAN  = the window's previous text with S cut out (prompt + task tokens [+ the first
           timestamp the model writes from CLEAN, on timestamp paths])
  LOOP_k = CLEAN's previous text + a fake previous window that wrote S k times (k = 1, 2, 3),
           formatted the way the path writes it (`<|t0|> S<|t1|><|t1|> S<|t2|>...` or S S S EOT)

Same audio, same step. Measured per state: raw log-probs at the decision position, logit diff
LD = log p(S[0]) - log p(C[0]) (C = CLEAN's own greedy continuation), and a short
whisper.cpp-filtered greedy decode (Loop = it writes S's first 4 words). Plus the 5 natural
Phase 0 cases. Output: results/phase1/dataset.json.

    .venv/bin/python -I scripts/phase1_dataset.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_window as hw  # noqa: E402
import phase1_common as pc  # noqa: E402
from hf_window import BEG, EOT  # noqa: E402

KS = (1, 2, 3)
MAX_S = 3
CLEAN_STEPS = 60


def sentence_spans(W, past):
    """Token spans [a, b) of sentences inside the text runs of `past`."""
    out = []
    for a, b in hw._segments_of(past):
        s = a
        for j in range(a, b):
            t = W.decode([past[j]], special=False).rstrip()
            if t.endswith((".", "?", "!")) or j == b - 1:
                out.append((s, j + 1, a, b))
                s = j + 1
    return out


def has_repeat(W, ids, n=5):
    w = pc.words(W.decode(ids, special=False))
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(grams) != len(set(grams))


def cut_span(past, a, b, run_a, run_b, ts_path):
    drop = set(range(a, b))
    if ts_path and a == run_a and b == run_b:
        if a - 1 >= 0 and past[a - 1] >= BEG:
            drop.add(a - 1)
        if b < len(past) and past[b] >= BEG:
            drop.add(b)
    return [t for i, t in enumerate(past) if i not in drop]


def fake_window(S, k, ts_path):
    if not ts_path:
        return list(S) * k + [EOT]
    n = max(50, min(400, 15 * len(S)))  # ~0.3 s per token, in 20 ms timestamp steps
    n = min(n, 1500 // k)
    out, t = [], 0
    for _ in range(k):
        out += [BEG + t] + list(S) + [BEG + t + n]
        t += n
    return out


def synthetic_candidates(d: pc.Dec, rng):
    W = d.W
    cands = []
    for f in sorted(pc.WIN_DIR.glob("windows_*.json")):
        res = json.loads(f.read_text())
        path, clip = res["path"], res["clip"]
        nt = res["config"]["no_timestamps"]
        for w in res["windows"]:
            past = w["past"]
            if w["index"] == 0 or not past:
                continue
            capped = past[-(hw.MAX_PROMPT_CTX - 1):]
            if has_repeat(W, capped):
                continue  # previous text already holds a Loop; CLEAN would not be clean
            spans = [s for s in sentence_spans(W, capped)
                     if 5 <= s[1] - s[0] <= 40 and len(pc.words(W.decode(capped[s[0]:s[1]], special=False))) >= 4]
            if not spans:
                continue
            last = spans[-1]
            others = spans[:-1]
            rng.shuffle(others)
            for a, b, ra, rb in [last] + others[:4]:
                cands.append(dict(path=path, clip=clip, nt=nt, window=w["index"], seek=w["seek"],
                                  S=capped[a:b], clean_past=cut_span(capped, a, b, ra, rb, not nt),
                                  is_last=(a, b) == last[:2]))
    return cands


def natural_cases(d):
    W = d.W
    rep = json.loads((hw.ROOT / "results/phase0/hf_repro.json").read_text())["cases"]
    paths = {"ts04_onset": "timestamps", "ts04_copy": "timestamps", "bare01_copy": "timestamps-bare",
             "nt02_copy": "nt", "ts03_onset": "timestamps"}
    clips = {"ts04_onset": "find_da_5min_04", "ts04_copy": "find_da_5min_04", "bare01_copy": "find_da_5min_01",
             "nt02_copy": "find_da_5min_02", "ts03_onset": "find_da_5min_03"}
    natural = []
    for name, c in rep.items():
        inp = c["inputs"]
        nt = paths[name] == "nt"
        seek = int(round(c["seek_s"] * 100))
        loop = inp["loop"]
        prefix = list(inp["prefix"])
        j = 0
        while j < len(loop) and loop[j] >= BEG:
            j += 1
        prefix += loop[:j]
        k = j
        while k < len(loop) and loop[k] < EOT:
            k += 1
        S = loop[j:k]
        onset = c["case"]["kind"] == "onset"
        clean_prompt = inp["variants"]["no_prev_text"] if onset else inp["variants"]["prev_text_minus_loop_sentence"]
        wk = d.encode(paths[name], clips[name], seek)
        g = d.greedy([clean_prompt], [wk], CLEAN_STEPS, [nt], prefixes=[prefix])[0]
        # decide at the first token where S and CLEAN's own continuation part (shared start -> prefix)
        j = 0
        while j < len(S) - 1 and j < len(g) and S[j] == g[j]:
            j += 1
        prefix, S, g = prefix + S[:j], S[j:], g[j:]
        C0 = g[0]
        gl = d.greedy([inp["prompt"]], [wk], len(S) + 10, [nt], prefixes=[prefix])[0]
        lp_l = d.logprobs_last([inp["prompt"] + prefix], [wk]).numpy()[0]
        lp_c = d.logprobs_last([clean_prompt + prefix], [wk]).numpy()[0]
        natural.append(dict(name=name, path=paths[name], clip=clips[name], split=pc.split_of(clips[name]), nt=nt,
                            seek=seek, S=S, S_text=W.decode(S, special=False), prefix=prefix,
                            prefix_text=W.decode(prefix[-6:]),
                            clean_prompt=clean_prompt, clean_cont=[t for t in g][:30], clean_text=W.decode(g),
                            C0=int(C0), S0=int(S[0]), clean_ld=float(lp_c[S[0]] - lp_c[C0]),
                            clean_loops=bool(pc.writes_s(W, g, S)),
                            loop={"nat": dict(prompt=inp["prompt"], ld=float(lp_l[S[0]] - lp_l[C0]),
                                              argmax=int(np.argmax(lp_l)), gen=gl, gen_text=W.decode(gl),
                                              loops=bool(pc.writes_s(W, gl, S)))}))
    return natural


def natural_stats(natural):
    return {n["name"]: dict(loop_ld=n["loop"]["nat"]["ld"], clean_ld=n["clean_ld"], decision_after=n.get("prefix_text", ""),
                            loop_writes_S=n["loop"]["nat"]["loops"], clean_writes_S=n["clean_loops"]) for n in natural}


def main():
    t0 = time.time()
    rng = random.Random(0)
    d = pc.Dec()
    W = d.W
    if "--natural-only" in sys.argv:
        out = json.loads((pc.P1 / "dataset.json").read_text())
        out["natural"] = natural_cases(d)
        out["stats"]["natural"] = natural_stats(out["natural"])
        print(json.dumps(out["stats"]["natural"], indent=1))
        (pc.P1 / "dataset.json").write_text(json.dumps(out, ensure_ascii=False))
        return
    cands = synthetic_candidates(d, rng)
    print(f"{len(cands)} candidate (window, S)", flush=True)
    wins = [d.encode(c["path"], c["clip"], c["seek"]) for c in cands]
    d.cache_kv = False
    # ---- CLEAN decodes
    clean_prompts = [hw.build_prompt(c["clean_past"], c["nt"]) for c in cands]
    gens = d.greedy(clean_prompts, wins, CLEAN_STEPS, [c["nt"] for c in cands])
    pairs = []
    per_window = {}
    for c, wk, pr, g in zip(cands, wins, clean_prompts, gens):
        prefix = [g[0]] if (not c["nt"] and g and g[0] >= BEG) else []
        cont = g[len(prefix):]
        text_cont = [t for t in cont if t < EOT]
        if not text_cont or cont[0] >= EOT:
            continue  # CLEAN ends the window at once (or writes a timestamp first): no decision token
        C0, S0 = cont[0], c["S"][0]
        if C0 == S0 or pc.writes_s(W, g, c["S"], 3):
            continue
        key = (c["path"], c["clip"], c["window"])
        if per_window.get(key, 0) >= MAX_S:
            continue
        per_window[key] = per_window.get(key, 0) + 1
        # clean continuation for the clean-cost metric: up to 30 tokens, to EOT
        cc = cont[:30]
        pairs.append(dict(path=c["path"], clip=c["clip"], split=pc.split_of(c["clip"]), nt=c["nt"],
                          window=c["window"], seek=c["seek"], S=c["S"], S_text=W.decode(c["S"], special=False),
                          is_last=c["is_last"], prefix=prefix, clean_past=c["clean_past"], clean_prompt=pr, clean_cont=cc,
                          clean_text=W.decode(g), C0=int(C0), S0=int(S0)))
    print(f"{len(pairs)} (window, S) after filters; {time.time() - t0:.0f}s", flush=True)
    # ---- LOOP states
    loop_states = []
    for i, p in enumerate(pairs):
        for k in KS:
            lp = p["clean_past"] + fake_window(p["S"], k, not p["nt"])
            loop_states.append((i, k, hw.build_prompt(lp, p["nt"])))
    lwins = [d.encode(pairs[i]["path"], pairs[i]["clip"], pairs[i]["seek"]) for i, _, _ in loop_states]
    lseqs = [pr + pairs[i]["prefix"] for i, _, pr in loop_states]
    lps = d.logprobs_last(lseqs, lwins).numpy()
    lgen = d.greedy([pr for _, _, pr in loop_states], lwins,
                    max(len(p["S"]) for p in pairs) + 10,
                    [pairs[i]["nt"] for i, _, _ in loop_states], prefixes=[pairs[i]["prefix"] for i, _, _ in loop_states])
    cwins = [d.encode(p["path"], p["clip"], p["seek"]) for p in pairs]
    cps = d.logprobs_last([p["clean_prompt"] + p["prefix"] for p in pairs], cwins).numpy()
    for p, lpc in zip(pairs, cps):
        p["clean_ld"] = float(lpc[p["S0"]] - lpc[p["C0"]])
        p["clean_lp_S0"] = float(lpc[p["S0"]])
        p["clean_lp_C0"] = float(lpc[p["C0"]])
        p["clean_loops"] = False  # filtered above: CLEAN's 60-token decode never writes S
        p["loop"] = {}
    for (i, k, pr), lpr, g in zip(loop_states, lps, lgen):
        p = pairs[i]
        p["loop"][str(k)] = dict(prompt=pr, ld=float(lpr[p["S0"]] - lpr[p["C0"]]), lp_S0=float(lpr[p["S0"]]),
                                 lp_C0=float(lpr[p["C0"]]), argmax=int(np.argmax(lpr)),
                                 gen=g, gen_text=W.decode(g), loops=bool(pc.writes_s(W, g, p["S"])))
    natural = natural_cases(d)
    # ---- stats
    stats = {}
    for k in KS:
        L = [p["loop"][str(k)] for p in pairs]
        stats[f"k{k}"] = dict(n=len(L), loop_rate=float(np.mean([x["loops"] for x in L])),
                              prefers_S=float(np.mean([x["ld"] > 0 for x in L])),
                              mean_ld=float(np.mean([x["ld"] for x in L])))
    stats["clean"] = dict(n=len(pairs), prefers_S=float(np.mean([p["clean_ld"] > 0 for p in pairs])),
                          mean_ld=float(np.mean([p["clean_ld"] for p in pairs])))
    valid = [(i, k) for i, p in enumerate(pairs) for k in KS
             if (p["loop"][str(k)]["loops"] or p["loop"][str(k)]["ld"] > 0) and p["clean_ld"] < 0]
    stats["valid_pairs"] = len(valid)
    stats["valid_by_split"] = {s: sum(pairs[i]["split"] == s for i, _ in valid) for s in "AB"}
    stats["valid_loops_and_prefers"] = sum(pairs[i]["loop"][str(k)]["loops"] and pairs[i]["loop"][str(k)]["ld"] > 0
                                           for i, k in valid)
    by_path = {}
    for p in pairs:
        for k in KS:
            b = by_path.setdefault(f"{p['path']}|k{k}", [0, 0])
            b[0] += 1
            b[1] += p["loop"][str(k)]["loops"]
    stats["loop_rate_by_path"] = {k: dict(n=v[0], loops=v[1], rate=v[1] / v[0]) for k, v in by_path.items()}
    stats["natural"] = natural_stats(natural)
    stats["seconds"] = round(time.time() - t0)
    print(json.dumps(stats, indent=1), flush=True)
    out = dict(stats=stats, pairs=pairs, natural=natural, valid=valid,
               notes="pairs[i].loop[k]: LOOP state with S written k times in a fake previous window; "
                     "clean_*: the CLEAN state (S cut out). valid = (pair, k) with LOOP writing or preferring S "
                     "and CLEAN preferring C.")
    (pc.P1 / "dataset.json").write_text(json.dumps(out, ensure_ascii=False))
    print("wrote results/phase1/dataset.json")


if __name__ == "__main__":
    main()
