"""Phase 1: print the markdown tables for results/phase1/localize.md from localize.json and
dataset.json (no model needed). Also adds a fuzzy Loop rate to each verify condition: a LOOP
state counts when its greedy text holds at least half of S's words (3+ letters), which also
catches paraphrased rewrites of S that the exact first-4-words test misses.

    .venv/bin/python -I scripts/phase1_report.py > /tmp/tables.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "results/phase1"
_W = re.compile(r"\w+", re.UNICODE)
_TS = re.compile(r"<\|[^|]*\|>")


def wset(t):
    return {w.lower() for w in _W.findall(_TS.sub(" ", t)) if len(w) >= 3}


def main():
    loc = json.loads((P1 / "localize.json").read_text())
    ds = json.loads((P1 / "dataset.json").read_text())
    pairs = ds["pairs"]
    valid = [(i, str(k)) for i, k in ds["valid"]]
    s = ds["stats"]
    print("## Dataset\n")
    print("| k copies of S | LOOP states | greedy writes S | prefers S (LD>0) | mean LD |")
    print("|---|---|---|---|---|")
    for k in ("1", "2", "3"):
        x = s[f"k{k}"]
        print(f"| {k} | {x['n']} | {x['loop_rate']:.1%} | {x['prefers_S']:.1%} | {x['mean_ld']:.2f} |")
    x = s["clean"]
    print(f"| CLEAN (0) | {x['n']} | 0 (filter) | {x['prefers_S']:.1%} | {x['mean_ld']:.2f} |")
    print(f"\nvalid pairs {s['valid_pairs']} (A {s['valid_by_split']['A']}, B {s['valid_by_split']['B']}); "
          f"both loops and prefers {s['valid_loops_and_prefers']}\n")
    print("| path | k=1 | k=2 | k=3 |\n|---|---|---|---|")
    for p in ("timestamps", "timestamps-bare", "nt"):
        r = [s["loop_rate_by_path"][f"{p}|k{k}"] for k in (1, 2, 3)]
        print(f"| {p} | " + " | ".join(f"{a['loops']}/{a['n']}" for a in r) + " |")

    # ---- verify, with fuzzy rate
    v = loc.get("verify")
    if v:
        LB = [(i, k) for i, k in valid if pairs[i]["split"] == "B"]
        print(f"\n## Verify on find-B ({v['n_loop_B']} LOOP, {v['n_clean_B']} CLEAN)\n")
        print("| condition | Loop rate (exact) | fuzzy | LD | CLEAN cont log-p | CLEAN greedy unchanged | natural Loops |")
        print("|---|---|---|---|---|---|---|")
        for name, c in v["conditions"].items():
            fz = []
            for (i, k), g in zip(LB, c["gens"]):
                sw = wset(pairs[i]["S_text"])
                fz.append(len(sw & wset(g)) / max(1, len(sw)) >= 0.5)
            c["fuzzy_loop_rate"] = float(np.mean(fz))
            print(f"| `{name}` | {c['loop_rate']:.2f} | {c['fuzzy_loop_rate']:.2f} | {c['loop_ld']:.2f} | "
                  f"{c['clean_cont_lp']:.3f} | {c['clean_greedy_unchanged']:.2f} | "
                  f"{sum(c['natural']['per_case'].values())}/5 |")
        (P1 / "localize.json").write_text(json.dumps(loc, ensure_ascii=False, indent=1))

    sw = loc.get("sweep")
    if sw:
        print(f"\n## Sweep (find-A, {sw['n_loop']} LOOP / {sw['n_clean']} CLEAN), base {sw['base']}\n")
        print("| head | dLD | flip | argmax=S | dCLEAN LD | dCLEAN cont log-p |\n|---|---|---|---|---|---|")
        for r in sorted(sw["rows"], key=lambda r: r["d_ld"])[:25]:
            print(f"| L{r['L']}{'H' if r['kind'] == 'self' else 'X'}{r['h']} | {r['d_ld']:.3f} | {r['flip']:.2f} | "
                  f"{r['argmax_S']:.2f} | {r['d_clean_ld']:.3f} | {r['d_clean_cont']:.3f} |")
        rows = sw["rows"]
        for kind in ("self", "cross"):
            d = np.array([r["d_ld"] for r in rows if r["kind"] == kind])
            print(f"\n{kind}: heads with dLD < -0.2: {(d < -0.2).sum()}, < -0.5: {(d < -0.5).sum()}, sum of all dLD {d.sum():.2f}")
    pt = loc.get("patch")
    if pt:
        print(f"\n## Patching (LOOP LD {pt['loop_ld']:.2f}, CLEAN LD {pt['clean_ld']:.2f})\n")
        print("| layer | attn_out | cross_out | mlp_out | resid_post |\n|---|---|---|---|---|")
        by = {}
        for r in pt["sites"]:
            if r["pos"] == "decision":
                by.setdefault(r["L"], {})[r["site"]] = r["frac_restored"]
        for L in sorted(by):
            b = by[L]
            print(f"| {L} | {b['attn_out']:.2f} | {b['cross_out']:.2f} | {b['mlp_out']:.2f} | {b['resid_post']:.2f} |")
        tail = {(r["L"], r["site"]): r["frac_restored"] for r in pt["sites"] if r["pos"] == "tail"}
        dec = {(r["L"], r["site"]): r["frac_restored"] for r in pt["sites"] if r["pos"] == "decision"}
        print("\nmax |tail - decision|:", max(abs(tail[k] - dec[k]) for k in tail))
        print("\n| head | frac restored (decision) | flipped |\n|---|---|---|")
        for r in sorted([h for h in pt.get("heads", []) if h["pos"] == "decision"], key=lambda r: -r["frac_restored"])[:12]:
            print(f"| L{r['L']}{'H' if r['kind'] == 'self' else 'X'}{r['h']} | {r['frac_restored']:.3f} | {r['flipped']:.2f} |")


if __name__ == "__main__":
    sys.exit(main())
