"""Phase 2: the Phase 1 HF proxy on an edited checkpoint (the weights themselves, no hooks).

    .venv/bin/python -I scripts/phase2_proxy.py <id|baseline>   -> results/phase2/proxy/<id>.json

Loads models/hf/edits/<id>/ (or the unedited models/hf/whisper-large-v3 for `baseline`) in fp16 on
MPS, so it runs exactly the fp16-rounded edited weights crispasr gets, and runs Phase 1's `verify`
evaluation unchanged (`phase1_localize.evaluate`): the 117 find-B LOOP states and their 58 CLEAN
states (greedy Loop rate over 24 tokens, LOOP logit diff log p(S0) - log p(C0), CLEAN continuation
log-prob per token, CLEAN greedy unchanged over 30 tokens), plus the 5 natural Phase 0 cases (60
tokens). Adds Phase 1's fuzzy Loop rate (greedy text holds >= half of S's 3+-letter words).
Only forward passes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_window as hw  # noqa: E402
import phase1_common as pc  # noqa: E402
import phase1_localize as pl  # noqa: E402
from phase1_report import wset  # noqa: E402

ROOT = pc.ROOT
OUT = ROOT / "results/phase2/proxy"


def main():
    eid = sys.argv[1]
    model_dir = hw.MODEL_DIR if eid == "baseline" else ROOT / "models/hf/edits" / eid
    torch.set_grad_enabled(False)
    t0 = time.time()
    d = pc.Dec(hw.Whisper(model_dir))
    pl.pc_dec = d
    d.cache_kv = False
    ds = pc.load_dataset()
    loop, clean, nat_loop, nat_clean = pl.build_items(d, ds)
    LB = [it for it in loop if it["split"] == "B"]
    CB = [it for it in clean if it["split"] == "B"]
    r = pl.evaluate(d, LB, CB, {}, 24, ds)
    rn = pl.evaluate(d, nat_loop, nat_clean, {}, 60, ds)
    pairs = ds["pairs"]
    r["fuzzy_loop_rate"] = float(np.mean([
        len(wset(pairs[it["pair"]]["S_text"]) & wset(g)) / max(1, len(wset(pairs[it["pair"]]["S_text"]))) >= 0.5
        for it, g in zip(LB, r["gens"])]))
    r["natural"] = dict(loop_rate=rn["loop_rate"], loop_ld=rn["loop_ld"],
                        per_case=dict(zip([it["name"] for it in nat_loop], rn["per_item_loops"])),
                        clean_cont_lp=rn["clean_cont_lp"], clean_greedy_unchanged=rn["clean_greedy_unchanged"],
                        gens=dict(zip([it["name"] for it in nat_loop], rn["gens"])))
    r["n_loop_B"], r["n_clean_B"] = len(LB), len(CB)
    r["model_dir"] = str(Path(model_dir).relative_to(ROOT))
    r["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{eid}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
    print(f"{eid}: loop {r['loop_rate']:.2f} (fuzzy {r['fuzzy_loop_rate']:.2f}) LD {r['loop_ld']:.2f} | "
          f"clean lp {r['clean_cont_lp']:.3f} unchanged {r['clean_greedy_unchanged']:.2f} | natural "
          f"{sum(r['natural']['per_case'].values())}/5 {r['natural']['per_case']} | {r['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
