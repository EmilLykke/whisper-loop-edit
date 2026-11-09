"""Phase 1: rebuild crispasr's decode windows for every Find clip on the paths that need no
VAD log (timestamps-bare, nt), plus the timestamps windows Phase 0 already rebuilt.

    .venv/bin/python -I scripts/phase1_windows.py        # -> results/phase1/windows/*.json

Windows already rebuilt in results/phase0/hf_repro are copied, not recomputed.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_window as hw  # noqa: E402
import phase1_common  # noqa: E402,F401  (UTF-8-tolerant case_inputs)

OUT = hw.ROOT / "results/phase1/windows"
CLIPS = [f"find_da_5min_0{i}" for i in range(1, 5)] + [f"find_da_2min_0{i}" for i in range(1, 4)] + \
        [f"find_da_continuous_90s_0{i}" for i in range(1, 4)]
# timestamps needs crispasr's VAD segments; Phase 0 saved them for 5min_03 and 5min_04 only
JOBS = [("timestamps", "find_da_5min_03"), ("timestamps", "find_da_5min_04")] + \
       [(p, c) for p in ("timestamps-bare", "nt") for c in CLIPS]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = None
    for path, clip in JOBS:
        out = OUT / f"windows_{path}_{clip}.json"
        if out.exists():
            continue
        old = hw.REPRO_DIR / f"windows_{path}_{clip}.json"
        if old.exists():
            shutil.copy(old, out)
            print("copied", old.name, flush=True)
            continue
        W = W or hw.Whisper()
        print("resolving", path, clip, flush=True)
        res = hw.resolve_case(W, path, clip, log=lambda s: print("  ", s, flush=True))
        for rec in res["windows"]:
            rec["text"] = W.decode(rec["tokens"])
            rec["prompt_text"] = W.decode(rec["prompt"])
        res["dtype"] = "float16"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
