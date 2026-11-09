"""Transcribe a few clips with crispasr under several GGML models and check the text matches.

    .venv/bin/python -I scripts/smoke_crispasr.py --model ours=models/ggml/a.bin \
        --model official=models/ggml/b.bin clip1.wav clip2.wav

Runs `crispasr -m <model> -l da -f <wav> --no-prints -nt -bs 5` (legacy `-nt` Decode Path,
beam 5) and prints one JSON line per run, then whether every clip got the same text from
every model. Exit code 1 if any clip differs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CRISPASR = Path(os.environ.get("CODICTATE_DIR", REPO.parent / "codictate")) / "vendors" / "crispasr" / "crispasr"


def transcribe(model: Path, wav: Path, extra: list[str]) -> tuple[str, float]:
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(CRISPASR.parent))
    cmd = [str(CRISPASR), "-m", str(model), "-l", "da", "-f", str(wav),
           "--no-prints", "-nt", "-bs", "5", *extra]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return res.stdout.strip(), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="label=path")
    ap.add_argument("--extra", default="", help="extra crispasr args, space separated")
    ap.add_argument("wavs", nargs="+", type=Path)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    extra = args.extra.split()

    texts: dict[str, dict[str, str]] = {}
    for wav in args.wavs:
        for label, path in models.items():
            text, secs = transcribe(Path(path), wav, extra)
            texts.setdefault(wav.name, {})[label] = text
            print(json.dumps({"clip": wav.name, "model": label, "seconds": round(secs, 2),
                              "text": text}, ensure_ascii=False))

    all_same = True
    for clip, by_model in texts.items():
        same = len(set(by_model.values())) == 1
        all_same &= same
        print(f"{clip}: {'IDENTICAL' if same else 'DIFFERENT'}")
    sys.exit(0 if all_same else 1)


if __name__ == "__main__":
    main()
