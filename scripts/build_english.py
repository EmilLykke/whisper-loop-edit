"""Final run, report-only extra: 5 English Long Clips from FLEURS en_us dev, same recipe as the
Find and Held-out Sets (build_clips.py's `buildLongClip` port, 1 s gaps, continuous clips trimmed
and joined with 0.05 s).

    .venv/bin/python -I scripts/build_english.py download   # FLEURS en_us dev tsv + tar, pinned
    .venv/bin/python -I scripts/build_english.py build      # data/clips/english/, data/manifests/english.json

Uses build_clips.py's functions with its module constants pointed at en_us dev: one recording
per sentence (seeded pick), seeded order with SEED below, clips 2 x 5 min, 2 x 2 min,
1 x continuous 90 s. English is never part of the bar (ADR-0002); this only checks that E3
does not break English long-form decoding.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_clips as bc  # noqa: E402

bc.RAW = bc.REPO / "data" / "raw" / "fleurs_en_us"
bc.HF_DIR = bc.RAW / "data" / "en_us"
bc.EXTRACTED = bc.RAW / "extracted"
bc.SPLITS = ("dev",)
bc.SEED = "whisper-loop-edit/2026-11-09/english"

PLAN = [
    ("en_5min_01", 300, bc.GAP_SECONDS, False),
    ("en_5min_02", 300, bc.GAP_SECONDS, False),
    ("en_2min_01", 120, bc.GAP_SECONDS, False),
    ("en_2min_02", 120, bc.GAP_SECONDS, False),
    ("en_continuous_90s_01", 90, bc.CONTINUOUS_GAP_SECONDS, True),
]


def download() -> None:
    import tarfile

    from huggingface_hub import hf_hub_download

    for f in ("data/en_us/dev.tsv", "data/en_us/audio/dev.tar.gz"):
        print(hf_hub_download(bc.HF_REPO, f, repo_type="dataset", revision=bc.HF_REVISION, local_dir=bc.RAW), flush=True)
    dest = bc.EXTRACTED
    if (dest / "dev").is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bc.HF_DIR / "audio" / "dev.tar.gz") as tar:
        for m in tar.getmembers():
            name = Path(m.name)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != "dev":
                raise SystemExit(f"unexpected tar member {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise SystemExit(f"unexpected tar member type {m.name!r}")
        tar.extractall(dest, filter="data")


def build() -> None:
    recs, rows = bc.read_recordings()
    groups = bc.group_sentences(recs)
    picks = [min(g, key=lambda r: bc.key("recording", r.split, r.file)) for g in groups.values()]
    remaining = sorted(picks, key=lambda r: bc.key("order", "english", r.split, r.file))
    out_dir = bc.CLIPS / "english"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for suffix, target, gap_s, trim in PLAN:
        name = f"english_{suffix}"
        audio, windows, used = bc.build_long_clip(remaining, target, gap_s, trim)
        remaining = [r for r in remaining if r.file not in used]
        data = bc.wav_bytes(audio)
        (out_dir / f"{name}.wav").write_bytes(data)
        clips.append({
            "name": name, "lang": "en", "path": f"data/clips/english/{name}.wav", "target_s": target,
            "duration_s": round(audio.size / bc.SAMPLE_RATE, 6), "samples": int(audio.size), "gap_s": gap_s,
            "trimmed": trim, "sha256": hashlib.sha256(data).hexdigest(), "utterances": windows,
        })
        print(f"{name}: {audio.size / bc.SAMPLE_RATE:7.2f} s, {len(windows)} utterances")
    manifest = {
        "set": "english",
        "source": {"dataset": bc.HF_REPO, "language": "en_us", "splits": list(bc.SPLITS), "revision": bc.HF_REVISION},
        "seed": bc.SEED, "sample_rate": bc.SAMPLE_RATE, "format": "WAV PCM16 mono",
        "recipe": "build_clips.py buildLongClip port, see scripts/build_english.py; report-only (ADR-0002)",
        "summary": {"tsv_rows": rows, "recordings_with_wav": len(recs),
                    "utterances_used": sum(len(c["utterances"]) for c in clips),
                    "total_duration_s": round(sum(c["duration_s"] for c in clips), 3)},
        "clips": clips,
    }
    (bc.MANIFESTS / "english.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    {"download": download, "build": build}[sys.argv[1]]()
