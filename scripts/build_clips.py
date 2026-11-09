"""Build the Find Set and Held-out Set of Long Clips from FLEURS da_dk train + dev.

Mirrors how Codictate's long-audio check builds its clips
(../codictate/src/bun/utils/whisper/engines/long-audio.manual.ts: `readWavSamples`,
`speechBounds`, `trimSilence`, `buildLongClip`, `fleursDanishPool`), so a Loop found here is
the same kind of Loop that check finds. Deviations, and why, are listed in
data/manifests/README.md.

    .venv/bin/python -I scripts/build_clips.py download   # FLEURS da_dk train+dev, pinned
    .venv/bin/python -I scripts/build_clips.py build      # WAVs + data/manifests/*.json
    .venv/bin/python -I scripts/build_clips.py check      # rebuild in a temp dir, compare sha256

The FLEURS test split is never downloaded or read: it is Codictate's WER benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "fleurs_da_dk"
HF_DIR = RAW / "data" / "da_dk"  # where hf_hub_download puts the files
EXTRACTED = RAW / "extracted"  # tar contents: extracted/{train,dev}/*.wav
CLIPS = REPO / "data" / "clips"
MANIFESTS = REPO / "data" / "manifests"

HF_REPO = "google/fleurs"
HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPLITS = ("train", "dev")  # never "test"

SEED = "whisper-loop-edit/2026-11-08"

# Codictate's constants (long-audio.manual.ts).
SAMPLE_RATE = 16_000
GAP_SECONDS = 1.0
CONTINUOUS_GAP_SECONDS = 0.05
MAX_CLIP_SECONDS = 25

SETS = ("find", "heldout")
# One round is 10 clips. The Held-out Set was extended from 1 round to 3 (20 more clips) after
# Phase 2, by walking on through its own remaining pool: rounds 2-3 come after round 1, so the
# first 10 clips are unchanged. Find stays at 1 round.
ROUNDS = {"find": 1, "heldout": 3}


def clip_plan(rounds: int) -> list[tuple[str, float, float, bool]]:
    """Per set, built in this order from one shared pool: (name suffix, target s, gap s, trim).
    Round r numbers 5 min clips 4r-3..4r, 2 min and continuous clips 3r-2..3r."""
    plan: list[tuple[str, float, float, bool]] = []
    for r in range(rounds):
        plan += [(f"da_5min_{4 * r + i:02d}", 300, GAP_SECONDS, False) for i in range(1, 5)]
        plan += [(f"da_2min_{3 * r + i:02d}", 120, GAP_SECONDS, False) for i in range(1, 4)]
        plan += [
            (f"da_continuous_90s_{3 * r + i:02d}", 90, CONTINUOUS_GAP_SECONDS, True)
            for i in range(1, 4)
        ]
    return plan


# -- Download -----------------------------------------------------------------------------


def download() -> None:
    from huggingface_hub import hf_hub_download

    files = [f"data/da_dk/{s}.tsv" for s in SPLITS] + [
        f"data/da_dk/audio/{s}.tar.gz" for s in SPLITS
    ]
    for f in files:
        path = hf_hub_download(
            HF_REPO, f, repo_type="dataset", revision=HF_REVISION, local_dir=RAW
        )
        print(path, flush=True)
    for split in SPLITS:
        dest = EXTRACTED
        if (dest / split).is_dir():
            print(f"{dest / split} exists, not extracting again")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(HF_DIR / "audio" / f"{split}.tar.gz") as tar:
            for m in tar.getmembers():
                name = Path(m.name)
                if name.is_absolute() or ".." in name.parts or name.parts[0] != split:
                    raise SystemExit(f"unexpected tar member {m.name!r}")
                if not (m.isfile() or m.isdir()):
                    raise SystemExit(f"unexpected tar member type {m.name!r}")
            tar.extractall(dest, filter="data")
        print(f"extracted {split} into {dest / split}")


# -- WAV I/O, as Codictate does it --------------------------------------------------------


def read_wav_samples(path: Path) -> np.ndarray | None:
    """`readWavSamples`: 16 kHz mono PCM16 or float32 to int16; anything else is None."""
    buf = path.read_bytes()
    offset = 12
    fmt = channels = rate = bits = 0
    while offset + 8 <= len(buf):
        cid = buf[offset : offset + 4]
        (size,) = struct.unpack_from("<I", buf, offset + 4)
        body = offset + 8
        if cid == b"fmt ":
            fmt, channels, rate = struct.unpack_from("<HHI", buf, body)
            (bits,) = struct.unpack_from("<H", buf, body + 14)
            if fmt == 0xFFFE:
                (fmt,) = struct.unpack_from("<H", buf, body + 24)
        elif cid == b"data":
            if channels != 1 or rate != SAMPLE_RATE:
                return None
            end = min(len(buf), body + size)
            if fmt == 1 and bits == 16:
                n = (end - body) >> 1
                return np.frombuffer(buf, "<i2", n, body).astype(np.int16)
            if fmt == 3 and bits == 32:
                n = (end - body) >> 2
                v = np.frombuffer(buf, "<f4", n, body).astype(np.float64)
                v = np.clip(v, -1.0, 1.0) * 32767.0
                # JS Math.round: half rounds up, i.e. floor(x + 0.5).
                return np.floor(v + 0.5).astype(np.int16)
            return None
        offset = body + size + (size & 1)
    return None


def wav_bytes(samples: np.ndarray) -> bytes:
    """`writeWav`: the same 44-byte header, 16 kHz mono PCM16."""
    n = samples.size * 2
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + n)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        + b"data"
        + struct.pack("<I", n)
    )
    return header + samples.astype("<i2").tobytes()


def speech_bounds(samples: np.ndarray) -> tuple[int, int]:
    """`speechBounds`: drop leading/trailing 20 ms frames under 2% of the loudest frame."""
    frame = SAMPLE_RATE // 50
    frames = samples.size // frame
    s = samples[: frames * frame].astype(np.int64).reshape(frames, frame)
    rms = np.sqrt((s * s).sum(axis=1) / frame)
    quiet = max(0.0, float(rms.max()) if frames else 0.0) * 0.02
    first, last = 0, frames - 1
    while first < last and rms[first] < quiet:
        first += 1
    while last > first and rms[last] < quiet:
        last -= 1
    return first * frame, (last + 1) * frame


# -- Pool and split -----------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    split: str
    sentence_id: str
    file: str
    text: str  # raw_transcription: TSV column 3, what Codictate's check uses
    gender: str

    @property
    def wav(self) -> Path:
        return EXTRACTED / self.split / self.file


def key(*parts: str) -> str:
    """A seeded, Python-version-independent sort key."""
    return hashlib.sha256("\0".join((SEED, *parts)).encode()).hexdigest()


def read_recordings() -> tuple[list[Recording], int]:
    """Every TSV row with a file, a text and a WAV on disk (`fleursDanishPool`'s filter)."""
    rows = 0
    out: list[Recording] = []
    for split in SPLITS:
        for line in (HF_DIR / f"{split}.tsv").read_text("utf-8").split("\n"):
            cols = line.split("\t")
            if len(cols) < 3 or not cols[1] or not cols[2]:
                continue
            rows += 1
            r = Recording(split, cols[0], cols[1], cols[2], cols[6] if len(cols) > 6 else "")
            if r.wav.exists():
                out.append(r)
    return out, rows


def group_sentences(recs: list[Recording]) -> dict[str, list[Recording]]:
    """Sentences, keyed by id, with ids that share a transcription merged into one group, so
    no sentence text can land in both sets."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_text: dict[str, str] = {}
    for r in recs:
        find(r.sentence_id)
        if r.text in by_text:
            a, b = find(by_text[r.text]), find(r.sentence_id)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            by_text[r.text] = r.sentence_id
    groups: dict[str, list[Recording]] = {}
    for r in recs:
        groups.setdefault(find(r.sentence_id), []).append(r)
    return groups


def set_pools(groups: dict[str, list[Recording]]) -> dict[str, list[Recording]]:
    """Half the sentences to each set; one recording per sentence (as `fleursDanishPool`),
    picked and ordered by the seeded key."""
    order = sorted(groups, key=lambda g: key("sentence", g))
    half = len(order) // 2
    assigned = {"find": order[:half], "heldout": order[half:]}
    pools: dict[str, list[Recording]] = {}
    for name, gids in assigned.items():
        picks = [
            min(groups[g], key=lambda r: key("recording", r.split, r.file)) for g in gids
        ]
        pools[name] = sorted(picks, key=lambda r: key("order", name, r.split, r.file))
    return pools


# -- Clips --------------------------------------------------------------------------------


def build_long_clip(
    pool: list[Recording], target: float, gap_s: float, trim: bool
) -> tuple[np.ndarray, list[dict], set[str]]:
    """`buildLongClip`, over the set's remaining pool: utterances in pool order joined with
    `gap_s` of silence; one is skipped when over 25 s or when it would pass target * 1.05; stop
    at target * 0.95. Returns the samples, the windows, and the files used."""
    pieces: list[np.ndarray] = []
    windows: list[dict] = []
    used: set[str] = set()
    gap = np.zeros(round(gap_s * SAMPLE_RATE), dtype=np.int16)
    at = 0.0
    for r in pool:
        read = read_wav_samples(r.wav)
        if read is None:
            continue
        if trim:
            t0, t1 = speech_bounds(read)
            samples = read[t0:t1]
        else:
            t0, t1 = 0, read.size
            samples = read
        seconds = samples.size / SAMPLE_RATE
        if seconds > MAX_CLIP_SECONDS or at + seconds > target * 1.05:
            continue
        if pieces:
            pieces.append(gap)
            at += gap.size / SAMPLE_RATE
        pieces.append(samples)
        s0, s1 = speech_bounds(samples)
        windows.append(
            {
                "fleurs_split": r.split,
                "fleurs_id": r.sentence_id,
                "file": r.file,
                "start_s": round(at, 6),
                "end_s": round(at + seconds, 6),
                "speech_start_s": round(at + s0 / SAMPLE_RATE, 6),
                "speech_end_s": round(at + s1 / SAMPLE_RATE, 6),
                "source_from_s": round(t0 / SAMPLE_RATE, 6),
                "source_to_s": round(t1 / SAMPLE_RATE, 6),
                "gender": r.gender,
                "text": r.text,
            }
        )
        used.add(r.file)
        at += seconds
        if at >= target * 0.95:
            break
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    return audio, windows, used


def build(clips_dir: Path, manifests_dir: Path | None) -> dict[str, dict]:
    recs, rows = read_recordings()
    groups = group_sentences(recs)
    pools = set_pools(groups)
    summary = {
        "tsv_rows": rows,
        "recordings_with_wav": len(recs),
        "recordings_by_split": {s: sum(r.split == s for r in recs) for s in SPLITS},
        "sentence_ids": len({r.sentence_id for r in recs}),
        "sentence_groups": len(groups),
    }
    print(json.dumps(summary, indent=2))
    manifests: dict[str, dict] = {}
    all_used: set[str] = set()
    for set_name in SETS:
        remaining = list(pools[set_name])
        out_dir = clips_dir / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = []
        for suffix, target, gap_s, trim in clip_plan(ROUNDS[set_name]):
            name = f"{set_name}_{suffix}"
            audio, windows, used = build_long_clip(remaining, target, gap_s, trim)
            if used & all_used:
                raise SystemExit(f"{name}: recording reused")
            all_used |= used
            remaining = [r for r in remaining if r.file not in used]
            data = wav_bytes(audio)
            (out_dir / f"{name}.wav").write_bytes(data)
            clips.append(
                {
                    "name": name,
                    "path": f"data/clips/{set_name}/{name}.wav",
                    "target_s": target,
                    "duration_s": round(audio.size / SAMPLE_RATE, 6),
                    "samples": int(audio.size),
                    "gap_s": gap_s,
                    "trimmed": trim,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "utterances": windows,
                }
            )
            print(f"{name}: {audio.size / SAMPLE_RATE:7.2f} s, {len(windows)} utterances")
        sentences_in_set = {r.sentence_id for r in pools[set_name]}
        used_sentences = {w["fleurs_id"] for c in clips for w in c["utterances"]}
        manifests[set_name] = {
            "set": set_name,
            "source": {
                "dataset": HF_REPO,
                "language": "da_dk",
                "splits": list(SPLITS),
                "revision": HF_REVISION,
            },
            "seed": SEED,
            "sample_rate": SAMPLE_RATE,
            "format": "WAV PCM16 mono",
            "recipe": "codictate src/bun/utils/whisper/engines/long-audio.manual.ts buildLongClip; see README.md",
            "summary": {
                "sentences_assigned": len(sentences_in_set),
                "sentences_used": len(used_sentences),
                "utterances_used": sum(len(c["utterances"]) for c in clips),
                "total_duration_s": round(sum(c["duration_s"] for c in clips), 3),
            },
            "clips": clips,
        }
    find_ids = {w["fleurs_id"] for c in manifests["find"]["clips"] for w in c["utterances"]}
    held_ids = {w["fleurs_id"] for c in manifests["heldout"]["clips"] for w in c["utterances"]}
    find_txt = {w["text"] for c in manifests["find"]["clips"] for w in c["utterances"]}
    held_txt = {w["text"] for c in manifests["heldout"]["clips"] for w in c["utterances"]}
    assert not (find_ids & held_ids) and not (find_txt & held_txt), "sentence in both sets"
    for s in SETS:
        print(f"{s}: {json.dumps(manifests[s]['summary'])}")
    if manifests_dir is not None:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for s in SETS:
            (manifests_dir / f"{s}.json").write_text(
                json.dumps(manifests[s], indent=2, ensure_ascii=False) + "\n", "utf-8"
            )
    return manifests


def check() -> None:
    """Rebuild into a temp dir and compare every WAV's sha256 with the committed manifests."""
    with tempfile.TemporaryDirectory() as tmp:
        built = build(Path(tmp), None)
    bad = 0
    for s in SETS:
        committed = json.loads((MANIFESTS / f"{s}.json").read_text("utf-8"))
        if committed != built[s]:
            print(f"{s}: manifest differs from a fresh build")
            bad += 1
        for c in committed["clips"]:
            on_disk = REPO / c["path"]
            if on_disk.exists() and hashlib.sha256(on_disk.read_bytes()).hexdigest() != c["sha256"]:
                print(f"{c['path']}: sha256 differs from manifest")
                bad += 1
    print("check: OK" if bad == 0 else f"check: {bad} mismatches")
    sys.exit(1 if bad else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["download", "build", "check"])
    a = p.parse_args()
    if a.command == "download":
        download()
    elif a.command == "build":
        build(CLIPS, MANIFESTS)
    else:
        check()


if __name__ == "__main__":
    main()
