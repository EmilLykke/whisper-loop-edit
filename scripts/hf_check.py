"""Release check: does E3 change long-form decoding in Hugging Face transformers, not only in crispasr?

    .venv/bin/python -I scripts/hf_check.py --model baseline|E3 --config fallback|nofallback [--clips 10]

Decodes the first N Held-out Long Clips (data/manifests/heldout.json order) whole, with
transformers' sequential long-form `generate` (Danish, return_timestamps, no prompt, no VAD,
greedy), in one of two configs:

  fallback    OpenAI / HF's recommended long-form settings: condition_on_prev_tokens=True,
              temperature (0.0, 0.2, ..., 1.0), compression_ratio_threshold=1.35,
              logprob_threshold=-1.0, no_speech_threshold=0.6.
  nofallback  condition_on_prev_tokens=True, temperature 0 only, no thresholds. The closest
              transformers gets to whisper.cpp's behaviour on a Loop.

Writes, per clip, crispasr-shaped files so Phase 0's scoreRun reads them unchanged:
  results/hf_check/transcripts/<config>/<model>/hf/<clip>.run1.{json,txt,meta.json}
Then `bun scripts/hf_check_measure.ts` scores them.
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"baseline": ROOT / "models/hf/whisper-large-v3", "E3": ROOT / "models/hf/edits/E3"}
OUT = ROOT / "results/hf_check/transcripts"
CONFIGS = {
    "fallback": dict(condition_on_prev_tokens=True, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     compression_ratio_threshold=1.35, logprob_threshold=-1.0, no_speech_threshold=0.6),
    "nofallback": dict(condition_on_prev_tokens=True, temperature=0.0),
}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--clips", type=int, default=10)
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.generation import utils as gen_utils

    # transformers 5.22 on MPS defers the stop check and then reads `cache.layers`, which an
    # EncoderDecoderCache lacks. Same workaround as hf_longform.py: it changes no decision.
    if hasattr(gen_utils, "DeferredStopCheck"):
        gen_utils.DeferredStopCheck.is_supported = staticmethod(lambda *a, **k: False)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    mdir = MODELS[args.model]
    proc = WhisperProcessor.from_pretrained(str(mdir))
    model = WhisperForConditionalGeneration.from_pretrained(str(mdir), torch_dtype=dtype).to(device).eval()

    clips = json.loads((ROOT / "data/manifests/heldout.json").read_text())["clips"][: args.clips]
    out_dir = OUT / args.config / args.model / "hf"
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(language="da", task="transcribe", return_timestamps=True, return_segments=True,
              num_beams=1, **CONFIGS[args.config])
    for clip in clips:
        base = out_dir / f"{clip['name']}.run1"
        if Path(f"{base}.meta.json").exists():
            continue
        audio = read_wav(ROOT / clip["path"])
        feats = proc(audio, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest",
                     return_attention_mask=True)
        x = feats.input_features.to(device, dtype)
        am = feats.attention_mask.to(device)
        if x.shape[-1] < 3000:
            x = torch.nn.functional.pad(x, (0, 3000 - x.shape[-1]))
            am = torch.nn.functional.pad(am, (0, 3000 - am.shape[-1]))
        t0 = time.perf_counter()
        error = None
        try:
            with torch.inference_mode():
                out = model.generate(x, attention_mask=am, **kw)
            segs = []
            for s in out["segments"][0]:
                toks = [int(t) for t in s["tokens"]]
                segs.append({"from": round(float(s["start"]) * 1000), "to": round(float(s["end"]) * 1000),
                             "text": " " + proc.tokenizer.decode(toks, skip_special_tokens=True).strip()})
        except Exception as e:  # recorded, never hidden
            error, segs = repr(e), []
        wall = round(time.perf_counter() - t0, 1)
        Path(f"{base}.json").write_text(json.dumps(
            {"transcription": [{"offsets": {"from": s["from"], "to": s["to"]}, "text": s["text"]} for s in segs]},
            ensure_ascii=False, indent=1))
        Path(f"{base}.txt").write_text(" ".join(s["text"].strip() for s in segs if s["text"].strip()) + "\n")
        Path(f"{base}.meta.json").write_text(json.dumps(
            {"engine": "transformers", "model": args.model, "config": args.config, "clip": clip["name"],
             "generate_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
             "device": device, "dtype": str(dtype), "error": error, "wallSeconds": wall}, indent=1))
        print(json.dumps({"model": args.model, "config": args.config, "clip": clip["name"], "seconds": wall,
                          "segments": len(segs), "error": error}), flush=True)


if __name__ == "__main__":
    main()
