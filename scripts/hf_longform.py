"""Long-form decoding of a whole Long Clip with large-v3 in HF/PyTorch, two ways.

  --engine hf   transformers' own sequential long-form `generate` (return_timestamps,
                condition_on_prev_tokens, temperature fallback, logprob / no-speech thresholds,
                optional compression-ratio threshold, optional prompt_ids). The closest a stock
                HF user gets to crispasr's settings.
  --engine cpp  whisper.cpp's `whisper_full_with_state` loop re-implemented on the HF model
                (scripts/hf_window.py): whole-recording mel, seek by the last timestamp,
                prompt_past capped at 223 tokens, the user prompt carried in prompt_past,
                temperature ladder 0, 0.2, ... with the prompt dropped from 0.5, entropy (last 32
                tokens) and avg-logprob checks, whisper.cpp's sampled beam search with its
                per-decoder std::mt19937 streams, VAD-filtered audio and time mapping.

    .venv/bin/python -I scripts/hf_longform.py --engine cpp --path timestamps --clip find_da_5min_04
    .venv/bin/python -I scripts/hf_longform.py --engine hf --path timestamps --clip find_da_5min_04 --beams 5

`--path` picks crispasr's Decode Path settings (VAD + beam 5 + prompt for `timestamps`; no VAD,
greedy, no prompt for `timestamps-bare`; `nt` adds <|notimestamps|>). Output goes to
results/phase0/hf_repro/longform/<engine>_<path>_<clip>[_tag].json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_window import (BEG, CHUNK_FRAMES, EOT, HOP, MAX_PROMPT_CTX, N_FFT, PATHS, PREV, PROMPT_TEXT, REPRO_DIR,
                       SR, TRANSCRIPTS, CppRng, Whisper, build_prompt, case_inputs, log_mel_whispercpp,
                       map_to_original)

OUT_DIR = REPRO_DIR / "longform"
TEMPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def crispasr_segments(path: str, clip: str) -> list[dict]:
    js = json.loads((TRANSCRIPTS / path / f"{clip}.run1.json").read_text())
    return [{"from_s": s["offsets"]["from"] / 1000, "to_s": s["offsets"]["to"] / 1000, "text": s["text"].strip()}
            for s in js["transcription"]]


def repeats(segments: list[dict]) -> list[dict]:
    """Codictate's repeatedParts idea: a segment of 3+ words identical to the one before."""
    out = []
    for a, b in zip(segments, segments[1:]):
        if a["text"] == b["text"] and len(a["text"].split()) >= 3:
            out.append({"at_s": b["from_s"], "text": b["text"]})
    return out


def most_repeated(segments: list[dict]) -> dict:
    counts = {}
    for s in segments:
        if len(s["text"].split()) >= 3:
            counts[s["text"]] = counts.get(s["text"], 0) + 1
    if not counts:
        return {"text": None, "count": 0}
    text, n = max(counts.items(), key=lambda kv: kv[1])
    first = next(s["from_s"] for s in segments if s["text"] == text)
    return {"text": text, "count": n, "first_s": first}


def run_cpp(W: Whisper, path: str, clip: str, rng_offset: int = 0, log=print) -> dict:
    cfg, _, audio, mapping = case_inputs(path, clip)
    nt = cfg["no_timestamps"]
    mel = log_mel_whispercpp(audio)
    seek_end = 1 + (len(audio) + 200 - N_FFT) // HOP
    # decoder 0's stream is created with the state (seed 0), decoders 1..4 per whisper_full (seed j)
    rngs = [CppRng(rng_offset + j) for j in range(5)]
    past = W.prompt_tokens() if cfg["prompt"] else []
    seek = 0
    windows, segments = [], []
    while seek + 10 < seek_end:
        enc = W.encode(mel, seek)
        if seek > 0 and seek + 500 >= seek_end:
            past = []
        tried = []
        for it, T in enumerate(TEMPS):
            prompt = build_prompt(past if T < 0.5 else [], nt)
            strat = ("cppbeam" if cfg["beam"] > 1 else "greedy") if T == 0 else "greedy"
            r = W.decode_window(enc, prompt, strategy=strat, beam=5, temperature=T, rngs=rngs, seek=seek,
                                seek_end=seek_end, no_timestamps=nt)
            fail = r["failed"] or (r["avg_logprobs"] < -1.0 and r["no_speech_prob"] < 0.6)
            tried.append({"T": T, "avg_logprobs": r["avg_logprobs"], "failed": r["failed"],
                          "failed_entropy": any(d.get("failed_entropy") for d in r["decoders"]),
                          "text": W.decode(r["tokens"])})
            if not fail or it == len(TEMPS) - 1:
                break
        toks = r["tokens"]
        no_speech = r["no_speech_prob"] > 0.6 and r["avg_logprobs"] < -1.0
        new_past = list(prompt[1:len(prompt) - (4 if nt else 3)]) if prompt[0] == PREV else []
        if not no_speech:
            new_past += toks
        past = new_past
        # segments (crispasr v0.8.41: a segment starts at the last timestamp of the run before it)
        if not no_speech and toks:
            if nt:
                segments.append({"from_s": map_to_original(seek, mapping) / 100,
                                 "to_s": map_to_original(seek + CHUNK_FRAMES, mapping) / 100,
                                 "text": W.decode(toks, special=False).strip()})
            else:
                t0, run = seek, []
                for k, t in enumerate(toks):
                    if t >= BEG:
                        if run:
                            t1 = seek + 2 * (t - BEG)
                            segments.append({"from_s": map_to_original(t0, mapping) / 100,
                                             "to_s": map_to_original(t1, mapping) / 100,
                                             "text": W.decode(run, special=False).strip()})
                            run = []
                        t0 = seek + 2 * (t - BEG)
                    elif t < EOT:
                        run.append(t)
                if run:
                    segments.append({"from_s": map_to_original(t0, mapping) / 100,
                                     "to_s": map_to_original(seek + r["seek_delta"], mapping) / 100,
                                     "text": W.decode(run, special=False).strip()})
        seek_delta = r["seek_delta"]
        if not nt and len(toks) > 1 and toks[-2] < BEG and toks[-1] > BEG:
            seek_delta = min(seek_end - seek, CHUNK_FRAMES)  # single timestamp ending: skip the chunk
        windows.append({"seek_s": seek / 100, "orig_start_s": map_to_original(seek, mapping) / 100,
                        "T": T, "tried": tried, "tokens": toks, "text": W.decode(toks),
                        "n_prompt": len(prompt), "seek_delta": seek_delta})
        log(json.dumps({"seek_s": seek / 100, "orig_s": round(map_to_original(seek, mapping) / 100, 2), "T": T,
                        "n": len(toks), "text": W.decode(toks, special=False)[:140]}, ensure_ascii=False))
        seek += seek_delta
    return {"windows": windows, "segments": segments}


def run_hf(W: Whisper, path: str, clip: str, beams: int, prompt: bool, crt: float | None,
           condition: bool = True) -> dict:
    """transformers' sequential long-form generate on the same audio buffer crispasr decoded."""
    from transformers import WhisperProcessor
    from transformers.generation import utils as gen_utils

    # transformers 5.22 on MPS defers the stop check and then reads `cache.layers`, which an
    # EncoderDecoderCache lacks (AttributeError at the end of every greedy window). Use the
    # plain stop check; it changes no decision, only when the loop notices it is done.
    gen_utils.DeferredStopCheck.is_supported = staticmethod(lambda *a, **k: False)

    _, _, audio, mapping = case_inputs(path, clip)
    proc = WhisperProcessor.from_pretrained(str(Path(W.model.config._name_or_path)))
    feats = proc(audio, sampling_rate=SR, return_tensors="pt", truncation=False, padding="longest",
                 return_attention_mask=True)
    kw = dict(language="da", task="transcribe", return_timestamps=True, condition_on_prev_tokens=condition,
              temperature=TEMPS, logprob_threshold=-1.0, no_speech_threshold=0.6,
              compression_ratio_threshold=crt, num_beams=beams, return_segments=True)
    if prompt:
        kw["prompt_ids"] = torch.tensor(proc.get_prompt_ids(PROMPT_TEXT), device=W.device)
    x = feats.input_features.to(W.device, W.dtype)
    am = feats.attention_mask.to(W.device)
    if x.shape[-1] < 3000:
        x = torch.nn.functional.pad(x, (0, 3000 - x.shape[-1]))
        am = torch.nn.functional.pad(am, (0, 3000 - am.shape[-1]))
    with torch.inference_mode():
        out = W.model.generate(x, attention_mask=am, **kw)
    segs = []
    for s in out["segments"][0]:
        a = float(s["start"]) * 100
        b = float(s["end"]) * 100
        segs.append({"from_s": map_to_original(a, mapping) / 100 if mapping else a / 100,
                     "to_s": map_to_original(b, mapping) / 100 if mapping else b / 100,
                     "text": W.decode([int(t) for t in s["tokens"]], special=False).strip()})
    return {"segments": segs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("cpp", "hf"), required=True)
    ap.add_argument("--path", choices=list(PATHS), required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--beams", type=int, default=None, help="hf: num_beams (default: the path's)")
    ap.add_argument("--prompt", choices=("auto", "on", "off"), default="auto", help="hf: prompt_ids")
    ap.add_argument("--crt", type=float, default=None, help="hf: compression_ratio_threshold (default off)")
    ap.add_argument("--rng-offset", type=int, default=0, help="cpp: shift the decoders' mt19937 seeds")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    W = Whisper(dtype=args.dtype)
    cfg = PATHS[args.path]
    t0 = time.perf_counter()
    if args.engine == "cpp":
        res = run_cpp(W, args.path, args.clip, args.rng_offset)
        settings = {"rng_offset": args.rng_offset}
    else:
        beams = args.beams or cfg["beam"]
        prompt = cfg["prompt"] if args.prompt == "auto" else args.prompt == "on"
        res = run_hf(W, args.path, args.clip, beams, prompt, args.crt)
        settings = {"num_beams": beams, "prompt_ids": prompt, "compression_ratio_threshold": args.crt,
                    "temperature": TEMPS, "logprob_threshold": -1.0, "no_speech_threshold": 0.6,
                    "condition_on_prev_tokens": True, "return_timestamps": True,
                    "audio": "VAD-filtered buffer (crispasr's segments)" if cfg["vad"] else "whole clip"}
    res["crispasr"] = crispasr_segments(args.path, args.clip)
    for k in ("segments", "crispasr"):
        res[f"{k}_repeats"] = repeats(res[k])
        res[f"{k}_most_repeated"] = most_repeated(res[k])
    res["same_segment_texts_as_crispasr"] = [s["text"] for s in res["segments"]] == [s["text"] for s in
                                                                                     res["crispasr"]]
    res.update({"engine": args.engine, "path": args.path, "clip": args.clip, "dtype": args.dtype,
                "settings": settings, "seconds": round(time.perf_counter() - t0, 1)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.engine}_{args.path}_{args.clip}" + (f"_{args.tag}" if args.tag else "")
    (OUT_DIR / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({"name": name, "seconds": res["seconds"], "same_as_crispasr": res["same_segment_texts_as_crispasr"],
                      "most_repeated": res["segments_most_repeated"],
                      "crispasr_most_repeated": res["crispasr_most_repeated"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
