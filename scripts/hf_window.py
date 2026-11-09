"""Whisper large-v3 in HF/PyTorch, driven the way whisper.cpp (crispasr) drives it.

One 30 s window at a time: build the decoder input whisper.cpp builds
(`<|startofprev|> prompt_past <|sot|> <|da|> <|transcribe|> [<|notimestamps|>]`), encode the
audio at a seek offset of a whole-recording log-mel (global clamp, 30 s zero tail, as
whisper.cpp computes it), then

  - decode it: `greedy`, `beam` (plain top-k beam search) or `cppbeam` (whisper.cpp's own
    "beam search", which *samples* k candidates per decoder from the filtered distribution
    with a per-decoder std::mt19937, then keeps the best by summed log-prob; emulated
    bit-for-bit on the RNG side, see `CppRng`),
  - teacher-force a continuation and read per-token log-probs (raw log-softmax and
    whisper.cpp's filtered log-probs),
  - dump hidden states / attentions at every position (`forward(..., hidden=True)`).

Logit filtering follows crispasr v0.8.29 `whisper_process_logits` (src/crispasr.cpp:6896):
suppress blank at start, no <|notimestamps|>/sot/nospeech/lm/task/lang/prev, timestamps in
pairs, max initial timestamp 1.0 s, timestamps non-decreasing, and "timestamps win if their
summed prob beats the best text token". Non-speech tokens are NOT suppressed (crispasr's
`suppress_nst` is off), unlike HF `generate`'s default `suppress_tokens`.

As a script, it reconstructs crispasr windows from a crispasr JSON and runs the Loop
experiments (see `results/phase0/hf_repro.md`):

    .venv/bin/python -I scripts/hf_window.py validate --path timestamps --clip find_da_5min_04
    .venv/bin/python -I scripts/hf_window.py repro    # all Loop cases -> results/phase0/hf_repro.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/hf/whisper-large-v3"
CODICTATE = Path(os.environ.get("CODICTATE_DIR", ROOT / "../codictate"))
MEL_FILTERS = CODICTATE / ".tmp/edda-mirror/whisper-src/whisper/assets/mel_filters.npz"
PROMPT_TEXT = "Hej, dette er en diktering. Jeg skriver i hele sætninger, med kommaer hvor jeg holder pause."

SR = 16000
N_FFT = 400
HOP = 160
CHUNK_FRAMES = 3000  # 30 s of mel frames (10 ms each); seek is in these units

EOT, SOT, DA, TRANSLATE, TRANSCRIBE, SOLM, PREV, NOSP, NOT, BEG = (
    50257, 50258, 50285, 50359, 50360, 50361, 50362, 50363, 50364, 50365)
LANG_FIRST, LANG_LAST = 50259, 50358
N_VOCAB = 51866
SPACE = 220
N_TEXT_CTX = 448
MAX_PROMPT_CTX = N_TEXT_CTX // 2  # min(n_max_text_ctx=16384, n_text_ctx/2)
N_MAX_STEPS = N_TEXT_CTX // 2 - 4  # whisper.cpp decode loop bound


def ts_token(seconds: float) -> int:
    return BEG + int(round(seconds / 0.02))


def ts_seconds(token: int) -> float:
    return (token - BEG) * 0.02


# --------------------------------------------------------------------------------------
# audio


def load_wav(path: Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32")
    assert rate == SR and audio.ndim == 1, path
    return audio


def vad_filter(audio: np.ndarray, segments: list[dict], overlap_s: float = 0.1):
    """whisper.cpp `whisper_vad` (crispasr.cpp:7423): speech segments (start/end in
    centiseconds, from crispasr's log) copied end to end, each but the last extended by
    `overlap_s`, with 0.1 s of zeros between them. Returns (samples, mapping) where mapping is
    whisper.cpp's sorted (processed_cs, original_cs) table."""
    n = len(audio)
    ov = int(overlap_s * SR)
    sil = int(0.1 * SR)
    out, mapping = [], []
    offset = 0
    for i, seg in enumerate(segments):
        s = min(seg["start"] * 160, n - 1)
        e = min(seg["end"] * 160, n - 1)
        orig_len = e - s
        if i < len(segments) - 1:
            e = min(e + ov, n - 1)
        if e - s <= 0:
            continue
        mapping.append((offset // 160, seg["start"]))
        mapping.append(((offset + orig_len) // 160, seg["end"]))
        out.append(audio[s:e])
        offset += e - s
        if i < len(segments) - 1:
            mapping.append((offset // 160, seg["end"]))
            mapping.append(((offset + sil) // 160, segments[i + 1]["start"]))
            out.append(np.zeros(sil, dtype=np.float32))
            offset += sil
    mapping.sort(key=lambda m: m[0])
    dedup = []
    for m in mapping:
        if not dedup or dedup[-1][0] != m[0]:
            dedup.append(m)
    return np.concatenate(out), dedup


def map_to_original(t_cs: int, mapping) -> float:
    """whisper.cpp map_processed_to_original_time: linear interpolation in the table."""
    if not mapping:
        return t_cs
    if t_cs <= mapping[0][0]:
        return mapping[0][1]
    if t_cs >= mapping[-1][0]:
        return mapping[-1][1]
    lo = 0
    hi = len(mapping) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mapping[mid][0] <= t_cs:
            lo = mid
        else:
            hi = mid
    (p0, o0), (p1, o1) = mapping[lo], mapping[hi]
    if p1 == p0:
        return o0
    return o0 + (t_cs - p0) * (o1 - o0) / (p1 - p0)


def map_to_processed(t_orig_cs: float, mapping) -> float:
    """Inverse of map_to_original (monotone piecewise-linear)."""
    if not mapping:
        return t_orig_cs
    for (p0, o0), (p1, o1) in zip(mapping, mapping[1:]):
        if o0 <= t_orig_cs <= o1:
            return p0 if o1 == o0 else p0 + (t_orig_cs - o0) * (p1 - p0) / (o1 - o0)
    return mapping[0][0] if t_orig_cs < mapping[0][1] else mapping[-1][0]


def log_mel_whispercpp(samples: np.ndarray) -> np.ndarray:
    """crispasr `log_mel_spectrogram` (crispasr.cpp:3729): 200-sample reflect pad at the start,
    30 s + 200 zeros at the end, periodic Hann, |FFT|^2, mel, log10(max(x,1e-10)), clamp to
    global max - 8, (x+4)/4. Returns (128, n_len) float32; n_len includes the 30 s tail."""
    filters = np.load(MEL_FILTERS)["mel_128"].astype(np.float64)
    pad2 = N_FFT // 2
    n = len(samples)
    padded = np.zeros(n + SR * 30 + 2 * pad2, dtype=np.float64)
    padded[pad2:pad2 + n] = samples
    padded[:pad2] = samples[1:1 + pad2][::-1]
    n_len = (len(padded) - N_FFT) // HOP
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_len)[:, None]
    mel = np.empty((128, n_len), dtype=np.float64)
    step = 4096
    for a in range(0, n_len, step):
        fr = padded[idx[a:a + step]] * hann
        power = np.abs(np.fft.rfft(fr, axis=1)) ** 2  # (frames, 201)
        mel[:, a:a + step] = np.log10(np.maximum(power @ filters.T, 1e-10)).T
    mmax = mel.max() - 8.0
    mel = (np.maximum(mel, mmax) + 4.0) / 4.0
    return mel.astype(np.float32)


# --------------------------------------------------------------------------------------
# whisper.cpp tokenizer (for the --prompt string)

_BYTE_ENCODER = None


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes_table(tokenizer) -> dict[bytes, int]:
    dec = {v: k for k, v in _bytes_to_unicode().items()}
    table = {}
    for tok, i in tokenizer.get_vocab().items():
        if i >= EOT:
            continue
        try:
            table[bytes(dec[c] for c in tok)] = i
        except KeyError:
            pass
    return table


# std::regex in the classic "C" locale: [[:alpha:]] and [[:digit:]] are ASCII only, so the
# UTF-8 bytes of æ/ø/å fall into the punctuation branch.
_CPP_SPLIT = re.compile(
    rb"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def tokenize_whispercpp(text: str, table: dict[bytes, int]) -> list[int]:
    """crispasr `tokenize` (crispasr.cpp:3825): regex split, then greedy longest match."""
    out = []
    for m in _CPP_SPLIT.finditer(text.encode("utf-8")):
        word = m.group(0)
        i = 0
        while i < len(word):
            for j in range(len(word), i, -1):
                if word[i:j] in table:
                    out.append(table[word[i:j]])
                    i = j
                    break
            else:
                i += 1
    return out


# --------------------------------------------------------------------------------------
# whisper.cpp's RNG: std::mt19937 + libc++ std::discrete_distribution


class CppRng:
    """std::mt19937(seed) feeding libc++ discrete_distribution<int>(probs).

    libc++: weights -> double, normalised, partial sums without the last element;
    draw u = generate_canonical<double,53>(g) = (g() + g()*2^32) / 2^64; index =
    upper_bound(partial, u)."""

    def __init__(self, seed: int):
        self.bg = np.random.MT19937()
        self.bg._legacy_seeding(seed)  # == init_genrand(seed) == std::mt19937(seed)
        self.n_draws = 0

    def canonical(self) -> float:
        lo, hi = self.bg.random_raw(2)
        self.n_draws += 1
        return (float(lo) + float(hi) * 4294967296.0) / 18446744073709551616.0

    def discrete(self, probs: np.ndarray) -> int:
        w = probs.astype(np.float64)
        w = w / w.sum()
        partial = np.cumsum(w[:-1])
        return int(np.searchsorted(partial, self.canonical(), side="right"))


# --------------------------------------------------------------------------------------
# logit filtering, whisper.cpp style


@dataclass
class DecState:
    tokens: list = field(default_factory=list)  # list of (id, plog)
    has_ts: bool = False
    seek_delta: int = CHUNK_FRAMES
    result_len: int = 0
    sum_logprobs_all: float = 0.0
    completed: bool = False
    failed: bool = False

    def copy(self):
        return DecState(list(self.tokens), self.has_ts, self.seek_delta, self.result_len,
                        self.sum_logprobs_all, self.completed, self.failed)


def filter_logits(logits: np.ndarray, toks: list[int], has_ts: bool, seek_delta: int,
                  temperature: float = 0.0, no_timestamps: bool = False, max_initial_ts: float = 1.0):
    """Returns (logprobs, probs) exactly as whisper_process_logits leaves them (float64 here).
    `logits` is the raw float32 row. `toks` are the tokens decoded so far in this window."""
    lg = logits.astype(np.float64).copy()
    if temperature > 0:
        lg /= temperature
    ninf = -np.inf
    is_initial = len(toks) == 0
    if is_initial:
        lg[EOT] = ninf
        lg[SPACE] = ninf
    lg[NOT] = ninf
    if no_timestamps:
        lg[BEG:] = ninf
    lg[SOT] = ninf
    lg[NOSP] = ninf
    lg[SOLM] = ninf
    lg[TRANSLATE] = ninf
    lg[TRANSCRIBE] = ninf
    lg[PREV] = ninf
    lg[LANG_FIRST:LANG_LAST + 1] = ninf
    last_ts = len(toks) > 0 and toks[-1] >= BEG
    pen_ts = len(toks) < 2 or toks[-2] >= BEG
    if last_ts:
        if pen_ts:
            lg[BEG:] = ninf
        else:
            lg[:EOT] = ninf
    if is_initial and max_initial_ts > 0:
        tid0 = int(round(max_initial_ts / (30.0 / 1500)))
        lg[BEG + tid0 + 1:] = ninf
    if has_ts:
        lg[BEG:BEG + seek_delta // 2] = ninf
    finite = np.isfinite(lg)
    m = lg[finite].max()
    lse = math.log(np.exp(lg[finite] - m).sum()) + m
    logprobs = np.where(finite, lg - lse, ninf)
    ts_lp = logprobs[BEG:]
    ts_f = np.isfinite(ts_lp)
    if ts_f.any():
        mm = ts_lp[ts_f].max()
        ts_logprob = math.log(np.exp(ts_lp[ts_f] - mm).sum()) + mm
    else:
        ts_logprob = ninf
    if ts_logprob > logprobs[:BEG].max():
        lg[:BEG] = ninf
        logprobs[:BEG] = ninf
    probs = np.where(np.isfinite(lg), np.exp(logprobs), 0.0)
    return logprobs, probs


def raw_logprobs(logits: np.ndarray) -> np.ndarray:
    lg = logits.astype(np.float64)
    m = lg.max()
    return lg - (math.log(np.exp(lg - m).sum()) + m)


def entropy_last32(ids: list[int]) -> float:
    tail = ids[-32:]
    if not tail:
        return 0.0
    _, counts = np.unique(tail, return_counts=True)
    p = counts / len(tail)
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------------------
# model


class Whisper:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None, dtype: str = "float16"):
        from transformers import WhisperForConditionalGeneration, WhisperTokenizer

        torch.set_grad_enabled(False)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=self.dtype).to(self.device)
        self.model.eval()
        self.tok = WhisperTokenizer.from_pretrained(model_dir)
        self.table = token_bytes_table(self.tok)
        self._mel_cache = {}

    # ---- text
    def decode(self, ids, special=True) -> str:
        """Text of `ids`; timestamps shown as <|t.tt|> (special=True) or dropped."""
        out, run = [], []
        for i in map(int, ids):
            if i >= BEG or (i >= EOT and special):
                if run:
                    out.append(self.tok.decode(run))
                    run = []
                if special:
                    out.append(f"<|{ts_seconds(i):.2f}|>" if i >= BEG else self.tok.convert_ids_to_tokens(i))
            elif i < EOT:
                run.append(i)
        if run:
            out.append(self.tok.decode(run))
        return "".join(out)

    def prompt_tokens(self, text: str = PROMPT_TEXT) -> list[int]:
        return tokenize_whispercpp(text, self.table)

    # ---- audio
    def encode(self, mel: np.ndarray, seek: int) -> torch.Tensor:
        win = np.zeros((128, CHUNK_FRAMES), dtype=np.float32)
        i0 = min(seek, mel.shape[1])
        i1 = min(seek + CHUNK_FRAMES, mel.shape[1])
        win[:, : i1 - i0] = mel[:, i0:i1]
        x = torch.from_numpy(win)[None].to(self.device, self.dtype)
        with torch.inference_mode():
            return self.model.model.encoder(x).last_hidden_state

    # ---- decoder
    def forward(self, enc: torch.Tensor, ids: list[int], hidden: bool = False, attentions: bool = False):
        """One teacher-forced pass. Returns logits (len(ids), V) float32 numpy, and optionally the
        decoder outputs (hidden states per layer, self/cross attentions) as torch tensors."""
        x = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            out = self.model.model.decoder(input_ids=x, encoder_hidden_states=enc, use_cache=False,
                                           output_hidden_states=hidden, output_attentions=attentions)
            logits = self.model.proj_out(out.last_hidden_state)[0].float().cpu().numpy()
        return (logits, out) if (hidden or attentions) else logits

    def _step(self, ids: torch.Tensor, enc: torch.Tensor, cache):
        with torch.inference_mode():
            out = self.model.model.decoder(input_ids=ids, encoder_hidden_states=enc, past_key_values=cache,
                                           use_cache=True)
            logits = self.model.proj_out(out.last_hidden_state[:, -1]).float().cpu().numpy()
        return logits, out.past_key_values

    def decode_window(self, enc: torch.Tensor, prompt: list[int], strategy: str = "greedy", beam: int = 5,
                      temperature: float = 0.0, rngs: list | None = None, seek: int = 0,
                      seek_end: int = 10 ** 9, no_timestamps: bool = False, max_steps: int = N_MAX_STEPS,
                      prefix: list | None = None) -> dict:
        """Decode one window from `prompt` (the full decoder input, ending in the task tokens).

        strategy: greedy (argmax; or one sample per decoder when temperature > 0, `beam`
        decoders = best_of), beam (true top-k beam search, no sampling), cppbeam (whisper.cpp's
        sampled beam search, needs `rngs`, one CppRng per decoder).
        Returns whisper.cpp's per-window result for the best decoder plus all decoders."""
        n_dec = 1 if (strategy == "greedy" and temperature == 0) else beam
        prefix = list(prefix or [])  # [(id, plog)] forced at the start of the window
        with torch.inference_mode():
            out = self.model.model.decoder(input_ids=torch.tensor([prompt], device=self.device),
                                           encoder_hidden_states=enc, use_cache=True)
            logits_p = self.model.proj_out(out.last_hidden_state[:, -1]).float().cpu().numpy()
            cache = out.past_key_values
        no_speech_prob = float(np.exp(raw_logprobs(logits_p[0])[NOSP]))
        if prefix:
            logits0, cache = self._step(torch.tensor([[t for t, _ in prefix]], device=self.device), enc, cache)
        else:
            logits0 = logits_p
        d0 = DecState()
        for k, (t, lp) in enumerate(prefix):
            d0.tokens.append((t, lp))
            d0.sum_logprobs_all += lp
            if t > BEG:
                d0.has_ts, d0.seek_delta, d0.result_len = True, 2 * (t - BEG), k + 1
        decs = [d0.copy() for _ in range(n_dec)]
        if n_dec > 1:
            cache.batch_repeat_interleave(n_dec)
        cur_logits = np.repeat(logits0, n_dec, axis=0)
        delta_min = 10
        for i in range(len(prefix), max_steps):
            filt = []
            for j, d in enumerate(decs):
                if d.completed or d.failed:
                    filt.append(None)
                    continue
                ids = [t for t, _ in d.tokens]
                filt.append(filter_logits(cur_logits[j], ids, d.has_ts, d.seek_delta, temperature, no_timestamps))
            if strategy in ("beam", "cppbeam") and temperature == 0:
                cands = []
                for j, d in enumerate(decs):
                    if filt[j] is None:
                        continue
                    lp, pr = filt[j]
                    if strategy == "cppbeam":
                        picks = [rngs[j].discrete(pr) for _ in range(beam)]
                    else:
                        picks = list(np.argsort(-lp, kind="stable")[:beam])
                    for t in picks:
                        nd = d.copy()
                        nd.tokens.append((int(t), float(lp[t])))
                        nd.sum_logprobs_all += float(lp[t])
                        cands.append((j, nd))
                cands.sort(key=lambda c: (-c[1].sum_logprobs_all, c[0]))
                src = list(range(n_dec))
                cur_c = 0
                for j in range(n_dec):
                    if decs[j].completed or decs[j].failed:
                        continue
                    if cur_c >= len(cands):
                        cur_c = 0
                    cj, cd = cands[cur_c]
                    cur_c += 1
                    while (len(cands) > cur_c and i > 0
                           and [t for t, _ in cands[cur_c][1].tokens] == [t for t, _ in cd.tokens]):
                        cur_c += 1
                    decs[j] = cd.copy()
                    src[j] = cj
                if src != list(range(n_dec)):
                    cache.reorder_cache(torch.tensor(src, device=self.device))
            else:
                for j, d in enumerate(decs):
                    if filt[j] is None:
                        continue
                    lp, pr = filt[j]
                    if temperature == 0:
                        t = int(np.argmax(pr))
                    else:
                        t = rngs[j].discrete(pr)
                    d.tokens.append((t, float(lp[t])))
                    d.sum_logprobs_all += float(lp[t])
            # update decoder state (whisper.cpp: completion, seek_delta, failure rules)
            for d in decs:
                if d.completed or d.failed:
                    continue
                t = d.tokens[-1][0]
                if t > BEG:
                    new = 2 * (t - BEG)
                    if d.has_ts and d.seek_delta > new and d.result_len < i:
                        d.failed = True
                        continue
                    d.seek_delta = new
                    d.result_len = i + 1
                    d.has_ts = True
                if t == EOT or (d.has_ts and seek + d.seek_delta + delta_min >= seek_end):
                    if d.result_len == 0 and not no_timestamps:
                        if seek + d.seek_delta + delta_min >= seek_end:
                            d.result_len = i + 1
                        else:
                            d.failed = True
                            continue
                    if no_timestamps:
                        d.result_len = i + 1
                        d.seek_delta = CHUNK_FRAMES
                    d.completed = True
                    continue
                if i == max_steps - 1 and (d.result_len == 0 or d.seek_delta < CHUNK_FRAMES // 2):
                    d.failed = True
            if all(d.completed or d.failed for d in decs):
                break
            # next logits for the active decoders (inactive rows are fed a dummy token)
            nxt = torch.tensor([[d.tokens[-1][0] if d.tokens else EOT] for d in decs], device=self.device)
            cur_logits, cache = self._step(nxt, enc, cache)
        # rank (whisper_sequence_score, length_penalty = -1 -> divide by result_len)
        best, best_score = 0, -np.inf
        results = []
        for j, d in enumerate(decs):
            ids = [t for t, _ in d.tokens[: d.result_len]]
            lps = [lp for _, lp in d.tokens[: d.result_len]]
            r = {"tokens": ids, "result_len": d.result_len, "seek_delta": d.seek_delta,
                 "failed": d.failed, "all_tokens": [t for t, _ in d.tokens]}
            if not d.failed and d.result_len > 0:
                r["sum_logprobs"] = float(sum(lps))
                r["avg_logprobs"] = r["sum_logprobs"] / d.result_len
                r["entropy"] = entropy_last32(ids)
                r["score"] = r["avg_logprobs"]
                if d.result_len > 32 and r["entropy"] < 2.4:
                    r["failed"] = True
                    r["failed_entropy"] = True
                elif r["score"] > best_score:
                    best, best_score = j, r["score"]
            else:
                r["avg_logprobs"] = -np.inf
            results.append(r)
        out = dict(results[best])
        out["best_decoder"] = best
        out["no_speech_prob"] = no_speech_prob
        out["decoders"] = results
        return out

    def teacher_force(self, enc: torch.Tensor, prompt: list[int], cont: list[int], window_prefix: list[int] = (),
                      no_timestamps: bool = False, temperature: float = 0.0, logits: np.ndarray | None = None) -> dict:
        """Log-probs of `cont` given `prompt` + `window_prefix` (tokens already decoded in this
        window). Filtered log-probs replay whisper.cpp's rules token by token."""
        prefix = list(window_prefix)
        ids = list(prompt) + prefix + list(cont)
        if logits is None:
            logits = self.forward(enc, ids)
        start = len(prompt)
        has_ts, seek_delta = False, CHUNK_FRAMES
        toks = []
        rows = []
        for k, t in enumerate(prefix + list(cont)):
            row = logits[start + k - 1]
            raw = raw_logprobs(row)
            flp, fpr = filter_logits(row, toks, has_ts, seek_delta, temperature, no_timestamps)
            if k >= len(prefix):
                rows.append({
                    "id": int(t), "text": self.decode([t]),
                    "raw_logprob": float(raw[t]), "filt_logprob": float(flp[t]),
                    "p": float(fpr[t]),
                    "rank_raw": int((raw > raw[t]).sum()),
                    "argmax": int(np.argmax(fpr)), "argmax_text": self.decode([int(np.argmax(fpr))]),
                })
            toks.append(int(t))
            if t > BEG:
                has_ts = True
                seek_delta = 2 * (t - BEG)
        raw = [r["raw_logprob"] for r in rows]
        filt = [r["filt_logprob"] for r in rows]
        return {"n": len(rows), "sum_raw": float(sum(raw)), "mean_raw": float(np.mean(raw)) if raw else 0.0,
                "sum_filt": float(sum(filt)), "mean_filt": float(np.mean(filt)) if filt else 0.0,
                "tokens": rows}


def build_prompt(prompt_past: list[int], no_timestamps: bool = False, lang: int = DA) -> list[int]:
    """whisper.cpp: [<|startofprev|> + last (MAX_PROMPT_CTX-1) of prompt_past] + sot/lang/task [+ not]."""
    p = []
    if prompt_past:
        p = [PREV] + list(prompt_past[-(MAX_PROMPT_CTX - 1):])
    p += [SOT, lang, TRANSCRIBE]
    if no_timestamps:
        p.append(NOT)
    return p


# --------------------------------------------------------------------------------------
# reconstructing crispasr windows from its JSON

TRANSCRIPTS = ROOT / "results/phase0/transcripts"
REPRO_DIR = ROOT / "results/phase0/hf_repro"

PATHS = {
    # path: (vad, beam, prompt, no_timestamps)
    "timestamps": dict(vad=True, beam=5, prompt=True, no_timestamps=False),
    "timestamps-bare": dict(vad=False, beam=1, prompt=False, no_timestamps=False),
    "nt": dict(vad=False, beam=1, prompt=False, no_timestamps=True),
}


def parse_vad_log(stderr_text: str) -> list[dict]:
    segs = []
    for m in re.finditer(r"VAD segment \d+: start = ([\d.]+), end = ([\d.]+)", stderr_text):
        segs.append({"start": int(round(float(m.group(1)) * 100)), "end": int(round(float(m.group(2)) * 100))})
    return segs


def load_manifest(clip: str) -> dict:
    m = json.loads((ROOT / "data/manifests/find.json").read_text())
    return next(c for c in m["clips"] if c["name"] == clip)



def case_inputs(path: str, clip: str):
    """crispasr JSON, the decoded audio buffer (VAD-filtered on the timestamps path) and the
    VAD time mapping for one (path, clip)."""
    cfg = PATHS[path]
    js = json.loads((TRANSCRIPTS / path / f"{clip}.run1.json").read_text())
    audio = load_wav(ROOT / f"data/clips/find/{clip}.wav")
    mapping = []
    if cfg["vad"]:
        vad_segments = json.loads((REPRO_DIR / f"vad_{clip}.json").read_text())["segments"]
        audio, mapping = vad_filter(audio, vad_segments)
    return cfg, js, audio, mapping


def _proc_cs(ms: int, mapping) -> float:
    return map_to_processed(ms / 10, mapping) if mapping else ms / 10


def _p_match(tf_rows, cpp_p):
    d = [abs(r["p"] - p) for r, p in zip(tf_rows, cpp_p) if p is not None]
    return {"max_abs_dp": float(max(d)) if d else 0.0, "mean_abs_dp": float(np.mean(d)) if d else 0.0,
            "not_argmax": int(sum(1 for r, p in zip(tf_rows, cpp_p) if p is not None and r["argmax"] != r["id"]))}


def resolve_case(W: "Whisper", path: str, clip: str, temps=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), log=print) -> dict:
    """Rebuild every decode window crispasr ran for (path, clip), as decoder inputs for HF.

    crispasr's JSON gives each segment's tokens with their probability p, but not the
    timestamp that begins a segment (v0.8.41 writes it into the segment's `from` time, mapped
    through the VAD table) nor a trailing timestamp at a window's end. A segment opens a new
    window when it starts with a timestamp or its end time goes back. Seek candidates come
    from chaining (previous seek + its last timestamp, or + 30 s) and from the mapped times;
    begin timestamps are restored from the times and snapped to HF's most probable timestamp
    within +-0.06 s. Wherever crispasr's output leaves a choice (seek, trailing timestamp,
    temperature), the choice is the one whose HF teacher-forced p best matches crispasr's p.
    Temperature: whisper.cpp falls back to 0.2, 0.4, ... and drops the previous-text prompt
    from 0.5 on, so a window's p tells which pass produced it."""
    cfg, js, audio, mapping = case_inputs(path, clip)
    nt = cfg["no_timestamps"]
    mel = log_mel_whispercpp(audio)
    seek_end = 1 + (len(audio) + 200 - N_FFT) // HOP
    segs = js["transcription"]

    def end_ts(s):
        ts = [t["id"] for t in s["tokens"] if t["id"] >= BEG]
        return ts[-1] if ts else None

    groups = []
    for k, s in enumerate(segs):
        new = nt or not groups or s["tokens"][0]["id"] >= BEG or (end_ts(s) or 0) < (end_ts(segs[k - 1]) or 0)
        if new:
            groups.append([k])
        else:
            groups[-1].append(k)

    def build_tokens(g, seek):
        toks, cpp_p, restored = [], [], []
        first = segs[g[0]]
        if not nt and first["tokens"][0]["id"] < BEG:
            # crispasr drops a window's leading timestamp unless it is <|0.00|> (its segment
            # loop tests `id > token_beg`) and writes it into the segment's `from` instead
            lead = int(round((_proc_cs(first["offsets"]["from"], mapping) - seek) / 2))
            if lead > 0:
                toks.append(BEG + min(lead, 50))  # max_initial_ts = 1.0 s
                cpp_p.append(None)
                restored.append(0)
        for n, k in enumerate(g):
            s = segs[k]
            if n > 0 and not nt:
                f = _proc_cs(s["offsets"]["from"], mapping)
                toks.append(min(BEG + 1500, max(toks[-1], BEG + int(round((f - seek) / 2)))))
                cpp_p.append(None)
                restored.append(len(toks) - 1)
            toks += [t["id"] for t in s["tokens"]]
            cpp_p += [t["p"] for t in s["tokens"]]
        return toks, cpp_p, restored

    def restore(enc, prompt, toks, restored):
        for _ in range(4):
            logits = W.forward(enc, prompt + toks + ([] if nt else [EOT]))
            changed = False
            for pos in restored:
                row = logits[len(prompt) + pos - 1]
                prev_ts = [2 * (t - BEG) for t in toks[:pos] if t > BEG]
                lp, _ = filter_logits(row, toks[:pos], bool(prev_ts), max(prev_ts or [CHUNK_FRAMES]))
                lo = max(toks[pos - 1] if pos > 0 else BEG, toks[pos] - 3)
                hi = min(BEG + (50 if pos == 0 else 1500), toks[pos] + 3)
                pick = lo + int(np.argmax(lp[lo:hi + 1]))
                if pick != toks[pos]:
                    toks[pos] = pick
                    changed = True
            if not changed:
                return logits
        return logits

    init = W.prompt_tokens() if cfg["prompt"] else []
    windows = []
    prev = None
    for w, g in enumerate(groups):
        first = segs[g[0]]
        text_start = not nt and first["tokens"][0]["id"] < BEG
        # ---- seek candidates
        if nt:
            cands = {first["offsets"]["to"] // 10 - CHUNK_FRAMES: "segment end - 30 s"}
        else:
            cands = {}
            if not text_start:
                mapped = _proc_cs(first["offsets"]["from"], mapping) - 2 * (first["tokens"][0]["id"] - BEG)
            if prev is None:
                cands[0 if text_start else int(round(mapped))] = "first window"
            else:
                last = 2 * (prev["tokens"][-1] - BEG)
                cands[prev["seek"] + last] = "previous seek + last timestamp"
                cands.setdefault(prev["seek"] + CHUNK_FRAMES, "previous seek + 30 s")
                if not text_start:
                    cands.setdefault(prev["seek"] + 2 * int(round((mapped - prev["seek"]) / 2)), "mapped")
        base_past = []
        if prev is not None:
            base_past = (prev["past"][-(MAX_PROMPT_CTX - 1):] if prev["prompt_used"] else []) + prev["tokens"]
        elif init:
            base_past = list(init)
        scored = []
        for seek, why in cands.items():
            past = [] if (seek > 0 and seek + 500 >= seek_end) else base_past
            enc = W.encode(mel, seek)
            toks, cpp_p, restored = build_tokens(g, seek)
            prompt = build_prompt(past, nt)
            logits = W.forward(enc, prompt + toks)
            m = min(_p_match(W.teacher_force(enc, prompt, toks, no_timestamps=nt, logits=logits, temperature=T)
                             ["tokens"], cpp_p)["mean_abs_dp"] for T in (0.0, 0.2, 0.4))
            # p-match first (in steps of 0.002, below which it is noise), then closeness to the
            # time crispasr wrote (our VAD map can be a few cs off), then plain chaining
            dist = abs(seek - mapped) if (not nt and not text_start) else 0
            scored.append(((int(m / 0.002), dist, why != "previous seek + last timestamp"), m, seek, why))
        scored.sort()
        scored = [(m, s, why) for _, m, s, why in scored]
        _, seek, seek_note = scored[0]
        if len(scored) > 1:
            seek_note += " (p-match " + ", ".join(f"{s / 100:.2f}s:{m:.4f}" for m, s, _ in scored) + ")"
        # ---- trailing timestamp of the previous window, implied by the seek
        if prev is not None and not nt:
            delta = seek - prev["seek"]
            last = 2 * (prev["tokens"][-1] - BEG)
            end_dist = prev["end_dist"]
            cand = BEG + min(delta, CHUNK_FRAMES) // 2
            if len(prev["tokens"]) >= N_MAX_STEPS:
                # the window ran into whisper.cpp's 220-token cap on a lone timestamp: no pair,
                # and the single-timestamp ending makes it skip the whole 30 s
                take = False
                prev["trailing_note"] = f"hit the {N_MAX_STEPS}-token cap on a lone timestamp -> seek + 30 s"
            elif delta == last or (delta == CHUNK_FRAMES and last != CHUNK_FRAMES):
                take = end_dist[cand] > end_dist[EOT]
                prev["trailing_note"] = (f"lone end timestamp or pair: HF p(EOT)={end_dist[EOT]:.3f}, "
                                         f"p(<|{ts_seconds(cand):.2f}|>)={end_dist[cand]:.3f} -> "
                                         + ("pair" if take else "lone"))
            else:
                take = True
                prev["trailing_note"] = f"pair implied by seek delta {delta / 100:.2f}s"
            if take:
                prev["tokens"].append(cand)
                prev["cpp_p"].append(None)
                prev["restored"].append(len(prev["tokens"]) - 1)
                base_past = (prev["past"][-(MAX_PROMPT_CTX - 1):] if prev["prompt_used"] else []) + prev["tokens"]
        past = [] if (seek > 0 and seek + 500 >= seek_end) else base_past
        enc = W.encode(mel, seek)
        toks, cpp_p, restored = build_tokens(g, seek)
        # ---- temperature hypotheses
        best = None
        for use_prompt in ((True, False) if past else (True,)):
            prompt = build_prompt(past if use_prompt else [], nt)
            logits = restore(enc, prompt, toks, restored)
            for T in temps:
                if past and (T < 0.5) != use_prompt:
                    continue
                tf = W.teacher_force(enc, prompt, toks, no_timestamps=nt, temperature=T, logits=logits)
                m = _p_match(tf["tokens"], cpp_p)
                if best is None or m["mean_abs_dp"] < best["match"]["mean_abs_dp"]:
                    best = {"T": T, "prompt_used": bool(use_prompt and past), "prompt": prompt, "match": m,
                            "logits": logits, "tf": tf, "toks": list(toks)}
        toks = best["toks"]
        end_dist = None
        if not nt:
            _, end_dist = filter_logits(best["logits"][len(best["prompt"]) + len(toks) - 1], toks, True,
                                        2 * (toks[-1] - BEG))
        rec = {"index": w, "seek": seek, "seek_s": seek / 100, "seek_note": seek_note,
               "orig_start_s": map_to_original(seek, mapping) / 100 if mapping else seek / 100,
               "text_start": text_start, "segments": g, "T": best["T"], "prompt_used": best["prompt_used"],
               "past": list(past), "prompt": best["prompt"], "tokens": toks, "cpp_p": cpp_p,
               "restored": restored, "match": best["match"], "end_dist": end_dist}
        log(json.dumps({k: rec[k] for k in ("index", "seek_s", "orig_start_s", "T", "prompt_used")}
                       | {"n_prompt": len(rec["prompt"]), "n_tokens": len(toks)}
                       | {k: round(v, 4) for k, v in rec["match"].items()} | {"seek_note": seek_note}))
        windows.append(rec)
        prev = rec
    for rec in windows:
        rec.pop("end_dist", None)
    return {"path": path, "clip": clip, "config": cfg, "seek_end": seek_end, "mapping": mapping,
            "init_prompt": init, "windows": windows}



# --------------------------------------------------------------------------------------
# Loop experiments at the reconstructed windows

CASES = [
    # name, path, clip, window, prefix segments kept from the window, text that marks the Loop
    dict(name="ts04_onset", path="timestamps", clip="find_da_5min_04", window=2, keep=4,
         key="lange skib var på vej fra København", kind="onset",
         note="window 55.5 s (orig 92.75 s); the window ends 2.8 s into 'Det 100-meter lange skib ...'; "
              "seg 10 wrote it with a made-up ending at 28.00, the Loop starts at the first rewrite (29.00)"),
    dict(name="ts04_copy", path="timestamps", clip="find_da_5min_04", window=3, keep=0,
         key="lange skib var på vej fra København", kind="copy",
         note="next window (orig 135.76-~182 s, ten other utterances); crispasr wrote the sentence 11x, "
              "at the T=0.2 fallback"),
    dict(name="bare01_copy", path="timestamps-bare", clip="find_da_5min_01", window=2, keep=0,
         key="vandstanden kun stiger", kind="copy",
         note="window 53-83 s; the previous window ended on the real 'Selvom vandstanden ...' (44.5-55.9 s)"),
    dict(name="nt02_copy", path="nt", clip="find_da_5min_02", window=4, keep=0,
         key="et forhold til kvindes helbred", kind="copy",
         note="window 120-150 s written as a copy of the 90-120 s window, at the T=0.2 fallback"),
    dict(name="ts03_onset", path="timestamps", clip="find_da_5min_03", window=1, keep=4,
         key="Ecuador kræver at kubanske", kind="onset",
         note="window 27 s (orig 34.69 s); 'Ecuador ...' written at 27.00, rewritten at 28.00 and 29.00"),
]


def _end_ts_positions(toks):
    return [k for k in range(1, len(toks)) if toks[k] > BEG and toks[k - 1] < BEG]


def _segments_of(toks):
    """(start, end) token spans of text runs, each with its surrounding timestamps."""
    spans, k = [], 0
    while k < len(toks):
        if toks[k] < EOT:
            a = k
            while k < len(toks) and toks[k] < EOT:
                k += 1
            spans.append((a, k))
        else:
            k += 1
    return spans


def drop_text(W, past, key, nt):
    """`past` without the text runs that contain `key` (and their timestamps / EOT)."""
    out, k = [], 0
    drop = set()
    for a, b in _segments_of(past):
        if key in W.decode(past[a:b], special=False):
            drop.update(range(a, b))
            j = b
            while j < len(past) and past[j] >= EOT:  # its end timestamp(s) / EOT
                drop.add(j)
                j += 1
    return [t for i, t in enumerate(past) if i not in drop]


def reference_continuation(W, clip, rec, mapping, nt, after_cs=0):
    """Reference utterances that lie wholly in the window (and start after `after_cs`, window
    time), formatted like the model would write them: <|t0|> text<|t1|> ... (or plain text +
    EOT on the -nt path). Utterances cut by the window edges are left out."""
    man = load_manifest(clip)
    seek = rec["seek"]
    toks, utts = [], []
    for u in man["utterances"]:
        a = (map_to_processed(u["speech_start_s"] * 100, mapping) if mapping else u["speech_start_s"] * 100) - seek
        b = (map_to_processed(u["speech_end_s"] * 100, mapping) if mapping else u["speech_end_s"] * 100) - seek
        if a < after_cs or b > CHUNK_FRAMES or a < 0:
            continue
        utts.append(u["text"])
        text = W.tok.encode(" " + u["text"], add_special_tokens=False)
        if nt:
            toks += text
        else:
            t0 = max(BEG + int(round(a / 2)), toks[-1] if toks else BEG)
            t1 = max(BEG + int(round(b / 2)), t0)
            toks += [t0] + text + [t1]
    if nt:
        toks.append(EOT)
    return toks, utts


def loop_count(text, key):
    return text.count(key)


def run_case(W: "Whisper", case: dict, seeds=(0, 1, 2, 3), log=print) -> dict:
    res = json.loads((REPRO_DIR / f"windows_{case['path']}_{case['clip']}.json").read_text())
    cfg = res["config"]
    nt = cfg["no_timestamps"]
    rec = res["windows"][case["window"]]
    mapping = [tuple(m) for m in res["mapping"]]
    _, _, audio, _ = case_inputs(case["path"], case["clip"])
    mel = log_mel_whispercpp(audio)
    seek_end = res["seek_end"]
    seek = rec["seek"]
    enc = W.encode(mel, seek)
    toks = rec["tokens"]
    ends = _end_ts_positions(toks)
    cut = ends[case["keep"] - 1] + 1 if case["keep"] else 0
    prefix_ids = toks[:cut]
    loop_ids = toks[cut:]
    prompt_cpp = rec["prompt"]
    past = rec["past"] if rec["prompt_used"] else []
    user = W.prompt_tokens()
    after = 2 * (prefix_ids[-1] - BEG) if prefix_ids else 0
    ref_ids, ref_utts = reference_continuation(W, case["clip"], rec, mapping, nt, after)
    if case["kind"] == "onset":
        ref_ids = [EOT]  # the rest of the window is the utterance already written; stop
        ref_utts = ["(end the window here)"]

    variants = {"crispasr": prompt_cpp, "no_prev_text": build_prompt([], nt)}
    variants["user_prompt_only"] = build_prompt(user, nt)
    variants["carry_user_prompt"] = [PREV] + user + past[-(MAX_PROMPT_CTX - len(user) - 1):] + build_prompt([], nt)
    if case["kind"] == "copy":
        variants["prev_text_minus_loop_sentence"] = build_prompt(drop_text(W, past, case["key"], nt), nt)
    out = {"case": case, "seek": seek, "seek_s": seek / 100, "orig_start_s": rec["orig_start_s"],
           "crispasr_T": rec["T"], "crispasr_prompt_used": rec["prompt_used"], "match": rec["match"],
           "inputs": {"prompt": prompt_cpp, "prompt_text": W.decode(prompt_cpp), "prefix": prefix_ids,
                      "prefix_text": W.decode(prefix_ids), "loop": loop_ids, "loop_text": W.decode(loop_ids),
                      "reference": ref_ids, "reference_text": W.decode(ref_ids), "reference_utterances": ref_utts,
                      "user_prompt": user, "variants": {k: v for k, v in variants.items()}},
           "variants": {}}
    if case["kind"] == "onset":
        variants["prefix_before_first_instance"] = prompt_cpp  # same prompt, shorter prefix (below)

    for vname, prompt in variants.items():
        pre = prefix_ids
        if vname == "prefix_before_first_instance":
            pre = toks[: ends[case["keep"] - 2] + 1] if case["keep"] > 1 else []
        tfp = W.teacher_force(enc, prompt, pre, no_timestamps=nt) if pre else None
        pre_lp = [(t["id"], t["filt_logprob"]) for t in tfp["tokens"]] if tfp else []
        v = {"n_prompt": len(prompt), "decodes": {}}
        t0 = time.perf_counter()
        # free decodes
        runs = [("greedy_T0", "greedy", None), ("beam5_T0", "beam", None)]
        if not nt:
            runs += [(f"cppbeam5_T0_seed{s}", "cppbeam", s) for s in seeds]
        for label, strat, sd in runs:
            rngs = [CppRng(1000 * sd + j) for j in range(5)] if sd is not None else None
            r = W.decode_window(enc, prompt, strategy=strat, beam=5, rngs=rngs, seek=seek, seek_end=seek_end,
                                no_timestamps=nt, prefix=pre_lp)
            text = W.decode(r["tokens"][len(pre):])
            v["decodes"][label] = {"text": text, "loops": loop_count(text, case["key"]),
                                   "avg_logprobs": r["avg_logprobs"], "failed": r["failed"],
                                   "entropy": r.get("entropy"), "no_speech_prob": r["no_speech_prob"],
                                   "would_fall_back": bool(r["failed"] or (r["avg_logprobs"] < -1.0
                                                                          and r["no_speech_prob"] < 0.6)),
                                   "tokens": r["tokens"][len(pre):]}
        # whisper.cpp's fallback ladder, as the path runs it (T = 0, 0.2, ...; prompt dropped from 0.5)
        strat0 = "cppbeam" if cfg["beam"] > 1 else "greedy"
        for sd in seeds[:2]:
            rngs = [CppRng(1000 * sd + j) for j in range(5)]
            ladder = []
            for T in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                p_use = prompt if T < 0.5 else build_prompt([], nt)
                pre_T = pre_lp if T < 0.5 or not pre else [(t, lp) for t, lp in pre_lp]
                r = W.decode_window(enc, p_use, strategy=strat0 if T == 0 else "greedy", beam=5,
                                    temperature=T, rngs=rngs, seek=seek, seek_end=seek_end, no_timestamps=nt,
                                    prefix=pre_T)
                fail = r["failed"] or (r["avg_logprobs"] < -1.0 and r["no_speech_prob"] < 0.6)
                text = W.decode(r["tokens"][len(pre):])
                ladder.append({"T": T, "text": text, "loops": loop_count(text, case["key"]),
                               "avg_logprobs": r["avg_logprobs"], "failed": r["failed"], "fell_back": bool(fail)})
                if not fail:
                    break
            v["decodes"][f"ladder_seed{sd}"] = ladder
        # teacher forcing: Loop vs reference (vs the no-previous-text decode, filled in below)
        v["tf"] = {}
        for cname, cont in (("loop", loop_ids), ("reference", ref_ids)):
            tf = W.teacher_force(enc, prompt, cont, window_prefix=pre, no_timestamps=nt)
            v["tf"][cname] = {k: tf[k] for k in ("n", "sum_raw", "mean_raw", "sum_filt", "mean_filt")}
            v["tf"][cname]["first_token"] = tf["tokens"][0]
            if vname == "crispasr":
                v["tf"][cname]["tokens"] = tf["tokens"]
        v["seconds"] = round(time.perf_counter() - t0, 1)
        out["variants"][vname] = v
        d = v["decodes"]
        log(json.dumps({"case": case["name"], "variant": vname, "seconds": v["seconds"],
                        "loops": {k: (x["loops"] if isinstance(x, dict) else [y["loops"] for y in x])
                                  for k, x in d.items()},
                        "tf_mean_raw": {k: round(x["mean_raw"], 3) for k, x in v["tf"].items()}},
                       ensure_ascii=False))
    # the model's own transcript without previous text, scored under every variant
    clean = out["variants"]["no_prev_text"]["decodes"]["beam5_T0"]["tokens"]
    out["inputs"]["clean"] = clean
    out["inputs"]["clean_text"] = W.decode(clean)
    for vname, prompt in variants.items():
        if vname == "prefix_before_first_instance":
            continue
        tf = W.teacher_force(enc, prompt, clean, window_prefix=prefix_ids, no_timestamps=nt)
        out["variants"][vname]["tf"]["clean"] = {k: tf[k] for k in ("n", "sum_raw", "mean_raw", "sum_filt",
                                                                     "mean_filt")}
        out["variants"][vname]["tf"]["clean"]["first_token"] = tf["tokens"][0] if tf["tokens"] else None
    return out


def cmd_repro(args):
    W = Whisper(dtype=args.dtype)
    names = args.cases.split(",") if args.cases else [c["name"] for c in CASES]
    out_file = REPRO_DIR / "cases.json"
    allres = json.loads(out_file.read_text()) if out_file.exists() else {}
    for c in CASES:
        if c["name"] in names:
            allres[c["name"]] = run_case(W, c)
            allres[c["name"]]["dtype"] = args.dtype
            out_file.write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print("wrote", out_file)


def cmd_summary(args):
    """Condense cases.json, the window validations and the long-form runs into
    results/phase0/hf_repro.json and print the tables used in hf_repro.md."""
    cases = json.loads((REPRO_DIR / "cases.json").read_text())
    out = {"cases": {}, "windows": {}, "longform": {}}
    for f in sorted(REPRO_DIR.glob("windows_*.json")):
        w = json.loads(f.read_text())
        out["windows"][f"{w['path']}/{w['clip']}"] = [
            {k: x.get(k) for k in ("index", "seek_s", "orig_start_s", "T", "prompt_used", "match", "seek_note",
                                   "trailing_note")} | {"n_prompt": len(x["prompt"]), "n_tokens": len(x["tokens"])}
            for x in w["windows"]]
    print("\n## window reconstruction (mean |dp| vs crispasr p, per window)")
    for k, ws in out["windows"].items():
        print(k, " ".join(f"{x['seek_s']:.2f}s:T{x['T']}:{x['match']['mean_abs_dp']:.4f}" for x in ws))
    for name, c in cases.items():
        cs = {"case": c["case"], "seek_s": c["seek_s"], "orig_start_s": c["orig_start_s"],
              "crispasr_T": c["crispasr_T"], "match": c["match"], "inputs": c["inputs"], "variants": {}}
        for vname, v in c["variants"].items():
            dec = {}
            for k, x in v["decodes"].items():
                if isinstance(x, list):
                    dec[k] = [{kk: y[kk] for kk in ("T", "loops", "avg_logprobs", "failed", "fell_back", "text")}
                              for y in x]
                else:
                    dec[k] = {kk: x[kk] for kk in ("loops", "avg_logprobs", "failed", "would_fall_back", "text")}
            tf = {}
            for k, x in v["tf"].items():
                tf[k] = {kk: x[kk] for kk in ("n", "sum_raw", "mean_raw", "sum_filt", "mean_filt")}
                ft = x.get("first_token")
                if ft:
                    tf[k]["first_token"] = {kk: ft[kk] for kk in ("text", "raw_logprob", "p", "argmax_text")}
                if "tokens" in x:
                    tf[k]["tokens"] = [{kk: t[kk] for kk in ("id", "text", "raw_logprob", "filt_logprob")}
                                       for t in x["tokens"]]
            cs["variants"][vname] = {"n_prompt": v["n_prompt"], "decodes": dec, "tf": tf}
        out["cases"][name] = cs
        print(f"\n## {name}  (seek {c['seek_s']:.2f}s, orig {c['orig_start_s']:.2f}s, crispasr T={c['crispasr_T']})")
        print("variant | n_prompt | loops greedy/beam5/cppbeam seeds | ladder (T:loops) | "
              "TF mean raw loop/ref/clean | TF sum raw loop/ref/clean")
        for vname, v in c["variants"].items():
            d = v["decodes"]
            cpp = "/".join(str(x["loops"]) for k, x in d.items() if k.startswith("cppbeam"))
            lad = " ".join("[" + ",".join(f"{y['T']}:{y['loops']}" for y in x) + "]"
                           for k, x in d.items() if k.startswith("ladder"))
            tf = v["tf"]
            m = "/".join(f"{tf[k]['mean_raw']:.3f}" if k in tf else "-" for k in ("loop", "reference", "clean"))
            sm = "/".join(f"{tf[k]['sum_raw']:.1f}({tf[k]['n']})" if k in tf else "-"
                          for k in ("loop", "reference", "clean"))
            print(f"{vname} | {v['n_prompt']} | {d['greedy_T0']['loops']}/{d['beam5_T0']['loops']}/{cpp or '-'} | "
                  f"{lad} | {m} | {sm}")
    for f in sorted((REPRO_DIR / "longform").glob("*.json")):
        r = json.loads(f.read_text())
        out["longform"][f.stem] = {k: r[k] for k in ("engine", "path", "clip", "settings", "seconds",
                                                    "segments_most_repeated", "crispasr_most_repeated",
                                                    "segments_repeats", "same_segment_texts_as_crispasr")}
        out["longform"][f.stem]["segments"] = r["segments"]
        if "windows" in r:
            out["longform"][f.stem]["windows"] = [{k: w[k] for k in ("seek_s", "orig_start_s", "T", "text")}
                                                  for w in r["windows"]]
        mr = r["segments_most_repeated"]
        cr = r["crispasr_most_repeated"]
        print(f"\nlongform {f.stem}: {mr['count']}x from {mr.get('first_s', 0) or 0:.2f}s '{(mr['text'] or '')[:70]}'"
              f" | crispasr {cr['count']}x from {cr.get('first_s', 0) or 0:.2f}s | identical segment texts: "
              f"{r['same_segment_texts_as_crispasr']} | {r['seconds']}s")
    (ROOT / "results/phase0/hf_repro.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\nwrote results/phase0/hf_repro.json")


# --------------------------------------------------------------------------------------
# CLI


def cmd_validate(args):
    W = Whisper(dtype=args.dtype)
    res = resolve_case(W, args.path, args.clip)
    out = REPRO_DIR / f"windows_{args.path}_{args.clip}.json"
    if not args.no_save:
        for rec in res["windows"]:
            rec["text"] = W.decode(rec["tokens"])
            rec["prompt_text"] = W.decode(rec["prompt"])
        res["dtype"] = args.dtype
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="rebuild crispasr's windows and check them against its token p")
    v.add_argument("--path", required=True, choices=list(PATHS))
    v.add_argument("--clip", required=True)
    v.add_argument("--dtype", default="float16")
    v.add_argument("--no-save", action="store_true")
    r = sub.add_parser("repro", help="free decodes, teacher forcing and prompt ablations at the Loop windows")
    r.add_argument("--cases", default="")
    r.add_argument("--dtype", default="float16")
    sub.add_parser("summary", help="write results/phase0/hf_repro.json and print the report tables")
    args = ap.parse_args()
    {"validate": cmd_validate, "repro": cmd_repro, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
