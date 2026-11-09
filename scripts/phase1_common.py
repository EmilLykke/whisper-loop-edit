"""Phase 1 shared code: a hand-written large-v3 decoder forward with hooks at every site.

The forward reads the HF module weights directly (no copies) and computes the same thing as
`WhisperDecoder` (pre-LN, sdpa, scale folded into q), but on a left-padded batch so every
sequence's decision position is the last column. Each window's encoder output is computed
once and its cross-attention K/V can be cached per layer.

Hooks: `hooks[(layer, site)] = fn(x) -> x | None`, sites per layer, in order:
    resid_pre   (B, T, D)      residual entering the layer
    z_self      (B, T, H, dh)  self-attention head outputs, before out_proj
    self_probs  (B, H, T)      self-attention probs of the LAST position (read-only)
    attn_out    (B, T, D)      self-attention write (out_proj output, bias included)
    z_cross     (B, T, H, dh)  cross-attention head outputs, before out_proj
    cross_probs (B, H, 1500)   cross-attention probs of the LAST position (read-only)
    cross_out   (B, T, D)
    mlp_act     (B, T, F)      gelu(fc1(ln(x)))
    mlp_out     (B, T, D)      fc2 write
    resid_post  (B, T, D)
Returning a tensor replaces the activation. Nothing here computes a gradient.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_window as hw  # noqa: E402
from hf_window import BEG, EOT  # noqa: E402

ROOT = hw.ROOT


def _case_inputs(path: str, clip: str):
    """hf_window.case_inputs, but tolerant of crispasr JSON with a token split mid-UTF-8
    (only token ids and p are used)."""
    cfg = hw.PATHS[path]
    js = json.loads((hw.TRANSCRIPTS / path / f"{clip}.run1.json").read_bytes().decode("utf-8", "replace"))
    audio = hw.load_wav(ROOT / f"data/clips/find/{clip}.wav")
    mapping = []
    if cfg["vad"]:
        vad_segments = json.loads((hw.REPRO_DIR / f"vad_{clip}.json").read_text())["segments"]
        audio, mapping = hw.vad_filter(audio, vad_segments)
    return cfg, js, audio, mapping


hw.case_inputs = _case_inputs  # resolve_case looks it up as a module global
P1 = ROOT / "results/phase1"
WIN_DIR = P1 / "windows"
N_LAYERS, N_HEADS, D, DH = 32, 20, 1280, 64
SITES = ("resid_pre", "attn_out", "cross_out", "mlp_out", "resid_post")

FIND_A = {"find_da_5min_01", "find_da_5min_03", "find_da_2min_01", "find_da_2min_03", "find_da_continuous_90s_02"}
FIND_B = {"find_da_5min_02", "find_da_5min_04", "find_da_2min_02", "find_da_continuous_90s_01",
          "find_da_continuous_90s_03"}


def split_of(clip: str) -> str:
    return "A" if clip in FIND_A else "B"


class Dec:
    def __init__(self, W: hw.Whisper | None = None):
        self.W = W or hw.Whisper()
        self.dev = self.W.device
        dec = self.W.model.model.decoder
        self.layers = dec.layers
        self.embed = dec.embed_tokens
        self.pos = dec.embed_positions.weight
        self.ln_f = dec.layer_norm
        self.proj = self.W.model.proj_out
        self.enc = {}       # window key -> (1, 1500, D)
        self.kv = {}        # (window key, layer) -> (K, V), each (1, H, 1500, dh)
        self.cache_kv = True
        self._mel = {}

    # ---------------------------------------------------------------- audio
    def mel(self, path: str, clip: str):
        key = (path, clip)
        if key not in self._mel:
            _, _, audio, _ = hw.case_inputs(path, clip)
            self._mel[key] = hw.log_mel_whispercpp(audio)
        return self._mel[key]

    def encode(self, path: str, clip: str, seek: int) -> str:
        key = f"{path}|{clip}|{seek}"
        if key not in self.enc:
            self.enc[key] = self.W.encode(self.mel(path, clip), seek)
        return key

    def drop_kv(self):
        self.kv.clear()
        if self.dev == "mps":
            torch.mps.empty_cache()

    def cross_kv(self, L: int, wins: list[str]):
        att = self.layers[L].encoder_attn
        out_k, out_v = [], []
        for w in wins:
            kv = self.kv.get((w, L))
            if kv is None:
                e = self.enc[w]
                k = att.k_proj(e).view(1, -1, N_HEADS, DH).transpose(1, 2)
                v = att.v_proj(e).view(1, -1, N_HEADS, DH).transpose(1, 2)
                kv = (k, v)
                if self.cache_kv:
                    self.kv[(w, L)] = kv
            out_k.append(kv[0])
            out_v.append(kv[1])
        return torch.cat(out_k), torch.cat(out_v)

    # ---------------------------------------------------------------- forward
    def batch(self, seqs: list[list[int]]):
        """Left-pad. Returns ids (B,T), positions (B,T), key mask (B,T) True = real."""
        T = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        pos = torch.zeros(len(seqs), T, dtype=torch.long)
        real = torch.zeros(len(seqs), T, dtype=torch.bool)
        for b, s in enumerate(seqs):
            n = len(s)
            ids[b, T - n:] = torch.tensor(s)
            pos[b, T - n:] = torch.arange(n)
            real[b, T - n:] = True
        return ids.to(self.dev), pos.to(self.dev), real.to(self.dev)

    def _mask(self, real: torch.Tensor, Tq: int):
        """(B,1,Tq,Tk) bool: causal, real keys only; pad queries see themselves (avoid NaN)."""
        Tk = real.shape[1]
        causal = torch.ones(Tq, Tk, dtype=torch.bool, device=self.dev).tril(Tk - Tq)
        m = causal[None] & real[:, None, :]
        eye = torch.zeros(Tq, Tk, dtype=torch.bool, device=self.dev)
        eye[torch.arange(Tq), torch.arange(Tk - Tq, Tk)] = True
        qpad = ~real[:, Tk - Tq:]  # (B, Tq)
        m = m | (eye[None] & qpad[:, :, None])
        return m[:, None]

    @torch.inference_mode()
    def forward(self, seqs, wins, hooks=None, start=0, resid=None, real=None, stop=N_LAYERS,
                past=None, return_past=False, last_n=1):
        """Run layers [start, stop). `resid` (B,T,D) replaces the embeddings when start > 0.
        `past`: list per layer of (K, V) self-attn cache plus `real` covering past+new.
        Returns dict(logits=(B,V) float32 at the last position [or (B,last_n,V)], resid=final x, past=...)."""
        hooks = hooks or {}
        if past is None:
            ids, pos, real_ = self.batch(seqs) if resid is None or start == 0 else (None, None, real)
            if real is None:
                real = real_
            if start == 0:
                x = self.embed(ids) + self.pos[pos]
            else:
                x = resid
            Tq = x.shape[1]
        else:
            # seqs: (B, 1) new tokens; real: (B, Tk) incl. the new column; positions from real
            ids = torch.tensor(seqs, device=self.dev)
            posn = real.sum(1, keepdim=True) - 1
            x = self.embed(ids) + self.pos[posn]
            Tq = 1
        mask = self._mask(real, Tq)
        new_past = []
        B = x.shape[0]
        for L in range(start, stop):
            lay = self.layers[L]

            def H(site, t):
                f = hooks.get((L, site))
                if f is None:
                    return t
                r = f(t)
                return t if r is None else r

            x = H("resid_pre", x)
            sa = lay.self_attn
            h = lay.self_attn_layer_norm(x)
            q = (sa.q_proj(h) * sa.scaling).view(B, Tq, N_HEADS, DH).transpose(1, 2)
            k = sa.k_proj(h).view(B, Tq, N_HEADS, DH).transpose(1, 2)
            v = sa.v_proj(h).view(B, Tq, N_HEADS, DH).transpose(1, 2)
            if past is not None:
                k = torch.cat([past[L - start][0], k], 2)
                v = torch.cat([past[L - start][1], v], 2)
            if return_past:
                new_past.append((k, v))
            if (L, "self_probs") in hooks:
                s = (q[:, :, -1:] @ k.transpose(2, 3))[:, :, 0].float()
                s = s.masked_fill(~mask[:, 0, -1][:, None, :], float("-inf"))
                hooks[(L, "self_probs")](torch.softmax(s, -1))
            z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0).transpose(1, 2)
            z = H("z_self", z)
            a = sa.out_proj(z.reshape(B, Tq, D))
            a = H("attn_out", a)
            x = x + a

            ca = lay.encoder_attn
            h = lay.encoder_attn_layer_norm(x)
            q = (ca.q_proj(h) * ca.scaling).view(B, Tq, N_HEADS, DH).transpose(1, 2)
            K, V = self.cross_kv(L, wins)
            if (L, "cross_probs") in hooks:
                s = (q[:, :, -1:] @ K.transpose(2, 3))[:, :, 0].float()
                hooks[(L, "cross_probs")](torch.softmax(s, -1))
            z = F.scaled_dot_product_attention(q, K, V, scale=1.0).transpose(1, 2)
            z = H("z_cross", z)
            c = ca.out_proj(z.reshape(B, Tq, D))
            c = H("cross_out", c)
            x = x + c

            h = lay.final_layer_norm(x)
            act = F.gelu(lay.fc1(h))
            act = H("mlp_act", act)
            m = lay.fc2(act)
            m = H("mlp_out", m)
            x = x + m
            x = H("resid_post", x)
        out = {"resid": x, "real": real}
        if stop == N_LAYERS:
            y = self.ln_f(x[:, -1] if last_n == 1 else x[:, -last_n:])
            out["logits"] = self.proj(y).float()
        if return_past:
            out["past"] = new_past
        return out

    # ---------------------------------------------------------------- helpers
    def chunks(self, n, bs):
        for i in range(0, n, bs):
            yield slice(i, min(n, i + bs))

    def logprobs_last(self, seqs, wins, hooks_fn=None, bs=32):
        """log-softmax at the last position for every sequence; hooks_fn(slice) -> hooks."""
        out = []
        for sl in self.chunks(len(seqs), bs):
            r = self.forward(seqs[sl], wins[sl], hooks=hooks_fn(sl) if hooks_fn else None)
            out.append(torch.log_softmax(r["logits"], -1).cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def greedy(self, prompts, wins, n_steps, no_ts, prefixes=None, hooks=None, bs=24):
        """whisper.cpp-filtered greedy decode of up to n_steps tokens per sequence (batched,
        self-attn KV cache). `prompts` end in the task tokens; `prefixes` are window tokens
        already decoded (part of the input). Hooks apply at every step (weight-like edits)."""
        prefixes = prefixes or [[] for _ in prompts]
        res = [None] * len(prompts)
        keep = self.cache_kv
        for sl in self.chunks(len(prompts), bs):
            if not keep:
                self.kv.clear()
            self.cache_kv = True  # cross K/V once per batch, not per step
            idx = list(range(sl.start, sl.stop))
            seqs = [list(prompts[i]) + list(prefixes[i]) for i in idx]
            steps = min(n_steps, hw.N_TEXT_CTX - max(len(q) for q in seqs))  # 448 positions
            r = self.forward(seqs, [wins[i] for i in idx], hooks=hooks, return_past=True)
            past, real = r["past"], r["real"]
            toks = [list(prefixes[i]) for i in idx]
            done = [False] * len(idx)
            logits = r["logits"].cpu().numpy()
            for step in range(steps):
                nxt = []
                for j, i in enumerate(idx):
                    if done[j]:
                        nxt.append(EOT)
                        continue
                    t = toks[j]
                    tsv = [2 * (x - BEG) for x in t if x > BEG]
                    lp, _ = hw.filter_logits(logits[j], t, bool(tsv), max(tsv or [hw.CHUNK_FRAMES]),
                                             no_timestamps=no_ts[i])
                    a = int(np.argmax(lp))
                    t.append(a)
                    if a == EOT:
                        done[j] = True
                    nxt.append(a)
                if all(done) or step == steps - 1:
                    break
                real = torch.cat([real, torch.ones(len(idx), 1, dtype=torch.bool, device=self.dev)], 1)
                r = self.forward([[a] for a in nxt], [wins[i] for i in idx], hooks=hooks, past=past,
                                 real=real, return_past=True)
                past = r["past"]
                logits = r["logits"].cpu().numpy()
            for j, i in enumerate(idx):
                res[i] = toks[j][len(prefixes[i]):]
        self.cache_kv = keep
        if not keep:
            self.drop_kv()
        return res


# -------------------------------------------------------------------- text helpers
_WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def contains_seq(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def writes_s(W, gen_ids, s_ids, n_words=4) -> bool:
    """Does the generated text contain S's first n_words words?"""
    sw = words(W.decode(s_ids, special=False))[:n_words]
    return contains_seq(words(W.decode(gen_ids, special=False)), sw)


def load_dataset():
    return json.loads((P1 / "dataset.json").read_text())


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = sums[inv] / cnt[inv]
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohens_d(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    s = math.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / s) if s > 0 else 0.0
