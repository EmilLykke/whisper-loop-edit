# Phase 1: where the decoder writes the Loop

**Verdict: the kill condition is not met.** The Loop is written by a small set of decoder
**self-attention** heads that copy from the previous text: an induction circuit with an early part
(L12H13, L13H7, L13H18, L15H17, plus L2H19, L18H19) and a late copy-writer part (L25H18, L25H19,
L24H0, L26H6, L21H14). On find-B, zeroing those 11 heads takes the greedy Loop rate on LOOP states
from **68% to 0%** (0 of 117), and the natural Phase 0 Loops from 2/5 to 0/5. Two random sets of 11
heads in the same layers leave it at 74% and 68%. Zeroing just **two heads, L12H13 + L25H18, gives
6%** (random same-layer pair: 68%). The price is real: the CLEAN continuation's log-prob drops
from -0.369 to -0.50 per token for 11 heads (-0.46 for 2 heads). Random head sets cost about 0.01.

A direction separates Loop from clean too (find-B AUC **0.97** at layer 14 for LOOP vs CLEAN
states, **0.86** at layer 24 for "will loop" vs "will not" among LOOP states), but only the layer-24
one is causal. Projecting it out of the writers of layers 13-24 halves the Loop rate (68% to 32%)
at almost no log-prob cost (-0.006 per token). The layer-14 direction changes nothing when removed:
it records that the previous text holds a repeat; it does not decide to copy. No MLP neurons
matter. Cross-attention does not collapse in LOOP states, and Glazer et al.'s L18H13 does nothing
here.

So of the three options: **(i) a small set of heads: yes**, cleanly, with a clean cost to manage in
Phase 2. **(ii) a single direction: separates, partly causal** (layer 24, cheap but only halves the
Loop rate). **(iii) neurons: no.**

Run 2026-11-08 on the M4 Max, torch 2.15 MPS, transformers 5.22, HF large-v3 fp16. Find Set only.
Only forward passes: ablation, patching, attention and activation reads, mean differences,
sklearn logistic regression on activations. No gradient was computed anywhere.

## 1. Contrastive dataset

`scripts/phase1_windows.py` rebuilt crispasr's decode windows (Phase 0's `resolve_case`) for all
10 Find clips on `timestamps-bare` and `nt` (the paths that need no VAD log), plus the two
`timestamps` clips Phase 0 had VAD segments for: 22 (path, clip) runs, 152 windows, mean |Δp| to
crispasr ≤ 0.029 on every window except one at 0.062 (bare 5min_01's prompt-cleared tail). One
crispasr JSON has a token split mid-UTF-8; it is read tolerantly (only ids and p are used).

`scripts/phase1_dataset.py` then builds, for every window with previous text that holds no Loop
already, and for up to 3 sentences S from that previous text (the last one plus random others,
5-40 tokens):

- **CLEAN**: crispasr's previous text for that window with S cut out (and its timestamps).
- **LOOP_k**: CLEAN's previous text plus a fake previous window that wrote S k times, in the
  path's own format (`<|0.00|> S<|t|><|t|> S<|2t|>...` on timestamp paths, `S S S<|endoftext|>`
  on `nt`). k = 1, 2, 3.

Same audio, same step. The decision position is the window's first text token (after the
first timestamp the model writes from CLEAN, on timestamp paths). C = CLEAN's own greedy
continuation, LD = log p(S[0]) - log p(C[0]) (raw log-softmax). A state **loops** when its
whisper.cpp-filtered greedy decode writes S's first 4 words. Pairs were dropped where S[0] =
C[0] or where CLEAN's 60-token decode itself contains S (S is in the audio).

| k copies of S | LOOP states | greedy writes S | prefers S (LD > 0) | mean LD |
|---|---|---|---|---|
| 1 | 231 | 3.9% | 12.1% | -4.61 |
| 2 | 231 | 29.9% | 38.5% | -1.54 |
| 3 | 231 | 34.6% | 43.3% | -0.92 |
| CLEAN | 231 | 0% (filter) | 0% | -7.01 |

| Path | k=1 | k=2 | k=3 |
|---|---|---|---|
| `timestamps` | 0/22 | 4/22 | 6/22 |
| `timestamps-bare` | 7/103 | 36/103 | 41/103 |
| `nt` | 2/106 | 29/106 | 33/106 |

One copy of S in the previous text rarely makes the model write it again. Two copies do, a third of
the time. **Valid pairs** (LOOP writes or prefers S, CLEAN prefers C): **220** (155 of them both
write and prefer S). Split by clip: **find-A** = 5min_01, 5min_03, 2min_01, 2min_03, cont_02 (103
pairs); **find-B** = 5min_02, 5min_04, 2min_02, cont_01, cont_03 (117 pairs).

The 5 natural Phase 0 cases are included (LOOP = crispasr's input, CLEAN = previous text minus
the Loop sentence, or no previous text for the two onsets), decided at the first token where the
Loop and CLEAN's continuation part. At that token 4 of 5 prefer the Loop (LD +3.4, +3.5, +0.2,
+0.6; ts03_onset -0.6), and greedy writes it in 2 (ts04_copy, nt02_copy), as in Phase 0.

## 2a. Head ablation sweep (find-A)

All 1,280 decoder heads (32 layers × 20 heads × self/cross), zeroed one at a time (= zero that
head's 64 columns of `out_proj.weight`), on 96 find-A LOOP states and their 52 CLEAN states.
Baseline: LOOP LD +1.89 (S is the argmax in 85%), CLEAN LD -4.98, CLEAN continuation log-prob
-0.284 per token. H = self-attention, X = cross-attention.

| Head | ΔLD (LOOP) | LD flips to < 0 | argmax = S | ΔLD (CLEAN) | Δ CLEAN log-p/token |
|---|---|---|---|---|---|
| **L12H13** | -0.98 | 26% | 0.61 | +0.08 | -0.186 |
| L0H5 | -0.96 | 30% | 0.57 | +1.22 | **-0.879** (breaks everything) |
| **L25H18** | -0.75 | 27% | 0.58 | +0.01 | -0.034 |
| L2H19 | -0.44 | 16% | 0.70 | +0.10 | +0.006 |
| L24H0 | -0.28 | 12% | 0.75 | +0.01 | -0.004 |
| L25H19 | -0.24 | 9% | 0.80 | +0.05 | -0.011 |
| L19X4 | -0.23 | 10% | 0.72 | -0.28 | +0.002 |
| L26H6 | -0.23 | 12% | 0.73 | -0.01 | 0.000 |
| L6X10 | -0.20 | 9% | 0.77 | -0.10 | +0.004 |
| L21H14 | -0.19 | 9% | 0.75 | -0.01 | -0.001 |
| L13H7 | -0.19 | 8% | 0.79 | -0.01 | -0.041 |
| L13H18 | -0.19 | 9% | 0.76 | +0.03 | -0.014 |
| L15H17 | -0.19 | 13% | 0.72 | +0.01 | -0.003 |
| L18H19 | -0.18 | 14% | 0.71 | +0.02 | 0.000 |
| L30H7 | -0.16 | 13% | 0.73 | +0.02 | -0.002 |
| L18X13 (Glazer) | +0.00 | 0% | 0.85 | +0.02 | -0.001 |

Single heads move LD by at most 1 nat of a 6.9-nat LOOP-CLEAN gap: the circuit is spread over
several heads. Only 8 self-attention heads (and 1 cross) reach ΔLD < -0.2. The self-attention
heads' ΔLD sum to -6.64; the cross-attention heads' sum to +2.53. Ablating some heads makes the
Loop *stronger* (L10H9 +0.50, L26H1 +0.38, L19X11 +0.27). Full table: `localize.json` → `sweep`.

## 2b. Activation patching (find-A)

Each site's output at the decision position, patched from CLEAN into LOOP (96 states). Fraction of
the LD gap restored (LOOP +1.89, CLEAN -4.40, averaged per LOOP state):

| Layer | attn_out | cross_out | mlp_out | resid_post |
|---|---|---|---|---|
| 0-12 | ≤ 0.03 | ≤ 0.01 | ≤ 0.01 | ≤ 0.06 |
| 13 | 0.05 | 0.00 | 0.02 | 0.08 |
| 14 | -0.07 | 0.00 | -0.05 | 0.10 |
| 16 | 0.01 | 0.01 | 0.01 | 0.20 |
| 18 | 0.05 | 0.00 | 0.00 | 0.31 |
| 21 | 0.05 | 0.01 | 0.00 | 0.52 |
| 24 | 0.03 | 0.01 | 0.02 | 0.66 |
| **25** | **0.24** | 0.01 | 0.04 | 0.83 |
| 26 | -0.02 | 0.01 | 0.09 | 0.89 |
| 27 | 0.08 | 0.03 | 0.05 | 0.95 |
| 31 | 0.01 | 0.01 | -0.05 | 1.00 |

The residual at the decision position carries almost nothing until layer 13, then picks up the
Loop gradually from layer 13 to 27. It is read in from the previous text by attention, not built
at the decision position. The one site that matters on its own is **layer 25's self-attention
(24%)**, and in it **L25H18 alone restores 16%** (flips 32% of states). Next: L25H19 5%, L21H14 4%,
L13H7 4%, L26H6 4%, L15H17 4%, L18H19 3%, L24H0 3%, L30H7 3%. Cross-attention outputs restore ≤ 3%
at every layer. Patching the whole shared tail (task tokens + first timestamp, 4 positions) gives
the same numbers as patching the decision position alone (max difference 0.01): the earlier
task-token positions carry nothing extra.

## 2c. Induction and copy scores

Attention of the decision position (valid LOOP states, both splits):

| Head | to the first token of each copy of S (`ind_S`) | after writing S[0], S[1]: to S[2] in the copies |
|---|---|---|
| **L25H18** | **0.60** | **0.81** |
| L15H10 | 0.49 | 0.63 |
| L13H7 | 0.33 | 0.74 |
| L12H13 | 0.06 | **0.78** |
| L13H9 | 0.03 | 0.66 |
| L25H19 | 0.27 | 0.50 |
| L27H9 | 0.26 | 0.48 |

These are induction heads in the textbook sense: once the model has started S, they attend to the
next token of S in the earlier copies. L25H18 already does it at the first token: from the segment-
start timestamp it attends to the token after each earlier segment start, which in a LOOP state is
S. In CLEAN states the same heads spread that attention over the starts of all earlier segments
(L25H18: 0.33 total, ≤ 0.1 each). L12H13 is the strongest early induction head once copying is under
way, and the sweep's top head. Many layer 4-16 heads put ≥ 80% of the decision position's attention
on the S copies (L7H17 0.97, L16H9 0.89), but S copies are always the most recent text in our LOOP
states, so for these "attend to recent text" heads that number is partly recency. The ablation and
patching results, not these scores, are what single out the heads above.

## 2d. Directions

Residual stream after each layer at the decision position. Mean-difference direction from
find-A, scored on find-B (AUC and Cohen's d of the projection, accuracy of a find-A threshold), and
an L2 logistic probe (sklearn, C = 0.05, standardised) trained on find-A, accuracy on find-B.

**Contrast 1, LOOP (valid) vs CLEAN** (find-B: 117 vs 58, majority baseline 0.67):

| Layer | AUC B | d B | mean-diff acc B | probe acc B |
|---|---|---|---|---|
| 8 | 0.86 | 1.42 | 0.82 | 0.82 |
| 12 | 0.82 | 1.29 | 0.73 | 0.85 |
| 13 | 0.97 | 2.27 | 0.83 | 0.93 |
| **14** | **0.97** | **2.93** | **0.93** | **0.97** |
| 18 | 0.96 | 3.11 | 0.92 | 0.95 |
| 24 | 0.96 | 2.56 | 0.90 | 0.92 |
| 31 | 0.73 | 0.65 | 0.66 | 0.69 |

Averaging the last 4 positions (as the blog did) gives the same picture (AUC 0.98 from layer 14 to 24).
The direction appears at layers 13-14. It is written by L14's MLP (projection difference 1.40), L14's
self-attention (1.25) and L13's self-attention (0.81). By head: L14H8 0.46, L13H1 0.25, L14H15 0.25,
L13H18 0.20. By neuron: L14 N3648 (0.63) and N1507 (0.49).

**Contrast 2, LOOP states that loop vs LOOP states that do not** (same kind of previous text, the
decision differs; find-B 80 vs 277, majority baseline 0.78):

| Layer | AUC B | d B | probe acc B |
|---|---|---|---|
| 14 | 0.78 | 1.06 | 0.80 |
| 18 | 0.81 | 1.16 | 0.82 |
| 21 | 0.84 | 1.28 | 0.82 |
| **24** | **0.86** | **1.36** | **0.83** |
| 26 | 0.75 | 0.90 | 0.83 |

Its writers are spread out: L14 MLP 0.23, L14 attn 0.18, L13 attn 0.13, L24 attn and MLP 0.09 each,
L23/L19/L22 cross-attention 0.05-0.07. By head: L24H0 0.10, L13H7 0.07, L14H7 0.06.

On find-B, removing contrast 1's layer-14 direction does nothing (see section 3): it marks "the
previous text repeats a sentence", which is true for LOOP_1 states that never loop. Contrast 2's
layer-24 direction is partly causal.

**Direct logit attribution** (`scripts/phase1_dla.py`; frozen final LN, valid LOOP vs CLEAN,
contribution to LD): L25H18 **+1.18** of the 7.17 total, then L26H6 +0.37, L30H7 +0.35, L27H10
+0.28, L27H9 +0.26, L15H10 +0.23, L27H14 +0.21, L13H7 +0.18, L22H0 +0.18, L25H19 +0.15. By site:
L25 attention +1.32, L27 attention +0.84, L27 MLP +0.55, L30 attention +0.48. L25H18 is the head
that writes S's token into the output; L12H13 acts indirectly, through the later heads.

## 2e. Cross-attention

At the decision position, cross-attention looks the same in LOOP and CLEAN states. Head-mean
entropy and mass on the window's first 3 s differ by ≤ 0.03 at every layer (for example L20:
audio 0.14 vs 0.15, entropy 4.08 vs 3.84 nats). The heads with the largest drop in audio mass
(L20X5 0.39 vs 0.52, L20X8 0.35 vs 0.47, L21X13 0.44 vs 0.55) are small effects. **L18X13** (Glazer
et al.'s head): audio 0.15 vs 0.14, AUC 0.59, ablation ΔLD +0.001. The Loop is not the audio
attention collapsing. It is the self-attention copy winning over an audio read that still works.
Zeroing the 5 cross-attention heads with the largest ablation effect reduces the Loop rate on
find-B (68% to 37%), but it changes the CLEAN output most of all conditions (greedy unchanged
40%): those heads read the audio, and removing them makes the model guess.

## 3. Checks on find-B

Every intervention is the activation twin of a Weight Edit, applied at every position and every
decode step: `heads=` scales heads' `out_proj` column slices, `neurons=` scales `fc2` columns,
`wproj=` applies `W ← (I - u uᵀ) W` (bias too) to the writers of the given layers (a = self-attn
`out_proj`, c = cross-attn `out_proj`, m = `fc2`), `rproj=` projects u out of the residual after one
layer (diagnostic only, not a Weight Edit). u = the find-A unit direction of layer L, contrast 1, 2
or a random unit vector (r). 117 find-B LOOP states, 58 CLEAN states, greedy 24 tokens. "Fuzzy"
counts a LOOP state as looping when its text holds ≥ half of S's words, which also catches
paraphrased copies. CLEAN log-p: mean log-prob per token of CLEAN's own 30-token continuation.
Unchanged: CLEAN's greedy 30 tokens identical to the unedited ones. Natural: the 5 Phase 0 cases,
greedy.

| Condition | Loop rate | fuzzy | LOOP LD | CLEAN log-p | CLEAN unchanged | natural Loops |
|---|---|---|---|---|---|---|
| baseline | 0.68 | 0.69 | 2.04 | -0.369 | 1.00 | 2/5 |
| **11 heads** (L12H13 L13H7 L13H18 L15H17 L2H19 L18H19 L25H18 L25H19 L24H0 L26H6 L21H14) | **0.00** | 0.00 | -1.66 | -0.500 | 0.45 | **0/5** |
| control: 11 random heads, same layers, seed 0 | 0.74 | 0.74 | 2.24 | -0.373 | 0.78 | 2/5 |
| control: seed 1 | 0.68 | 0.68 | 2.04 | -0.371 | 0.79 | 4/5 |
| 11 heads at 0.5 | 0.16 | 0.18 | 0.29 | -0.428 | 0.57 | 0/5 |
| 10 heads (without L12H13) | 0.03 | 0.04 | -1.09 | -0.446 | 0.52 | 0/5 |
| early 6 (L12H13 L13H7 L13H18 L15H17 L2H19 L18H19) | 0.01 | 0.03 | -0.61 | -0.463 | 0.50 | 0/5 |
| early 6 at 0.5 | 0.31 | 0.32 | 1.03 | -0.409 | 0.62 | 2/5 |
| control: 6 random heads, same layers | 0.69 | 0.69 | 1.94 | -0.378 | 0.62 | 2/5 |
| late 5 (L25H18 L25H19 L24H0 L26H6 L21H14) | 0.21 | 0.20 | 0.04 | -0.415 | 0.69 | 2/5 |
| L12H13 + late 5 | 0.03 | 0.03 | -0.56 | -0.483 | 0.53 | 1/5 |
| **L12H13 + L25H18** | **0.06** | 0.11 | 0.42 | -0.462 | 0.52 | 1/5 |
| control: L12H4 + L25H7 | 0.68 | 0.68 | 2.07 | -0.381 | 0.69 | 3/5 |
| L12H13 | 0.14 | 0.16 | 0.80 | -0.441 | 0.60 | 1/5 |
| L12H13 at 0.5 | 0.59 | 0.61 | 2.06 | -0.397 | 0.71 | 2/5 |
| L25H18 | 0.45 | 0.46 | 1.20 | -0.394 | 0.74 | 3/5 |
| 5 cross heads (L19X4 L6X10 L24X3 L21X17 L26X9) | 0.37 | 0.37 | 1.38 | -0.387 | 0.40 | 3/5 |
| **`wproj` L13-24 a+m, u = L24 contrast 2** | **0.32** | 0.32 | 0.67 | **-0.375** | 0.78 | 3/5 |
| control: same, random u | 0.69 | 0.70 | 2.03 | -0.370 | 1.00 | 3/5 |
| `wproj` L0-24 a+c+m, u = L24 contrast 2 | 0.21 | 0.22 | -0.01 | -0.379 | 0.62 | 3/5 |
| control: same, random u | 0.68 | 0.68 | 2.01 | -0.369 | 0.90 | 3/5 |
| `rproj` L24, contrast 2 | 0.30 | 0.30 | 0.95 | -0.375 | 0.78 | 3/5 |
| `rproj` L24, contrast 2, α = 2 | 0.09 | 0.09 | -0.60 | -0.386 | 0.53 | 3/5 |
| control: `rproj` L24 random | 0.68 | 0.69 | 2.03 | -0.369 | 0.98 | 3/5 |
| `wproj` L13-14 a+m, u = L14 contrast 1 | 0.65 | 0.68 | 2.45 | -0.378 | 0.90 | 2/5 |
| `rproj` L14, contrast 1 | 0.68 | 0.72 | 2.36 | -0.376 | 0.84 | 2/5 |
| control: `rproj` L14 random | 0.67 | 0.68 | 2.00 | -0.368 | 0.93 | 2/5 |
| neurons L14 N3648 + N1507 zeroed | 0.72 | 0.73 | 2.56 | -0.370 | 0.98 | 2/5 |
| control: 2 random L14 neurons | 0.68 | 0.69 | 2.04 | -0.369 | 1.00 | 2/5 |

Notes:

- Every control (random heads in the same layers, random directions in the same writers, random
  neurons) leaves the Loop rate at 0.68-0.74. The effects above belong to those components.
- Head ablation is strongly non-linear: L12H13 at 0.5 does almost nothing (0.59), at 0 a lot (0.14).
  The early heads are where the circuit is fragile; the late writers are partly redundant.
- With the heads removed, the model mostly writes the window's real audio. ts04_copy now writes
  "... for at hente det servale i laven, gødning ..." (the real speech), bare01 writes "Den erodrede
  sandbanker langs med floden.", nt02 writes a made-up short line instead of the copied window. One
  paraphrased copy slipped past the exact test (ts04_onset under 11 heads: "Det var en af de største
  skib, der havde været på vej til København"); the fuzzy rate agrees with the exact one within
  0.05 everywhere.
- The cost. On the natural cases the CLEAN continuation log-prob falls from -0.26 to -0.73 (11 heads),
  -0.70 (early 6), -0.62 (2 heads), -0.33 (late 5), but only to -0.27 (`wproj` L13-24). The early
  induction heads also help ordinary transcription (they keep spelling and names consistent with the
  previous text, which this metric rewards). Two caveats: "unchanged" is a strict test (any token
  differs within 30 tokens), and random controls already score 0.62-0.79 on it. The real check is
  `nt-vad` WER in Phase 2.
- The natural ts03_onset ("Ecuador ...", greedy stops in Phase 0) starts writing S under the L12H13
  edits. Removing that head also lowers p(end of text) after the pair. Watch onset Loops in Phase 2.

## Weight Edit candidates for Phase 2

All are explicit formulas over the original weights, keep every name and shape, and touch no
`proj_out` / `embed_tokens`, so `scripts/hf_to_ggml.sh` converts them as is. Compute in fp32 from the
fp16 originals and round once to fp16 (conversion.md). Zeroing columns is exact in fp16. The
projections change entries by up to |u|·|uᵀW|, far above fp16's resolution, but compare HF-side
results using the rounded weights anyway.

1. **Induction-head scale (main candidate).** For each head (L, h) in H:

   `W_O^L[:, 64h : 64h+64] ← α · W_O^L[:, 64h : 64h+64]`, with
   `W_O^L = model.decoder.layers.L.self_attn.out_proj.weight` (bias unchanged).

   Same thing in ggml: `decoder.blocks.L.attn.out.weight`, same column slice.
   Variants to run, cheapest first: **H = {L12H13, L25H18}** (2 slices, find-B Loop 68% → 6%);
   H = early 6 {L2H19, L12H13, L13H7, L13H18, L15H17, L18H19} (→ 1%); H = all 11 (→ 0%). α ∈ {0,
   0.25}; 0.5 is too weak for the early heads. Equivalent and also legal: scale rows
   `64h:64h+64` of that layer's `self_attn.v_proj.weight` and `.bias`. Control Edit: the same number
   of random heads in the same layers (already done: 68-74%).

2. **Copy-decision direction out of the mid-layer writers.** u = find-A unit mean difference at
   layer 24 between LOOP states that loop and those that do not (`results/phase1/directions_findA.pt`,
   `U[24][1]`, fp32, 1280). For L in 13..24 and W in {`self_attn.out_proj`, `fc2`}:

   `W.weight ← W.weight - α · u (uᵀ W.weight)`, `W.bias ← W.bias - α · u (uᵀ W.bias)`

   24 matrices + 24 biases. find-B: α = 1 gives 68% → 32% at -0.006 log-p per token (cheapest of
   all). Including `encoder_attn.out_proj` and layers 0-12 gives 21%. The residual-projection twin
   at α = 2 reaches 9%, so try α ∈ {1, 1.5, 2} on the writers. Control Edit: random u, same
   tensors (done: 68-69%).

3. **Combination: late copy writers + direction.** Candidate 1 with H = late 5 {L21H14, L24H0, L25H18,
   L25H19, L26H6} (alone 21%, cost -0.046 on find-B, -0.07 on the natural cases, the cheapest head set)
   plus candidate 2. It avoids the early induction heads, which carry most of the clean cost. Not run
   together yet; it is the first thing to try if candidate 1 fails the `nt-vad` WER guardrail.

## Limits

- The LOOP states are synthetic: S copies always sit at the end of the previous text, formatted as
  one fake window. The 5 natural cases agree with them (the 11- and 6-head edits end all 5), but they
  are few.
- The cost metrics here are log-probs of the model's own short continuations. Long-clip WER,
  Lost Audio and the `nt-vad` guardrail are Phase 2's job.
- The sweep ablates one head at a time with zero ablation on 96 find-A states, so heads that only
  matter together (backups) are under-ranked. The early-6 and late-5 sets came from that ranking
  plus patching and induction scores, then were checked on find-B.
- `timestamps` contributes only 22 of 231 pairs (VAD segments exist for 2 clips). Its k=3 Loop rate
  (27%) is in line with the other paths.

## Files

- `scripts/phase1_common.py`: hand-written decoder forward on a left-padded batch with hooks at every
  site (z per head, attention probs, writes, residuals), batched whisper.cpp-filtered greedy with KV
  cache. Matches HF logits to ≤ 0.03 (fp16).
- `scripts/phase1_windows.py` → `results/phase1/windows/` (22 window files).
- `scripts/phase1_dataset.py` → `results/phase1/dataset.json` (pairs with token ids, prompts, decodes,
  LDs; `valid`; natural cases). `--natural-only` redoes the natural cases.
- `scripts/phase1_localize.py` `sweep | patch | capture | verify` → `results/phase1/localize.json`
  (`sweep`, `patch`, `capture`, `verify` with every greedy text) and `results/phase1/directions_findA.pt`
  (per layer, contrast 1 and 2 unit directions from find-A).
- `scripts/phase1_dla.py` → `localize.json` `dla`; `scripts/phase1_report.py` prints the tables and
  adds the fuzzy Loop rate.

```sh
.venv/bin/python -I scripts/phase1_windows.py      # ~25 min
.venv/bin/python -I scripts/phase1_dataset.py      # ~10 min
.venv/bin/python -I scripts/phase1_localize.py sweep     # ~95 min
.venv/bin/python -I scripts/phase1_localize.py patch     # ~15 min
.venv/bin/python -I scripts/phase1_localize.py capture --cap-file <scratch>/capture.pt   # ~5 min, 1.5 GB file
.venv/bin/python -I scripts/phase1_dla.py --cap-file <scratch>/capture.pt
.venv/bin/python -I scripts/phase1_localize.py verify --no-auto --resume --extra "heads=12.self.13,25.self.18" ...
.venv/bin/python -I scripts/phase1_report.py
```
