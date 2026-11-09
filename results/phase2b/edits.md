# Phase 2b: E3 combined with a fix for made-up text in silence (partial)

**Status: partial, stopped 2026-11-08 ~22:15 when the session closed.** The bar is unchanged from
Phase 2. E7 was fully measured and does not meet it. E9 got only its HF proxy, which shows heavy
damage. C3 was not run. E8 was not run, because the rule was to run it only if E7 hurt short WER,
and E7 did not.

Same harness, machine, argv and scoring as Phase 2 (`results/phase2/edits.md`). Find Set only.
No Held-out or FLEURS test audio was read. `edits.json` was written by
`bun scripts/phase2_measure.ts --out results/phase2b/edits.json baseline E3 C1 C2 E7`.

## Edits

- **E7 = E3 + E4 at α = 1.0**: zero the 11 Phase 1 induction heads, plus
  `W ← W − u(uᵀW)` (α = 1) with the layer-24 direction on `self_attn.out_proj` and `fc2` of
  decoder layers 13-24. 52 tensors. ggml tensor-verified (52/52 equal), sha256 `d43c3d78…`.
  Controls: C1 (heads) and C2 (direction, α = 2).
- **E9 = E3 + Calm-Whisper heads**: zero self-attention heads #1, #6 and #11 in every one of the
  32 decoder layers, plus E3's 11 (L26H6 overlaps): 106 heads, 32 tensors. ggml tensor-verified
  (32/32 equal), sha256 `0b102890…`. Its control C3 is as many random heads per layer
  (seed 2, never one of E9's, never a Phase 1 top-30 head).
- Edit records: `edit_json/E7.json`, `edit_json/E9.json`. HF proxies:
  `results/phase2/proxy/E7.json`, `results/phase2/proxy/E9.json`.

## Results

| Edit | find-B Loop | clean log-p / greedy unchanged | natural | Loops ts / bare / nt / silero / vad | Loops ts+bare+nt (bar: ≤ 3) | short WER (bar: ≤ 9.68%) | `nt-vad` lost | bar |
|---|---|---|---|---|---|---|---|---|
| baseline | 68% | -0.369 | 2/5 | 5 / 10 / 2 / 0 / 0 | 17 | 9.38% | 9 | - |
| E3 | 0% | -0.500 | 0/5 | 0 / 26 / 0 / 0 / 0 | 26 | 9.04% | 8 | ✘ Loops +53% |
| **E7** | 0% | -0.502 / 45% | 0/5 | 1 / 8 / 2 / 0 / 0 | **11 (-35%)** | 9.16% | 8 | ✘ Loops -35% |
| E9 | 3% | **-0.781 / 2%** | 0/5 | not run | - | not run | - | partial: proxy shows heavy damage |
| C1 | 68% | -0.380 | 3/5 | 6 / 14 / 2 / 0 / 0 | 22 | 9.60% | 8 | (control) |
| C2 | 66% | -0.386 | 3/5 | 6 / 14 / 0 / 0 / 0 | 20 | 9.23% | 8 | (control) |
| C3 | - | - | - | not run | - | - | - | - |

E7's other bar conditions pass: lost utterances summed over 5 paths 57 (baseline 101), none
worse per path (5 / 3 / 35 / 6 / 8), `nt-vad` 0 Loops, controls C1 and C2 have no effect.
Spot-check rerun byte-identical. 34 of 155 short utterances changed.

## What E7 shows

- Adding the direction to E3 cuts the made-up filler on `timestamps-bare` (26 to 8 Loops) but
  brings back 2 short Loops on `nt` (E3 had 0) and 1 on `timestamps`. Net 11, so E7 is the
  best edit so far on the three paths (E4 9 was close, but had more copies), and still far from
  ≤ 3.
- 10 of E7's 11 Loops are made-up text, not copies: "af de fleste af de fleste af de fleste"
  (`nt`, `find_da_5min_02`), "vi har været i en af de fleste af de steder …" (112 words, bare
  `find_da_2min_03`), "de to var i det samme sammenhæng …". The one copy is a 47-word
  "at bo og lave frivilligt arbejde …" on bare `find_da_5min_02`. So the copy mechanism is gone.
  What is left is the model looping on its own filler phrase ("af de fleste"), mostly where
  there is no VAD.

## E9 and Calm-Whisper

Calm-Whisper: Y. Wang, A. Alhmoud, S. Alsahly, M. Alqurishi, M. Ravanelli, "Calm-Whisper:
Reduce Whisper Hallucination On Non-Speech By Calming Crazy Heads Down", Interspeech 2025,
arXiv:2505.12969. What the paper says:

- The heads are **decoder self-attention** heads, numbered **0-based** ("#0-#19"). One index
  covers that head in every decoder layer ("heads #1, #6, and #11 in each decoder layer").
- The 99.97% → 24% figure holds, but it comes from **masking alone**, and that costs speech:
  masking #1, #6, #11 drops UrbanSound8K non-speech hallucination from 99.97% to 24.10%, while
  LibriSpeech WER rises from 2.12% to 3.57% (clean) and from 4.07% to 5.98% (other). Their final
  model (15.51% hallucination, +0.07 pp WER) is fine-tuned, so ADR 0001 rules it out. The paper
  does not say how the masking is done; E9 zeroes the head's `out_proj` slice, as E3 does.

E9's proxy fits that cost, and it is worse in Danish on top of E3: find-B Loop 3%, but clean
continuation log-p -0.781 (E3 -0.500, baseline -0.369), and clean greedy output unchanged in 2%
of the clean find-B continuations (E3 and E7: 45%, baseline 100% by definition).
E9 will very likely fail the short-WER guardrail. Its long clips were not run.

## Disk

E7 ggml kept (`models/ggml/edits/ggml-large-v3-E7-f16.bin`), with E3's. E9's ggml was deleted
unmeasured. All safetensors were deleted. Each one can be rebuilt bit for bit from its formula
(see LOG.md).
