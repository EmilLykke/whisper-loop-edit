# Phase 2: Weight Edits on the Find Set

**Verdict: no edit meets the pre-registered bar.** All of them fail its first condition (Loops
on `timestamps` + `timestamps-bare` + `nt` down at least 80%), because `timestamps-bare` keeps
looping under every edit. On the other two paths the head edit E3 removes every Loop (7 to 0),
E5 nearly does (7 to 1, but it breaks the short-utterance guardrail) and E4 halves them (7 to 2).
On `timestamps-bare`, E3 trades copied sentences for made-up filler that repeats itself (10 Loops
to 26), and the best edit there, E4, still has 7. Both Control Edits leave the Loops where they
were (C1 22, C2 20, baseline 17), so whatever the edits do change belongs to them. Every edit
keeps `nt-vad` free of Loops. Every edit but E5 keeps short-utterance WER within 0.3 pp.

Run 2026-11-08 on the M4 Max, crispasr v0.8.41, 14 threads, torch 2.15 MPS. Find Set only: the
10 Long Clips and the 155 FLEURS da_dk train/dev utterances they are made of. No Held-out or FLEURS
test audio was read.

## Summary

Loops are counted by Phase 0's detector. Lost is utterances with under 30% heard. WER is
pooled over the 10 clips. "Short WER" is pooled over the 155 single utterances on `nt-vad`
(3,197 reference words, so 1 word = 0.03 pp). Bar: ✔ meets all five conditions, ✘ fails the
ones listed.

| Edit | find-B Loop rate | natural Loops | Loops ts / bare / nt / silero / vad | `timestamps` heard / lost | `nt` heard / lost | long-clip WER % ts / bare / nt / silero / vad | short WER | `nt-vad` lost | bar |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 68% | 2/5 | 5 / 10 / 2 / 0 / 0 | 78.9% / 18 | 63.7% / 43 | 30.9 / 44.9 / 37.7 / 13.6 / 13.7 | 9.38% | 9 | - |
| E1 2 heads | 6% | 1/5 | 7 / 25 / 1 / 0 / 0 | 83.3% / 7 | 67.3% / 34 | 22.1 / 47.9 / 34.1 / 13.4 / 14.5 | 9.01% | **10** | ✘ Loops +94% |
| E2 late 5 | 21% | 2/5 | 4 / 9 / 1 / 0 / 0 | 77.3% / 18 | 63.4% / 43 | 25.8 / 40.1 / 37.7 / 13.4 / 14.0 | 9.63% | 9 | ✘ Loops -18% |
| E3 all 11 | **0%** | **0/5** | **0** / 26 / **0** / 0 / 0 | 82.9% / 7 | 68.2% / 35 | 19.3 / 37.2 / 33.2 / 13.4 / 13.2 | 9.04% | 8 | ✘ Loops +53% |
| E4 dir α=1.5 | 6% | 2/5 | 2 / 7 / **0** / 0 / 0 | 82.7% / 7 | 66.3% / 36 | 20.5 / 28.5 / 34.6 / 13.4 / 13.3 | 9.13% | 9 | ✘ Loops -47% |
| E5 dir α=2 | **0%** | 1/5 | **0** / 9 / 1 / 0 / 0 | 81.5% / 8 | 65.4% / 35 | 20.3 / 21.6 / 35.9 / 14.5 / 15.2 | **12.11%** | 5 | ✘ Loops -41%, short WER +2.73 pp |
| E6 late 5 + dir α=1 | 11% | 1/5 | 5 / 7 / 2 / 0 / 0 | 80.2% / 14 | 62.3% / **44** | 24.5 / 30.2 / 38.8 / 13.4 / 14.6 | 9.45% | **10** | ✘ Loops -18% |
| C1 (control for E3) | 68% | 3/5 | 6 / 14 / 2 / 0 / 0 | 84.7% / 3 | 65.9% / 38 | 19.9 / 49.0 / 35.1 / 13.5 / 13.3 | 9.60% | 8 | (control) |
| C2 (control for E5) | 66% | 3/5 | 6 / 14 / 0 / 0 / 0 | 77.6% / 16 | 64.8% / 40 | 40.2 / 47.0 / 36.2 / 13.4 / 13.8 | 9.23% | 8 | (control) |

### Each condition of the bar

| Edit | Loops ts+bare+nt (baseline 17) | Lost, 5 paths summed (baseline 101) | Lost per path, none worse | Short WER ≤ 9.68% | `nt-vad` 0 Loops | control without the effect |
|---|---|---|---|---|---|---|
| E1 | 33 ✘ | 68 ✔ | ✘ (`nt-vad` 10) | ✔ 9.01 | ✔ | ✔ C1 |
| E2 | 14 ✘ | 89 ✔ | ✔ | ✔ 9.63 | ✔ | ✔ C1 |
| E3 | 26 ✘ | 63 ✔ | ✔ | ✔ 9.04 | ✔ | ✔ C1 (22 Loops) |
| E4 | 9 ✘ | 59 ✔ | ✔ | ✔ 9.13 | ✔ | ✔ C2 |
| E5 | 10 ✘ | 56 ✔ | ✘ (`nt-silero` 7) | ✘ 12.11 | ✔ | ✔ C2 (20 Loops) |
| E6 | 14 ✘ | 76 ✔ | ✘ (`nt-vad` 10, `nt` 44) | ✔ 9.45 | ✔ | ✔ C1, C2 |

"Lost Audio not worse" is scored on the sum over the five paths (it is reported per path too).
"Control without the effect" means its own Loop drop on the three paths is under 80%. C1 serves
as the control for every head edit (it zeroes 11 heads, more than E1 and E2), and C2 for every
direction edit (α = 2, more than E4).

### Without `timestamps-bare` (not the bar, for reading)

Loops on `timestamps` + `nt`: baseline 7, E1 8, E2 5, **E3 0**, **E5 1**, E4 2, E6 7, C1 8,
C2 6. Extra words in Loops on the three paths: baseline 1,610, E1 1,088, E2 853, E3 570, E4 447,
E5 158, E6 629, C1 924, C2 1,860.

## What to read into the long-clip numbers, and what not to

- **Any edit, even a random one, breaks the worst Phase 0 Loop.** `timestamps` `da_5min_04` (one
  sentence 43 times, 16 utterances lost, WER 100%) runs clean under C1 (WER 27%, 1 lost) as well
  as under E3 (26%, 2 lost). That one clip moves `timestamps` lost from 18 to 3-8 and pooled WER
  from 30.9% to about 20% for edits and controls alike, so **the `timestamps` heard / lost / WER
  gains are not evidence for any edit.** C2 shows the other side: its `da_5min_03` writes one
  sentence over 504 extra words (WER 40.2%). Long-clip decoding is chaotic. A small change
  anywhere sends a 5 min decode down a different path, so per-clip outcomes swing both ways. The
  Loop *count*, compared with the two controls, is the steadier signal (controls: 6 / 14 / 2 and
  6 / 14 / 0).
- **Two kinds of Loop.** Phase 0's Loops mostly write heard speech again (a copied sentence,
  often misheard). The edits that remove the induction heads leave a second kind: made-up text
  the audio never says, repeated (YouTube-style filler, "Danske tekster Jesper Buhl Scandinavian
  Text Service 2018"). A rough split (a Loop is *made-up* when under 10% of its word trigrams are
  in the clip's reference) counts made-up Loops on the three paths as baseline 5, E1 15, E2 6,
  **E3 17**, E4 1, E5 3, E6 5, C1 10, C2 2. Copied ones: baseline 12, E1 19, E2 8, **E3 9**,
  E4 8, E5 7, E6 9, C1 12, C2 18. So E3's extra bare-path Loops are mostly made-up filler.
  Without the induction heads the model still repeats itself, but it repeats what it is
  inventing in a silence, not the previous text. The split is rough (a 0.10-0.35 share can be
  either kind). The worst examples below were read by eye.
- **`timestamps-bare` is where the remaining Loops live.** No VAD means 1 s gaps of digital
  silence decoded as windows, no beam and no prompt. Every edit's worst Loop is on this path
  except C2's. Codictate does not ship this argv (it is ADR-0010's first check).
- **Spot check:** for every edit, a second run of (`timestamps`, `da_5min_04`) gave
  byte-identical JSON to the first. Decoding stays deterministic, as in Phase 0.

## Per edit

Relative Frobenius change is per touched tensor, `‖W_edit,fp16 − W₀‖ / ‖W₀‖`, from `edit.json`.

- **E1, zero L12H13 + L25H18** (2 `out_proj` tensors, rel. change 0.24 / 0.29). The proxy matches
  Phase 1's activation twin to the third decimal (6%, -0.462, 1/5; ts03_onset now loops). On long
  clips it is the worst edit: bare 10 to 25 Loops, `timestamps` 5 to 7, and `nt-vad` loses one
  more utterance (10). Its worst Loop is made-up ("det var en del af det der var i gang" repeated,
  413 extra words, bare `da_5min_02` 87 s).
- **E2, zero the late 5** (4 tensors, 0.24-0.42). Proxy 21%, the cheapest head edit on clean
  log-p (-0.415) and on short WER (+0.25 pp, 13 utterances changed). The Loops barely move
  (14). Worst: "uden at være forsigtig ved udskytning af pæren" repeated, 391 extra words, bare
  `da_5min_04` 181 s (made-up wording around a real utterance).
- **E3, zero all 11** (9 tensors, 0.18-0.42). Proxy 0% and 0/5 natural, as in Phase 1. On the
  shipped `timestamps` argv and on legacy `nt` it writes **no Loop at all** (C1: 6 and 2). Lost
  Audio does not get worse anywhere (`nt-vad` 8, `nt` 35, bare 7 against 25). Short WER 9.04%
  (-0.34 pp; 5 utterances worse, 10 better). Bare goes up to 26 Loops, 13 of them on
  `da_5min_02` alone, mostly made-up filler written into silence. Worst: "en af de fleste af de
  københavnere i europa han har været en af de største indenfor landet ..." repeated, 97 extra
  words, bare `da_5min_04` 237 s.
- **E4, layer-24 direction α = 1.5** (48 tensors: weights 0.044-0.057, biases up to 0.11). Proxy
  6% at a small clean cost (-0.403). It is the edit with the fewest Loops on the three paths (9)
  and the fewest made-up ones (1). `nt` has 0, `timestamps` 2 (a "subsistenslandbrug" copy, 33
  words, and "det 100 meter lange skib var på vej for" twice, the trace of the 43x Loop), bare 7.
  Bare lost goes from 25 to 1. Short WER 9.13%. Worst: "år han har været i rwanda i år ..."
  repeated, 163 extra words, bare `da_5min_04` 226 s.
- **E5, the same direction at α = 2** (48 tensors: weights 0.058-0.076, biases up to 0.15). At
  α = 2 the edit is a reflection: it flips the u component. Proxy 0%, but the clean cost is the
  largest (-0.694, CLEAN greedy unchanged 28%). Long-clip Loops 0 / 9 / 1. **Short WER 12.11%
  (+2.73 pp, 64 utterances worse)**, `nt-silero` and `nt-vad` WER up 0.9 and 1.5 pp, and
  `nt-silero` loses one more utterance. It is the only edit that clearly damages ordinary
  transcription. `nt-vad` loses only 5 utterances (baseline 9), but with that damage it is not
  a clean win. One single utterance swaps "Danske tekster af Nicolai Winther" for "Danske
  tekster af Jesper Buhl Scandinavian Text Service 2018". Worst: a 41-word copy of a misheard
  sentence, bare `da_5min_02` 153 s.
- **E6, late 5 + direction α = 1** (50 tensors). Proxy 11%, clean -0.420. Loops 5 / 7 / 2 (14).
  `nt-vad` loses 10 and `nt` 44, one more each than baseline. Worst: "den japanske judokæmper
  hitoshi saito vinderen af to olympiske guldmedaller er død ..." repeated, 284 extra words, bare
  `da_5min_04` 222 s.
- **C1, 11 random heads** (L12H17, L13H11, L13H9, L15H5, L2H5, L18H0, L25H1, L25H0, L24H4, L26H16,
  L21H12: one per E3 slot, same layer, seed 0, none of E3's heads and none of Phase 1's top-30
  self-attention heads; 9 tensors, 0.16-0.35, about the size of E3). Proxy 68% (no effect), Loops
  6 / 14 / 2. So E3's `timestamps`/`nt` result comes from those 11 heads, not from removing any 11
  heads. Bare loses 30 utterances. Worst: "det er en del af det som er vigtigst for en
  uddannelsesforskning" repeated, 242 extra words, bare `da_2min_01`.
- **C2, random direction α = 2** (u: Gaussian, seed 1024, orthogonalised against U[24][1], so cos
  = 0; 48 tensors, weights 0.054-0.058, a little smaller than E5's 0.058-0.076, because the real u
  lies along directions the writers actually use). Proxy 66%, Loops 6 / 14 / 0, short WER 9.23%.
  A random reflection of the same size costs almost nothing and does nothing to Loops. Worst
  (of any run here): `timestamps` `da_5min_03`, "de fleste af de bomeranger man kan prøve at
  kaste i blæsevær ..." repeated over 504 extra words, 172-178 s.

## The edits, as built

`scripts/phase2_edit.py build <id>` loads `models/hf/whisper-large-v3/model.safetensors` (fp16),
applies the formula in fp32, rounds each touched tensor once to fp16 and writes
`models/hf/edits/<id>/` (safetensors, the tokenizer/preprocessor/config files, `edit.json`).
Every tensor name, shape and dtype is unchanged (asserted). `edit.json` (copied to
`results/phase2/edit_json/<id>.json`) holds the formula, the heads or the direction (u itself, α,
layers), seeds, every touched tensor with its ggml name, relative Frobenius change, max |ΔW|, fp16
rounding error and the sha256 of the safetensors and the ggml.

| Edit | Formula | Tensors | safetensors sha256 | ggml sha256 |
|---|---|---|---|---|
| E1 | `out_proj.weight[:, 64h:64h+64] ← 0` for L12H13, L25H18 | 2 | `ae346c0b…` | `4eed39be…` |
| E2 | same, L21H14, L24H0, L25H18, L25H19, L26H6 | 4 | `4cc7db1a…` | `8c38fead…` |
| E3 | same, all 11 | 9 | `2c08dd6c…` | `ce1c3710…` |
| E4 | `W ← W − α u(uᵀW)`, `b ← b − α u(uᵀb)`, `self_attn.out_proj` + `fc2`, L13-24, u = U[24][1], α = 1.5 | 48 | `e7eefe1d…` | `158fb729…` |
| E5 | same, α = 2 | 48 | `9e6bbde9…` | `0d5fc44e…` |
| E6 | E2's heads + E4's formula at α = 1 (they commute: the projection acts on rows, zeroed columns stay zero) | 50 | `b05d8b62…` | `4235b011…` |
| C1 | E3's formula, C1's 11 heads | 9 | `d6218a59…` | `e23c87a2…` |
| C2 | E5's formula, random u ⟂ U[24][1] (seed 1024) | 48 | `b8ca55b7…` | `1e1cc60f…` |

**Conversion check: every edit, not just one.** After `scripts/hf_to_ggml.sh`,
`phase2_edit.py verify` maps each of the 1,259 ggml tensors to its HF name with the converter's
own rule and compares bits with the edited fp16 HF tensor (1-D and positional tensors are stored
as f32 holding the fp16 value). Result for all 8: 1,259 checked, every touched tensor present and
equal, 0 mismatches. The HF proxy loads the same edited safetensors in fp16, so HF and crispasr
run identical weights.

## Method

1. **HF proxy** (`scripts/phase2_proxy.py <id>`): Phase 1's `verify` evaluation unchanged
   (`phase1_localize.evaluate`), run on the edited weights with no hooks: 117 find-B LOOP states
   (greedy 24 tokens, Loop = writes S's first 4 words), 58 CLEAN states (continuation log-prob
   per token, greedy unchanged over 30 tokens), and the 5 natural Phase 0 cases. Baseline
   reproduces Phase 1 exactly (68%, -0.369, 2/5), and E1, E2, E3 reproduce Phase 1's activation
   twins exactly (6% / -0.462, 21% / -0.415, 0% / -0.500). Per-edit JSON in `proxy/`, with every
   greedy text. Extra columns there: LOOP logit diff (E3 -1.66, E4 -1.87, E5 -3.25, baseline
   +2.04), fuzzy Loop rate (within 0.05 of the exact rate everywhere; E1 6% exact, 11% fuzzy).
2. **crispasr** (`scripts/run_paths.ts --model <ggml> --out results/phase2/transcripts/<id>
   --runs 1`): Phase 0's five argv, unchanged, on all 10 Find clips, run 1. Plus run 2 of
   (`timestamps`, `da_5min_04`) per edit. The baseline is Phase 0's own run 1 (same harness,
   argv, crispasr, threads; its ggml is tensor-identical to the official file). Re-scoring it with
   this script reproduces `loops_find.md` to the decimal.
3. **Short utterances** (`scripts/phase2_utts.ts`): each of the 155 utterances decoded alone from
   its FLEURS source WAV (`data/raw/fleurs_da_dk/extracted/{train,dev}/`, 16 kHz float32, fed as
   is) with the `nt-vad` argv and slices from Codictate's `planVadSlices`, as Codictate's benchmark
   runs a FLEURS clip. WER: `computeWer` per utterance with Codictate's `normalizeForWer`,
   operations summed over all 155. Baseline 9.38% (300 of 3,197 words). For scale, the two
   controls move it -0.15 and +0.22 pp, so the +0.3 pp guardrail is about one control's noise.
4. **Scoring** (`scripts/phase2_measure.ts` → `edits.json`): Phase 0's `scoreRun` (now exported
   from `measure.ts`), pooled per path, plus the bar, the copy/made-up split, the worst Loop per
   edit and every single utterance whose text changed against baseline.

## Weight files kept and deleted (disk)

The disk had 6-20 GiB free, and one edit is 3 GB of safetensors plus 3 GB of ggml, so weights
were pruned as the run went (`results/phase2/pipeline.log` logs each deletion):

- `models/hf/edits/<id>/model.safetensors`: deleted for every edit once its ggml was converted
  and verified and its HF proxy was done. `edit.json`, the tokenizer files and the sha256 stay.
- ggml: **only `models/ggml/edits/ggml-large-v3-E3-f16.bin` is kept.** C1's was deleted before E4
  was built (free space under 10 GiB). E1's was deleted by the coordinator, and E2, E4, E5, C2 and
  E6 were deleted each as soon as its crispasr runs and single-utterance decodes were done. The
  `.convert.log` files and the ggml sha256 in `edit.json` stay.
- The coordinator also deleted `models/ggml/ggml-large-v3-official.bin` (re-downloadable at its
  pinned sha256; its tensors are bit-identical to `ggml-large-v3-ours-f16.bin`, which stays).
- E3's ggml was rebuilt once mid-run (an earlier version of the pipeline had deleted it). The
  rebuild gave the same safetensors sha256 (`2c08dd6c…`) and ggml sha256 (`ce1c3710…`) as the
  first build, so `phase2_edit.py build <id>` + `hf_to_ggml.sh` reproduce any pruned file bit for
  bit.

## Reproduce

```sh
scripts/phase2_pipeline.sh                 # baseline E3 C1 E1 E2 E4 E5 C2 E6, ~2.5 h, resumable
# or per step:
.venv/bin/python -I scripts/phase2_edit.py build E4
scripts/hf_to_ggml.sh models/hf/edits/E4 models/ggml/edits/ggml-large-v3-E4-f16.bin
.venv/bin/python -I scripts/phase2_edit.py verify E4 models/ggml/edits/ggml-large-v3-E4-f16.bin
.venv/bin/python -I scripts/phase2_proxy.py E4
bun scripts/run_paths.ts --model models/ggml/edits/ggml-large-v3-E4-f16.bin --out results/phase2/transcripts/E4 --runs 1
bun scripts/phase2_utts.ts --model models/ggml/edits/ggml-large-v3-E4-f16.bin --out results/phase2/transcripts/E4/utts
bun scripts/phase2_measure.ts              # -> results/phase2/edits.json + the summary table
```

Per edit, about 13 min of crispasr on the long clips, 4.5 min on the utterances, 75 s of proxy and
30 s to build and convert.
