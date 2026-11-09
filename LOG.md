# Log

Phase status for the spike. Details live in `results/<phase>/`.

## Phase 0: reproduce (2026-11-08) - passed

- Conversion HF safetensors -> ggml is faithful: 1,259 tensors bit-identical to the official `ggml-large-v3.bin` (`results/phase0/conversion.md`).
- Find Set (10 Long Clips, FLEURS da train+dev) built with Codictate's own clip code; Held-out Set built and sealed.
- crispasr large-v3 f16 Loops on the Find Set, deterministically (100/100 decodes byte-identical across runs): `timestamps` on all four 5 min clips, `timestamps` without VAD/beam/prompt on 5 of 10, legacy `-nt` on 2. Legacy `-nt --vad` and `nt-vad` no Loops (`results/phase0/loops_find.md`).
- Long-clip WER pooled: `timestamps` 30.9%, bare 44.9%, `-nt` 37.7%, `nt-silero` 13.6%, `nt-vad` 13.7%.
- HF large-v3 fp16 on MPS, from crispasr's reconstructed decoder state, writes or prefers the same Loop in 5/5 cases (`results/phase0/hf_repro.md`). Teacher-forced mean log-prob per token, Loop vs reference: -0.04 vs -0.67 (ts04), -0.16 vs -0.63 (bare01), -0.04 vs -0.33 (nt02).
- Trigger: previous-text conditioning. Removing the previous text, or only the earlier copies of the looped sentence, ends all Loops. The dictation prompt is not the cause.
- Side findings: `nt-vad` loses 9 utterances on the Find Set without any Loop (early end of text inside a slice). whisper.cpp has no compression-ratio check, and its entropy check (last 32 tokens) misses 19-20 token Loops.

## Phase 1: localize (2026-11-08) - passed (kill condition not met)

- Contrastive Find Set dataset: 152 rebuilt crispasr windows (22 path/clip runs), 231 (window, sentence S) pairs; previous text with S written k times vs S cut out. Induced greedy Loop rate 3.9% / 29.9% / 34.6% for k = 1/2/3. 220 valid pairs (find-A 103, find-B 117, split by clip) + 5 natural cases (`results/phase1/localize.md`).
- The Loop is written by decoder self-attention induction heads, not cross-attention: early L12H13, L13H7, L13H18, L15H17 (+L2H19, L18H19), late copy writers L25H18 (DLA +1.18 of 7.17 nats; patching it alone restores 16%), L25H19, L24H0, L26H6, L21H14. Cross-attention does not collapse; L18H13 has no effect.
- find-B, greedy Loop rate 68% → 0% zeroing those 11 heads (natural 2/5 → 0/5), → 6% zeroing L12H13 + L25H18; random same-layer heads 68-74%. Clean cost: CLEAN continuation log-p -0.369 → -0.50 (11) / -0.46 (2) per token.
- Directions: LOOP vs CLEAN separates at layer 14 (find-B AUC 0.97, probe 0.97) but is not causal; loop vs no-loop at layer 24 (AUC 0.86) is: projecting it out of the layer 13-24 writers gives 68% → 32% at -0.006 log-p. MLP neurons: no effect.
- Phase 2 candidates: scale induction heads' `out_proj` column slices (α 0-0.25); `W ← W - α u uᵀW` on layer 13-24 `out_proj`/`fc2` with the layer-24 direction; late copy heads + direction.

## Phase 2: edits on the Find Set (2026-11-08) - bar not met

Grid of 6 Weight Edits + 2 Control Edits, each converted to ggml (tensors verified bit-equal) and decoded with crispasr on all 10 Find clips x 5 Decode Paths, plus 155 single utterances on `nt-vad` as the damage proxy (`results/phase2/edits.md`).

- No edit reaches Loops -80% on `timestamps` + `timestamps-bare` + `nt` combined (baseline 17). Best: E4 (layer-24 direction, a=1.5) 9 Loops (-47%), short-utterance WER 9.13% vs 9.38%. Controls: C1 22, C2 20 - no effect, so E4's drop is the direction, not the size.
- E3 (zero 11 induction heads) ends every copy Loop on `timestamps` and `nt` (7 -> 0; controls 8 and 6), short WER 9.04%, but on `timestamps-bare` (no VAD) Loops rise 10 -> 26: with the copy heads gone, made-up filler in the 1 s silences ("... denne video", subtitle credits) repeats instead. A different failure: non-speech hallucination, not self-conditioning.
- E5 (a=2) breaks the guardrail: short WER 12.11%.
- The HF proxy does not predict long-form crispasr (E1: proxy 6%, long-clip Loops up 94%). Whole-clip decoding is chaotic: even a random control can break or create a single big Loop, so only Loop counts against the controls mean anything. The +0.3 pp guardrail is about one control's worth of noise.

## Held-out Set extended to 30 clips (2026-11-08)

- Before any Held-out decode: 20 more Long Clips from the unused rest of the Held-out pool, same recipe and seed (`da_5min_05..12`, `da_2min_04..09`, `da_continuous_90s_04..09`), 310 utterances, 3592.47 s. Held-out now 30 clips, 460 utterances, 5384.75 s.
- First 10 Held-out clips and the whole Find Set byte-identical (sha256). No sentence id or text shared with Find or FLEURS test. `build_clips.py check` OK. Still sealed (`data/manifests/README.md`).

## Phase 2b: E3 + a fix for made-up text in silence (2026-11-08) - partial, bar not met

The user kept the Phase 2 bar unchanged (Q15 a) and allowed one more round on the Find Set. It
stopped at about 22:15 when the session closed. Details: `results/phase2b/edits.md`.

- **E7 = E3 + layer-24 direction α = 1** (fully measured): Loops 1 / 8 / 2 on ts / bare / nt =
  11 (-35%, bar needs ≤ 3). Short WER 9.16%, `nt-vad` 0 Loops, 8 lost. Every other condition
  passes, and controls C1 22 and C2 20 have no effect. 10 of the 11 Loops are made-up filler
  ("af de fleste af de fleste ..."), not copies. Best so far on the three paths, still far off.
- **E9 = E3 + Calm-Whisper heads #1/#6/#11 in all 32 layers** (106 heads): HF proxy only.
  find-B Loop 3%, but clean log-p -0.781 and clean greedy unchanged 2% (E3 45%), which is heavy
  damage. Long clips and short WER not run.
- C3 (control for E9) not run. E8 (E3 + α = 0.5) skipped by the pre-set rule, because E7 did not
  hurt short WER.
- Calm-Whisper (Wang et al., Interspeech 2025, arXiv:2505.12969): decoder self-attention heads,
  0-based, the same index in every layer. Masking #1, #6, #11 takes UrbanSound8K non-speech
  hallucination from 99.97% to 24.10%, but LibriSpeech WER goes from 2.12 to 3.57 (clean) and
  from 4.07 to 5.98 (other). The 15.51% / +0.07 pp result needs fine-tuning, which ADR 0001
  rules out. Our earlier summary quoted the 99.97% to 24% drop correctly, but left out that WER
  cost.

### Where we are / next steps

- Bar unchanged. No edit meets it. On `timestamps` + `nt`, E3 has 0 Loops. On all three paths,
  the best is E4 (9) or E7 (11). What is left is made-up filler repeating where there is no
  VAD (`timestamps-bare`).
- Remaining work: either finish E9 (long clips + short WER) and C3, or drop E9 now on the proxy
  damage (my call as editor: drop it, because a 2% unchanged rate is the kind of break that
  failed E5's guardrail). Then the user decides between Q15 (b), amend the bar to drop
  `timestamps-bare` (E3 then meets every condition), and (c), a negative write-up.
- Held-out Set is now 30 clips (`data/manifests/heldout.json`), still sealed. FLEURS da test is
  still sealed too.
- On disk: base ggml, `ggml-large-v3-E3-f16.bin`, `ggml-large-v3-E7-f16.bin`. All safetensors
  are deleted.
- Rebuild any edit from its formula, bit for bit (sha256 in `results/phase2*/edit_json/<id>.json`):

      .venv/bin/python -I scripts/phase2_edit.py build E9           # -> models/hf/edits/E9/
      scripts/hf_to_ggml.sh models/hf/edits/E9 models/ggml/edits/ggml-large-v3-E9-f16.bin
      .venv/bin/python -I scripts/phase2_edit.py verify E9 models/ggml/edits/ggml-large-v3-E9-f16.bin

- Finish the round (build, verify, proxy, 5 paths x 10 Find clips, 155 utterances, score):

      KEEP_GGML="E3 E7" MEASURE_ARGS="--out results/phase2b/edits.json baseline E3 C1 C2 E7 E9 C3" \
        scripts/phase2_pipeline.sh E9 C3 >> results/phase2b/pipeline.log 2>&1

  (~18 min per edit. E9's proxy is already in `results/phase2/proxy/E9.json`, so it is skipped.)

## Bar amended (2026-11-09)

- User chose Q17 (b): `timestamps-bare` leaves the bar, after the Find Set results, done openly in `docs/adr/0002-timestamps-bare-leaves-the-bar.md`. E9 dropped on proxy damage (editor's call), so C3 is not needed. The final run's bar is frozen in ADR 0002 before any Held-out decode. Candidate: E3.

## Final run (2026-11-09) - FAIL

E3 (unchanged, ggml `ce1c3710…`) on the 30 Held-out clips, run 1. Control Edits: C1 (rebuilt bit-identical) and C4 (new, seed 4, same draw as C1). Scripts: `final_pipeline.sh`, `final_rest.sh`, `final_measure.ts`; everything in `results/final/results.json`, the printed tables and every Loop in `results/final/measure.txt`.

| # | Condition (ADR 0002) | baseline | E3 | Result |
|---|---|---|---|---|
| 1 | Loops `timestamps` + `-nt` at least −80% | 15 + 10 = 25 | 4 + 2 = 6 (−76%) | **FAIL** |
| 2 | Control Edits do not reach 1 | - | C1 22 (−12%), C4 26 (+4%) | pass |
| 3 | Lost Audio no worse, ts / nt / silero / vad | 59 / 119 / 19 / 14 | 26 / 107 / 18 / **15** | **FAIL** (`nt-vad` +1) |
| 4 | `nt-vad` 0 Loops | 0 | 0 | pass |
| 5 | FLEURS `da_dk` test WER on `nt-vad` at most +0.3 pp | 12.56% | 12.45% (−0.11) | pass |

- Loop size, `timestamps` + `-nt`: baseline 1,380 extra words (21 Loops of 10+ words), E3 85 (1), C1 730 (15), C4 2,313 (22). Copy Loops: baseline 23, E3 4, C1 21, C4 22.
- E3's Loops by eye: 5 of 6 are 3–4 word re-writes ("Lodin sagde desuden at ændringen …" twice, "til at booke" twice, "nogle af dem" and "en af de" in a second, misheard place). They fit the Glossary, and every model has them. The 6th is made-up filler on `-nt` (5min_10, "hvor de har været i en krig" ×N, 69 words).
- `nt-vad` Lost Audio: 12 in common. E3 loses 3 the baseline keeps and keeps 2 it loses. C1 15, C4 14: within the controls' range, but the bar counts it.
- Pooled WER (ts / nt / silero / vad): baseline 31.0 / 38.3 / 15.5 / 14.4, E3 19.7 / 36.2 / 15.2 / 14.2. FLEURS test: all 930 recordings, none failed. Errors: baseline 2,500, E3 2,477, C1 2,530 of 19,898 words.
- `timestamps-bare` (report only): on the Held-out Set, E3 has *fewer* Loops: 41 → 23, lost 84 → 11, WER 56.5 → 23.7%. That is the opposite of the Find Set (10 → 26), which ADR 0002 was based on.
- Extras (report only, baseline → E3):
  - **Codictate's long-audio clips:**
    - Danish Loops: ts 1 → 0, nt 0 → 0, vad 0 → 0.
    - Danish WER: ts 17.3 → 11.3%, nt 19.3 → 19.7%, vad 11.2 → 12.0%.
    - English ts: Loops 2 → 0 (the baseline writes 160 made-up words), WER 17.6 → 1.7%. nt and vad unchanged.
  - **English Long Clips** (FLEURS `en_us` dev, 5 clips):
    - ts: Loops 8 → 2, WER 106.6 → 24.6%.
    - nt: Loops 3 → 3, WER 48.2 → 51.9%. Made-up filler on both models.
    - vad: unchanged (7.3 / 7.2%).
  - **Private 65 s dictation:** all 4 check phrases kept on ts, nt and vad by both models. Transcripts are not committed.
  - **q5_0** (`crispasr-legacy-quantize`), Held-out ts + nt: Loops 33 → 7 (−79%), ts WER 32.4 → 20.3%, nt 40.7 → 36.9%.
- Reading: E3 removes almost all of the copy Loop volume on the Held-out Set, and the controls show it is the heads, not the edit's size. It does no harm to FLEURS WER, `nt-vad`, English `nt-vad` or q5_0. But by the frozen bar it fails, narrowly, on Loop count and on one `nt-vad` utterance.

## Release prep (2026-11-09)

User chose to publish E3 as `whisper-large-v3-copy-heads-off` (personal Hugging Face account) with this repo public on GitHub. Prepared locally only; nothing uploaded or pushed.

- **transformers check** (`results/hf_check/report.md`, 10 Held-out clips, whole-clip `generate`, greedy, no prompt, no VAD, no controls): recommended fallback settings, Loops 21 → 3, Loops of 10+ words 10 → 0, WER 32.3 → 29.5%; no fallback, Loops 18 → 7, Loop words 3,132 → 388, WER 112.0 → 30.6%. E3's remaining long Loops are made-up filler, as in the final run.
- **Portability:** Codictate is read from `CODICTATE_DIR` (default `../codictate`) through `scripts/codictate.ts`; Decode Path argv for Find and Held-out are byte-identical before and after, VAD slice files identical, and re-scoring the final run gives the same results.json. `requirements.txt` pinned from the .venv.
- **Absolute paths** (`/Users/...`) scrubbed from 5,308 tracked result files (crispasr JSON and meta); text-only, scoring unchanged. Older commits still contain them.
- **`release/make_copy_heads_off.py`** rebuilds E3 from `openai/whisper-large-v3` at `06f233fe`, tested end to end: sha256 `2c08dd6c…` matches. Model card and staged files in `release/hf/` (large files gitignored, hard links to `models/`).
- LICENSE (Apache-2.0) and NOTICE added for the code.

## Published (2026-11-09)

- Code: https://github.com/EmilLykke/whisper-loop-edit (one squashed commit; the full local history is kept only in the local branch `history-local`).
- Model: https://huggingface.co/emillykkegrann/whisper-large-v3-copy-heads-off (safetensors, ggml f16, ggml q5_0, make script, model card). Uploaded sha256 values match the card.
