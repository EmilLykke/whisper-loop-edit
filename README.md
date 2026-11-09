# Whisper Loop Edit

Research spike, November 2026. Question: can a hand-written edit to Whisper large-v3's weights stop it from looping on long Danish audio, without making it worse elsewhere?

The edited model is on Hugging Face: [emillykkegrann/whisper-large-v3-copy-heads-off](https://huggingface.co/emillykkegrann/whisper-large-v3-copy-heads-off) (transformers safetensors, ggml f16 and q5_0).

**Answer: mostly yes, but it failed the bar we set in advance.** Zeroing 11 decoder self-attention heads removes almost all of the copy Loop on sealed audio (long Loops 21 → 1, words lost to Loops −94%), and random edits of the same size don't. Danish WER does not get worse, and it does not get better. By the frozen bar it fails narrowly: Loops −76% where −80% was required, and `nt-vad` lost one utterance more than the unedited model.

Terms (Loop, Lost Audio, Decode Path, Weight Edit, Control Edit, Find Set, Held-out Set) are defined in [GLOSSARY.md](GLOSSARY.md). The full lab notebook, phase by phase, is [LOG.md](LOG.md).

## Where the idea came from

"Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, 2026; [blog](https://rikiyat.github.io/blog/weight-space-programming/), [code](https://github.com/RikiyaT/lm_understand-logs)) had an LLM agent read a small model's internals and write explicit weight edits, with no training, that removed its reasoning loops. Whisper large-v3 writes the same sentence over and over on long Danish recordings. We asked whether the same approach could remove that.

## Approach

Rules ([ADR 0001](docs/adr/0001-gradient-free-edits-only.md)):

- **No training.** Every Weight Edit is an explicit formula over the original weights (zero a head, project out a direction). Shapes stay the same, so the edited model converts to ggml and runs in crispasr like any other model. Claude (Opus 5.5) was the editor.
- **Find Set / Held-out Set.** Edits were searched for on 10 Long Clips built from FLEURS `da_dk` train/dev. 30 other Long Clips and FLEURS `da_dk` test stayed sealed until one final run.
- **Control Edits.** Every edit was compared with random edits of the same size in the same place.
- **Guardrails.** `nt-vad` (Codictate's slicing path) stays free of Loops, and FLEURS `da_dk` test WER may get at most 0.3 points worse.

Phases:

1. **Reproduce.** Our own ggml conversion matched the official one tensor for tensor. Loops reproduced deterministically in crispasr on the `timestamps` and `-nt` Decode Paths, and in Hugging Face transformers from the same decoder state. Trigger: the previous text Whisper is conditioned on. Remove earlier copies of a sentence from it and the Loop stops.
2. **Localize.** 220 contrastive pairs (previous text with a sentence repeated vs cut out). The Loop is written by induction heads in decoder self-attention (L12H13, L25H18 and 9 more), not by cross-attention as the literature suggested (Glazer et al. 2025). Zeroing the 11 heads took an induced Loop rate from 68% to 0%; random heads did nothing.
3. **Edit on the Find Set.** Eight edits plus controls. E3 (zero the 11 heads) ended every copy Loop on `timestamps` and `-nt`, but on `timestamps-bare` (no VAD) Loops of made-up filler rose 10 → 26. Combining E3 with a direction edit (E7) or with Calm-Whisper's silence heads (E9) did not help; E9 damaged the model.
4. **Bar amended, openly.** `timestamps-bare` was dropped from the bar after seeing the Find Set results ([ADR 0002](docs/adr/0002-timestamps-bare-leaves-the-bar.md)), and the bar was frozen before the final run.
5. **Final run** on the 30 sealed Held-out clips, FLEURS test, Codictate's own long clips, English Long Clips and q5_0.

## Final result (E3, Held-out Set)

| # | Condition | baseline → E3 | |
|---|---|---|---|
| 1 | Loops on `timestamps` + `-nt` at least −80% | 25 → 6 (−76%) | fail |
| 2 | Control Edits do not do the same | C1 −12%, C4 +4% | pass |
| 3 | Lost Audio no worse on any path | `nt-vad` 14 → 15 | fail |
| 4 | `nt-vad` has no Loops | 0 → 0 | pass |
| 5 | FLEURS `da_dk` test WER at most +0.3 points | 12.56 → 12.45% | pass |

Beyond the bar (report only):

- 5 of E3's 6 remaining Loops are 3–4 word phrases written twice, a kind every model has. The sixth is made-up filler, not a copy.
- `timestamps` WER: Held-out 31.0 → 19.7%; Codictate's Danish long clips 17.3 → 11.3%; Codictate's English long clip 17.6 → 1.7%.
- The effect survives q5_0 Quantization (Loops −79%).
- It also holds in Hugging Face transformers long-form decoding (10 Held-out clips, `results/hf_check/report.md`): with OpenAI's recommended fallback settings, Loops 21 → 3 and long Loops 10 → 0; without fallback, WER 112% → 30.6%.
- `timestamps-bare` improved on the Held-out Set (Loops 41 → 23), the opposite of the Find Set. So the Find Set result that ADR 0002 rested on was most likely noise.

## What we learned

- Whisper's copy Loop has a concrete, small mechanism: a handful of induction heads copying earlier text from the previous-text prompt. A gradient-free edit can switch it off.
- Removing it does not make Danish better. On `nt-vad`, which slices audio and has no Loops to begin with, nothing changes. The blog's Gemma finding holds here too: an edit can remove an obstacle, not add ability.
- Long-clip decoding is chaotic. Even a random edit can break or create one big Loop, so only comparisons against Control Edits mean anything, and a single small Find Set misleads (the `timestamps-bare` flip).
- Whisper making up text in silence is a separate failure from copying, and these edits do not fix it. VAD does.

## Reproduce

Only the edit itself (no Codictate needed):

    pip install torch safetensors==0.8.0 huggingface_hub
    python release/make_copy_heads_off.py out/   # prints sha256, must match the published file

The whole study ran on an Apple Silicon Mac (Metal). Requirements:

- Python 3.12 with `requirements.txt` (pinned): `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- Bun 1.3.
- A checkout of [Codictate](https://github.com/EmilLykke/codictate) at commit `0c25733`, read-only. Point `CODICTATE_DIR` at it (default `../codictate`). It supplies the Decode Path argv, VAD slicing, transcript reading and WER scoring, so this repo measures exactly what Codictate shipped. Inside it:
  - `bun run scripts/pre-build.ts` fetches crispasr v0.8.41 and the Silero VAD model into `vendors/`.
  - `bun run scripts/mirror-edda.ts --dry-run` builds `.tmp/edda-mirror/` (the pinned ggml converter, `crispasr-legacy-quantize` and Whisper's mel filters).
  - FLEURS `da_dk` test and LibriSpeech under `benchmarks/datasets/` (see its `benchmarks/README.md`) are needed only for the FLEURS guardrail and Codictate's long clips.

Steps:

    .venv/bin/python -I scripts/build_clips.py download && .venv/bin/python -I scripts/build_clips.py build
    hf download openai/whisper-large-v3 --revision 06f233fe06e710322aca913c1bc4249a0d71fce1 --local-dir models/hf/whisper-large-v3
    scripts/hf_to_ggml.sh models/hf/whisper-large-v3 models/ggml/ggml-large-v3-ours-f16.bin
    .venv/bin/python -I scripts/phase2_edit.py build E3        # also C1, C4, ...
    scripts/hf_to_ggml.sh models/hf/edits/E3 models/ggml/edits/ggml-large-v3-E3-f16.bin
    scripts/final_pipeline.sh && scripts/final_rest.sh && bun scripts/final_measure.ts

Phase 0-2 commands are in LOG.md and in each script's header. Whole-clip decoding is chaotic: other hardware, thread counts or crispasr builds can move individual Loops, so compare against the Control Edits rather than against our exact counts.

## Layout

- `scripts/`: clip building, ggml conversion, Decode Path runners, Phase 1 probes, edit builder (`phase2_edit.py`), scoring.
- `data/manifests/`: Find, Held-out, Codictate and English clip manifests (audio is gitignored and rebuilt by `build_clips.py`).
- `results/phase0..phase2b/`, `results/final/`: every number in the log, with transcripts.
- `models/` (gitignored): every edit rebuilds bit for bit from its formula; commands are in LOG.md.
- `release/`: the standalone edit script and the Hugging Face model card.

## License

Code: Apache-2.0 (LICENSE, NOTICE; one file is copied from Codictate, also Apache-2.0). Edited weights: MIT, like openai/whisper-large-v3. Data references FLEURS and LibriSpeech (CC BY 4.0).
