# Release check: E3 in Hugging Face transformers (2026-11-09)

Before publishing, we checked whether E3's effect is specific to whisper.cpp/crispasr or also
shows up in transformers' own long-form decoding, which most Hugging Face users run.

- **Clips:** the first 10 Held-out Long Clips in manifest order (4 × 5 min, 3 × 2 min,
  3 × 90 s continuous; 1,792 s, Danish). These were already unsealed by the final run; this is
  a second engine on the same audio, not a new held-out test.
- **Decoding:** transformers 5.22.2, torch 2.15 MPS fp16, `generate` on the whole clip
  (sequential long-form), `language="da"`, `return_timestamps=True`, greedy, no prompt, no VAD
  (`scripts/hf_check.py`).
  - `fallback`: OpenAI/HF's recommended settings: `condition_on_prev_tokens=True`, temperature
    (0.0, 0.2, ..., 1.0), `compression_ratio_threshold=1.35`, `logprob_threshold=-1.0`,
    `no_speech_threshold=0.6`.
  - `nofallback`: `condition_on_prev_tokens=True`, temperature 0 only, no thresholds; the
    closest to whisper.cpp's behaviour on a Loop.
- **Scoring:** the final run's scorer, unchanged (Phase 0 `scoreRun`, copy/made-up tag), via
  `scripts/hf_check_measure.ts`. Data: `results.json`, transcripts under `transcripts/`.
- **No Control Edits** were run here; the controls are in the final run (whisper.cpp).

| config | model | Loops (copy / made-up) | Loops 10+ words | Loop words | Lost Audio | WER |
|---|---|---|---|---|---|---|
| fallback | large-v3 | 21 (14 / 7) | 10 | 388 | 9 | 32.3% |
| fallback | E3 | 3 (1 / 2) | 0 | 11 | 2 | 29.5% |
| nofallback | large-v3 | 18 (10 / 8) | 11 | 3,132 | 37 | 112.0% |
| nofallback | E3 | 7 (2 / 5) | 4 | 388 | 3 | 30.6% |

Reading:

- **The effect carries over to transformers.** With the recommended fallback, the stock model
  still loops (21 Loops, 10 of them 10+ words, mostly copies); E3 has 3 Loops of 3–5 words and
  no long one. The fallback catches the worst runaways in both models, but not the copies.
- **Without fallback the stock model runs away** (one Loop of 1,539 words on 5min_03, WER 112%);
  E3 stays at 30.6%. E3's remaining long Loops there are made-up filler, not copies
  ("hvis du har en kæreste så er det en god idé …" on 5min_02, "… i denne video" on 2min_02),
  the same failure the final run saw: E3 removes copying, not invented text.
- 10 clips, no controls, one run: supporting evidence, not a second bar.
