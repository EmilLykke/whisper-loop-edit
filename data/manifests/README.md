# Long Clip manifests

`find.json` is the Find Set (10 Long Clips) and `heldout.json` the Held-out Set (30 Long
Clips; see GLOSSARY.md), built from FLEURS `da_dk` **train and dev** only. The FLEURS test split is
Codictate's WER benchmark: it is never downloaded or read here. The Held-out Set is sealed:
built, never transcribed until the final measurement (ADR-0001).
It was unsealed for the final run on 2026-11-09 (ADR-0002, `results/final/`).

`codictate.json` (Codictate's own long-audio check clips, `scripts/codictate_clips.ts`) and
`english.json` (5 English Long Clips from FLEURS `en_us` dev, `scripts/build_english.py`, seed
`whisper-loop-edit/2026-11-09/english`) are report-only extras of the final run.

The audio is not committed (`data/raw/`, `data/clips/` are gitignored). Every WAV's sha256 is
in its manifest.

## Regenerate

```sh
.venv/bin/python -I scripts/build_clips.py download  # google/fleurs @ 70bb2e84, train+dev tsv and tar.gz
.venv/bin/python -I scripts/build_clips.py build     # data/clips/{find,heldout}/*.wav + these manifests
.venv/bin/python -I scripts/build_clips.py check     # rebuild in a temp dir; manifests and sha256 must match
```

Downloads land in `data/raw/fleurs_da_dk/data/da_dk/`; the tars are extracted to
`data/raw/fleurs_da_dk/extracted/{train,dev}/`. The build is deterministic: no RNG, every
choice is a sort on `sha256(SEED, ...)`, with `SEED = "whisper-loop-edit/2026-11-08"`.

## Recipe

Mirrors Codictate's long-audio check,
`../codictate/src/bun/utils/whisper/engines/long-audio.manual.ts` (`readWavSamples`,
`speechBounds`, `trimSilence`, `buildLongClip`, `fleursDanishPool`), ported function by
function:

- **Audio.** FLEURS's 16 kHz mono float32 WAVs are read as Codictate reads them: clamped to
  [-1, 1], times 32767, rounded the way JS `Math.round` rounds (half up), to int16. Clips are
  written as 16 kHz mono PCM16 with the same 44-byte header.
- **Reference text.** `raw_transcription` (TSV column 3, cased and punctuated), as Codictate's
  `fleursDanishPool` uses it, not the normalized `transcription`.
- **One recording per sentence**, as `fleursDanishPool` does, so no sentence is spoken twice in
  a set (a repeated sentence would look like a Loop).
- **Joining (`buildLongClip`).** Walk the pool in order. Skip an utterance longer than 25 s, or
  one that would take the clip past `target * 1.05` (gap not counted, as in Codictate). Put
  the gap before every utterance but the first. Stop once the clip reaches `target * 0.95`.
- **With pauses:** 1 s of digital silence between utterances, utterances untrimmed
  (`da_5min_*` target 300 s, `da_2min_*` target 120 s).
- **Continuous** (`da_continuous_90s_*`, target 90 s): each utterance first trimmed of its own
  leading and trailing silence (`speechBounds`: 20 ms frames, quieter than 2% of the loudest
  frame), then joined with 0.05 s, which is under the `nt-vad` planner's 98 ms pause minimum.
- **Per-utterance times.** `start_s`/`end_s` are where the utterance (trimmed or not) sits in
  the clip; `speech_start_s`/`speech_end_s` are its speech by `speechBounds`, as Codictate's
  `Window` records them. `source_from_s`/`source_to_s` are the part of the source WAV used
  (the trim, for continuous clips).

## Where this differs from Codictate, and why

1. **Source split.** train + dev instead of test, so Codictate's benchmark stays untouched.
2. **Split by sentence.** Sentences (TSV `id`; ids sharing a `raw_transcription` would be
   merged, though none do) are split 50/50 between the sets by seeded order, so no sentence
   is in both sets, in any speaker's voice.
3. **Order and recording choice.** Codictate takes the first recording of each sentence in
   `test.tsv` order. Here the recording per sentence and the order within a set are both
   seeded (sha256), because train.tsv's order is not a natural one to keep and a fixed seed
   gives the same determinism.
4. **Many clips per kind, no reuse.** Codictate builds one clip of each kind and its `da_2min`
   and `da_5min` both start at the top of the same pool (the 2 min clip is the first part of
   the 5 min one). Here a set's 10 clips are built in order (5 min x4, 2 min x3, continuous
   x3) from one shared pool, and a used recording leaves the pool, so no recording appears
   twice within or across sets. An utterance skipped for overshooting one clip stays in the
   pool for the next.

## Held-out extension (2026-11-08, after Phase 2)

Phase 2 showed whole-clip decoding swings a lot even under a Control Edit, so 10 clips (17
baseline Loops on Find) are too few to judge a -80% bar. The Held-out Set was extended to 30
clips, before any Held-out decode, by running the same 10-clip plan two more times
(`ROUNDS["heldout"] = 3` in `build_clips.py`) over what was left of the Held-out pool:
`da_5min_05..12`, `da_2min_04..09`, `da_continuous_90s_04..09`.

No new seed: the Held-out half of the sentence split already held 614 unused sentences in
seeded order, and walking on through it is the same recipe as the first 10 clips. Rounds 2-3
come after round 1, so `heldout_*_01..04`/`01..03` and the whole Find Set are byte-identical to
before (same sha256, same manifest entries; verified). Still sealed: built, not transcribed.

5. **No `CONTINUOUS_FROM`.** Codictate starts its continuous clip at a fixed test recording
   where a loss was found. That recording is in the test split, so continuous clips here start
   wherever the pool is.

## Summary

From `build` (also under `summary` in each manifest):

- FLEURS `da_dk` train + dev: 2860 recordings (train 2465, dev 395), all with a readable WAV,
  of 1528 sentences (train and dev share none; no two ids share a transcription).
- 764 sentences assigned to each set. Find uses 155 of them (141 train, 14 dev recordings),
  1806.15 s in all; Held-out uses 460 (414 train, 46 dev), 5384.75 s (the first 10 clips 150
sentences, 1792.28 s; the 20 added clips 310, 3592.47 s). The rest stay unused (609 on the
Find side, 304 on the Held-out side).

| Clip kind | Find durations (s) | Held-out durations (s) |
|---|---|---|
| `da_5min_01..04`, 1 s gaps | 286.44, 300.62, 293.68, 292.88 | 294.38, 299.38, 289.60, 285.50 |
| `da_2min_01..03`, 1 s gaps | 118.52, 119.88, 121.22 | 118.78, 119.88, 121.20 |
| `da_continuous_90s_01..03`, 0.05 s, trimmed | 89.28, 93.45, 90.18 | 90.24, 86.88, 86.44 |
| `da_5min_05..12` (Held-out only) | | 286.84, 286.68, 287.34, 292.20, 294.60, 286.60, 291.88, 290.06 |
| `da_2min_04..09` (Held-out only) | | 124.36, 121.32, 119.20, 123.92, 120.22, 117.98 |
| `da_continuous_90s_04..09` (Held-out only) | | 94.52, 93.83, 89.33, 86.70, 90.97, 93.92 |

## Verified

- `check` rebuilds both sets into a temp dir and matched every sha256 and both manifests
  (rerun after the extension: OK, 40 clips).
- The port was checked against Codictate's own code: `readWavSamples`, `writeWav`,
  `speechBounds`, `trimSilence` and `buildLongClip` copied verbatim from
  `long-audio.manual.ts` and run under bun over the same pool order built all 20 clips
  byte-identical (same sha256) with the same per-utterance times.
- No sentence id or reference text in either set appears in FLEURS `da_dk` test.tsv (checked
  against Codictate's local copy, again after the extension: 0 of 460 Held-out ids and texts;
  the build itself never reads it).
