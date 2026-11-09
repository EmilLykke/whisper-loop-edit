# Phase 0: Loops and Lost Audio of unedited large-v3 on the Find Set

**Verdict.** Loops reproduce on the two **timestamps** Decode Paths and, less often, on
legacy **`-nt`**. They do not show up on the two `-nt` paths that decode VAD regions or
slices (`nt-silero`, `nt-vad`). Today's shipped `timestamps` argv loops on all four 5 min
clips. The worst case is `da_5min_04`: one sentence is written 43 times from 130 s to the end,
and 16 of 24 utterances are lost (WER 100.4%). The 2 min and continuous 90 s clips do not loop
on `timestamps`. `nt-vad`, the guardrail path, writes no Loop, but it still loses 9
utterances across the set. Every (path, clip) pair gave byte-identical JSON on both runs.

Run 2026-11-08 on an Apple M4 Max, crispasr v0.8.41, 14 threads (Codictate's
`max(4, availableParallelism())`), `models/ggml/ggml-large-v3-ours-f16.bin` (tensors
bit-identical to the official file, see `conversion.md`). Find Set only (10 Long Clips,
1806 s, 155 utterances). No Held-out or FLEURS test audio was read.

## Results

Each cell is **Loops / heard / lost utterances / WER**. All runs were identical, so each
cell is both runs.

| Clip | `timestamps` | `timestamps-bare` | `nt` | `nt-silero` | `nt-vad` |
|---|---|---|---|---|---|
| da_5min_01 | **1** / 85.9% / 0 / 16.0% | **1** / 19.8% / 20 / 97.6% | 0 / 66.5% / 5 / 34.9% | 0 / 90.1% / 0 / 10.5% | 0 / 87.1% / 1 / 13.5% |
| da_5min_02 | **1** / 82.0% / 1 / 22.6% | **2** / 82.9% / 0 / 39.3% | **1** / 63.8% / 6 / 40.1% | 0 / 85.1% / 1 / 15.7% | 0 / 81.4% / 3 / 20.0% |
| da_5min_03 | **2** / 84.8% / 0 / 31.6% | 0 / 84.6% / 0 / 18.9% | 0 / 63.9% / 6 / 37.5% | 0 / 92.5% / 0 / 8.3% | 0 / 90.1% / 1 / 11.2% |
| da_5min_04 | **1** / 36.6% / 16 / 100.4% | **4** / 73.4% / 3 / 76.1% | 0 / 66.9% / 6 / 34.2% | 0 / 77.9% / 3 / 23.5% | 0 / 85.9% / 2 / 17.0% |
| da_2min_01 | 0 / 94.8% / 0 / 5.7% | 0 / 93.3% / 0 / 11.9% | **1** / 56.7% / 4 / 43.3% | 0 / 93.8% / 0 / 6.7% | 0 / 95.2% / 0 / 5.2% |
| da_2min_02 | 0 / 90.1% / 0 / 12.9% | 0 / 91.8% / 0 / 9.9% | 0 / 69.8% / 2 / 31.9% | 0 / 87.1% / 0 / 15.1% | 0 / 93.5% / 0 / 7.8% |
| da_2min_03 | 0 / 89.0% / 0 / 13.8% | **2** / 89.5% / 0 / 39.8% | 0 / 72.4% / 3 / 27.6% | 0 / 87.8% / 1 / 13.8% | 0 / 93.9% / 0 / 7.7% |
| da_continuous_90s_01 | 0 / 89.3% / 0 / 11.2% | 0 / 87.6% / 0 / 13.6% | 0 / 57.4% / 3 / 42.6% | 0 / 92.9% / 0 / 8.3% | 0 / 92.9% / 0 / 7.7% |
| da_continuous_90s_02 | 0 / 82.8% / 1 / 17.7% | **1** / 72.4% / 2 / 55.2% | 0 / 51.6% / 4 / 48.4% | 0 / 81.3% / 1 / 18.8% | 0 / 72.9% / 2 / 27.1% |
| da_continuous_90s_03 | 0 / 91.4% / 0 / 9.6% | 0 / 90.4% / 0 / 11.2% | 0 / 57.2% / 4 / 42.8% | 0 / 92.5% / 0 / 8.6% | 0 / 93.0% / 0 / 8.0% |

Totals over the set:

| Path | Clips with a Loop | Loops | Extra words in Loops | `repeatedParts` | Lost utterances | Heard | WER (pooled) | Wall time, all 10 clips |
|---|---|---|---|---|---|---|---|---|
| `timestamps` | 4 | 5 | 488 | 45 | 18 | 78.9% | 30.9% | 157 s |
| `timestamps-bare` | 5 | 10 | 1015 | 58 | 25 | 73.7% | 44.9% | 200 s |
| `nt` | 2 | 2 | 107 | 0 | 43 | 63.7% | 37.7% | 106 s |
| `nt-silero` | 0 | 0 | 0 | 0 | 6 | 87.4% | 13.6% | 134 s |
| `nt-vad` | 0 | 0 | 0 | 0 | 9 | 87.6% | 13.7% | 138 s |

No transcript on any path repeats the dictation prompt (`echoedPrompt`), and none has a
non-Latin letter. Per-run wall times, the lost utterances (clip times, share heard,
loudness), every Loop's text and place, and the WER ops are in `loops_find.json`.

## Which paths reproduce Loops

- **`timestamps` (today's argv): yes, on all four 5 min clips, on none of the shorter
  ones.** Two of these are big: `da_5min_04` (one sentence 43 times, everything after 131 s
  lost) and `da_5min_03` (a sentence written 3 times, and the last utterance written twice).
  The other two are small rewrites: "har svært ved at" written twice, and "der bruger
  lægerøret frø hjemmehørende i" once from a half-finished first try. This matches ADR-0010's
  second check ("large-v3 ... repeats on Danish").
- **`timestamps-bare` (ADR-0010's first check, no VAD, no beam, no prompt): yes, and worse.**
  5 of 10 clips loop, including the 2 min and continuous clips. On `da_5min_01` one sentence
  is written 28 times from 53 s to 277 s, and 20 utterances are lost. This matches what
  large-v3-q5_0 did on 2026-10-13 ("repeating parts up to four times and losing the last
  utterance").
- **`nt` (legacy plain `-nt`): yes, twice, but mostly it loses audio.** `da_5min_02` writes
  the 90-120 s window's text again as the 120-150 s window, and the real speech of 127-158 s
  is lost. `da_2min_01` ends on a made-up line that repeats itself ("Det var en af de
  fleste, der var med i det, og det var en af de fleste, der var med i det"). The main
  failure here is Lost Audio at the 30 s window edges: 43 utterances lost, 52-72% heard.
  `da_5min_02` also ends with a made-up subtitle credit, "Danske tekster af Nicolai Winther".
- **`nt-silero` (legacy `-nt --vad`): no Loops on Danish.** ADR-0010 saw large-v3 write the
  first slice's text again on this argv, but that was on *English* continuous speech. Danish
  here does not do it.
- **`nt-vad` (today's guardrail): no Loops**, as ADR-0001 requires.

Codictate's own gate, `repeatedParts` (a segment of 3+ words identical to the one before),
catches every big timestamps-path Loop (45 and 58 repeated parts). It sees nothing on the
`-nt` paths, where one segment is a whole 30 s window or slice, and it misses the small
in-sentence rewrites. That is why the Loop count above uses a broader detector (see Method).

## Worst Loops

1. **`timestamps`, `da_5min_04`.** The reference says "Det 100-meter lange skib var på vej
   for at hente dets sædvanlige ladning gødning, ..." (131-146 s). The transcript says
   "Det 100 meter lange skib var på vej fra København." 43 times, one segment each, from
   130.05 s to 289.92 s. 410 extra words, and all 16 utterances from 131 s on are lost
   (heard 0-24% each). The speech before the Loop is fine.
2. **`timestamps-bare`, `da_5min_01`.** "Selvom vandstanden kun stiger fod efter
   oversvømmelsen, håber eksperter, at det vil være nok til at genoprette." 28 times from
   53 s to 277 s. 457 extra words, 20 utterances lost (58-286 s), heard 19.8%.
3. **`timestamps`, `da_5min_03`.** "Ecuador kræver at kubanske borgere modtager et
   invitationsbrev før indrejse til Ecuador via internationale lufthavne eller græder" as
   three segments in a row (64.46-72.46 s) before "lufthavne eller grænsepassager.", and
   then the clip's last utterance ("Som resultat heraf er der opstået en stor skandale i det
   græske retssystem ...") written twice (273.6 s and 288.2 s).
4. **`timestamps-bare`, `da_5min_02`.** "Det betyder hej." 7 times (263-269 s), "Mennesker
   begyndte at gå oprejst for 2 millioner år siden ..." 3 times, and an earlier sentence
   twice.
5. **`nt`, `da_5min_02`.** Window 4 (90-120 s), "Hvis du har et forhold til kvindes helbred,
   så er det ikke en succes. Han sagde i et interview ... Ulovlig spyrertrusk indeholder
   forskellige farlige urenheder ...", is written again word for word as window 5.
   88 extra words, and the speech of 116-158 s is lost.

## Lost Audio without Loops

Many lost utterances are not Loops. Two causes show up:

- **Very quiet FLEURS recordings.** Some Find Set utterances are 18-31 dB below their clip's
  level (FLEURS da_dk train has uneven recording gain). `timestamps`, `nt-silero` and `nt-vad`
  all lose the same ones: `da_5min_02` 25-40 s (-18 dB), `da_5min_04` 28-32 s (-20 dB),
  `da_continuous_90s_02` 20-26 s (-6 dB, in a quiet clip). `nt-silero` and `timestamps` also
  lose `da_5min_04` 46-59 s (-31 dB). Any later "heard" gain has to be read against these.
- **`nt-vad` stops early inside a slice.** 5 of its 9 lost utterances are at normal loudness.
  On `da_5min_01` the slice from 13.1 s holds three utterances, and the model writes two and
  ends (30-43 s lost, a loud, clipped recording at +12.6 dB). On `da_5min_04` the slice from
  165.0 s holds "Derefter skal pæren udskiftes ..." and "Russisk jordkontrol ...", and only
  the first is written (173-181 s lost). Also lost: `da_5min_02` 250-259 s and
  `da_continuous_90s_02` 26-39 s. This is the early end-of-text ADR-0010 describes, now
  inside a ≤30 s slice: there is no window left to skip, but the rest of that slice is still
  never decoded. ADR-0010's long-audio check found "none lost" for large-v3 on `nt-vad`, but
  on Codictate's test-split clips, so this is new. It matters for the guardrail: a Weight
  Edit should not make it worse, and it may be fixable by the same mechanism.

## Determinism

Each (clip, path) pair ran twice, 100 decodes in all. **Every pair gave byte-identical
crispasr JSON** (50 of 50, `cmp`), so identical text, segments and times too. No third run
was needed. Temperature fallback does not make runs differ here: whisper.cpp's sampler RNG
uses a fixed seed per process, so even a fallback that samples gives the same result every
time. One run per configuration is enough for later comparisons on this machine and build.
Changing thread count or backend could still move results.

## Decode Paths: exact argv and provenance

`<wav>`, `<base>` and `<slices>` stand for `data/clips/find/<clip>.wav`,
`results/phase0/transcripts/<path>/<clip>.run<N>` and
`results/phase0/transcripts/nt-vad/<clip>.slices.json`. `crispasr` is
`../codictate/vendors/crispasr/crispasr` (v0.8.41), started with
`DYLD_LIBRARY_PATH=../codictate/vendors/crispasr LC_ALL=LANG=en_US.UTF-8`. `<silero>` is
`../codictate/vendors/silero-vad/ggml-silero-v6.2.0.bin`. `<prompt>` is
`Hej, dette er en diktering. Jeg skriver i hele sætninger, med kommaer hvor jeg holder pause.`

| Path | argv | Where it comes from |
|---|---|---|
| `timestamps` | `crispasr -m <model> -t 14 --language da -f <wav> --no-prints --vad -vm <silero> -vt 0.25 -bs 5 --prompt <prompt> -ojf -of <base>` | Built by calling `buildWhisperHarnessCommand` from `../codictate` at `0c25733` (HEAD), with `speechModelId: 'large-v3-turbo'` (a timestamps model; large-v3 itself is on `nt-vad`; apart from the decoding, the argv does not depend on the model) and `language: 'da'`. Unchanged since `ead40ab` (2026-10-17, `-vt 0.25`). |
| `timestamps-bare` | `crispasr -m <model> -t 14 --language da -f <wav> --no-prints -ojf -of <base>` | The timestamps branch at `de54009` (2026-10-13 16:25, the commit that added ADR-0010), which is what the first check ran. `328408a` (2026-10-13 21:37) then added `--vad -vm -vt 0.35 -bs 5 --prompt`. |
| `nt` | `crispasr -m <model> -t 14 --language da -f <wav> --no-prints -nt` + `-ojf -of <base>` | `whisper-harness-command.ts` at `9621486` (2026-09-15), the last version before `de54009`. The transcript was read from stdout, and `.txt` here is that stdout. `-ojf -of` were added only to get the JSON. On `da_continuous_90s_01` the stdout was the same with and without them. |
| `nt-silero` | `crispasr --backend whisper -m <model> -t 14 --language da -f <wav> --no-prints -nt --vad -vm <silero> -ojf -of <base>` | The `nt-vad` branch from `de54009` through `23fc0db` (2026-10-17). `019b5d8` (2026-10-18) replaced it with `--vad-import`. large-v3 decoded on this after failing the timestamps check. Transcript from the JSON, with today's `nt-vad` join. |
| `nt-vad` | `crispasr --backend whisper -m <model> -t 14 --language da -f <wav> --no-prints -nt --vad-import <slices> -sns -bs 5 --prompt <prompt> -ojf -of <base>` | Built by calling `buildWhisperHarnessCommand` at `0c25733` with `speechModelId: 'large-v3'` (no `whisperDecoding`, so `nt-vad`; `ntVadPrompt` on). Slices come from `planVadSlices` (default options) and `vadImportJson`, which is what `writeNtVadSlices` does when the app passes no tuning. |

Transcripts are read the way Codictate read each path, with `crispasrTranscript` from
`../codictate/src/bun/utils/whisper/engines/crispasr-json.ts`: the `timestamps` join for
both timestamps paths, the `nt-vad` join for `nt-silero` and `nt-vad`, and stdout for `nt`.

## Method

- **heard / lost:** `heardPerWindow` from `long-audio.manual.ts`: reference words found in
  order, pooled over the utterances, and an utterance with under 30% heard is lost. Clip
  times come from the manifest. **WER:** `computeWer` from `benchmarks/stt/wer.ts` against
  the clip's utterance texts joined with spaces, with `normalizeForWer` (whisper-basic).
  **Prompt echo / foreign script:** `echoedPrompt`, `nonLatinShare`. Codictate's private
  functions are copied verbatim into `scripts/codictate_long_audio.ts`, with line numbers.
  Everything exported is imported from `../codictate` directly, read-only.
- **Loop** (GLOSSARY: a span of 3+ words written again when the audio says it once): a
  transcript word is *repeated* when it sits in a word trigram that occurs 2+ times in the
  transcript and more often than in the reference. It is *extra* when, on top of that, the
  minimum-edit alignment to the reference (`alignWords`) does not match it to the same
  reference word. A run of repeated words, joined across gaps of up to 3 other words, is one
  Loop when it holds 3+ extra words. The alignment test throws out repeats that come from a
  misheard word the reference itself repeats. One example on `timestamps` `da_5min_01`:
  "i fanen med radioaktivt" written twice for "i fagen med radioaktivt", which the reference
  says twice. Every flagged Loop was checked by eye against the segments. All are real
  repeats, with two caveats, both on `timestamps-bare`. On `da_5min_04` one 3-word hit, "det
  er en", is a coincidence: two different made-up sentences both start that way. And the
  same clip's "Vinneren af to olympiske guldmedaller ..." run is split in two (11 + 229
  words) around a made-up line. So that clip has 2 real Loop events, not 4. On `da_2min_03`
  the 4-word hit "af de fleste mand" is a made-up filler phrase written twice; the audio
  never says it at all.
  `repeatedParts` is reported beside it, applied to crispasr's JSON segments on every path.

## Reproduce

```sh
bun scripts/run_paths.ts                 # 5 paths x 10 clips x runs 1,2; skips what exists
bun scripts/run_paths.ts --runs 3 --paths timestamps --clips find_da_5min_04   # an extra run
bun scripts/measure.ts                   # -> results/phase0/loops_find.json + printed table
```

Files: `results/phase0/transcripts/<path>/<clip>.run<N>.{json,txt,meta.json}` (meta has
the argv, wall time and stderr tail), plus `nt-vad/<clip>.slices.json`.
