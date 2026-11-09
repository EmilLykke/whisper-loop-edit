# Phase 0: the Loops in HF/PyTorch

**Verdict: the Loop can be seen in PyTorch.** Every Loop we tried was rebuilt at the 30 s window where
crispasr writes it, with the same decoder input crispasr used. HF large-v3 then writes the same Loop or
prefers it clearly, in 5 of 5 cases. Removing the previous-text prompt makes it go away every time. So
the Loop is driven by the model conditioning on its own earlier output. The user `--prompt` does not
cause it. The phase-0 kill condition is not met. Later phases can read activations at the looping step
from the inputs saved here.

| Case | Window (decoded audio) | Loop written by HF from crispasr's state? | Teacher forcing, mean log-prob per token: Loop / reference / model's own clean text | Without previous text |
|---|---|---|---|---|
| `ts04_copy`: `timestamps`, 5min_04, "Det 100 meter lange skib ..." ×11 | seek 85.46 s (orig 135.76 s) | **yes**: greedy, beam 5 and whisper.cpp beam (4 RNG seeds), all 11× | **-0.039** / -0.672 / -0.366 (sums -8.5 / -69.2 / -55.2) | 0 Loops, all decoders |
| `ts04_onset`: same Loop, first rewrite at the window's end | seek 55.50 s (orig 92.75 s), after seg 10 | **yes**: beam 5 and 3 of 4 whisper.cpp-beam seeds write the 2 rewrites crispasr wrote. Greedy does not. | branch after `<\|28.00\|><\|28.00\|>`: p(" Det") 0.311 vs p(EOT) 0.011 | p(" Det") 0.015 vs p(EOT) 0.937 |
| `bare01_copy`: `timestamps-bare`, 5min_01, "Selvom vandstanden ..." ×28 | seek 53 s | **yes, after one near-tie**: HF and crispasr split at token 7 (" fod" 0.103 vs " v" 0.096 in HF; crispasr picked " v" at p 0.098). With crispasr's token forced, greedy writes the sentence 3×, with crispasr's exact timestamps. | **-0.158** / -0.632 / -0.807 | 0 Loops. Loop text drops to -1.377 per token. |
| `nt02_copy`: legacy `-nt`, 5min_02, window 90-120 s written again at 120-150 s | seek 120 s | **yes**: whisper.cpp's fallback steps reproduced. Greedy at T=0 copies, then loops until the 220-token cap, so the window fails. T=0.2 then writes the copy, as crispasr did. Beam 5 copies too. | **-0.040** / -0.334 / -0.273 | 0 Loops. Copy text -2.521 per token. |
| `ts03_onset`: `timestamps`, 5min_03, "Ecuador kræver ..." ×3 | seek 27 s (orig 34.69 s), after the first "Ecuador" | **yes**: beam 5 and all 4 whisper.cpp-beam seeds write the 2 rewrites. Greedy stops. | branch: p(" Ecuador") 0.197 vs p(EOT) 0.366. Beam still prefers the Loop on its sequence score. | p(" Ecuador") 0.009 vs p(EOT) 0.978 |

How close HF is to whisper.cpp: rebuilt windows match crispasr's per-token probability `p` with a mean
|Δp| of 0.0001-0.005 on the Loop windows (≤ 0.017 on every window of the four runs). Every fallback window
crispasr counted (`fallbacks = N p`) is found at the right temperature. Long-form decoding of a whole clip
matches crispasr only in places, because both decoders amplify near-ties (see Mismatches).

Run 2026-11-08 on an M4 Max: torch 2.15.0 (MPS), transformers 5.22.2, `models/hf/whisper-large-v3`
(revision `06f233fe`), **float16 on MPS** throughout. We hit no numerical trouble. The one near-tie we
checked in float32 (bare01, token 7) came out the same, to 0.0002. crispasr v0.8.41 (binary), read
against the CrispASR v0.8.29 source in `../codictate/.tmp/edda-mirror/crispasr-src` (`src/crispasr.cpp`).
Find Set only.

## Method

### Rebuilding crispasr's decoder state (scripts/hf_window.py)

whisper.cpp's `whisper_full_with_state` (crispasr.cpp:7613) decodes in 30 s windows. Before each window:

- **Audio.** On `timestamps` the input is the VAD-filtered buffer (`whisper_vad`, crispasr.cpp:7423):
  Silero speech segments laid end to end, each but the last extended by 0.1 s, with 0.1 s of zeros
  between them. The segments come from crispasr's own log: we reran the same argv without
  `--no-prints`, got byte-identical transcription JSON, and saved them in `hf_repro/vad_<clip>.json`. The
  rebuilt buffer has crispasr's length to the sample (3,002,720 for 5min_04), and the 170-point time map
  matches too. The log-mel is computed once over the whole buffer, as whisper.cpp does it: 30 s zero tail,
  global max-8 clamp. The window is mel frames `[seek, seek+3000)`.
- **Decoder input** = `<|startofprev|>` + the last 223 tokens of `prompt_past` + `<|sot|><|da|><|transcribe|>`
  (+ `<|notimestamps|>` on `-nt`). `prompt_past` starts as the `--prompt` tokens. After each window it
  becomes the capped prompt plus every token the window decoded, timestamps included (and the EOT on
  `-nt`). It is dropped for fallback passes at T ≥ 0.5, and in the last 5 s. The `--prompt` is tokenized
  the way whisper.cpp does it: regex split in the C locale, then greedy longest match. That gives
  `diktering` as 3 tokens where HF BPE gives other ones.
- **Window tokens** come from crispasr's JSON (`results/phase0/transcripts/...run1.json`, which has
  token ids and p), with three repairs. These are crispasr v0.8.41 output quirks, found while matching p:
  (1) the timestamp that *begins* a segment is not in its token list, but its time is in the segment's
  `from`; (2) a window's leading timestamp is dropped unless it is `<|0.00|>` (the segment loop tests
  `id > token_beg`); (3) a trailing timestamp pair at the window's end is not written. Restored
  timestamps are snapped to HF's most likely value within ±0.06 s. Seek is chained from the previous
  window's last timestamp, or +30 s after a lone ending timestamp or the 220-token cap. Where crispasr's
  output leaves a choice (seek, trailing pair, temperature), we keep the option whose teacher-forced p
  matches crispasr's p best.
- **Logit filtering** copies `whisper_process_logits` (crispasr.cpp:6896): suppress blank at the start,
  timestamps in pairs and non-decreasing, max initial timestamp 1 s, "timestamps win if their summed
  probability beats the best text token". Non-speech tokens are *not* suppressed.
- **Decoders**: `greedy`, `beam` (true top-k beam search, length-normalised) and `cppbeam`.
  `cppbeam` is whisper.cpp's beam search, which does not take the top k: each decoder *samples* k
  candidates from the filtered distribution with its own `std::mt19937`. The RNG is emulated exactly (MT19937
  `init_genrand` + libc++ `discrete_distribution`/`generate_canonical`). The fallback ladder (T = 0, 0.2,
  …, 1.0; avg-logprob < -1, entropy of the last 32 tokens < 2.4, the 220-token cap check) is copied as
  well.

Check against crispasr, per window (mean |Δp| over the tokens crispasr reports; T = the inferred pass):

| Run | windows (seek: T: mean \|Δp\|) |
|---|---|
| `timestamps` 5min_04 | 0.00: 0: .0040 · 25.50: 0: .0050 · **55.50: 0: .0051** · **85.46: 0.2: .0001** · 115.46: 0: .0012 · 145.46: 0: .0006 · 175.46: 0: .0012 |
| `timestamps-bare` 5min_01 | 0: .0063 · 23: .0043 · **53: .0034** · 82: .0027 · 111: .0029 · 140: 0.2: .0022 · 169: .0021 · … · 285 (tail, prompt cleared): .0616 |
| `nt` 5min_02 | 0: .0020 · 30: .0014 · 60: .0035 · 90: .0133 · **120: 0.2: .0006** · 150: 0.6: .0014 · 180-300: ≤ .0060 |
| `timestamps` 5min_03 | 0: .0085 · **27: .0033** · 56: 0.6: .0035 · 81.44: .0107 · 111.46: .0082 · 141.46: 0.2: .0167 · 169.46: .0147 · 197.46: .0023 |

The inferred fallbacks add up to crispasr's printed counts: 1 (5min_04), 1 (bare 5min_01), 4 (nt 5min_02:
one pass at 0.2, three to reach 0.6), 4 (5min_03). Full per-window inputs (prompt ids, tokens, which tokens
were restored, notes) are in `hf_repro/windows_<path>_<clip>.json`.

### Experiments per Loop (`scripts/hf_window.py repro`)

At each Loop window, we used crispasr's decoder input, plus any window tokens before the first rewrite
(the "prefix"):

1. Free decoding: greedy, true beam 5, whisper.cpp beam 5 with 4 RNG seed sets, and the path's full
   fallback ladder. We count how often the Loop text appears in the output.
2. Teacher forcing of three continuations, giving per-token raw log-softmax, whisper.cpp-filtered
   log-prob and argmax:
   **Loop** (what crispasr wrote, to the end of the window); **reference** (the FLEURS text of the
   utterances lying wholly in the window, with timestamps from the manifest, mapped through VAD; for the
   onset cases, "end the window here"); **clean** (HF's beam-5 decode of the same window with no
   previous text). Clean is the fairer competitor, because it is in the model's own spelling and covers
   utterances cut by the window edges.
3. Prompt ablations, each run through 1 and 2: `no_prev_text` (only sot/da/transcribe);
   `user_prompt_only`; `carry_user_prompt` (`--carry-initial-prompt` style: user prompt + previous
   text); `prev_text_minus_loop_sentence` (previous text with every earlier copy of the Loop sentence
   cut out). For the onsets, `prefix_before_first_instance` restarts before the first made-up sentence.

## Results per case

`hf_repro.json` → `cases` has every decode text, every per-token log-prob (for crispasr's input) and
every input as token ids and text. Loop counts below: greedy / beam5 / whisper.cpp beam with seeds 0-3.

### ts04_copy (the 43× Loop): conditioned copy over 30 s of other speech

Input: 227 tokens. The previous text ends `... ville de ikke kunne ses helt. Det 100 meter lange skib var på
vej fra København.` ×3. The audio is 135.8-~182 s orig: the end of the "100-meter lange skib" utterance,
then "Det kan trækkes ud ...", "13-årige motorcykelracer ...", "Derefter skal pæren ...", and more.

| Variant | Loops | TF mean raw: Loop / ref / clean | TF sum raw (tokens) |
|---|---|---|---|
| crispasr input | 11 / 11 / 11,11,11,11 | **-0.039** / -0.672 / -0.366 | -8.5 (220) / -69.2 (103) / -55.2 (151) |
| no previous text | 0 / 0 / 0,0,0,0 | -0.311 / -0.582 / -0.232 | -68.4 / -59.9 / -35.0 |
| user prompt only | 0 / 0 / 0,0,0,0 | -0.322 / -0.612 / -0.298 | |
| user prompt + previous text | 11 / 11 / 11,11,11,11 | -0.038 / -0.677 / -0.390 | |
| previous text minus the 3 copies | 0 / 0 / 0,0,0,0 | -0.328 / -0.691 / -0.367 | |

The Loop is preferred by a wide margin: about 17× more probable per token than clean, even though it is
longer. The fallback ladder matches crispasr's too. The T=0 decoders that space the sentence 1 s apart
reach the 220-token cap at 11 s. whisper.cpp's "repetition loop" check (`seek_delta < 15 s` at the cap)
flags them, and the window falls back. The next temperature writes the same sentence at a 2 s spacing,
which passes. crispasr did the same: window 85.46 s is the one fallback in its log, at T = 0.2 (mean
|Δp| 0.0001 for that hypothesis). The entropy check never fires: a 19-20 token period has entropy ≈ 2.9
over the last 32 tokens, above the 2.4 limit. Without the previous text, the model writes the window
correctly, starting with the tail of the cut utterance: "for at hente dets sædvanlige ladning, gødning
...".

### ts04_onset: the first rewrite

Prefix: `<|0.00|> Men det er også ... <|27.00|><|27.00|> Det 100 meter lange skib var på vej fra København.<|28.00|>`.
The window ends 2.8 s into the real "Det 100-meter lange skib var på vej for at hente ..." (VAD time 82.69
of a window that runs to 85.50). So seg 10 is already the start of that sentence with a made-up ending.
The Loop is the next segment rewriting it. With crispasr's input, beam 5 and 3 of 4 whisper.cpp-beam seeds
rewrite it twice before the window ends, as crispasr did. Greedy does not. After the pair, the next token
is p(" Det") 0.311 against p(EOT) 0.011 (raw -1.18 vs -4.54). Without previous text it is 0.015 against
0.937: the model would stop, and the next window would start at 28 s. The user prompt alone also gives
"stop" (0.022 vs 0.916). The copy in the previous text is what turns "stop" into "write it again".

### bare01_copy (the 28× Loop)

Previous text (no user prompt on this path) ends with the real utterance, already written once with a
stutter: `"Selvom vandstanden kun stiger fod fod efter oversvømmelsen, ..."<|29.00|>`. The window is
53-83 s ("... eroderede sandbanker langs med floden.", "Du får som minimum brug for fodtøj ...",
"Jerusalem ...").

- Free decoding does **not** reproduce it directly. At token 7 HF picks " fod" (0.1026) over " v"
  (0.0957, for "vandstanden"). crispasr picked " v" at p 0.0985, its argmax. This is a ggml-vs-torch
  difference, not fp16: float32 gives 0.1028 / 0.0957. HF's own greedy path writes a different made-up
  sentence 3× ("Fod er en af de mest vigtigste steder i Europa ..."), which is also a Loop.
- **With crispasr's " v" forced** (`hf_repro/bare01_forced_branch.json`), greedy writes "Selvom
  vandstanden ... genoprette." 3× with exactly crispasr's timestamps (0-5, 6-15, 16-28/29), and beam 5
  writes it 4×. Same forced token without previous text: no Loop ("Selvom vores køer er rådrede
  sandbanker ...").
- Teacher forcing prefers the Loop: -0.158 per token vs -0.632 (reference) and -0.807 (clean). Without
  previous text the Loop drops to -1.377 and clean rises to -0.274. Cutting only the earlier copy of
  the sentence from the previous text does the same (-1.280). With the user prompt added in front of the
  previous text (`carry_user_prompt`), greedy writes the sentence 4×.

### nt02_copy (window written again on legacy `-nt`)

Input: 228 tokens. The previous text is windows 0-3, each ending in `<|endoftext|>` (whisper.cpp keeps
the EOT in `prompt_past` on `-nt`). At T=0, greedy copies window 3's text and then loops "Ulovlig
spyrertrusk indeholder ..." until the 220-token cap. That fails the window. The T=0.2 pass then writes
window 3 again and stops, exactly what crispasr wrote (its window 120 s is a T = 0.2 fallback, |Δp|
0.0006). True beam 5 at T=0 writes the copy directly. Teacher forcing: copy -0.040 per token vs reference
-0.334 vs clean -0.273. Without previous text the copy text costs -2.521 per token. With window 3's text
cut from the previous text there is no copy (-2.469).

### ts03_onset (Ecuador ×3)

Prefix ends `... via internationale lufthavne eller græder.<|27.00|>`. The window again ends inside the
utterance. Beam 5 and all whisper.cpp-beam seeds rewrite the sentence twice (crispasr: twice). Greedy
stops. Here the user prompt is still in the input (window 1: 29 prompt tokens + window 0). Branch after
the pair: p(" Ecuador") 0.197 vs p(EOT) 0.366. Greedy stops, but beam search ranks the rewrite higher,
because whisper.cpp and HF score a finished sequence by its average log-prob, and a confident 42-token
copy raises the average. Without previous text: 0.009 vs 0.978. **With only the user prompt removed**
(previous text kept): p(" Ecuador") 0.487 vs p(EOT) 0.005, and greedy writes the 2 rewrites too
(`onset_branch.json`). The user prompt slightly *protects* here; it does not trigger the Loop.

## Which removal kills the Loop

- **The previous-text prompt.** Removing it kills all 5 Loops in every decoder. Cutting only the earlier
  copies of the looped sentence (or the copied window) from it kills all 3 copy Loops. The Loop is
  self-conditioning: once a sentence is in `prompt_past`, the next window writes it again over
  different audio. In the onset cases, a sentence just written in the same window gets written again at
  the window's edge.
- **The user `--prompt`** is not the trigger. At the `timestamps` Loop windows of 5min_04 it has already
  scrolled out of the 223-token `prompt_past`. The Loop windows hold no user-prompt tokens at all, so
  "removing" it changes nothing there. Given alone, it never loops. Carried in front of the previous text,
  it does not stop the copy Loops. Where it is still present (ts03 window 27 s), removing it makes the
  Loop *stronger*. The `timestamps-bare` and `-nt` Loops happen with no user prompt at all.

## Long-form in HF (scripts/hf_longform.py)

**transformers' own long-form `generate`** on the same audio crispasr decoded (VAD-filtered buffer on
`timestamps`): `return_timestamps=True, condition_on_prev_tokens=True`, temperature
(0, 0.2, …, 1.0), `logprob_threshold=-1.0`, `no_speech_threshold=0.6`, `prompt_ids` = the Danish prompt
where the path uses one. whisper.cpp has no gzip check. Its entropy check is not something HF offers, so
we ran `compression_ratio_threshold` off (closest to whisper.cpp, whose entropy check never fired on
these clips) and at HF's recommended 1.35.

| Clip, settings | Most repeated segment | crispasr |
|---|---|---|
| 5min_04, beam 5, prompt | **"Det 100 meter lange skib var på vej fra København." 56× from 129.06 s**. The segments before it start at the same times as crispasr's (Kruger 94.75 s, DVD 113.31 s). | same sentence 43× from 130.05 s |
| 5min_04, beam 5, no prompt | same sentence 113× from 129.06 s | |
| 5min_04, greedy, prompt | no Loop at 129 s (a 2× repeat at 222 s) | |
| 5min_04, beam 5, prompt, `compression_ratio_threshold=1.35` | **no Loop** (the gzip check sends the window to fallback) | |
| 5min_03, beam 5, prompt | no Ecuador Loop. "Så disse kunne sejle hen over middelhavet ..." 43× from 231.9 s | Ecuador 3× at 64.46 s |
| bare 5min_01, greedy | "De lavede to eksperimenter ..." 9× at 140 s, "Du kan se pyramiderne ..." 26× from 204 s, no Loop at 53 s | "Selvom vandstanden ..." 28× from 53 s |
| bare 5min_01, greedy, `crt=1.35` / with prompt | no Loop | |

transformers cannot do long-form without timestamps, so there is no native `-nt` run. On MPS,
transformers 5.22's deferred stop check reads `cache.layers` on an `EncoderDecoderCache` and crashes.
`hf_longform.py` turns that check off, which changes no decoding decision.

**whisper.cpp's loop re-implemented on the HF model** (`--engine cpp`: same seek, prompt_past, VAD, fallback
and RNG streams as crispasr). It reproduces **the Ecuador Loop on 5min_03 at the same place (3× from
64.46 s)**. On the other clips it drifts from crispasr in an early window and never reaches the Loop's state:

- bare 5min_01 splits in window 0, at a near-tie (crispasr's window 0 ends at 23 s, ours at 30 s).
- nt 5min_02 writes window 90 s differently, so there is nothing to copy.
- 5min_04 writes window 55.5 s differently in 5 of 5 RNG seed sets, so the 130 s Loop never starts. One
  seed set loops elsewhere instead ("Det er vigtigt at være forsigtig ved udskytten af pæren." 11× from
  170 s).

## Mismatches between HF and whisper.cpp, and why

1. **Numerics.** ggml (Metal, f16 weights) and torch (MPS, fp16 or fp32) give token probabilities that
   differ by ~0.005 on average and up to ~0.15 on single tokens. That is harmless except at near-ties,
   and a long-form decode meets several. One flip early on (bare 5min_01 window 0, nt 5min_02 window 90 s)
   changes everything after it. This is why window-level rebuilding, not whole-clip decoding, is the
   way to get crispasr's exact state.
2. **whisper.cpp's beam search samples.** `whisper_sample_token_topk` (crispasr.cpp:7323) builds a
   top-k list and then ignores it, drawing k tokens from the distribution with each decoder's RNG. Its
   output therefore depends on the RNG streams, which carry over across windows, and on tiny
   probability differences. True beam search (HF) is a different decoder. At the Loop windows both write
   the Loop. Over a whole clip they take different paths.
3. **HF native long-form is a different algorithm**: its own seek and segment logic, non-speech
   `suppress_tokens`, and prompt handling. Even so, it reproduces the 5min_04 Loop at 129 s with beam 5,
   prompted or not.
4. **crispasr's JSON hides tokens.** Segment-start timestamps, a non-zero leading timestamp and trailing
   pairs are missing. They had to be restored to get the decoder input right (see Method).
5. **No gzip check in whisper.cpp.** Its entropy check looks at only the last 32 tokens, which a 19-20
   token Loop passes. Its 220-token check catches only Loops paced under 15 s per 220 tokens, and the
   fallback then writes the same Loop at a slower pace. HF's `compression_ratio_threshold=1.35` stopped
   both long-form Loops we tested.

## Inputs saved for later phases (contrastive pairs)

`hf_repro.json` → `cases.<name>.inputs` has, as token ids and text: `prompt` (crispasr's decoder input up
to the task tokens), `prefix` (window tokens before the first rewrite), `loop`, `reference`, `clean`,
`user_prompt`, and every ablation's prompt in `variants`. The audio window is the clip, (path), `seek`
(mel frames, 10 ms) in the buffer `hf_window.case_inputs()` rebuilds. Natural contrastive pairs:

- same audio + prefix, `prompt` vs `variants.no_prev_text` (or `prev_text_minus_loop_sentence`): Loop vs
  no Loop at the same step;
- same input, the Loop continuation vs `clean`, token by token (per-token log-probs for crispasr's input
  are in `variants.crispasr.tf.loop.tokens`).

To get activations: `Whisper.forward(enc, prompt + prefix + cont, hidden=True, attentions=True)` returns
every layer's hidden states and the self- and cross-attention maps.

## Reproduce

```sh
# crispasr's VAD segments (same argv as the timestamps path, without --no-prints), then:
.venv/bin/python -I scripts/hf_window.py validate --path timestamps --clip find_da_5min_04   # also 5min_03,
.venv/bin/python -I scripts/hf_window.py validate --path timestamps-bare --clip find_da_5min_01  # nt 5min_02
.venv/bin/python -I scripts/hf_window.py repro          # ~20 min, -> hf_repro/cases.json
.venv/bin/python -I scripts/hf_window.py summary        # -> hf_repro.json + the tables above
.venv/bin/python -I scripts/hf_longform.py --engine hf  --path timestamps --clip find_da_5min_04 --beams 5
.venv/bin/python -I scripts/hf_longform.py --engine cpp --path timestamps --clip find_da_5min_04 [--rng-offset N]
```

`hf_repro/onset_branch.json` and `hf_repro/bare01_forced_branch.json` came from short one-off runs of
`hf_window` functions; their content and inputs are described above.
