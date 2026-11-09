# Whisper Loop Edit

Whisper large-v3 sometimes gets stuck on long recordings. It writes a sentence, then writes it again, and again. In one 5-minute Danish test it wrote the same sentence 43 times.

We found the small part of the model that causes this, and switched it off by editing the model's weights directly. No training. The idea comes from "Sparks of Weight-Space Programming" (MIT, November 2026: [blog](https://rikiyat.github.io/blog/weight-space-programming/), [code](https://github.com/RikiyaT/lm_understand-logs)), where an AI agent edited a language model's weights to stop it from going in circles. Here, Claude (Anthropic's Opus 5.5) did the editing.

The edited model is on Hugging Face: [emillykkegrann/whisper-large-v3-copy-heads-off](https://huggingface.co/emillykkegrann/whisper-large-v3-copy-heads-off).

## The short answer

| Question | Answer | |
|---|---|---|
| Can a direct weight edit remove the loops? | **Yes, mostly.** On 30 recordings the edit never saw, long loops went from 21 to 1, and the words written by loops dropped 94%. | 🟢 Good |
| Is that real, or luck? | **Real.** Two random edits of the same size did nothing like it. | 🟢 Good |
| Does it break anything? | **Nothing we found.** Danish accuracy on normal short recordings stayed the same. It still works when the model is compressed. | 🟢 Good |
| Did it pass the strict pass mark we set before the final test? | **Just missed.** We asked for 80% fewer loops and got 76%. One test lost one more sentence than the original (15 against 14). | 🟡 Meh |
| Is it better at Danish? | **No.** Accuracy did not change. Removing a fault does not teach the model more Danish. | ⚪ Neutral |
| Does it stop Whisper making up text in silence? | **No.** That is a different fault. Use voice detection (VAD) in front of the model. | 🔴 Not fixed |

**Verdict: the method works.** We found what causes the copy loop and removed it with a small edit, without training. It fell just short of the strict pass mark we set for ourselves, so we call it a strong result, not a finished fix.

## How we tested it

1. **We made the problem happen on purpose.** We built long Danish recordings (90 seconds to 5 minutes) by joining short [FLEURS](https://huggingface.co/datasets/google/fleurs) recordings of people reading sentences aloud. The original model looped on them, the same way every time.
2. **We found the cause.** Whisper listens in 30-second windows. For each new window, it gets the text it already wrote as a hint. When that hint contains an earlier copy of a sentence, Whisper copies the sentence again. When we took the copy out of the hint, the loop stopped.
3. **We found the part that copies.** The text half of Whisper has 640 small parts called attention heads. We tested which ones do the copying, and found 11. With those 11 switched off, test loops went from 68% to 0%. With 11 random heads switched off, nothing changed.
4. **We tried edits on 10 practice recordings.** Claude built and tested 8 different edits there. The best one simply switches off the 11 heads.
5. **We tested once on 30 hidden recordings.** These 30 recordings (90 minutes) stayed sealed until the edit and the pass mark were locked. We ran the original model, the edited model, and two random edits on them.

Rules that keep the test fair ([ADR 0001](docs/adr/0001-gradient-free-edits-only.md)):

- **No training.** Every edit is a formula on the original weights. Anyone can rerun it.
- **Practice and hidden recordings are separate.** The edit was chosen on the practice set only.
- **Random edits for comparison.** If a random edit of the same size does the same, the result is luck.
- **A pass mark, set in advance.** We changed it once, openly, before the final test ([ADR 0002](docs/adr/0002-timestamps-bare-leaves-the-bar.md)). See "Other results" below.

## Results on the 30 hidden recordings

Measured with whisper.cpp (crispasr), which transcribes a long recording in one go, with and without timestamps. Lower is better in every row.

| | Original | **Edited** | Random edit A | Random edit B |
|---|---|---|---|---|
| Loops | 25 | **6** | 22 | 26 |
| Long loops (10 words or more) | 21 | **1** | 15 | 22 |
| Extra words written by loops | 1,380 | **85** | 730 | 2,313 |
| Word error rate, with timestamps | 31.0% | **19.7%** | 24.5% | 36.1% |
| Word error rate, without timestamps | 38.3% | **36.2%** | 38.3% | 41.0% |

### The strict pass mark

| What we required | What happened | Passed? |
|---|---|---|
| At least 80% fewer loops | 76% fewer (25 to 6) | ❌ Just missed |
| Random edits must not do the same | Random edits: 12% fewer, and 4% more | ✅ |
| No more lost speech, in any mode | One more lost sentence when the audio is cut into slices (14 to 15). The random edits lost 15 and 14. | ❌ Just missed |
| No loops when the audio is cut into slices | 0 and 0 | ✅ |
| Danish accuracy no more than 0.3 points worse (FLEURS test, 930 recordings) | Word error rate 12.56% to 12.45% | ✅ |

Why the misses are small: 5 of the 6 loops left are 3 or 4 words written twice, like "til at booke" two times in a row. The original model and both random edits make that kind of slip too. The sixth is made-up text, not a copy.

### Other results (not part of the pass mark)

- **English.** On Codictate's three English test recordings (about 7 minutes), the original wrote 160 made-up words in a loop. The edited model wrote none. Word error rate went from 17.6% to 1.7%.
- **Codictate's Danish test recordings.** Word error rate with timestamps went from 17.3% to 11.3%.
- **Compressed model (q5_0).** About a third of the size. Loops still dropped 79%.
- **Hugging Face transformers.** 10 of the hidden recordings, with OpenAI's recommended settings: loops went from 21 to 3 ([report](results/hf_check/report.md)).
- **Audio with no voice detection.** We removed this mode from the pass mark, because on the practice recordings the edit made it worse (10 to 26 loops). On the hidden recordings it made it better (41 to 23). So the practice result was most likely chance.

## What we learned

- **The copy loop has a small, clear cause.** A few attention heads copy earlier text from the hint. You can switch them off without training.
- **Removing a fault does not add skill.** Where there were no loops to begin with (audio cut into short slices, which is what Codictate does), nothing changed. The MIT paper found the same: an edit can remove an obstacle, but it cannot teach the model something new.
- **Long recordings are unpredictable.** Any change to the model, even a random one, can make a single recording loop or stop looping. Only a comparison with random edits shows what is real. 10 practice recordings were not enough to see that.
- **Making up text in silence is a separate problem.** This edit does not fix it. Voice detection does.

## Technical details

- **The edit (E3):** zero the 64 columns of `model.decoder.layers.L.self_attn.out_proj.weight` for 11 decoder self-attention heads: L2H19, L12H13, L13H7, L13H18, L15H17, L18H19 (they find the earlier copy), and L21H14, L24H0, L25H18, L25H19, L26H6 (they write it out). These are induction heads. Cross-attention, which earlier work blamed ([Glazer et al. 2025](https://arxiv.org/abs/2508.15882)), does not collapse in our cases.
- **Names in the files:** practice recordings = Find Set, hidden recordings = Held-out Set, random edits = Control Edits C1 and C4, a loop = Loop, lost speech = Lost Audio. Modes: `timestamps`, `-nt` (no timestamps), `nt-vad` (cut into slices of 30 s or less, what Codictate ships), `timestamps-bare` (no voice detection). All terms are in [GLOSSARY.md](GLOSSARY.md).
- **Edits that did worse:** E1 and E2 (only 2 or 5 of the heads), E4 and E5 (removing a loop direction from the layer 13-24 writers; E5 hurt accuracy), E6 (5 heads plus that direction), E7 (E3 plus the direction) and E9 (E3 plus Calm-Whisper's silence heads, which damaged the model). Every step, number and decision is in [LOG.md](LOG.md).

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
