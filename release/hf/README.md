---
license: mit
base_model: openai/whisper-large-v3
language:
- da
- en
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- whisper
- whisper.cpp
- ggml
- danish
- long-form
- hallucination
- repetition
- weight-editing
- interpretability
---

# whisper-large-v3-copy-heads-off

Whisper large-v3 with 11 decoder self-attention heads switched off. Those heads copy earlier text, and they are what makes large-v3 write the same sentence over and over on long recordings. Nothing was trained: the edit is one line of arithmetic on the original weights, and [a short script](make_copy_heads_off.py) rebuilds this exact file from `openai/whisper-large-v3`.

## The short answer

| Question | Answer | |
|---|---|---|
| Can a direct weight edit remove the loops? | **Yes, mostly.** On 30 recordings the edit never saw, long loops went from 21 to 1, and the words written by loops dropped 94%. | 🟢 Good |
| Is that real, or luck? | **Real.** Two random edits of the same size did nothing like it. | 🟢 Good |
| Does it break anything? | **Nothing we found.** Danish accuracy on normal short recordings stayed the same. It still works when the model is compressed. | 🟢 Good |
| Did it pass the strict pass mark we set before the final test? | **Just missed.** We asked for 80% fewer loops and got 76%. One test lost one more sentence than the original (15 against 14). | 🟡 Partly |
| Is it better at Danish? | **Only on long audio.** On long recordings transcribed in one go, the Danish word error rate went from 31.0% to 19.7%, because the repeats are gone. On short recordings it did not change: removing a fault does not teach the model more Danish. | 🟡 Partly |
| Does it stop Whisper making up text in silence? | **No, and it was not meant to.** That is a different fault, and voice detection (VAD) already handles it. Codictate uses voice detection. | ⚪ Out of scope |

**Verdict: the method works.** We found what causes the copy loop and removed it with a small edit, without training. It fell just short of the strict pass mark we set for ourselves, so we call it a strong result, not a finished fix.

Method, code and every number: [github.com/EmilLykke/whisper-loop-edit](https://github.com/EmilLykke/whisper-loop-edit).

## Terms

- **Loop**: the transcript writes a span of 3 or more words again when the audio says it once.
- **Lost Audio**: an utterance that never reaches the transcript (often because a Loop ran over it).
- **Decode Path**: how the long recording is fed to the model. `timestamps`: whisper.cpp long-form with timestamps, Silero VAD, beam 5. `-nt`: whisper.cpp long-form without timestamps, 30 s windows with previous text carried over. `nt-vad`: the recording cut by VAD into slices of at most 30 s, each decoded alone (what [Codictate](https://github.com/EmilLykke/codictate) ships).
- **Control Edit**: the same edit on randomly chosen heads in the same layers, to show what any edit of that size does.

## The edit

For each (layer, head) below, zero that head's 64 columns of the decoder self-attention output projection, so the head can no longer write to the residual stream:

```
model.decoder.layers.L.self_attn.out_proj.weight[:, 64*h : 64*h + 64] = 0
```

Heads (decoder layer, head; 0-based): L2H19, L12H13, L13H7, L13H18, L15H17, L18H19, L21H14, L24H0, L25H18, L25H19, L26H6.

Biases and every other tensor are unchanged, so names, shapes and dtype (fp16) match the original. 9 of 1,259 tensors change.

How the heads were found: on long Danish clips we rebuilt the decoder state at the moment large-v3 starts a Loop, then built 220 pairs of windows, one with an earlier copy of a sentence in the previous text and one with it cut out. Activation patching and direct logit attribution point at induction heads in decoder self-attention: early heads that find the earlier copy (L12H13 and others) and late heads that write it out (L25H18 and others). Zeroing the 11 heads takes the induced Loop rate from 68% to 0%; random heads leave it at 68–74%. Cross-attention, which earlier work blamed (Glazer et al. 2025), does not collapse in our cases.

## Results

All long-audio numbers are whisper.cpp ([crispasr](https://github.com/CrispStrobe/CrispASR) v0.8.41, a whisper.cpp fork, Metal) on an Apple Silicon Mac, Danish, with Codictate's exact command lines.

### The strict pass mark (30 hidden Danish recordings, 90 minutes, sealed until the final test)

| What we required | large-v3 | this model | Passed? |
|---|---|---|---|
| At least 80% fewer Loops (`timestamps` + `-nt`) | 25 | 6 (76% fewer) | ❌ Just missed |
| Random edits of the same size must not do the same | | they gave 12% fewer and 4% more | ✅ |
| No more sentences left out (Lost Audio), in any mode | `nt-vad` (slices): 14 of 460 | 15 of 460 (random edits: 15 and 14) | ❌ Just missed, by one sentence |
| No Loops when the audio is cut into slices (`nt-vad`) | 0 | 0 | ✅ |
| FLEURS `da_dk` test word error rate no more than 0.3 points worse | 12.56% | 12.45% | ✅ |

Why the misses are small: 5 of the 6 Loops left are 3 or 4 words written twice, a slip the original and both random edits make too. The sixth is made-up text, not a copy.

### Other results (not part of the pass mark)

| | large-v3 | this model |
|---|---|---|
| Held-out, Loops of 10+ words (`timestamps` + `-nt`) | 21 | 1 |
| Held-out, words written by Loops | 1,380 | 85 |
| Held-out, `timestamps` WER | 31.0% | 19.7% |
| Held-out, `-nt` WER | 38.3% | 36.2% |
| Held-out, `nt-vad` WER | 14.4% | 14.2% |
| Codictate's Danish long clips, `timestamps` WER | 17.3% | 11.3% |
| Codictate's English long clips, `timestamps` WER | 17.6% | 1.7% |
| 5 English Long Clips (FLEURS `en_us`), `timestamps` Loops / WER | 8 / 106.6% | 2 / 24.6% |
| 5 English Long Clips, `-nt` Loops / WER | 3 / 48.2% | 3 / 51.9% |
| ggml q5_0, Held-out Loops (`timestamps` + `-nt`) | 33 | 7 (−79%) |

### In Hugging Face transformers

A smaller check on the first 10 of those clips (30 minutes), decoding the whole clip with transformers' own long-form `generate` (greedy, no prompt, no VAD). One run, no Control Edits.

| Settings | | Loops | Loops of 10+ words | Words written by Loops | WER |
|---|---|---|---|---|---|
| OpenAI's recommended fallback (temperature ladder, compression ratio 1.35, log-prob −1.0) | large-v3 | 21 | 10 | 388 | 32.3% |
| | this model | 3 | 0 | 11 | 29.5% |
| No fallback (temperature 0) | large-v3 | 18 | 11 | 3,132 | 112.0% |
| | this model | 7 | 4 | 388 | 30.6% |

The fallback stops the worst runaways but not the copies; this model removes most of those. Its remaining long Loops without fallback are made-up filler, not copies.

## Limits

- **Made-up text in silence is not fixed.** With no VAD, Whisper invents text in pauses ("… denne video", subtitle credits), and that filler can still repeat. Use VAD in front of this model. Removing the copy heads can even make this worse on some clips.
- **No gain on short audio.** If you already cut audio into slices under 30 s and decode each alone (`nt-vad`), there are no copy Loops to remove, and WER is unchanged.
- **Not better Danish.** The edit removes a failure. It does not add knowledge: Danish WER on FLEURS test is the same within noise.
- **Tested mainly on whisper.cpp, Danish.** English was checked on 8 long clips only. The transformers check above is 10 clips.
- **Long-form decoding is chaotic.** Any change to the weights, even a random one, can break or create a single big Loop on a given clip. Judge on many clips, against a control.
- **The bar was amended once.** After the search phase we dropped a Decode Path with no VAD from the bar, openly and before the sealed run ([ADR 0002](https://github.com/EmilLykke/whisper-loop-edit/blob/main/docs/adr/0002-timestamps-bare-leaves-the-bar.md)). On the sealed clips the edit improved that path too (Loops 41 → 23).

## How to use

transformers:

```python
import torch
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="emillykkegrann/whisper-large-v3-copy-heads-off",
    torch_dtype=torch.float16,
    device="cuda",  # or "mps", "cpu"
)
out = asr(
    "long_recording.wav",
    return_timestamps=True,
    generate_kwargs={
        "language": "da",
        "condition_on_prev_tokens": True,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    },
)
print(out["text"])
```

These `generate_kwargs` are OpenAI's recommended long-form fallback settings. On Apple `mps` with transformers 5.22 we had to disable a deferred stop check (`transformers.generation.utils.DeferredStopCheck.is_supported`) to avoid an `AttributeError` at the end of each window; it does not change the output.

whisper.cpp / crispasr (the ggml files have the same tensor layout as the official `ggml-large-v3.bin`; we tested crispasr v0.8.41):

```bash
hf download emillykkegrann/whisper-large-v3-copy-heads-off ggml-large-v3-copy-heads-off-f16.bin --local-dir .
crispasr -m ggml-large-v3-copy-heads-off-f16.bin --language da -f long_recording.wav \
  --vad -vm ggml-silero-v6.2.0.bin -vt 0.25 -bs 5
```

## Files

| File | What | sha256 |
|---|---|---|
| `model.safetensors` (+ config, tokenizer) | transformers, fp16 | `2c08dd6c378dcc24870813a8b09e26b6734113b433d7e8c0e1d97658803854f1` |
| `ggml-large-v3-copy-heads-off-f16.bin` | whisper.cpp ggml, f16 | `ce1c3710c08dda0f5ec13b5005b9c3efbafabf83b02eff778cd7ada811fccace` |
| `ggml-large-v3-copy-heads-off-q5_0.bin` | whisper.cpp ggml, q5_0 | `e5614e50a234a6793ea5bb10e057d7b6e053414715ee4a4c092cc9ded55696ed` |
| `make_copy_heads_off.py` | rebuilds `model.safetensors` from `openai/whisper-large-v3` at revision `06f233fe` | |

## Credits

- The idea: Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space Programming" (MIT CSAIL, November 2026): [blog](https://rikiyat.github.io/blog/weight-space-programming/), [code](https://github.com/RikiyaT/lm_understand-logs). There an LLM agent edited a small LLM's weights directly, with no training, to remove reasoning loops. We tried the same on Whisper.
- Prior work on Whisper repetition and hallucination: Glazer et al. 2025, [arXiv:2508.15882](https://arxiv.org/abs/2508.15882); Wang et al., "Calm-Whisper", Interspeech 2025, [arXiv:2505.12969](https://arxiv.org/abs/2505.12969).
- Base model: OpenAI, [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (MIT).
- Data: [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0), [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0).
- Decode settings and scoring: [Codictate](https://github.com/EmilLykke/codictate) (Apache-2.0).
- Found and measured by Claude (Anthropic). By Emil Lykke Grann.

License: MIT, like the base model.
