# Phase 0: large-v3 in HF and GGML form, and a faithful conversion

**Verdict:** the conversion is faithful. Our GGML f16 from the HF safetensors has the same
1,259 tensors as the official `ggml-large-v3.bin` Codictate ships, every one bit-identical,
with the same mel filters, the same hparams and the same first 50,257 vocabulary entries. The
files differ only in 20,164 bytes of header, the 1,609 special-token strings our converter
writes, and in tensor order. crispasr gives the same text from both on every test clip, on
both the `-nt` and the timestamps Decode Path, and HF transformers on MPS gives the same text
again.

Run 2026-11-08 on an Apple M4 Max with crispasr v0.8.41 (`340d7085`), torch 2.15.0,
transformers 5.22.2 and huggingface_hub 2.4.0.

## Sources

| What | Where | Pin |
| --- | --- | --- |
| HF checkpoint | `openai/whisper-large-v3` → `models/hf/whisper-large-v3/` | revision `06f233fe06e710322aca913c1bc4249a0d71fce1` |
| Official GGML | `ggerganov/whisper.cpp` `ggml-large-v3.bin` → `models/ggml/ggml-large-v3-official.bin` | revision `5359861c739e955e79d9a303bcbc70fb988958b1` (`WHISPER_CPP_MODELS_REVISION`) |
| Converter | `../codictate/.tmp/edda-mirror/crispasr-src/models/convert-h5-to-ggml.py` | CrispASR v0.8.29 `9d1c5eef`, file sha256 `fe8b0d8a…5dc34b2d` |
| Mel filters | `../codictate/.tmp/edda-mirror/whisper-src/whisper/assets/mel_filters.npz` | openai/whisper `86098128`, sha256 `7450ae70…5f0f6d4c` |

From the HF repo we fetched only `model.safetensors` (fp16), `config.json`,
`generation_config.json`, the tokenizer and preprocessor files and `README.md`. The flax,
pytorch `.bin` and fp32 shard duplicates were skipped.

## Files

| File | Bytes | sha256 |
| --- | --- | --- |
| `models/hf/whisper-large-v3/model.safetensors` | 3,087,130,976 | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` (matches the HF LFS etag) |
| `models/ggml/ggml-large-v3-ours-f16.bin` | 3,095,053,647 | `1c57c1f0c22d454829894da5add0e409db281ca283529cf726aafc7816abdb16` |
| `models/ggml/ggml-large-v3-official.bin` | 3,095,033,483 | `64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2` |

**Official sha256 check: pass.** It equals the `large-v3` pin in
`../codictate/src/shared/speech-models.ts`.

## Our conversion against the official file

Not byte-identical (`cmp` first differs at byte 102,969, inside the vocabulary).
`scripts/compare_ggml.py` breaks it down:

| Part | Result |
| --- | --- |
| hparams (11 ints) | identical (n_vocab 51866, 32+32 layers, 1280 wide, 20 heads, 128 mels, ftype 1) |
| Mel filters (128 × 201 f32) | identical |
| Vocabulary | ids 0..50256 identical. Ours has 51,866 entries, official 50,257: ours also writes the 1,609 special tokens (`<\|endoftext\|>` … `<\|30.00\|>`), because the HF repo ships `added_tokens.json`. whisper.cpp generates those at run time. This is the same difference `../codictate/docs/EDDA_MIRROR.md` records for Edda, to the byte (20,164). |
| Tensors | 1,259 in both, same names, shapes and ftypes, **all 1,259 bit-identical** (max abs diff 0). A second check hashed every tensor (name, shape, bytes) in name order: the same sha256 for both files. |
| Tensor order | differs (ours starts `encoder.conv1.weight`, official starts `encoder.blocks.0.attn.out.bias`). whisper.cpp looks tensors up by name, so this does not matter. |
| Size | tensor payload 3,094,359,040 bytes in both; the whole 20,164-byte file difference is header |

There is no fp16 rounding difference: the HF checkpoint is already fp16, and the converter's
`.float()` then `.astype(float16)` is lossless for it.

## crispasr: ours against official

`scripts/smoke_crispasr.py`, i.e. `crispasr -m <model> -l da -f <wav> --no-prints -nt -bs 5`,
on three short FLEURS `da_dk` clips. Seconds are wall time per run, model load included
(warm page cache, default Metal backend).

| Clip | Length | Ours | Official | Text (both) |
| --- | --- | --- | --- | --- |
| `8476175897995602250` | 6.84 s | 1.38 s | 1.38 s | Aerosmith har aflyst de resterende koncerter på deres turné. |
| `11417824699554894875` | 7.14 s | 1.51 s | 1.52 s | Adskillelse og rekombination blander variationen frem og tilbage mellem de to puljer med hver generation. |
| `14524040401498118125` | 8.04 s | 1.63 s | 1.60 s | Alle navneord, inklusiv ordet sige for dig, begynder altid med stort og så i midten af en sætning. |

All three identical. On the timestamps Decode Path (no `-nt`), the first and third clip gave
the same segments and times from both files too (`[00:00:00.000 --> 00:00:06.840]` and
`[… --> 00:00:08.040]`, with `"sige"` in quotes on that path).

## HF transformers on MPS, float16

`scripts/smoke_hf.py`: `WhisperForConditionalGeneration` + `WhisperProcessor`, `dtype=float16`
on `mps`, `language="da"`, `task="transcribe"`, `num_beams=5`. The model loads in 1.2 s.

| Clip | Generate | HF text | Same as crispasr? |
| --- | --- | --- | --- |
| `8476175897995602250` | 3.38 s (first call, includes warm-up) | Aerosmith har aflyst de resterende koncerter på deres turné. | yes |
| `11417824699554894875` | 1.92 s | Adskillelse og rekombination blander variationen frem og tilbage mellem de to puljer med hver generation. | yes |
| `14524040401498118125` | 2.16 s | Alle navneord, inklusiv ordet sige for dig, begynder altid med stort og så i midten af en sætning. | yes |

## Timings

| Step | Time |
| --- | --- |
| HF download (13 files, 3.1 GB) | 3 min 48 s |
| Official GGML download (3.1 GB, sharing bandwidth) | 8 min 4 s |
| `scripts/hf_to_ggml.sh` (converter 22 s) | 28 s wall, 4.0 GB peak RSS |
| `scripts/compare_ggml.py` | under 1 s (memory-mapped, warm cache) |

## Converting an edited checkpoint later

```bash
scripts/hf_to_ggml.sh <edited-hf-dir> models/ggml/<name>-f16.bin
```

The script stages symlinks (the HF dir is never touched), checks the converter's and the
mel filters' sha256 against the pins above, derives `vocab.json`/`added_tokens.json` from
`tokenizer.json` when they are missing (as `mirror-edda.ts` does), and writes the
converter's log next to the output. Things to know before relying on it:

- **`proj_out.weight` is never written.** The converter skips it and whisper.cpp reuses
  `decoder.token_embedding.weight` as the output projection (large-v3 has
  `tie_word_embeddings=True`). A Weight Edit to the output projection must be made to
  `model.decoder.embed_tokens.weight`, which also changes the input embedding. An untied
  edit to `proj_out` alone would be dropped silently.
- **Everything is rounded to fp16, biases and LayerNorms included.** 1-D tensors and the
  positional embeddings are stored as f32 but pass through fp16 first. A tiny edit can round
  away, so compare HF-side results using the fp16-rounded edited weights.
- `save_pretrained` on the model alone writes no tokenizer files. Copy them from
  `models/hf/whisper-large-v3/` into the edited dir; the script stops with a message if
  neither `vocab.json` nor `tokenizer.json` is there.
- Our files carry the 1,609 special-token strings, so they will never be byte-identical to a
  whisper.cpp conversion. Compare tensors with `scripts/compare_ggml.py`, not file hashes.
- The converter lives in `../codictate/.tmp/`, which is scratch space. If it disappears,
  re-create it with `../codictate/scripts/mirror-edda.ts` steps 2-3 (or set `CONVERTER` and
  `WHISPER_SRC`); the sha256 checks catch a different version.

## Note on the test clips

The three clips are from FLEURS `da_dk` **test**, the only split in
`../codictate/benchmarks/datasets/fleurs/da_dk/` when this ran (the Find Set was not built
yet). ADR-0001 holds those clips back until the final measurement. This run used the
unedited model only, to check the conversion, and no edit was chosen from it, but later
smoke tests should use Find Set clips instead.
