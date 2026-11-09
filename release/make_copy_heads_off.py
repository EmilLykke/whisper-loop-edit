"""Build whisper-large-v3-copy-heads-off from openai/whisper-large-v3, bit for bit.

    pip install "torch>=2.4" "safetensors==0.8.0" "huggingface_hub>=1.0"
    python make_copy_heads_off.py [out_dir]          # default ./whisper-large-v3-copy-heads-off

The edit (no training): in 11 decoder self-attention heads, zero that head's 64 input columns
of the attention output projection, so the head can no longer write to the residual stream:

    model.decoder.layers.L.self_attn.out_proj.weight[:, 64*h : 64*h + 64] = 0

Biases and every other tensor are unchanged; names, shapes and dtype (fp16) are kept, so the
result loads in transformers and converts to ggml like the original. The script checks the
sha256 of the input and of the output against the published files.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
BASE_SHA256 = "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"
OUT_SHA256 = "2c08dd6c378dcc24870813a8b09e26b6734113b433d7e8c0e1d97658803854f1"

# (decoder layer, head), 0-based. Found by activation patching on contrastive Danish windows.
HEADS = [(12, 13), (13, 7), (13, 18), (15, 17), (2, 19), (18, 19),
         (25, 18), (25, 19), (24, 0), (26, 6), (21, 14)]
HEAD_DIM = 64

FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "added_tokens.json", "merges.txt", "normalizer.json", "preprocessor_config.json",
         "special_tokens_map.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "whisper-large-v3-copy-heads-off")
    out.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(REPO, "model.safetensors", revision=REVISION))
    got = sha256(src)
    if got != BASE_SHA256:
        sys.exit(f"base model.safetensors sha256 {got} != {BASE_SHA256}")
    sd = load_file(src)
    for layer, head in HEADS:
        w = sd[f"model.decoder.layers.{layer}.self_attn.out_proj.weight"]
        w[:, HEAD_DIM * head:HEAD_DIM * (head + 1)] = 0
    save_file(sd, out / "model.safetensors", metadata={"format": "pt"})
    for name in FILES:
        shutil.copy2(hf_hub_download(REPO, name, revision=REVISION), out / name)
    got = sha256(out / "model.safetensors")
    print(f"{out / 'model.safetensors'}  sha256 {got}")
    if got != OUT_SHA256:
        sys.exit(f"output sha256 differs from the published file ({OUT_SHA256})")
    print("matches the published whisper-large-v3-copy-heads-off")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
