#!/usr/bin/env bash
# Convert a Hugging Face Whisper checkpoint (safetensors) to whisper.cpp GGML f16, the same
# way Codictate built its Edda Mirror (../codictate/scripts/mirror-edda.ts, step 3-4):
# CrispASR v0.8.29's models/convert-h5-to-ggml.py plus an openai/whisper checkout for
# mel_filters.npz.
#
#   scripts/hf_to_ggml.sh <hf-model-dir> <out.bin>
#
# The HF dir needs config.json and model.safetensors (or a sharded index). vocab.json and
# added_tokens.json are used when present; otherwise both are derived from tokenizer.json,
# as mirror-edda.ts does. Nothing in <hf-model-dir> is modified: the converter reads a
# staging dir of symlinks.
#
# Env overrides: PYTHON, CONVERTER, WHISPER_SRC.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <hf-model-dir> <out.bin>" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIRROR_DIR="${CODICTATE_DIR:-$REPO_DIR/../codictate}/.tmp/edda-mirror"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
CONVERTER="${CONVERTER:-$MIRROR_DIR/crispasr-src/models/convert-h5-to-ggml.py}"
WHISPER_SRC="${WHISPER_SRC:-$MIRROR_DIR/whisper-src}"

# Pinned so a silently changed converter or filter bank cannot slip into a comparison.
# CrispASR v0.8.29 (9d1c5eef) and openai/whisper 86098128, the pair mirror-edda.ts uses.
CONVERTER_SHA256=fe8b0d8a06096a12e3e302a20e8875827244743c3605810a9c9c463c5dc34b2d
MEL_SHA256=7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c

src="$(cd "$1" && pwd)"
out="$2"
mkdir -p "$(dirname "$out")"
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"

check_sha() {
  local got
  got="$(shasum -a 256 "$1" | cut -d' ' -f1)"
  if [[ "$got" != "$2" ]]; then
    echo "sha256 mismatch for $1: got $got, want $2" >&2
    exit 1
  fi
}
check_sha "$CONVERTER" "$CONVERTER_SHA256"
check_sha "$WHISPER_SRC/whisper/assets/mel_filters.npz" "$MEL_SHA256"

[[ -f "$src/config.json" ]] || { echo "no config.json in $src" >&2; exit 1; }

work="$(mktemp -d "${TMPDIR:-/tmp}/hf_to_ggml.XXXXXX")"
trap 'rm -rf "$work"' EXIT
stage="$work/stage"
convert_out="$work/out"
mkdir -p "$stage" "$convert_out"

for f in "$src"/config.json "$src"/generation_config.json "$src"/*.safetensors \
         "$src"/model.safetensors.index.json "$src"/vocab.json "$src"/added_tokens.json \
         "$src"/tokenizer.json; do
  [[ -e "$f" ]] && ln -s "$f" "$stage/$(basename "$f")"
done

if [[ ! -e "$stage/vocab.json" ]]; then
  if [[ ! -e "$stage/tokenizer.json" ]]; then
    echo "no vocab.json or tokenizer.json in $src; copy the tokenizer files from the base" \
         "model dir (save_pretrained on the model alone does not write them)" >&2
    exit 1
  fi
  echo "--- Deriving vocab.json and added_tokens.json from tokenizer.json ---"
  "$PYTHON" -I -c '
import json, sys
src, out = sys.argv[1], sys.argv[2]
tok = json.load(open(f"{src}/tokenizer.json", encoding="utf8"))
vocab = tok["model"]["vocab"]
added = {t["content"]: t["id"] for t in tok["added_tokens"] if t["content"] not in vocab}
json.dump(vocab, open(f"{out}/vocab.json", "w", encoding="utf8"), ensure_ascii=False)
json.dump(added, open(f"{out}/added_tokens.json", "w", encoding="utf8"), ensure_ascii=False)
' "$src" "$stage"
fi

echo "--- Converting $src -> $out ---"
start=$(date +%s)
(cd "$work" && "$PYTHON" -I "$CONVERTER" "$stage" "$WHISPER_SRC" "$convert_out" > "${out%.bin}.convert.log")
mv "$convert_out/ggml-model.bin" "$out"
echo "converted in $(( $(date +%s) - start )) s"
ls -l "$out"
shasum -a 256 "$out"
