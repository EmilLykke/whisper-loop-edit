#!/usr/bin/env bash
# Final run (ADR-0002): build the Control Edits for E3, then decode the Held-out Set.
#
#   scripts/final_pipeline.sh build        C1 (rebuilt, sha checked against Phase 2) and C4 -> ggml
#   scripts/final_pipeline.sh decode P,..  every model on the Held-out Set, the given Decode Paths
#
# Models: baseline (models/ggml/ggml-large-v3-ours-f16.bin), E3 (kept since Phase 2), C1, C4.
# Transcripts: results/final/transcripts/<id>/<path>/heldout_*.run1.*
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/final/transcripts
model_of() {
  case $1 in
    baseline) echo models/ggml/ggml-large-v3-ours-f16.bin ;;
    *) echo models/ggml/edits/ggml-large-v3-$1-f16.bin ;;
  esac
}
case $1 in
  build)
    for id in C1 C4; do
      gg=$(model_of $id)
      [[ -f $gg ]] && continue
      echo "=== build $id $(date +%H:%M:%S), free $(df -g . | awk 'NR==2 {print $4}') GiB"
      $PY -I scripts/phase2_edit.py build $id | head -1
      scripts/hf_to_ggml.sh models/hf/edits/$id "$gg" | tail -1
      $PY -I scripts/phase2_edit.py verify $id "$gg"
      rm -f models/hf/edits/$id/model.safetensors
    done
    ;;
  decode)
    for id in baseline E3 C1 C4; do
      echo "=== decode $id $2 $(date +%H:%M:%S)"
      bun scripts/run_paths.ts --set heldout --model "$(model_of $id)" --out $OUT/$id --runs 1 --paths "$2" | grep -E 'ERROR' || true
    done
    ;;
esac
echo "=== done $1 $(date +%H:%M:%S)"
