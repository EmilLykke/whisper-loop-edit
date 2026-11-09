#!/usr/bin/env bash
# Phase 2: build, convert, verify and measure each edit, one after another (crispasr and MPS jobs
# never overlap, so nothing contends for the machine).
#
#   scripts/phase2_pipeline.sh [ids...]      default: baseline E3 C1 E1 E2 E4 E5 C2 E6
#
# Per edit: phase2_edit.py build -> hf_to_ggml.sh -> phase2_edit.py verify (every ggml tensor
# against the fp16 HF tensor) -> phase2_proxy.py (HF, find-B + natural) -> run_paths.ts on all
# 5 Decode Paths x 10 Find clips, run 1 -> a spot-check rerun (timestamps, find_da_5min_04, run 2)
# -> phase2_utts.ts (155 single utterances, nt-vad) -> phase2_measure.ts.
# `baseline` runs only the proxy and the single utterances (its long clips are Phase 0's).
#
# Disk: an edit is 3 GB of safetensors plus 3 GB of ggml, and the disk had 6-20 GiB free. So:
# - once an edit's ggml is converted and tensor-verified and its HF proxy is done, its
#   models/hf/edits/<id>/model.safetensors is deleted (edit.json keeps its sha256);
# - once its crispasr runs and single-utterance decodes are done, its ggml .bin is deleted too
#   (the .convert.log and edit.json, with the ggml sha256, stay), except E3's;
# - before each build, if under 10 GiB is free, finished edits' ggml are deleted first (controls
#   first), so the ~6 GB build peak fits.
# `phase2_edit.py build <id>` + `hf_to_ggml.sh` rebuild both files bit for bit (checked on E3: same
# safetensors and ggml sha256 as edit.json). KEEP_WEIGHTS=1 keeps everything; KEEP_GGML="E3 E7"
# (default "E3") names the edits whose ggml survives. MEASURE_ARGS is passed to phase2_measure.ts
# (Phase 2b: "--out results/phase2b/edits.json baseline E3 C1 C2 E7 E9 C3").
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=results/phase2/transcripts
ids=("$@")
[[ ${#ids[@]} -eq 0 ]] && ids=(baseline E3 C1 E1 E2 E4 E5 C2 E6)

free_gib() { df -g . | awk 'NR==2 {print $4}'; }
make_room() {  # $1 = the edit about to be built
  local f
  for f in models/ggml/edits/ggml-large-v3-C{1,2}-f16.bin models/ggml/edits/ggml-large-v3-E{1,2,4,5,6,3}-f16.bin; do
    (( $(free_gib) >= 10 )) && return 0
    [[ -f $f && $f != *-$1-f16.bin ]] || continue
    local e; e=$(basename "$f" -f16.bin); e=${e#ggml-large-v3-}
    [[ $(ls $T/$e/utts/*.meta.json 2>/dev/null | wc -l) -ge 155 ]] || continue  # only finished ones
    echo "pruning $f (free $(free_gib) GiB before building $1)"; rm -f "$f"
  done
}

for id in "${ids[@]}"; do
  echo "=== $id $(date +%H:%M:%S)"
  if [[ $id == baseline ]]; then
    [[ -f results/phase2/proxy/baseline.json ]] || $PY -I scripts/phase2_proxy.py baseline 2>&1 | grep -E '^baseline:'
    bun scripts/phase2_utts.ts --model models/ggml/ggml-large-v3-ours-f16.bin --out $T/baseline/utts
    continue
  fi
  hf=models/hf/edits/$id
  gg=models/ggml/edits/ggml-large-v3-$id-f16.bin
  need_model=0
  [[ -f results/phase2/proxy/$id.json ]] || need_model=1
  [[ -f $T/$id/nt-vad/find_da_continuous_90s_03.run1.meta.json ]] || need_model=1
  [[ $(ls $T/$id/utts/*.meta.json 2>/dev/null | wc -l) -ge 155 ]] || need_model=1
  if [[ $need_model == 1 ]]; then
    [[ -f $gg ]] || make_room "$id"
    [[ -f $hf/model.safetensors || -f $gg ]] || $PY -I scripts/phase2_edit.py build "$id" | head -1
    if [[ ! -f $gg ]]; then
      scripts/hf_to_ggml.sh "$hf" "$gg" | tail -1
      $PY -I scripts/phase2_edit.py verify "$id" "$gg"
    fi
    if [[ ! -f results/phase2/proxy/$id.json ]]; then
      [[ -f $hf/model.safetensors ]] || $PY -I scripts/phase2_edit.py build "$id" | head -1
      $PY -I scripts/phase2_proxy.py "$id" 2>&1 | grep -E "^$id:"
    fi
    [[ ${KEEP_WEIGHTS:-0} == 1 ]] || rm -f "$hf/model.safetensors"
    bun scripts/run_paths.ts --model "$gg" --out $T/$id --runs 1 | grep -E 'ERROR|5min_04' || true
    bun scripts/run_paths.ts --model "$gg" --out $T/$id --runs 2 --paths timestamps --clips find_da_5min_04
    bun scripts/phase2_utts.ts --model "$gg" --out $T/$id/utts
    if [[ ${KEEP_WEIGHTS:-0} != 1 && " ${KEEP_GGML:-E3} " != *" $id "* ]]; then
      echo "pruning $gg (crispasr + utterances finished)"; rm -f "$gg"
    fi
  fi
  bun scripts/phase2_measure.ts ${MEASURE_ARGS:-} > /dev/null || echo "measure failed (continuing)"
  df -h . | tail -1
done
echo "=== done $(date +%H:%M:%S)"
