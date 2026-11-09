#!/usr/bin/env bash
# Final run, after the Held-out decodes (final_pipeline.sh): FLEURS da_dk test on nt-vad
# (baseline, E3, C1), then the report-only extras (baseline vs E3): Codictate's long-audio clips,
# English Long Clips, the private dictation, and q5_0 on the Held-out Set (timestamps + nt).
set -euo pipefail
cd "$(dirname "$0")/.."
F=results/final
X=$F/extras
QUANT=${CODICTATE_DIR:-../codictate}/.tmp/edda-mirror/crispasr-src/build/bin/crispasr-legacy-quantize  # Codictate's quantize step (docs/EDDA_MIRROR.md)
model_of() {
  case $1 in
    baseline) echo models/ggml/ggml-large-v3-ours-f16.bin ;;
    baseline-q5_0) echo models/ggml/ggml-large-v3-ours-q5_0.bin ;;
    E3-q5_0) echo models/ggml/edits/ggml-large-v3-E3-q5_0.bin ;;
    *) echo models/ggml/edits/ggml-large-v3-$1-f16.bin ;;
  esac
}
for id in baseline E3 C1; do
  echo "=== fleurs $id $(date +%H:%M:%S)"
  bun scripts/final_fleurs.ts decode --model "$(model_of $id)" --out $F/fleurs/$id
done
for id in baseline E3; do
  echo "=== extras $id $(date +%H:%M:%S)"
  bun scripts/run_paths.ts --set codictate --model "$(model_of $id)" --out $X/codictate/$id --runs 1 --paths timestamps,nt,nt-vad | grep ERROR || true
  bun scripts/run_paths.ts --set english --model "$(model_of $id)" --out $X/english/$id --runs 1 --paths timestamps,nt,nt-vad | grep ERROR || true
  bun scripts/final_dictation.ts decode --model "$(model_of $id)" --out $X/dictation/$id
done
[[ -f $(model_of baseline-q5_0) ]] || $QUANT "$(model_of baseline)" "$(model_of baseline-q5_0)" q5_0 > $X/quant-baseline.log 2>&1
[[ -f $(model_of E3-q5_0) ]] || $QUANT "$(model_of E3)" "$(model_of E3-q5_0)" q5_0 > $X/quant-E3.log 2>&1
shasum -a 256 "$(model_of baseline-q5_0)" "$(model_of E3-q5_0)"
for id in baseline E3; do
  echo "=== q5_0 $id $(date +%H:%M:%S)"
  bun scripts/run_paths.ts --set heldout --model "$(model_of $id-q5_0)" --out $X/q5_0/$id --runs 1 --paths timestamps,nt | grep ERROR || true
done
echo "=== done rest $(date +%H:%M:%S)"
