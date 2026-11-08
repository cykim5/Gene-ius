#!/usr/bin/env bash
# Rebuild the variant graph on canonical feature_table.csv positions and rerun Output A
# headline + ablation + held-out-gene eval. Uses CUDA when PyTorch sees a GPU.
# Writes eval/results/reclassification_A.* and eval/results/ablation_A.*
#
#   bash scripts/retrain_all.sh
#
# Safe to re-run: deletes previous GNN checkpoints and graph caches first so nothing
# is scored from a partial local run. Does not train on this laptop unless you invoke it.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [ -f "$REPO/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$REPO/.venv/bin/activate"
fi
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"

echo "[retrain] repo $REPO"
"$PY" -W ignore - <<'PY'
import torch
print("[retrain] torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

DATA="${GENEIUS_DATA_DIR:-$REPO/data}"
rm -f "$DATA"/cache/variant_graph_*.pt
rm -rf "$DATA/gnn_a" "$DATA/gnn_a_am_impute" "$DATA/gnn_a_no_alphamissense" "$DATA/gnn_runs_v2"
mkdir -p "$REPO/eval/results"

echo "[retrain] headline GNN (a) am_missing"
"$PY" -W ignore -m geneius.models.gnn
echo "[retrain] headline GNN (b) am imputed"
"$PY" -W ignore -m geneius.models.gnn --am-impute --out "$DATA/gnn_a_am_impute"
echo "[retrain] headline GNN no AlphaMissense"
"$PY" -W ignore -m geneius.models.gnn --no-alphamissense

echo "[retrain] ablation + held-out genes (24 GNN runs)"
"$PY" -W ignore eval/ablation.py --train
echo "[retrain] score ablation"
"$PY" -W ignore eval/ablation.py
echo "[retrain] score headline"
"$PY" -W ignore eval/reclassification.py

echo "[retrain] wrote $REPO/eval/results/"
ls -l "$REPO/eval/results"/reclassification_A.* "$REPO/eval/results"/ablation_A.*
echo "[retrain] done. Update eval/RESULTS.md from those files (especially CACNA1C)."
