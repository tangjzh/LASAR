# MindCraft evaluation

```bash
# from repo root, conda env with deps installed
bash evaluation/scripts/eval/mindcraft.sh \
  CKPT_PATH \
  MINDCRAFT_DATA_ROOT \
  eval_out/mindcraft \
  "0"
```

Outputs: `eval_out/mindcraft/inference_results/predictions_chunk_*.json`, `metrics_results/metrics_results.json`.

Metrics: **QA-Acc**, **CMC**, **GCA**, **SR@WA** (GCA/SR@WA need navigation success labels; see `mindcraft_with_nav.sh`).

Re-run metrics only:

```bash
python evaluation/vlnce_baselines/mindcraft_metrics.py \
  --predictions-dir eval_out/mindcraft/inference_results \
  --dataset-path "$MINDCRAFT_DATA_ROOT" \
  --output-dir eval_out/mindcraft/metrics_results \
  --metrics all
```
