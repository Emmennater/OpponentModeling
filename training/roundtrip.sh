#!/usr/bin/env bash
# Checks that lczero-training reproduces 791556 before any training: the
# network goes into a checkpoint (net_to_model.py) and straight back out
# (model_to_net.py), the way a fine-tune starts and ends. Then compare the
# result with the original, from Windows (see compare_nets.py).
#
# The checkpoint and the exported network go to training/runs/roundtrip/
# (ignored by git), with configs/791556.yaml's settings but that path.
#
# Usage, from Windows (after setup_wsl.sh):
#     wsl -d Ubuntu -- bash training/roundtrip.sh
#     .venv\Scripts\python training\compare_nets.py lc0\bin\791556.pb.gz training\runs\roundtrip\om-791556\om-791556-0.pb.gz

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.venvs/lc0-training/bin/python"
OUT="$HERE/runs/roundtrip"
NET="$HERE/../lc0/bin/791556.pb.gz"
export TF_CPP_MIN_LOG_LEVEL=2  # TensorFlow's start-up messages under WSL are noise

rm -rf "$OUT"
mkdir -p "$OUT"
"$PYTHON" -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['training']['path'] = sys.argv[2]
yaml.safe_dump(cfg, open(sys.argv[3], 'w'))
" "$HERE/configs/791556.yaml" "$OUT" "$OUT/791556.yaml"

cd "$HERE/lczero-training/tf"
echo "== Network to checkpoint"
"$PYTHON" net_to_model.py --cfg "$OUT/791556.yaml" "$NET" > "$OUT/net_to_model.log" 2>&1 || {
  tail -30 "$OUT/net_to_model.log"; exit 1; }
tail -1 "$OUT/net_to_model.log"
echo "== Checkpoint to network"
"$PYTHON" model_to_net.py --cfg "$OUT/791556.yaml" > "$OUT/model_to_net.log" 2>&1 || {
  tail -30 "$OUT/model_to_net.log"; exit 1; }
ls -l "$OUT"/om-791556/*.pb.gz
