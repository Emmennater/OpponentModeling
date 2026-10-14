#!/usr/bin/env bash
# Trains net<N>, the network that plays generation N+1, with Lc0's own
# training code (lczero-training/tf, with lczero-training.patch): from the
# network that played generation N (791556 for the first, else net<N-1>),
# on the newest games of all generations so far, as configs/791556.yaml's
# dataset section says (only the engine's positions, results discounted by
# the half-moves to mate).
#
# The run goes to training/runs/net<N>/ (ignored by git): the network is put
# into a checkpoint first (net_to_model.py), then trained; a rerun resumes
# from the last checkpoint. lczero-training tests the held-out games (10%)
# before the first step, which is the starting network's loss, then every
# TEST_STEPS and at the end. The result is copied to training/nets/net<N>.pb.gz,
# where generate_games.py looks for the next generation's network.
#
# Usage, from Windows (after setup_wsl.sh), with the steps and learning rate
# to train with:
#     wsl -d Ubuntu -- bash training/train_net.sh 1 400 0.0001

set -euo pipefail

GENERATION=$1
STEPS=${2:-400}
LR=${3:-0.0001}
TEST_STEPS=50

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.venvs/lc0-training/bin/python"
NAME=$(printf 'net%02d' "$GENERATION")
if [ "$GENERATION" -eq 1 ]; then
  START="$HERE/../lc0/bin/791556.pb.gz"
else
  START="$HERE/nets/$(printf 'net%02d' $((GENERATION - 1))).pb.gz"
fi
RUN="$HERE/runs/$NAME"
export TF_CPP_MIN_LOG_LEVEL=2  # TensorFlow's start-up messages under WSL are noise

mkdir -p "$RUN" "$HERE/nets"
"$PYTHON" -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['name'] = sys.argv[2]
cfg['training'].update({
    'path': sys.argv[3], 'total_steps': int(sys.argv[4]), 'lr_values': [float(sys.argv[5])],
    'lr_boundaries': [], 'warmup_steps': min(50, int(sys.argv[4]) // 4),
    'test_steps': int(sys.argv[6]), 'checkpoint_steps': int(sys.argv[6]) * 2,
    'num_test_positions': 20000})
yaml.safe_dump(cfg, open(sys.argv[7], 'w'))
" "$HERE/configs/791556.yaml" "$NAME" "$HERE/runs" "$STEPS" "$LR" "$TEST_STEPS" "$RUN/config.yaml"

cd "$HERE/lczero-training/tf"
if [ ! -f "$RUN/checkpoint" ]; then
  echo "== $(basename "$START") to a checkpoint"
  "$PYTHON" net_to_model.py --cfg "$RUN/config.yaml" "$START" > "$RUN/net_to_model.log" 2>&1 || {
    tail -30 "$RUN/net_to_model.log"; exit 1; }
fi
echo "== Training $NAME: $STEPS steps at learning rate $LR"
"$PYTHON" train.py --cfg "$RUN/config.yaml" --output "$RUN/$NAME" 2>&1 | tee -a "$RUN/train.log" |
  grep --line-buffered -E "^step|^Using|Restoring|Model saved"
cp "$RUN/$NAME.pb.gz" "$HERE/nets/$NAME.pb.gz"
echo "== Wrote training/nets/$NAME.pb.gz"
