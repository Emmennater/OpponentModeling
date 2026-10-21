#!/usr/bin/env bash
# Trains net<N>, the network that plays generation N+1, with Lc0's own
# training code (lczero-training/tf, with lczero-training.patch): from the
# network that played generation N (791556 for the first, else net<N-1>),
# on the newest games of all generations so far, as configs/791556.yaml's
# dataset section says (both sides' positions for the value and moves left,
# the engine's for the policy, results discounted by the half-moves to mate).
#
# The run goes to training/runs/net<N>/ (ignored by git): the network is put
# into a checkpoint first (net_to_model.py), then trained. lczero-training
# tests the held-out games (10%) before the first step, which is the starting
# network's loss, then every TEST_STEPS and at the end. The result is copied
# to training/nets/net<N>.pb.gz, where generate_games.py looks for the next
# generation's network.
#
# A rerun resumes: from the last checkpoint (every 2 * TEST_STEPS steps), or,
# if the network was already written, straight to copying it. Stopping the
# script (Ctrl+C) stops the training too; one left running by a closed window
# is stopped before resuming.
#
# Only generations up to N are read, through links in the run's data/ folder:
# later ones were played by net<N> or after it. For experiments, WINDOW sets
# how many of the newest games are read instead of the config's num_chunks,
# START the network to train from, and TAG names the result net<N>-<TAG>, to
# keep it apart from net<N>.
#
# SEARCH=om trains the opponent-model branch (generate_games.py --search om):
# net<N>-om, from the network that played gen<N>-om (net<N-1>-om, or
# net<N-1> for the branch's first generation), on gen<K>-om in place of
# gen<K> wherever the branch has it.
#
# Usage, from Windows (after setup_wsl.sh), with the steps and learning rate
# to train with:
#     wsl -d Ubuntu -- bash training/train_net.sh 1 400 0.0001
#     wsl -d Ubuntu -- env WINDOW=5000 TAG=newest bash training/train_net.sh 2 400 0.0001
#     wsl -d Ubuntu -- env SEARCH=om bash training/train_net.sh 10 400 0.0001

set -euo pipefail

GENERATION=$1
STEPS=${2:-400}
LR=${3:-0.0001}
TEST_STEPS=50
WINDOW=${WINDOW:-}
TAG=${TAG:-}
case "${SEARCH:-plain}" in
  plain) BRANCH="" ;;
  om) BRANCH="-om" ;;
  *) echo "SEARCH must be plain or om"; exit 1 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.venvs/lc0-training/bin/python"
NAME=$(printf 'net%02d' "$GENERATION")$BRANCH${TAG:+-$TAG}
if [ -n "${START:-}" ]; then
  START="$(cd "$(dirname "$START")" && pwd)/$(basename "$START")"
elif [ "$GENERATION" -eq 1 ]; then
  START="$HERE/../lc0/bin/791556.pb.gz"
else
  START="$HERE/nets/$(printf 'net%02d' $((GENERATION - 1))).pb.gz"
  BRANCH_START="${START%.pb.gz}$BRANCH.pb.gz"
  if [ -n "$BRANCH" ] && [ -f "$BRANCH_START" ]; then START="$BRANCH_START"; fi
fi
RUN="$HERE/runs/$NAME"
export TF_CPP_MIN_LOG_LEVEL=2  # TensorFlow's start-up messages under WSL are noise

# A finished run says so in its log once the network is fully written.
finished() { grep -qs "Weights saved as '$RUN/$NAME.pb.gz'" "$RUN/train.log"; }
# Stops train.py with its data reader processes, which share its command line
# and would otherwise be left behind.
stop_training() { pkill -f "train.py --cfg $RUN/config.yaml"; }

mkdir -p "$RUN" "$HERE/nets"
if stop_training; then
  echo "== Stopped a training of $NAME left running"
  sleep 3
fi

if ! finished; then
  rm -rf "$RUN/data"
  mkdir -p "$RUN/data"
  for dir in "$HERE"/data/gen[0-9][0-9]; do
    if [ "$((10#${dir##*gen}))" -le "$GENERATION" ]; then
      if [ -n "$BRANCH" ] && [ -d "$dir$BRANCH" ]; then dir="$dir$BRANCH"; fi
      ln -s "$dir" "$RUN/data/$(basename "$dir")"
    fi
  done
  "$PYTHON" -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['name'] = sys.argv[2]
cfg['dataset']['input'] = sys.argv[8]
if sys.argv[9]:
    cfg['dataset']['num_chunks'] = int(sys.argv[9])
cfg['training'].update({
    'path': sys.argv[3], 'total_steps': int(sys.argv[4]), 'lr_values': [float(sys.argv[5])],
    'lr_boundaries': [], 'warmup_steps': min(50, int(sys.argv[4]) // 4),
    'test_steps': int(sys.argv[6]), 'checkpoint_steps': int(sys.argv[6]) * 2,
    'num_test_positions': 20000})
yaml.safe_dump(cfg, open(sys.argv[7], 'w'))
" "$HERE/configs/791556.yaml" "$NAME" "$HERE/runs" "$STEPS" "$LR" "$TEST_STEPS" "$RUN/config.yaml" \
    "$RUN/data/gen*/data-*/" "$WINDOW"
  echo "== Reading $(ls "$RUN/data" | tr '\n' ' ')${WINDOW:+(newest $WINDOW games)}"

  cd "$HERE/lczero-training/tf"
  if [ ! -f "$RUN/checkpoint" ]; then
    echo "== $(basename "$START") to a checkpoint"
    "$PYTHON" net_to_model.py --cfg "$RUN/config.yaml" "$START" > "$RUN/net_to_model.log" 2>&1 || {
      tail -30 "$RUN/net_to_model.log"; exit 1; }
  fi
  echo "== Training $NAME: $STEPS steps at learning rate $LR"
  touch "$RUN/train.log"
  # Unbuffered (-u), so the log, and the progress shown from it, keep up.
  "$PYTHON" -u train.py --cfg "$RUN/config.yaml" --output "$RUN/$NAME" >> "$RUN/train.log" 2>&1 &
  TRAIN=$!
  # A background job ignores Ctrl+C, so stopping this script stops it.
  trap 'stop_training; echo "Stopped; rerun to continue."; exit 130' INT TERM HUP
  tail -n 0 -f --pid=$TRAIN "$RUN/train.log" | grep --line-buffered -E "^step|^Using|Restoring|Model saved" &
  # lczero-training can hang shutting down its data readers after writing the
  # network: once it is written, train.py gets a minute to finish, then stops.
  while kill -0 $TRAIN 2> /dev/null; do
    if finished; then
      for _ in $(seq 60); do kill -0 $TRAIN 2> /dev/null || break; sleep 1; done
      stop_training || true
      break
    fi
    sleep 5
  done
  wait $TRAIN || true
  trap - INT TERM HUP
  if ! finished; then
    echo "Training failed: see $RUN/train.log"
    exit 1
  fi
fi

# The training config goes next to the network, first: generate_games.py reads
# from it whether the network needs Lc0's IdentityPlane.
cp "$RUN/config.yaml" "$HERE/nets/$NAME.yaml"
# Whole or not at all: the loop takes a network in nets/ as finished.
cp "$RUN/$NAME.pb.gz" "$HERE/nets/$NAME.pb.gz.part"
mv "$HERE/nets/$NAME.pb.gz.part" "$HERE/nets/$NAME.pb.gz"
echo "== Wrote training/nets/$NAME.pb.gz"
