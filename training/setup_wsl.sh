#!/usr/bin/env bash
# Sets up Lc0's own training pipeline (lczero-training's legacy TensorFlow
# code in tf/, the only one that trains residual networks like 791556) in WSL.
#
# - lczero-training is cloned into training/lczero-training (ignored by git),
#   at the commit below, with its libs/lc0 submodule for net.proto, and
#   lczero-training.patch applied.
# - The Python environment goes to ~/.venvs/lc0-training, on WSL's own disk:
#   TensorFlow is thousands of files, slow to import from /mnt/c.
# - TensorFlow 2.15 is the last version with Keras 2, which tf/ is written
#   for (2.16 moved to Keras 3). It needs Python 3.11 or older, which uv
#   downloads, since Ubuntu 24.04 has 3.12 and installing another needs sudo.
# - net.proto is compiled with grpcio-tools, a protoc that comes with pip.
#
# Run again at any time: each step skips what is already done.
#
# Usage, from Windows:
#     wsl -d Ubuntu -- bash training/setup_wsl.sh

set -euo pipefail

COMMIT=b00588cca13f5cd837156624b193827f68e45aa8  # master, 2026-10-13
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/lczero-training"
VENV="$HOME/.venvs/lc0-training"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv > /dev/null; then
  echo "== Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d "$REPO/.git" ]; then
  echo "== Cloning lczero-training"
  git clone https://github.com/LeelaChessZero/lczero-training.git "$REPO"
fi
echo "== lczero-training at $COMMIT"
git -C "$REPO" fetch --quiet origin "$COMMIT" 2> /dev/null || true
git -C "$REPO" checkout --quiet "$COMMIT"
git -C "$REPO" submodule update --init --depth 1 libs/lc0
# Opponent modeling: the data loader's value_discount and skip_unsearched
# options (see lczero-training.patch)
if git -C "$REPO" apply --reverse --check "$HERE/lczero-training.patch" 2> /dev/null; then
  echo "== lczero-training.patch already applied"
else
  echo "== Applying lczero-training.patch"
  git -C "$REPO" apply "$HERE/lczero-training.patch"
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "== Creating $VENV (Python 3.11)"
  uv venv --python 3.11 "$VENV"
fi
echo "== Installing packages"
VIRTUAL_ENV="$VENV" uv pip install \
  "tensorflow[and-cuda]==2.15.1" "numpy<2" pyyaml "protobuf<5" "grpcio-tools==1.62.3"

echo "== Compiling net.proto"
cd "$REPO"
"$VENV/bin/python" -m grpc_tools.protoc --proto_path=libs/lc0 --python_out=tf proto/net.proto
touch tf/proto/__init__.py

echo "== Checking TensorFlow and the GPU"
"$VENV/bin/python" -c "
import tensorflow as tf
print('TensorFlow', tf.__version__, 'mish:', hasattr(tf.keras.activations, 'mish'))
print('GPUs:', tf.config.list_physical_devices('GPU'))
"
