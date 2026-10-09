"""Exports a Maia-3 checkpoint to ONNX, for Lc0's opponent model.

The exported model takes:
  tokens:    float32 (batch, 64, 12 * history) -- one row per square, 12
             one-hot piece planes per position in the history window, oldest
             first, each position in its own side-to-move frame
  self_elo:  float32 (batch,) -- rating of the player to move
  oppo_elo:  float32 (batch,) -- rating of their opponent
and returns:
  policy:    float32 (batch, 4352) move logits, in the side-to-move frame:
             from * 64 + to for moves (squares a1 = 0 ... h8 = 63), then
             4096 + (from_file * 8 + to_file) * 4 + piece for promotions from
             rank 7 to rank 8, pieces ordered q, r, b, n
  wdl:       float32 (batch, 3) value logits, ordered loss, draw, win for the
             player to move (Maia-3's own order)

The model is checked against PyTorch on real positions, encoded by
StockfishTrapFinder's Maia3Net, which is the reference for the encoding.

Usage (from the project's .venv, which has CPU PyTorch, onnx and maia3):
    python tools/export_maia3.py maia3-23m lc0/nets/maia3-23m.onnx
"""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAIA3_PATH = ROOT.parent / "StockfishTrapFinder"


def load_model(name):
    from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
    from maia3.models import MAIA3Model

    spec = resolve_model_spec(name)
    checkpoint = resolve_checkpoint_path(spec)
    cfg = SimpleNamespace(**spec.config, device="cpu", checkpoint_path=checkpoint)
    if cfg.include_time_info:
        sys.exit(f"{name} uses time inputs, which the Lc0 opponent model does not provide")

    model = MAIA3Model(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # Older checkpoints call the guided-attention block "smolgen"
    state = {k.replace("smolgen", "gab"): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"checkpoint does not match the architecture: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval()
    return model, cfg


def original_outputs(model, tokens, self_elo, oppo_elo):
    """The policy and value logits from Maia-3's own forward pass."""
    # The model slices its last input column off (a training label)
    label = torch.zeros(tokens.shape[0], 64, 1, dtype=tokens.dtype)
    policy, wdl, _ = model(torch.cat([tokens, label], dim=-1), self_elo, oppo_elo)
    return policy, wdl


class Exported(torch.nn.Module):
    """Maia-3 with float ratings and only the outputs the opponent model uses.

    The forward pass is Maia-3's own (MAIA3Model.forward), except that the 256
    promotion logits are computed in one operation instead of a Python loop,
    which exports as some 770 tiny operations that are slow to launch on a GPU
    and a 257-input Concat that CUDA graphs cannot capture. main() checks the
    result against the original.

    Inputs and outputs are float32 either way. With `dtype` float16 the model
    itself runs in half precision, which GPUs compute much faster.
    """

    def __init__(self, model, dtype=torch.float32):
        super().__init__()
        self.model = model.to(dtype)
        self.dtype = dtype

    def forward(self, tokens, self_elo, oppo_elo):
        m = self.model
        tokens = tokens.to(self.dtype)
        self_embs = m.interpolate_elo(self_elo.to(self.dtype)).unsqueeze(1).expand(-1, 64, -1)
        oppo_embs = m.interpolate_elo(oppo_elo.to(self.dtype)).unsqueeze(1).expand(-1, 64, -1)
        x = m.token_projection(torch.cat([tokens, self_embs, oppo_embs], dim=-1))
        if hasattr(m, "abs_pe"):
            x = m.abs_pe(x)
        x = m.transformer(x)

        sq_from = m.proj_sq_from(x)
        sq_to = m.proj_sq_to(x)
        scores = torch.einsum("bid,bjd->bij", sq_from, sq_to) / math.sqrt(m.cfg.head_hid_dim)

        # Promotions from rank 7 (squares 48-55) to rank 8 (56-63): the move's
        # score plus a bias for the piece, ordered from file, to file, piece
        promo_biases = m.promo_bias_proj(sq_to[:, 56:64, :]) * math.sqrt(m.cfg.head_hid_dim)  # (B, 8, 4)
        promotions = scores[:, 48:56, 56:64].unsqueeze(-1) + promo_biases.unsqueeze(1)  # (B, 8, 8, 4)
        policy = torch.cat([scores.reshape(-1, 64 * 64), promotions.reshape(-1, 256)], dim=1)

        x = m.last_ln(x.mean(dim=1))
        wdl = m.fc_value(torch.nn.functional.relu(m.fc_value_hid(x)))
        return policy.float(), wdl.float()


def sample_boards(games_file, count):
    import chess

    boards = []
    with open(games_file, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            board = chess.Board()
            for san in fields[3].split():
                try:
                    board.push_san(san.rstrip("?!"))
                except ValueError:
                    break
                if any(board.legal_moves):
                    boards.append(board.copy(stack=8))
            if len(boards) >= count:
                break
    return boards[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Maia-3 checkpoint name, e.g. maia3-23m")
    parser.add_argument("output", help="ONNX file to write")
    parser.add_argument("--games", default=str(ROOT / "lichess_games.tsv"), help="games for the check")
    parser.add_argument("--maia3-path", default=str(DEFAULT_MAIA3_PATH), help="folder containing maia3_net.py")
    parser.add_argument("--fp16", action="store_true", help="run the model in half precision (inputs and outputs stay float32)")
    args = parser.parse_args()

    model, cfg = load_model(args.model)
    exported = Exported(model, torch.float16 if args.fp16 else torch.float32)

    # The TorchScript exporter has no rms_norm before opset 23, so it is
    # written out in elementary operations, which give the same result.
    def rms_norm(x, normalized_shape, weight=None, eps=None):
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        dims = tuple(range(-len(normalized_shape), 0))
        x = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + eps)
        return x * weight if weight is not None else x

    torch.nn.functional.rms_norm = rms_norm
    width = 12 * cfg.history

    tokens = torch.zeros(2, 64, width)
    elos = torch.tensor([1300.0, 1500.0])
    torch.onnx.export(
        exported,
        (tokens, elos, elos),
        args.output,
        input_names=["tokens", "self_elo", "oppo_elo"],
        output_names=["policy", "wdl"],
        dynamic_axes={"tokens": {0: "batch"}, "self_elo": {0: "batch"}, "oppo_elo": {0: "batch"},
                      "policy": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"Wrote {args.output} (history {cfg.history}, input width {width})")

    # Check against PyTorch on real positions, encoded by the reference wrapper
    import onnxruntime as ort

    sys.path.insert(0, args.maia3_path)
    from maia3_net import Maia3Net

    reference = Maia3Net(args.model, device="cpu", use_amp=False, self_elo=1300, oppo_elo=1700)
    boards = sample_boards(args.games, 256)
    batch = np.stack([reference._tokens_for(b)[:, :width] for b in boards]).astype(np.float32)
    self_elo = np.full(len(boards), 1300, np.float32)
    oppo_elo = np.full(len(boards), 1700, np.float32)

    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    policy, wdl = session.run(None, {"tokens": batch, "self_elo": self_elo, "oppo_elo": oppo_elo})

    with torch.no_grad():
        ref_policy, ref_wdl = original_outputs(load_model(args.model)[0], torch.from_numpy(batch),
                                               torch.from_numpy(self_elo), torch.from_numpy(oppo_elo))
    print(f"Largest difference from full-precision PyTorch: policy {np.abs(policy - ref_policy.numpy()).max():.2e}, "
          f"wdl {np.abs(wdl - ref_wdl.numpy()).max():.2e}")

    # And the legal-move probabilities against the reference wrapper's
    ref_probs, _ = reference.run(boards)
    worst, top_agree = 0.0, 0
    for i, board in enumerate(boards):
        moves, indices = reference._legal_indices(board)
        logits = policy[i, indices]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        worst = max(worst, max(abs(p - ref_probs[i][m]) for m, p in zip(moves, probs)))
        top_agree += moves[int(np.argmax(probs))] == max(ref_probs[i], key=ref_probs[i].get)
    print(f"Largest difference in legal-move probabilities from Maia3Net: {worst:.2e}; "
          f"top move agrees in {top_agree} of {len(boards)} positions")


if __name__ == "__main__":
    main()
