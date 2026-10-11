"""Pilot: fine-tunes the main network's policy head toward the moves the
opponent-model search prefers, and tests the result.

The main network's move probabilities (its policy) decide which of this
engine's moves the search tries first, and which moves become candidates at
the root. They were trained for play against a perfect opponent, so they may
favor safe moves over moves that set problems for a human, which the search
then finds only with many nodes. This pilot trains the policy toward the
opponent-model search's own preferences. Three steps:

1. Targets: the opponent-model search (5,000 nodes) with the playing
   settings below, on positions from real games, keeping the root's visits,
   prior and value for every move. Saved as it goes and resumed when rerun.
2. Training: the policy head's projection, queries and keys (the promotion
   weights, the body and the value head stay frozen), on 80% of the
   positions, toward half the original policy and half the move the search
   chose: the best-valued of its most visited moves. (The root's visits
   themselves are no target: sequential halving spreads them evenly over the
   candidates, which taught a flatter policy that put less probability on
   the search's choice.) Reported on the other 20%:
   how much probability the head puts on the search's choice, and how often
   it is the head's own top move, against the original head. The
   result is written into a copy of the network in Lc0's own format, and the
   priors of a few telling moves are compared through Lc0 itself.
3. Tests: the original and the fine-tuned network with the playing settings
   (see finetune_value_head.py).

Files go to lc0/finetune/<tag>/ (ignored by git). Needs what
finetune_value_head.py needs.

Usage (about 15 minutes for the targets, 1 for training, 5 for the tests):
    .venv\\Scripts\\python tools\\finetune_policy_head.py
"""

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_value_head as fv  # noqa: E402

BODY = "/block14/conv2/mish"  # the body's output, the policy head's input: 192 x 8 x 8
CHOICE_SHARE = 0.5  # the target's share on the search's choice; the rest is the original policy
PLAYING = {"OpponentModelRating": 1500, "OpponentModelOwnRating": 1500,
           "OpponentCaution": 0.15, "OpponentDiscount": 0.995, "OpponentBestLean": 1.0}
PRIOR_CHECKS = [  # name, moves, move whose prior is shown
    ("Start: 1.e4", "", "e2e4"),
    ("Start: 1.d4", "", "d2d4"),
    ("Belgrade: 5.Nd5", fv.BELGRADE, "c3d5"),
    ("Belgrade: 5.Nxd4", fv.BELGRADE, "f3d4"),
    ("Mate in 2: 17.Rxh6+", fv.MATE_IN_2, "h1h6"),
]


def grid_index(board, uci):
    """The move's cell in the policy's 64 x 64 grid (from x 64 + to), in the
    side to move's view, or None for a promotion. Castling is the king taking
    its own rook, as in Lc0."""
    move = chess.Move.from_uci(uci)
    if move.promotion:
        return None
    f, t = move.from_square, move.to_square
    if board.is_castling(move) and abs(chess.square_file(f) - chess.square_file(t)) == 2:
        t = chess.square(7 if chess.square_file(t) == 6 else 0, chess.square_rank(f))
    if board.turn == chess.BLACK:
        f, t = f ^ 56, t ^ 56
    return f * 64 + t


class PolicyHead(torch.nn.Module):
    """The policy head's projection, queries and keys, from the network's own."""

    def __init__(self, init):
        super().__init__()
        self.scale = float(init["/policy/scale/w"][0])
        self.layers = torch.nn.ModuleDict()
        for name, onnx_name in (("dense1", "/policy/dense1"), ("q", "/policy/Q"), ("k", "/policy/K")):
            layer = torch.nn.Linear(192, 192)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(init[f"{onnx_name}/matmul/w"].T))
                layer.bias.copy_(torch.tensor(init[f"{onnx_name}/add/w"]))
            self.layers[name] = layer

    def forward(self, body):
        x = body.float().permute(0, 2, 3, 1).reshape(-1, 64, 192)
        h = torch.nn.functional.selu(self.layers["dense1"](x))
        q, k = self.layers["q"](h), self.layers["k"](h)
        return (q @ k.transpose(1, 2)).reshape(-1, 4096) * self.scale


def train(targets):
    print(f"\nStep 2: training the policy head on {len(targets)} positions")
    targets = [t for t in targets if t.get("root")]
    random.Random(1).shuffle(targets)
    model = onnx.load(str(fv.NET_ONNX))
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    model.graph.output.append(onnx.helper.make_tensor_value_info(BODY, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    bodies, masks, dists = [], [], []
    for i in tqdm(range(0, len(targets), 256), desc="Features", unit="batch"):
        batch = targets[i:i + 256]
        bodies.append(sess.run([BODY], {"/input/planes": np.stack([fv.encode(t["moves"]) for t in batch])})[0]
                      .astype(np.float16))
        for t in batch:
            board = chess.Board()
            for m in t["moves"]:
                board.push_uci(m)
            mask = np.zeros(4096, bool)
            for move in board.legal_moves:
                cell = grid_index(board, move.uci())
                if cell is not None:
                    mask[cell] = True
            # The search's choice: the best-valued of its most visited moves
            stats = [(uci, n, q) for uci, (n, _, q) in t["root"].items() if len(uci) == 4]
            most = max((n for _, n, _ in stats), default=0)
            finalists = [(q, uci) for uci, n, q in stats if most > 0 and n >= 0.9 * most]
            choice = np.zeros(4096, np.float32)
            if finalists:
                cell = grid_index(board, max(finalists)[1])
                if cell is not None and mask[cell]:
                    choice[cell] = 1.0
            masks.append(mask)
            dists.append(choice)
    bodies = torch.tensor(np.concatenate(bodies))
    masks = torch.tensor(np.stack(masks))
    dists = torch.tensor(np.stack(dists))
    keep = dists.sum(1) > 0
    bodies, masks, dists = bodies[keep], masks[keep], dists[keep]
    n_train = int(0.8 * len(bodies))

    head = PolicyHead(init)
    # The target: half the original policy, half the search's choice
    with torch.no_grad():
        original = torch.cat([torch.softmax(head(bodies[i:i + 256]).masked_fill(~masks[i:i + 256], float("-inf")), 1)
                              for i in range(0, len(bodies), 256)])
    choices = dists.clone()
    dists = (1 - CHOICE_SHARE) * original + CHOICE_SHARE * choices

    def log_probs(idx):
        logits = head(bodies[idx]).masked_fill(~masks[idx], float("-inf"))
        return torch.log_softmax(logits, 1).masked_fill(~masks[idx], 0.0)

    def held_out(name):
        with torch.no_grad():
            lp = log_probs(torch.arange(n_train, len(bodies)))
            target = dists[n_train:]
            best = choices[n_train:].argmax(1)
            p_best = lp.exp().gather(1, best[:, None]).mean().item()
            top1 = (lp.masked_fill(~masks[n_train:], float("-inf")).argmax(1) == best).float().mean().item()
            ce = -(target * lp).sum(1).mean().item()
        return (f"  {name:<24} cross-entropy {ce:.3f}   probability on the search's choice {p_best:.3f}"
                f"   the search's choice is the top move {100 * top1:.1f}%")

    summary = [f"Held-out positions ({len(bodies) - n_train}): the policy against the search's choices",
               held_out("Original head")]
    opt = torch.optim.Adam(head.parameters(), lr=1e-4, weight_decay=1e-4)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in tqdm(range(100), desc="Training", unit="epoch"):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            b = perm[i:i + 64]
            loss = -(dists[b] * log_probs(b)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            test_loss = -(dists[n_train:] * log_probs(torch.arange(n_train, len(bodies)))).sum(1).mean().item()
        if test_loss < best_loss:
            best_loss, best_epoch = test_loss, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    summary.append(held_out(f"Fine-tuned (epoch {best_epoch})"))

    net_pb2 = fv.net_proto()
    net = net_pb2.Net()
    net.ParseFromString(gzip.open(fv.NET).read())
    w = net.weights
    state = {k: v.numpy().astype(np.float64) for k, v in head.state_dict().items()}
    for proto_name, layer in (("ip_pol", "dense1"), ("ip2_pol", "q"), ("ip3_pol", "k")):
        fv.write_layer(getattr(w, f"{proto_name}_w"), state[f"layers.{layer}.weight"].ravel())
        fv.write_layer(getattr(w, f"{proto_name}_b"), state[f"layers.{layer}.bias"])
    tuned = fv.WORK / "791556-om.pb.gz"
    tuned.write_bytes(gzip.compress(net.SerializeToString()))
    summary.append(f"Wrote {tuned}")
    print("\n".join(summary))
    return summary


def prior_checks():
    """Priors of a few telling moves, through Lc0 itself."""
    summary = ["Priors (Lc0, one node):", f"  {'Move':<24} {'Original':>9} {'Fine-tuned':>11}"]
    priors = {}
    for name, weights in (("Original", fv.NET), ("Fine-tuned", fv.WORK / "791556-om.pb.gz")):
        lc0 = fv.Lc0(weights, {"OpponentModelWeights": ""})
        for label, moves, move in PRIOR_CHECKS:
            _, children, _ = lc0.search(moves, 1)
            priors[(name, label)] = children.get(move, (0, float("nan"), 0))[1]
        lc0.close()
    for label, _, _ in PRIOR_CHECKS:
        summary.append(f"  {label:<24} {100 * priors[('Original', label)]:>8.1f}% "
                       f"{100 * priors[('Fine-tuned', label)]:>10.1f}%")
    print("\n".join(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="policy", help="experiment name: files go to lc0/finetune/<tag>")
    parser.add_argument("--rating", type=int, default=PLAYING["OpponentModelRating"])
    parser.add_argument("--caution", type=float, default=PLAYING["OpponentCaution"])
    parser.add_argument("--lean", type=float, default=PLAYING["OpponentBestLean"])
    parser.add_argument("--discount", type=float, default=PLAYING["OpponentDiscount"])
    args = parser.parse_args()
    fv.WORK = fv.SHARED / args.tag
    fv.WORK.mkdir(parents=True, exist_ok=True)
    fv.TARGET_OPTIONS = {"OpponentModelRating": args.rating, "OpponentModelOwnRating": args.rating,
                         "OpponentCaution": args.caution, "OpponentDiscount": args.discount,
                         "OpponentBestLean": args.lean}
    print(f"Experiment '{args.tag}': targets with {fv.TARGET_OPTIONS}")
    targets = fv.generate_targets()
    summary = train(targets) + [""] + prior_checks() + [""] + fv.tests(fv.TEST_OPTIONS, "your settings")
    (fv.WORK / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nSummary saved to {fv.WORK / 'summary.txt'}")


if __name__ == "__main__":
    main()
