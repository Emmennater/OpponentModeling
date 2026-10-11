"""Pilot: fine-tunes the main network's value head toward the opponent-model
search's values, and tests the result.

The main network judges a position as if both sides play perfectly. This
pilot checks whether its value head can learn the opponent-model search's
values instead, which include the modeled opponent's likely mistakes, and
whether that changes the engine's choices for the better. Three steps:

1. Targets: the opponent-model search (5,000 nodes; the opponent's rating,
   caution, lean and discount set by the options below) on positions from
   real games, saved as it goes, so an interrupted run resumes. Each target
   also keeps the root's statistics for every move (visits, prior, value),
   for training the policy head later.
2. Training: the value head's two fully connected layers, with the body and
   the value convolution frozen, on 80% of the positions. Reported on the
   other 20%, against the original head and against the original head
   rescaled to fit the targets, which is what the fine-tuned head has to beat
   to have learned something about positions rather than an offset. The
   result is written into a copy of the network in Lc0's own format.
3. Tests: the original and the fine-tuned network, on the Belgrade Gambit
   with equal search for 5.Nxd4 and 5.Nd5, opening choices, two Halloween
   Gambit positions and a mate in 2, with the settings in TEST_OPTIONS and
   again with the targets' own settings.

Files go to lc0/finetune/<tag>/ (ignored by git); the positions are shared,
in lc0/finetune/positions.json. Needs the repository's .venv
(PyTorch, onnx, onnxruntime, python-chess, grpcio-tools, tqdm), and Lc0 built
in lc0-src/build with the network in lc0/bin and Maia-3 in lc0/nets.

Step 1 saves its targets as it goes and resumes where it stopped; steps 2 and
3 always run (seconds and minutes). A summary is saved to
lc0/finetune/<tag>/summary.txt.

Usage (the first pilot, 2,250 targets plus the tests, took 10-15 minutes):
    .venv\\Scripts\\python tools\\finetune_value_head.py --tag pilot --rating 1700 --caution 0.15
    .venv\\Scripts\\python tools\\finetune_value_head.py --tag cautious --rating 1900 --caution 0.5
"""

import argparse

import gzip
import json
import os
import random
import re
import subprocess
import sys
import winreg
from pathlib import Path

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SHARED = ROOT / "lc0" / "finetune"
WORK = SHARED  # lc0/finetune/<tag>, set in main()
LC0 = ROOT / "lc0-src" / "build" / "lc0.exe"
NET = ROOT / "lc0" / "bin" / "791556.pb.gz"
MAIA = ROOT / "lc0" / "nets" / "maia3-23m-fp16.onnx"
NET_ONNX = SHARED / "791556.onnx"

TARGET_NODES = 5000
TARGET_OPTIONS = {}  # set in main()
TEST_OPTIONS = {"OpponentModelRating": 1500, "OpponentModelOwnRating": 1500,
                "OpponentCaution": 0.15, "OpponentDiscount": 0.995, "OpponentBestLean": 1.0}

BELGRADE = "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 d2d4 e5d4"
HALLOWEEN_8 = "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f3e5 c6e5 d2d4 e5g6 e4e5 f6g8 d1f3 d7d5"
HALLOWEEN_15 = HALLOWEEN_8 + " c3d5 c7c6 d5c3 d8d4 c1e3 d4e5 f1b5 c6b5 e1c1 g8f6 h1e1 f8e7 e3h6 e5f5"
MATE_IN_2 = ("e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 d2d4 e5d4 c3d5 f6e4 d1e2 f7f5 f3g5 f8e7 h2h4 e8g8 e2h5 "
             "h7h6 f1c4 g8h8 h5g6 e7g5 h4g5 e4g5 c1g5 d8e8 d5e7 e8g6 e7g6 h8h7 g6f8 h7h8")
ROOT_TESTS = [  # name, moves, nodes, what was expected before
    ("Start", "", 300000, "1.e4"),
    ("1.e4 e5", "e2e4 e7e5", 300000, "2.d4"),
    ("1.e4 e5 2.Nf3 Nc6", "e2e4 e7e5 g1f3 b8c6", 300000, "3.Nc3"),
    ("1.e4 c5", "e2e4 c7c5", 300000, "2.d4"),
    ("Belgrade, move 5", BELGRADE, 100000, "5.Nd5"),
    ("Halloween, move 8", HALLOWEEN_8, 300000, "8.Qxd5 (not 8.Nxd5)"),
    ("Halloween, move 15", HALLOWEEN_15, 300000, "15.Bg7 or 15.Qxf5 (not 15.Rxe7+)"),
    ("Mate in 2", MATE_IN_2, 100000, "17.Rxh6+"),
]


# ------------------------------------------------------------------ Lc0


def cuda_path():
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
        return winreg.QueryValueEx(key, "CUDA_PATH")[0]


class Lc0:
    """An Lc0 process with the opponent model."""

    NODE_LINE = re.compile(r"info string node .*\(WL: +(-?[0-9.]+)\) \(D: +([0-9.]+)\)")
    MOVE_LINE = re.compile(
        r"info string (\S+)\s+\(\s*\d+\s*\) N: +(\d+).*\(P: +([0-9.]+)%\).*\(Q: +(-?[0-9.]+)\)")

    def __init__(self, weights, options):
        cuda = cuda_path()
        env = dict(os.environ, PATH=f"{cuda}\\bin;{cuda}\\bin\\x64;{os.environ['PATH']}")
        self.proc = subprocess.Popen([str(LC0)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        settings = {"WeightsFile": weights, "OpponentModelWeights": MAIA, "VerboseMoveStats": "true", **options}
        for name, value in settings.items():
            self.send(f"setoption name {name} value {value}")

    def send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def search(self, moves, nodes, searchmoves=""):
        """Returns the root's (WL, D), its children's {move: (visits, prior, Q)}
        and the best move."""
        self.send("position startpos" + (f" moves {moves}" if moves else ""))
        self.send(f"go{' searchmoves ' + searchmoves if searchmoves else ''} nodes {nodes}")
        root, children, best = None, {}, None
        for line in iter(self.proc.stdout.readline, ""):
            if m := self.NODE_LINE.search(line):
                root = (float(m.group(1)), float(m.group(2)))
            elif (m := self.MOVE_LINE.search(line)) and m.group(1) != "node":
                children[m.group(1)] = (int(m.group(2)), float(m.group(3)) / 100, float(m.group(4)))
            if line.startswith("bestmove"):
                best = line.split()[1]
                break
        return root, children, best

    def close(self):
        self.send("quit")
        self.proc.wait()


# ------------------------------------------------------------------ Step 1


def generate_targets():
    positions = json.loads((SHARED / "positions.json").read_text())
    path = WORK / "targets.json"
    targets = json.loads(path.read_text()) if path.exists() else []
    if len(targets) >= len(positions):
        print(f"Step 1: {len(targets)} targets already generated")
        return targets
    print(f"Step 1: targets, {len(targets)} of {len(positions)} done")
    lc0 = Lc0(NET, TARGET_OPTIONS)
    for i in tqdm(range(len(targets), len(positions)), initial=len(targets), total=len(positions),
                  desc="Targets", unit="pos"):
        root, children, _ = lc0.search(" ".join(positions[i]["moves"]), TARGET_NODES)
        if root is not None:
            targets.append({**positions[i], "wl": root[0], "d": root[1], "root": children})
        if len(targets) % 50 == 0:
            path.write_text(json.dumps(targets))
    lc0.close()
    path.write_text(json.dumps(targets))
    return targets


# ------------------------------------------------------------------ Step 2

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
FEATURES = "/value/reshape"  # the value convolution's output, flattened: 32 x 64


def encode(moves):
    """Lc0's INPUT_CLASSICAL_112_PLANE encoding of the position after `moves`
    (UCI) from the start position: 8 history positions of 13 planes (our 6
    pieces, their 6, repetition), newest first, from the side to move's view
    (ranks mirrored for Black), then castling, side to move, the 50-move
    counter, zeros and ones. Checked against Lc0's own evaluations."""
    board = chess.Board()
    boards = [board.copy()]
    for m in moves:
        board.push_uci(m)
        boards.append(board.copy())
    current = boards[-1]
    us = current.turn
    flip = us == chess.BLACK
    planes = np.zeros((112, 64), np.float32)
    for i in range(8):
        idx = len(boards) - 1 - i
        if idx < 0:
            break
        b = boards[idx]
        for p, piece in enumerate(PIECES):
            for side, offset in ((us, 0), (not us, 6)):
                for sq in chess.SquareSet(b.pieces_mask(piece, side)):
                    planes[i * 13 + offset + p, (sq ^ 56) if flip else sq] = 1
        key = b._transposition_key()
        if any(boards[j]._transposition_key() == key for j in range(idx)):
            planes[i * 13 + 12, :] = 1
    planes[104, :] = current.has_queenside_castling_rights(us)
    planes[105, :] = current.has_kingside_castling_rights(us)
    planes[106, :] = current.has_queenside_castling_rights(not us)
    planes[107, :] = current.has_kingside_castling_rights(not us)
    planes[108, :] = flip
    planes[109, :] = current.halfmove_clock
    planes[111, :] = 1
    return planes.reshape(112, 8, 8)


class Head(torch.nn.Module):
    """The value head's two fully connected layers, from the network's own."""

    def __init__(self, init):
        super().__init__()
        self.fc1 = torch.nn.Linear(2048, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(init["/value/dense1/matmul/w"].T))
            self.fc1.bias.copy_(torch.tensor(init["/value/dense1/add/w"]))
            self.fc2.weight.copy_(torch.tensor(init["/value/dense2/matmul/w"].T))
            self.fc2.bias.copy_(torch.tensor(init["/value/dense2/add/w"]))

    def forward(self, x):
        return self.fc2(torch.nn.functional.mish(self.fc1(x)))


def net_proto():
    """Python bindings for Lc0's weights format, generated once."""
    out = SHARED / "proto"
    if not (out / "proto" / "net_pb2.py").exists():
        from grpc_tools import protoc
        out.mkdir(exist_ok=True)
        protoc.main(["protoc", f"-I{ROOT / 'lc0-src' / 'libs' / 'lczero-common'}",
                     f"--python_out={out}", "proto/net.proto"])
    sys.path.insert(0, str(out))
    from proto import net_pb2
    return net_pb2


def write_layer(layer, values):
    """Lc0 stores a layer as 16-bit values between a minimum and a maximum."""
    lo, hi = float(values.min()), float(values.max())
    q = np.round((values - lo) / (hi - lo) * 65535.0).clip(0, 65535).astype(np.uint16)
    layer.min_val, layer.max_val, layer.params = lo, hi, q.tobytes()


def train(targets):
    print(f"\nStep 2: training on {len(targets)} positions")
    if not NET_ONNX.exists():
        cuda = cuda_path()
        env = dict(os.environ, PATH=f"{cuda}\\bin;{cuda}\\bin\\x64;{os.environ['PATH']}")
        subprocess.run([str(LC0), "leela2onnx", f"--input={NET}", f"--output={NET_ONNX}"],
                       check=True, capture_output=True, env=env)
    targets = list(targets)
    random.Random(1).shuffle(targets)
    model = onnx.load(str(NET_ONNX))
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    model.graph.output.append(onnx.helper.make_tensor_value_info(FEATURES, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feats, orig = [], []
    for i in tqdm(range(0, len(targets), 256), desc="Features", unit="batch"):
        x = np.stack([encode(t["moves"]) for t in targets[i:i + 256]])
        wdl, f = sess.run(["/output/wdl", FEATURES], {"/input/planes": x})
        feats.append(f)
        orig.append(wdl)
    X = torch.tensor(np.concatenate(feats))
    orig = np.concatenate(orig)
    wl = np.array([t["wl"] for t in targets])
    d = np.array([t["d"] for t in targets])
    Y = torch.tensor(np.clip(np.stack([(1 + wl - d) / 2, d, (1 - wl - d) / 2], 1), 0, 1), dtype=torch.float32)
    Y = Y / Y.sum(1, keepdim=True)
    n_train = int(0.8 * len(targets))
    test = np.arange(n_train, len(targets))

    head = Head(init)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in tqdm(range(200), desc="Training", unit="epoch"):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            b = perm[i:i + 64]
            loss = -(Y[b] * torch.log_softmax(head(X[b]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            test_loss = -(Y[n_train:] * torch.log_softmax(head(X[n_train:]), 1)).sum(1).mean().item()
        if test_loss < best:
            best, best_state, best_epoch = test_loss, {k: v.clone() for k, v in head.state_dict().items()}, epoch
    head.load_state_dict(best_state)
    with torch.no_grad():
        tuned = torch.softmax(head(X), 1).numpy()

    v0 = orig[:, 0] - orig[:, 2]
    a, b = np.polyfit(v0[:n_train], wl[:n_train], 1)
    rows = [("Original head", v0), ("Original, rescaled to fit", np.clip(a * v0 + b, -1, 1)),
            (f"Fine-tuned (epoch {best_epoch})", tuned[:, 0] - tuned[:, 2])]
    summary = [f"Held-out positions ({len(test)}): predicting the search's value (W - L)"]
    for name, v in rows:
        err = v[test] - wl[test]
        summary.append(f"  {name:<28} correlation {np.corrcoef(v[test], wl[test])[0, 1]:.3f}"
                       f"   mean abs error {np.abs(err).mean():.3f}   mean error {err.mean():+.3f}")

    net_pb2 = net_proto()
    net = net_pb2.Net()
    net.ParseFromString(gzip.open(NET).read())
    w = net.weights
    state = {k: v.numpy().astype(np.float64) for k, v in head.state_dict().items()}
    write_layer(w.ip1_val_w, state["fc1.weight"].ravel())
    write_layer(w.ip1_val_b, state["fc1.bias"])
    write_layer(w.ip2_val_w, state["fc2.weight"].ravel())
    write_layer(w.ip2_val_b, state["fc2.bias"])
    tuned_net = WORK / "791556-om.pb.gz"
    tuned_net.write_bytes(gzip.compress(net.SerializeToString()))
    summary.append(f"Wrote {tuned_net}")
    print("\n".join(summary))
    return summary


# ------------------------------------------------------------------ Step 3


def san(moves, uci):
    board = chess.Board()
    for m in moves.split():
        board.push_uci(m)
    try:
        return board.san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return uci


def tests(options, label):
    print(f"\nStep 3: tests with {label}, original against fine-tuned network")
    jobs = [(name, n) for name, n in (("Original", NET), ("Fine-tuned", WORK / "791556-om.pb.gz"))]
    total = len(jobs) * (6 + len(ROOT_TESTS))
    results = {}
    with tqdm(total=total, desc="Tests", unit="search") as bar:
        for name, weights in jobs:
            lc0 = Lc0(weights, options)
            gaps = []
            for nodes in (20000, 80000, 300000):
                q = {}
                for move in ("f3d4", "c3d5"):
                    _, children, _ = lc0.search(BELGRADE, nodes, searchmoves=move)
                    q[move] = children.get(move, (0, 0, float("nan")))[2]
                    bar.update()
                gaps.append(q["c3d5"] - q["f3d4"])
            choices = []
            for _, moves, nodes, _ in ROOT_TESTS:
                _, _, best = lc0.search(moves, nodes)
                choices.append(san(moves, best) if best else "?")
                bar.update()
            lc0.close()
            results[name] = (gaps, choices)
    summary = [f"Tests with {label}: {options}",
               "Belgrade Gambit, 5.Nd5 - 5.Nxd4 with equal search (about 9k, 34k, 129k visits each):"]
    for name, (gaps, _) in results.items():
        summary.append(f"  {name:<11} " + "   ".join(f"{g:+.3f}" for g in gaps))
    summary.append("Choices:")
    summary.append(f"  {'Position':<20} {'Original':<12} {'Fine-tuned':<12} Before")
    for i, (pos, _, _, before) in enumerate(ROOT_TESTS):
        summary.append(f"  {pos:<20} {results['Original'][1][i]:<12} {results['Fine-tuned'][1][i]:<12} {before}")
    print("\n".join(summary))
    return summary


def main():
    global WORK, TARGET_OPTIONS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="pilot", help="experiment name: files go to lc0/finetune/<tag>")
    parser.add_argument("--rating", type=int, default=1700, help="the targets' opponent and own rating")
    parser.add_argument("--caution", type=float, default=0.15, help="the targets' OpponentCaution")
    parser.add_argument("--lean", type=float, default=0.0, help="the targets' OpponentBestLean")
    parser.add_argument("--discount", type=float, default=1.0, help="the targets' OpponentDiscount")
    args = parser.parse_args()
    WORK = SHARED / args.tag
    WORK.mkdir(parents=True, exist_ok=True)
    TARGET_OPTIONS = {"OpponentModelRating": args.rating, "OpponentModelOwnRating": args.rating,
                      "OpponentCaution": args.caution, "OpponentDiscount": args.discount,
                      "OpponentBestLean": args.lean}
    print(f"Experiment '{args.tag}': targets with {TARGET_OPTIONS}")
    targets = generate_targets()
    summary = (train(targets) + [""] + tests(TEST_OPTIONS, "your settings") + [""]
               + tests(TARGET_OPTIONS, "the targets' settings"))
    (WORK / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nSummary saved to {WORK / 'summary.txt'}")


if __name__ == "__main__":
    main()
