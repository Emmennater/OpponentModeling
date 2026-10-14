"""Compares two Lc0 networks: their weights, layer by layer, and what Lc0
makes of them, on a few opening positions.

Used to check that lczero-training's round trip (net_to_model.py, then
model_to_net.py, see roundtrip.sh) gives back the network it was given. The
weights are stored as 16-bit values between each layer's minimum and maximum
(LINEAR16), so a round trip can differ by about one step of that scale; the
report gives the largest difference in each layer, in steps. Lc0 then
evaluates each position with one node and each network: the root's value
(WL, D) and every move's prior should agree to the printed precision.

Usage (from Windows, with the repository's .venv):
    .venv\\Scripts\\python training\\compare_nets.py lc0\\bin\\791556.pb.gz training\\runs\\roundtrip\\791556-0.pb.gz
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import finetune_value_head as fv  # noqa: E402

POSITIONS = [  # name, moves
    ("Start", ""),
    ("1.e4 e5 2.Nf3 Nc6", "e2e4 e7e5 g1f3 b8c6"),
    ("1.e4 c5", "e2e4 c7c5"),
    ("1.d4 d5 2.c4", "d2d4 d7d5 c2c4"),
    ("Belgrade, move 5", fv.BELGRADE),
    ("Halloween, move 8", fv.HALLOWEEN_8),
]


def load(path, pb):
    net = pb.Net()
    net.ParseFromString(gzip.open(path).read())
    return net


def layers(message, prefix=""):
    """Yields (name, Layer) for every Layer in a Weights message, recursively."""
    for field, value in message.ListFields():
        if field.message_type is None:
            continue  # a number, e.g. pol_headcount
        name = f"{prefix}{field.name}"
        repeated = not hasattr(value, "DESCRIPTOR")  # a list of messages
        items = value if repeated else [value]
        for i, item in enumerate(items):
            item_name = f"{name}[{i}]" if repeated else name
            if item.DESCRIPTOR.name == "Layer":
                yield item_name, item
            else:
                yield from layers(item, item_name + ".")


def decode(layer):
    """A LINEAR16 layer's values, and the size of one 16-bit step."""
    raw = np.frombuffer(layer.params, dtype=np.uint16).astype(np.float64)
    step = (layer.max_val - layer.min_val) / 65535
    return layer.min_val + raw * step, step


def compare_weights(a, b, pb):
    for net in (a, b):
        if net.format.weights_encoding != pb.Format.LINEAR16:
            sys.exit("Only LINEAR16 networks are supported")
    if a.format.network_format != b.format.network_format:
        print("Network formats differ:")
        print(" ", a.format.network_format)
        print(" ", b.format.network_format)
    else:
        print("Network formats are identical")
    la, lb = dict(layers(a.weights)), dict(layers(b.weights))
    for name in sorted(la.keys() - lb.keys()):
        print(f"Only in the first: {name}")
    for name in sorted(lb.keys() - la.keys()):
        print(f"Only in the second: {name}")
    worst = []
    for name in sorted(la.keys() & lb.keys()):
        va, step = decode(la[name])
        vb, _ = decode(lb[name])
        if va.shape != vb.shape:
            print(f"{name}: sizes differ, {va.size} and {vb.size}")
            continue
        diff = float(np.max(np.abs(va - vb))) if va.size else 0.0
        worst.append((diff / step if step else 0.0, diff, name))
    worst.sort(reverse=True)
    print(f"{len(worst)} layers in both. Largest differences:")
    for steps, diff, name in worst[:10]:
        print(f"  {name}: {diff:.3g} ({steps:.2f} steps)")


def compare_engine(a, b):
    options = {"OpponentModelWeights": ""}  # plain Lc0, no opponent model
    results = []
    for path in (a, b):
        lc0 = fv.Lc0(path, options)
        results.append([lc0.search(moves, 1) for _, moves in POSITIONS])
        lc0.close()
    worst_v, worst_p = 0.0, 0.0
    for (name, _), (ra, ca, _), (rb, cb, _) in zip(POSITIONS, *results):
        dv = max(abs(ra[0] - rb[0]), abs(ra[1] - rb[1]))
        dp = max(abs(ca[m][1] - cb.get(m, (0, -1))[1]) for m in ca)
        worst_v, worst_p = max(worst_v, dv), max(worst_p, dp)
        print(f"  {name}: WL {ra[0]:+.5f} / {rb[0]:+.5f}, D {ra[1]:.5f} / {rb[1]:.5f}, "
              f"largest prior difference {dp * 100:.2f}%")
    print(f"Largest value difference {worst_v:.5f}, largest prior difference {worst_p * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()
    pb = fv.net_proto()
    print("== Weights")
    compare_weights(load(args.first, pb), load(args.second, pb), pb)
    print("== Lc0, one node")
    compare_engine(Path(args.first).resolve(), Path(args.second).resolve())


if __name__ == "__main__":
    main()
