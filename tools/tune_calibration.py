"""Tunes the calibration of Maia-3 against real games.

The calibration (see measure_model.py --calibrate) reweights Maia's
probabilities with a low-depth evaluation. Its levers are the evaluation's
depth, scale_cp, max_rate, min_prob, and a temperature applied to Maia's
probabilities first. This tool collects the slow inputs once and saves them:
Maia-3's probabilities, this engine's low-depth scores of every legal move at
each calibration depth, and a deeper normal Stockfish score of every move. It then scores a
grid of settings from the saved data, which takes seconds.

Each setting is scored by:
- the log-likelihood of the move actually played (higher is better), the
  measure of how well the probabilities predict real moves;
- for each loss L, the probability put on moves losing at least L centipawns
  according to a normal Stockfish search at the truth depth (MultiPV over
  every move), against how often the played move lost that much;
- the blunder error: the average of |log(predicted / actual)| over the four
  values of L, 0 when every level matches.

Run it under a Python with PyTorch, the maia3 package and Maia3Net (the
StockfishTrapFinder venv):

    python tools/tune_calibration.py lichess_games.tsv --collect data.pkl --games 400
    python tools/tune_calibration.py --data data.pkl
"""

import argparse
import itertools
import math
import multiprocessing
import pickle
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_model as mm  # noqa: E402

LEVELS = [100, 200, 300, 500]


# ---------------------------------------------------------------- collecting


def scores_at(tasks, depth, workers, seed):
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    jobs = [(shuffled[i::workers], depth, 200) for i in range(workers)]
    out = {}
    with multiprocessing.Pool(workers) as pool:
        for part, _ in pool.map(mm.stockfish_scores, jobs):
            out.update({i: scores for i, (scores, _) in part.items()})
    return out


MULTIPV_LINE = re.compile(r"^info depth (\d+) .*?multipv \d+ score (cp|mate) (-?\d+) .*? pv (\S+)")


def multipv_scores(args):
    """Worker: the score of every legal move, in centipawns for the mover,
    from a normal Stockfish search (opponent modeling off) to `depth` with a
    principal variation for every move."""
    tasks, depth = args
    engine = mm.UciProcess(mm.STOCKFISH, {"Opponent Depth": 0, "MultiPV": 256, "Threads": 1, "Hash": 64})
    out = {}
    for task in tasks:
        engine.send("ucinewgame")
        engine.set_position(task["moves"])
        scores = {}
        for line in engine.read_until("bestmove", f"go depth {depth}"):
            m = MULTIPV_LINE.match(line)
            if m and int(m.group(1)) == depth:
                kind, value, move = m.group(2), int(m.group(3)), m.group(4)
                scores[move] = value if kind == "cp" else (mm.MATE_CP if value > 0 else -mm.MATE_CP)
        out[task["id"]] = scores
    engine.close()
    return out


def truth_scores(tasks, depth, workers, seed):
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    out = {}
    with multiprocessing.Pool(workers) as pool:
        for part in pool.map(multipv_scores, [(shuffled[i::workers], depth) for i in range(workers)]):
            out.update(part)
    return out


def collect(args):
    reservoirs, seen = mm.sample_games(args.games_file, args.games, args.seed)
    bands = [b for b in mm.BANDS if mm.band_name(b) in args.bands.split(",")]
    tasks = [t for t in mm.positions_of([g for b in bands for g in reservoirs[b]]) if t["ply"] >= args.min_ply]
    tasks = [t for t in tasks if mm.band_of(t["rating"]) in bands]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} positions from bands {args.bands}", flush=True)

    data = {"tasks": tasks, "scores": {}}
    start = time.time()
    data["probs"] = mm.maia3_probs(tasks, args.maia3_path, args.maia3_model)
    print(f"Maia-3: {time.time() - start:.0f} s", flush=True)
    for depth in args.depths:
        start = time.time()
        data["scores"][depth] = scores_at(tasks, depth, args.workers, args.seed)
        print(f"Scores at depth {depth}: {time.time() - start:.0f} s", flush=True)
    start = time.time()
    data["scores"]["truth"] = truth_scores(tasks, args.truth_depth, args.workers, args.seed)
    data["truth_depth"] = args.truth_depth
    print(f"Truth scores (Stockfish MultiPV, depth {args.truth_depth}): {time.time() - start:.0f} s", flush=True)

    with open(args.collect, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {args.collect}")


# ---------------------------------------------------------------- scoring


def with_temperature(probs, temperature):
    if temperature == 1:
        return probs
    powered = {m: p ** (1 / temperature) for m, p in probs.items() if p > 0}
    total = sum(powered.values())
    return {m: p / total for m, p in powered.items()}


def evaluate(data, depth, temperature, min_prob, scale_cp, max_rate):
    """Returns (log-likelihood, top 1, {L: (predicted, actual)}) over the
    positions where every input is present."""
    truth_all = data["scores"]["truth"]
    calib_all = data["scores"].get(depth)
    ll, top1, pred, real = [], [], {l: [] for l in LEVELS}, {l: [] for l in LEVELS}
    for task in data["tasks"]:
        i, played = task["id"], task["played"]
        probs, truth = data["probs"].get(i), truth_all.get(i)
        if not probs or not truth or played not in truth:
            continue
        probs = with_temperature(probs, temperature)
        if max_rate > 0 or min_prob > 0:
            calib = calib_all.get(i)
            if not calib:
                continue
            probs = mm.calibrate(probs, calib, min_prob, scale_cp, max_rate)
        p = probs.get(played, 0.0)
        ll.append(math.log(max(p, mm.MIN_PROB)))
        top1.append(max(probs, key=probs.get) == played)
        best = max(truth.values())
        for level in LEVELS:
            pred[level].append(sum(q for m, q in probs.items() if m in truth and best - truth[m] >= level))
            real[level].append(best - truth[played] >= level)
    blunders = {l: (statistics.mean(pred[l]), statistics.mean(real[l])) for l in LEVELS}
    return statistics.mean(ll), statistics.mean(top1), blunders


def blunder_error(blunders):
    return statistics.mean(abs(math.log(max(p, 1e-6) / max(a, 1e-6))) for p, a in blunders.values())


def sweep(args):
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    depths = [d for d in args.depths if d in data["scores"]]
    print(f"{len(data['tasks'])} positions, truth depth {data['truth_depth']}, calibration depths {depths}")

    rows = []
    for temperature in args.temperatures:
        rows.append((None, temperature, 0, 0, 0) + evaluate(data, None, temperature, 0, 0, 0))
    grid = itertools.product(depths, args.temperatures, args.min_probs, args.scales, args.rates)
    for depth, temperature, min_prob, scale, rate in grid:
        rows.append((depth, temperature, min_prob, scale, rate) + evaluate(data, depth, temperature, min_prob, scale, rate))

    actual = rows[0][7]
    print("\nActually played: " + "  ".join(f"L={l}: {100 * actual[l][1]:.1f}%" for l in LEVELS))
    header = (f"{'Depth':>5} {'Temp':>5} {'MinP':>5} {'Scale':>6} {'Rate':>5} {'LogLik':>7} {'Top1':>6} "
              + " ".join(f"{'L=' + str(l):>6}" for l in LEVELS) + f" {'BlErr':>6}")
    raw = rows[: len(args.temperatures)]
    rows = sorted(rows[len(args.temperatures) :], key=lambda r: (blunder_error(r[7]) if args.sort == "blunders" else -r[5]))
    print(header)
    print("-" * len(header))
    for depth, temperature, min_prob, scale, rate, ll, top1, blunders in raw + rows[: args.top]:
        print(f"{'raw' if depth is None else depth:>5} {temperature:>5.2f} {min_prob:>5.2f} {scale:>6.0f} {rate:>5.2f} "
              f"{ll:>7.3f} {100 * top1:>5.1f}% "
              + " ".join(f"{100 * blunders[l][0]:>5.1f}%" for l in LEVELS)
              + f" {blunder_error(blunders):>6.3f}")


def floats(text):
    return [float(x) for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games_file", nargs="?")
    parser.add_argument("--collect", help="collect the inputs from games_file and save them here")
    parser.add_argument("--data", help="score the grid from inputs saved by --collect")
    parser.add_argument("--bands", default="1200-1399", help="mover's rating bands, comma-separated (default 1200-1399)")
    parser.add_argument("--games", type=int, default=400, help="games sampled per band (default 400)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N positions (for timing)")
    parser.add_argument("--min-ply", type=int, default=6)
    parser.add_argument("--depths", type=lambda t: [int(x) for x in t.split(",")], default=[1, 2, 3],
                        help="calibration depths (default 1,2,3)")
    parser.add_argument("--truth-depth", type=int, default=10, help="Stockfish depth blunders are judged by (default 10)")
    parser.add_argument("--maia3-model", default="maia3-23m")
    parser.add_argument("--maia3-path", default=str(mm.DEFAULT_MAIA3_PATH))
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperatures", type=floats, default=[1.0])
    parser.add_argument("--min-probs", type=floats, default=[0.0])
    parser.add_argument("--scales", type=floats, default=[200, 400, 800, 1600])
    parser.add_argument("--rates", type=floats, default=[0.3, 0.5, 0.7])
    parser.add_argument("--sort", choices=["blunders", "loglik"], default="blunders")
    parser.add_argument("--top", type=int, default=30, help="rows shown (default 30)")
    args = parser.parse_args()

    if args.collect:
        collect(args)
    elif args.data:
        sweep(args)
    else:
        parser.error("give --collect or --data")


if __name__ == "__main__":
    main()
