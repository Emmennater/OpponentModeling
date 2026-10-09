"""Measures the opponent model against moves played in real games.

For a sample of games in each rating band, every position is replayed and
the engine's `opponentmoves` command gives the low-depth score of every legal
move. For the move that was actually played, it records:

- whether the model kept it (it is within the threshold of the best move),
- whether it was the predicted move (the low-depth search's best move),
- its rank among the legal moves, by low-depth score,
- its centipawn loss: how far its low-depth score is below the best move's.

The loss distribution shows which threshold would have kept a given share of
the moves actually played, for any threshold, from a single run.

The games file has one game per line, tab-separated: white rating, black
rating, result, and the moves in SAN separated by spaces.

Usage:
    python tools/measure_model.py lichess_games.tsv --depth 2 --games 200
"""

import argparse
import math
import multiprocessing
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

import chess

ENGINE = Path(__file__).resolve().parent.parent / "stockfish" / "src" / "stockfish.exe"

BANDS = [(0, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 4000)]

MATE_CP = 10000  # Centipawn value used for mate scores
ANNOTATION = re.compile(r"[?!]+$")
MOVE_LINE = re.compile(r"^(\S+)\s+(cp|mate) (-?\d+)\s+(kept|discarded)$")


def band_of(rating):
    for low, high in BANDS:
        if low <= rating < high:
            return (low, high)
    return None


def band_name(band):
    low, high = band
    if low == 0:
        return f"<{high}"
    if high >= 4000:
        return f"{low}+"
    return f"{low}-{high - 1}"


def sample_games(path, per_band, seed):
    """Reservoir-samples `per_band` games from each rating band, using the
    average of the two ratings."""
    rng = random.Random(seed)
    reservoirs = {band: [] for band in BANDS}
    seen = {band: 0 for band in BANDS}

    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                white, black = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            band = band_of((white + black) // 2)
            if band is None:
                continue
            seen[band] += 1
            reservoir = reservoirs[band]
            if len(reservoir) < per_band:
                reservoir.append(fields)
            else:
                j = rng.randrange(seen[band])
                if j < per_band:
                    reservoir[j] = fields

    return reservoirs, seen


class Engine:
    def __init__(self, depth, threshold):
        self.proc = subprocess.Popen(
            [str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.send(f"setoption name Opponent Depth value {depth}")
        self.send(f"setoption name Centipawn Threshold value {threshold}")
        self.sync()

    def send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def sync(self):
        """Sends 'isready' and returns every line before 'readyok'."""
        self.send("isready")
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.rstrip("\n")
            if line == "readyok":
                return lines
            lines.append(line)

    def opponent_moves(self, uci_moves):
        self.send("position startpos moves " + " ".join(uci_moves) if uci_moves else "position startpos")
        self.send("opponentmoves")
        lines = self.sync()

        moves, errors = {}, 0
        for line in lines:
            m = MOVE_LINE.match(line)
            if m:
                move, kind, value, verdict = m.groups()
                cp = int(value) if kind == "cp" else (MATE_CP if int(value) > 0 else -MATE_CP)
                moves[move] = (cp, verdict == "kept")
            elif line.startswith("ERROR"):
                errors += 1
        return moves, errors

    def close(self):
        self.send("quit")
        self.proc.wait()


def measure_games(args):
    """Worker: replays the given games and returns one record per position."""
    games, depth, threshold = args
    engine = Engine(depth, threshold)
    records, errors = [], 0

    for fields in games:
        white, black = int(fields[0]), int(fields[1])
        board = chess.Board()
        uci_moves = []

        for ply, san in enumerate(fields[3].split()):
            san = ANNOTATION.sub("", san)
            try:
                move = board.parse_san(san)
            except ValueError:
                break

            scores, e = engine.opponent_moves(uci_moves)
            errors += e
            played = move.uci()

            if played in scores and len(scores) > 1:
                ranked = sorted(scores.values(), key=lambda s: -s[0])
                best_cp = ranked[0][0]
                played_cp, kept = scores[played]
                rank = 1 + sum(1 for cp, _ in ranked if cp > played_cp)
                records.append(
                    {
                        "rating": white if board.turn == chess.WHITE else black,
                        "ply": ply,
                        "legal": len(scores),
                        "kept_count": sum(1 for _, k in scores.values() if k),
                        "kept": kept,
                        "predicted": rank == 1,
                        "rank": rank,
                        "loss": max(0, best_cp - played_cp),
                    }
                )

            board.push(move)
            uci_moves.append(played)

    engine.close()
    return records, errors


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def report(records, min_ply, threshold):
    rows = [r for r in records if r["ply"] >= min_ply]
    print(f"\nPositions from ply {min_ply} on ({len(rows)} positions). Rating is the mover's.")
    header = (
        f"{'Rating':>10} {'Positions':>9} {'Kept':>6} {'Predicted':>9} {'Rank':>5} "
        f"{'Set size':>9} {'Loss p50':>8} {'p75':>5} {'p90':>5} {'p95':>5}"
    )
    print(header)
    print("-" * len(header))

    for band in BANDS + [None]:
        group = rows if band is None else [r for r in rows if band_of(r["rating"]) == band]
        if not group:
            continue
        losses = [r["loss"] for r in group]
        print(
            f"{'All' if band is None else band_name(band):>10} {len(group):>9} "
            f"{100 * statistics.mean(r['kept'] for r in group):>5.1f}% "
            f"{100 * statistics.mean(r['predicted'] for r in group):>8.1f}% "
            f"{statistics.median(r['rank'] for r in group):>5.0f} "
            f"{100 * statistics.mean(r['kept_count'] / r['legal'] for r in group):>8.1f}% "
            f"{percentile(losses, 50):>8.0f} {percentile(losses, 75):>5.0f} "
            f"{percentile(losses, 90):>5.0f} {percentile(losses, 95):>5.0f}"
        )

    print(
        f"\nKept: played move within {threshold} cp of the best at low depth. "
        "Predicted: played move was the low-depth best.\n"
        "Rank: median rank of the played move. Set size: share of legal moves kept. "
        "Loss: low-depth centipawn loss of the played move."
    )

    print("\nShare of played moves within each threshold:")
    thresholds = [0, 20, 50, 100, 150, 200, 300, 500]
    print(f"{'Rating':>10} " + " ".join(f"{t:>6}" for t in thresholds))
    for band in BANDS + [None]:
        group = rows if band is None else [r for r in rows if band_of(r["rating"]) == band]
        if not group:
            continue
        shares = [100 * sum(1 for r in group if r["loss"] <= t) / len(group) for t in thresholds]
        print(f"{'All' if band is None else band_name(band):>10} " + " ".join(f"{s:>5.1f}%" for s in shares))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games_file")
    parser.add_argument("--depth", type=int, default=2, help="Opponent Depth (default 2)")
    parser.add_argument("--threshold", type=int, default=200, help="Centipawn Threshold (default 200)")
    parser.add_argument("--games", type=int, default=200, help="games sampled per rating band (default 200)")
    parser.add_argument("--min-ply", type=int, default=6, help="first ply reported, to skip the opening (default 6)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not ENGINE.exists():
        sys.exit(f"Engine not found: {ENGINE}. Run build.bat first.")

    print(f"Sampling {args.games} games per rating band from {args.games_file} ...", flush=True)
    reservoirs, seen = sample_games(args.games_file, args.games, args.seed)
    games = [g for band in BANDS for g in reservoirs[band]]
    for band in BANDS:
        print(f"  {band_name(band):>10}: {len(reservoirs[band])} of {seen[band]} games")

    random.Random(args.seed).shuffle(games)
    chunks = [games[i :: args.workers] for i in range(args.workers)]
    print(f"Replaying {len(games)} games with {args.workers} engines "
          f"(Opponent Depth {args.depth}, Centipawn Threshold {args.threshold}) ...", flush=True)

    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(measure_games, [(c, args.depth, args.threshold) for c in chunks])

    records = [r for recs, _ in results for r in recs]
    errors = sum(e for _, e in results)

    report(records, 0, args.threshold)
    report(records, args.min_ply, args.threshold)

    print(f"\nFast/exact disagreements reported by the engine: {errors}")


if __name__ == "__main__":
    main()
