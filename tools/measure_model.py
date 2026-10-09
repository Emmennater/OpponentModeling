"""Measures an opponent model against moves played in real games.

For a sample of games in each rating band, every position is replayed and the
model ranks the legal moves. For the move that was actually played, it
records its rank, whether it was the model's top move (the predicted move),
and model-specific details. Two models can be measured:

- stockfish: the engine's opponent model. The `opponentmoves` command gives
  the low-depth score of every move. Also recorded: whether the model kept the
  played move (within the threshold of the best move), and its centipawn loss
  at low depth. The loss distribution shows which threshold would have kept a
  given share of the played moves, for any threshold, from a single run.
- maia: the Maia networks, run by Lc0, which predict human moves for a given
  rating. Each position uses the network closest to the mover's rating. Also
  recorded: the probability the network gave the played move.

The games file has one game per line, tab-separated: white rating, black
rating, result, and the moves in SAN separated by spaces.

Usage:
    python tools/measure_model.py lichess_games.tsv --model stockfish --depth 2
    python tools/measure_model.py lichess_games.tsv --model maia
"""

import argparse
import math
import multiprocessing
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
STOCKFISH = ROOT / "stockfish" / "src" / "stockfish.exe"
LC0 = ROOT / "lc0" / "bin" / "lc0.exe"
MAIA_NETS = ROOT / "lc0" / "nets"
MAIA_RATINGS = range(1100, 2000, 100)

BANDS = [(0, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 4000)]

MATE_CP = 10000  # Centipawn value used for mate scores
ANNOTATION = re.compile(r"[?!]+$")
STOCKFISH_MOVE = re.compile(r"^(\S+)\s+(cp|mate) (-?\d+)\s+(kept|discarded)$")
LC0_MOVE = re.compile(r"^info string (\S+)\s+\(\s*\d+\s*\)\s+N:.*\(P:\s*([\d.]+)%\)")


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


def maia_rating(rating):
    """The Maia network for a rating: Maia-N is trained on players rated N to N+99."""
    return min(max(rating // 100 * 100, MAIA_RATINGS[0]), MAIA_RATINGS[-1])


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


def positions_of(games):
    """Replays the games and returns one task per position: the moves leading
    to it, the move played there, and the mover's rating."""
    tasks = []
    for fields in games:
        white, black = int(fields[0]), int(fields[1])
        board = chess.Board()
        moves = []
        for ply, san in enumerate(fields[3].split()):
            try:
                move = board.parse_san(ANNOTATION.sub("", san))
            except ValueError:
                break
            if board.legal_moves.count() > 1:
                tasks.append(
                    {
                        "rating": white if board.turn == chess.WHITE else black,
                        "ply": ply,
                        "moves": list(moves),
                        "played": move.uci(),
                    }
                )
            board.push(move)
            moves.append(move.uci())
    return tasks


class UciProcess:
    def __init__(self, path, options):
        self.proc = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, cwd=str(path.parent),
        )
        for name, value in options.items():
            self.send(f"setoption name {name} value {value}")
        self.read_until("readyok", "isready")

    def send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix, command):
        """Sends `command` and returns every line before the one starting with `prefix`."""
        self.send(command)
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            line = line.rstrip("\n")
            if line.startswith(prefix):
                return lines
            lines.append(line)

    def set_position(self, moves):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))

    def close(self):
        self.send("quit")
        self.proc.wait()


def rank_of(scores, played):
    """1-based rank of the played move among `scores`, a dict of move -> score."""
    return 1 + sum(1 for s in scores.values() if s > scores[played])


def measure_stockfish(args):
    tasks, depth, threshold = args
    engine = UciProcess(STOCKFISH, {"Opponent Depth": depth, "Centipawn Threshold": threshold})
    records, errors = [], 0

    for task in tasks:
        engine.set_position(task["moves"])
        engine.send("opponentmoves")
        lines = engine.read_until("readyok", "isready")

        scores, kept = {}, {}
        for line in lines:
            m = STOCKFISH_MOVE.match(line)
            if m:
                move, kind, value, verdict = m.groups()
                scores[move] = int(value) if kind == "cp" else (MATE_CP if int(value) > 0 else -MATE_CP)
                kept[move] = verdict == "kept"
            elif line.startswith("ERROR"):
                errors += 1

        played = task["played"]
        if played not in scores:
            continue
        rank = rank_of(scores, played)
        records.append(
            {
                "rating": task["rating"],
                "ply": task["ply"],
                "rank": rank,
                "kept": kept[played],
                "kept_share": sum(kept.values()) / len(kept),
                "loss": max(0, max(scores.values()) - scores[played]),
            }
        )

    engine.close()
    return records, errors


def standard_castling(board, uci):
    """Lc0 writes castling as the king taking its own rook (e1h1). Returns the
    standard form (e1g1) for such moves, and other moves unchanged."""
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.KING and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) > 1:
        to_file = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        return chess.Move(move.from_square, chess.square(to_file, chess.square_rank(move.from_square))).uci()
    return uci


def measure_maia(args):
    tasks, net = args
    engine = UciProcess(
        LC0,
        {
            "WeightsFile": str(MAIA_NETS / f"maia-{net}.pb.gz"),
            "VerboseMoveStats": "true",
            "PolicyTemperature": "1.0",  # The network's own probabilities
        },
    )
    records = []

    for task in tasks:
        engine.set_position(task["moves"])
        lines = engine.read_until("bestmove", "go nodes 1")

        board = chess.Board()
        for move in task["moves"]:
            board.push_uci(move)

        probs = {}
        for line in lines:
            m = LC0_MOVE.match(line)
            if m and m.group(1) != "node":
                probs[standard_castling(board, m.group(1))] = float(m.group(2)) / 100

        played = task["played"]
        if played not in probs:
            continue
        records.append(
            {
                "rating": task["rating"],
                "ply": task["ply"],
                "rank": rank_of(probs, played),
                "prob": probs[played],
            }
        )

    engine.close()
    return records, 0


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def groups(rows):
    for band in BANDS + [None]:
        group = rows if band is None else [r for r in rows if band_of(r["rating"]) == band]
        if group:
            yield ("All" if band is None else band_name(band)), group


def report(records, model, min_ply, threshold):
    rows = [r for r in records if r["ply"] >= min_ply]
    print(f"\nPositions from ply {min_ply} on ({len(rows)} positions). Rating is the mover's.")

    extra = "   Kept  Set size Loss p50   p90" if model == "stockfish" else "  Prob of played  Log-likelihood"
    header = f"{'Rating':>10} {'Positions':>9} {'Top 1':>6} {'Top 3':>6} {'Top 5':>6} {'Rank':>5} " + extra
    print(header)
    print("-" * len(header))

    for name, group in groups(rows):
        line = (
            f"{name:>10} {len(group):>9} "
            + " ".join(f"{100 * statistics.mean(r['rank'] <= k for r in group):>5.1f}%" for k in (1, 3, 5))
            + f" {statistics.median(r['rank'] for r in group):>5.0f} "
        )
        if model == "stockfish":
            losses = [r["loss"] for r in group]
            line += (
                f"{100 * statistics.mean(r['kept'] for r in group):>6.1f}% "
                f"{100 * statistics.mean(r['kept_share'] for r in group):>8.1f}% "
                f"{percentile(losses, 50):>8.0f} {percentile(losses, 90):>5.0f}"
            )
        else:
            line += (
                f"{100 * statistics.mean(r['prob'] for r in group):>15.1f}% "
                f"{statistics.mean(math.log(max(r['prob'], 1e-6)) for r in group):>15.3f}"
            )
        print(line)

    print(
        "\nTop k: share of played moves among the model's k best. Rank: median rank of the played move."
    )
    if model == "stockfish":
        print(
            f"Kept: played move within {threshold} cp of the best at low depth. "
            "Set size: share of legal moves kept. Loss: low-depth centipawn loss of the played move."
        )
        print("\nShare of played moves within each threshold:")
        thresholds = [0, 20, 50, 100, 150, 200, 300, 500]
        print(f"{'Rating':>10} " + " ".join(f"{t:>6}" for t in thresholds))
        for name, group in groups(rows):
            shares = [100 * sum(1 for r in group if r["loss"] <= t) / len(group) for t in thresholds]
            print(f"{name:>10} " + " ".join(f"{s:>5.1f}%" for s in shares))
    else:
        print(
            "Prob of played: average probability the network gave the played move. "
            "Log-likelihood: average natural log of it (higher is better)."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games_file")
    parser.add_argument("--model", choices=["stockfish", "maia"], default="stockfish")
    parser.add_argument("--depth", type=int, default=2, help="Opponent Depth for stockfish (default 2)")
    parser.add_argument("--threshold", type=int, default=200, help="Centipawn Threshold for stockfish (default 200)")
    parser.add_argument("--games", type=int, default=200, help="games sampled per rating band (default 200)")
    parser.add_argument("--min-ply", type=int, default=6, help="first ply reported, to skip the opening (default 6)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2),
                        help="engine processes for stockfish (maia uses one per network)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    engine = STOCKFISH if args.model == "stockfish" else LC0
    if not engine.exists():
        sys.exit(f"Engine not found: {engine}")

    print(f"Sampling {args.games} games per rating band from {args.games_file} ...", flush=True)
    reservoirs, seen = sample_games(args.games_file, args.games, args.seed)
    for band in BANDS:
        print(f"  {band_name(band):>10}: {len(reservoirs[band])} of {seen[band]} games")

    tasks = positions_of([g for band in BANDS for g in reservoirs[band]])

    if args.model == "stockfish":
        random.Random(args.seed).shuffle(tasks)
        jobs = [(tasks[i :: args.workers], args.depth, args.threshold) for i in range(args.workers)]
        worker = measure_stockfish
        print(f"Measuring {len(tasks)} positions with {args.workers} engines "
              f"(Opponent Depth {args.depth}, Centipawn Threshold {args.threshold}) ...", flush=True)
    else:
        by_net = defaultdict(list)
        for task in tasks:
            by_net[maia_rating(task["rating"])].append(task)
        jobs = [(net_tasks, net) for net, net_tasks in sorted(by_net.items())]
        worker = measure_maia
        print(f"Measuring {len(tasks)} positions with {len(jobs)} Maia networks ...", flush=True)

    with multiprocessing.Pool(len(jobs)) as pool:
        results = pool.map(worker, jobs)

    records = [r for recs, _ in results for r in recs]
    errors = sum(e for _, e in results)

    report(records, args.model, 0, args.threshold)
    report(records, args.model, args.min_ply, args.threshold)

    if args.model == "stockfish":
        print(f"\nFast/exact disagreements reported by the engine: {errors}")


if __name__ == "__main__":
    main()
