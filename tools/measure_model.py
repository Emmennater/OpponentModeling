"""Measures opponent models against moves played in real games.

For a sample of games in each rating band, every position is replayed, and
each model ranks the legal moves there. For the move that was actually
played, the tool records its rank, whether it was the model's top move (the
predicted move), and, for models that give probabilities, the probability the
model gave it. Models:

- stockfish: this engine's opponent model. The `opponentmoves` command gives
  the low-depth score of every move. Also reported: whether the model kept the
  played move (within the threshold of the best move), and its centipawn loss
  at low depth, which shows which threshold would have kept a given share of
  the played moves.
- maia: the Maia networks, run by Lc0. Each position uses the network closest
  to the mover's rating (Maia-1100 to Maia-1900).
- maia3: Maia-3 (maia3-79m by default, see --maia3-model), which takes both
  players' ratings as input. It
  needs PyTorch and the maia3 package, and the Maia3Net wrapper from the
  StockfishTrapFinder project (see --maia3-path).

With --calibrate, each Maia model is also measured with its probabilities
reweighted by a low-depth evaluation, as in StockfishTrapFinder's MaiaModel:
moves below --min-prob are dropped, the evaluation's best move is added, and
each move gives up max_rate * (1 - exp(-loss / scale_cp)) of its probability
to the evaluation's best move, where loss is how far its low-depth score is
below the best move's. The scores come from this engine's `opponentmoves`.

With --truth-depth, the tool also checks how well each model's probabilities
match how often players actually blunder. For each loss L, it compares the
probability a model puts on moves losing at least L centipawns with how often
the played move lost that much. Losses are measured at the truth depth, which
should be deeper than --depth so that calibration is not judged by the same
scores it was built from.

The games file has one game per line, tab-separated: white rating, black
rating, result, and the moves in SAN separated by spaces.

Usage:
    python tools/measure_model.py lichess_games.tsv --models stockfish,maia
    python tools/measure_model.py lichess_games.tsv --models maia,maia3 --calibrate
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
DEFAULT_MAIA3_PATH = ROOT.parent / "StockfishTrapFinder"

BANDS = [(0, 1200), (1200, 1400), (1400, 1600), (1600, 1800), (1800, 2000), (2000, 2200), (2200, 4000)]

MATE_CP = 10000  # Centipawn value used for mate scores
MIN_PROB = 1e-6  # Floor for log-likelihood, for moves a model gives no probability
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


# ---------------------------------------------------------------- games


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
    """Replays the games and returns one task per position with more than one
    legal move: the moves leading to it, the move played there, and the
    players' ratings."""
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
                white_to_move = board.turn == chess.WHITE
                tasks.append(
                    {
                        "id": len(tasks),
                        "rating": white if white_to_move else black,
                        "opp_rating": black if white_to_move else white,
                        "ply": ply,
                        "moves": list(moves),
                        "played": move.uci(),
                    }
                )
            board.push(move)
            moves.append(move.uci())
    return tasks


def board_of(task):
    board = chess.Board()
    for move in task["moves"]:
        board.push_uci(move)
    return board


# ---------------------------------------------------------------- engines


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


def stockfish_scores(args):
    """Worker: this engine's low-depth score of every move, in centipawns for
    the mover, and whether the model kept it. Returns ({id: (scores, kept)}, errors)."""
    tasks, depth, threshold = args
    engine = UciProcess(STOCKFISH, {"Opponent Depth": depth, "Centipawn Threshold": threshold})
    out, errors = {}, 0

    for task in tasks:
        engine.set_position(task["moves"])
        engine.send("opponentmoves")
        scores, kept = {}, {}
        for line in engine.read_until("readyok", "isready"):
            m = STOCKFISH_MOVE.match(line)
            if m:
                move, kind, value, verdict = m.groups()
                scores[move] = int(value) if kind == "cp" else (MATE_CP if int(value) > 0 else -MATE_CP)
                kept[move] = verdict == "kept"
            elif line.startswith("ERROR"):
                errors += 1
        out[task["id"]] = (scores, kept)

    engine.close()
    return out, errors


def standard_castling(board, uci):
    """Lc0 writes castling as the king taking its own rook (e1h1). Returns the
    standard form (e1g1) for such moves, and other moves unchanged."""
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.KING and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) > 1:
        to_file = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        return chess.Move(move.from_square, chess.square(to_file, chess.square_rank(move.from_square))).uci()
    return uci


def maia_probs(args):
    """Worker: Maia's probability of every move, for the tasks of one network."""
    tasks, net = args
    engine = UciProcess(
        LC0,
        {
            "WeightsFile": str(MAIA_NETS / f"maia-{net}.pb.gz"),
            "VerboseMoveStats": "true",
            "PolicyTemperature": "1.0",  # The network's own probabilities
        },
    )
    out = {}

    for task in tasks:
        engine.set_position(task["moves"])
        board = board_of(task)
        probs = {}
        for line in engine.read_until("bestmove", "go nodes 1"):
            m = LC0_MOVE.match(line)
            if m and m.group(1) != "node":
                probs[standard_castling(board, m.group(1))] = float(m.group(2)) / 100
        out[task["id"]] = probs

    engine.close()
    return out


def maia3_probs(tasks, maia3_path, model):
    """Maia-3's probability of every move. Runs in this process on the GPU,
    grouping positions by the two ratings rounded to 100."""
    sys.path.insert(0, str(maia3_path))
    from maia3_net import Maia3Net

    net = Maia3Net(model)
    groups = defaultdict(list)
    for task in tasks:
        groups[(round(task["rating"], -2), round(task["opp_rating"], -2))].append(task)

    out = {}
    for (self_elo, oppo_elo), group in sorted(groups.items()):
        net.self_elo, net.oppo_elo = self_elo, oppo_elo
        policies, _ = net.run([board_of(task) for task in group])
        for task, policy in zip(group, policies):
            out[task["id"]] = {move.uci(): p for move, p in policy.items()}
    return out


# ---------------------------------------------------------------- calibration


def calibrate(probs, scores, min_prob, scale_cp, max_rate):
    """StockfishTrapFinder's MaiaModel reweighting, with this engine's scores.

    Drops moves below min_prob (keeping the most likely if none remain), adds
    the move with the best low-depth score, and moves part of each move's
    probability to the best-scoring candidate: max_rate * (1 - exp(-loss /
    scale_cp)) of it, where loss is how far the move's score is below that
    candidate's. Dropped moves get probability 0.
    """
    total = sum(probs.values()) or 1.0
    probs = {m: p / total for m, p in probs.items()}

    candidates = {m: p for m, p in probs.items() if p >= min_prob}
    if not candidates:
        top = max(probs, key=probs.get)
        candidates = {top: probs[top]}

    scored = {m: s for m, s in scores.items() if m in probs}
    if scored:
        search_best = max(scored, key=scored.get)
        candidates.setdefault(search_best, probs.get(search_best, 0.0))

    candidates = {m: p for m, p in candidates.items() if m in scores}
    if not candidates:
        return probs

    best = max(candidates, key=lambda m: scores[m])
    shifted, caught = {}, 0.0
    for move, p in candidates.items():
        loss = max(0, scores[best] - scores[move])
        rate = max_rate * (1 - math.exp(-loss / scale_cp))
        caught += p * rate
        shifted[move] = p * (1 - rate)
    shifted[best] += caught

    total = sum(shifted.values())
    if total <= 0:
        return {m: 1 / len(shifted) for m in shifted}
    return {m: p / total for m, p in shifted.items()}


# ---------------------------------------------------------------- reporting


def rank_of(values, played):
    """1-based rank of the played move among `values`, a dict of move -> value."""
    return 1 + sum(1 for v in values.values() if v > values[played])


def records_for(tasks, values_by_id, has_probs):
    records = []
    for task in tasks:
        values = values_by_id.get(task["id"])
        if not values or task["played"] not in values and not has_probs:
            continue
        played = task["played"]
        record = {"rating": task["rating"], "ply": task["ply"]}
        if played in values:
            record["rank"] = rank_of(values, played)
        else:
            record["rank"] = len(values) + 1
        if has_probs:
            record["prob"] = values.get(played, 0.0)
        records.append(record)
    return records


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


def report(name, records, min_ply, has_probs):
    rows = [r for r in records if r["ply"] >= min_ply]
    print(f"\n{name}: positions from ply {min_ply} on ({len(rows)} positions). Rating is the mover's.")
    header = f"{'Rating':>10} {'Positions':>9} {'Top 1':>6} {'Top 3':>6} {'Top 5':>6} {'Rank':>5}"
    if has_probs:
        header += f" {'Prob of played':>15} {'Log-likelihood':>15}"
    print(header)
    print("-" * len(header))
    for band, group in groups(rows):
        line = (
            f"{band:>10} {len(group):>9} "
            + " ".join(f"{100 * statistics.mean(r['rank'] <= k for r in group):>5.1f}%" for k in (1, 3, 5))
            + f" {statistics.median(r['rank'] for r in group):>5.0f}"
        )
        if has_probs:
            line += (
                f" {100 * statistics.mean(r['prob'] for r in group):>14.1f}%"
                f" {statistics.mean(math.log(max(r['prob'], MIN_PROB)) for r in group):>15.3f}"
            )
        print(line)


def report_stockfish_details(tasks, sf, min_ply, threshold):
    rows = []
    for task in tasks:
        if task["ply"] < min_ply or task["id"] not in sf:
            continue
        scores, kept = sf[task["id"]]
        played = task["played"]
        if played not in scores:
            continue
        rows.append(
            {
                "rating": task["rating"],
                "kept": kept[played],
                "kept_share": sum(kept.values()) / len(kept),
                "loss": max(0, max(scores.values()) - scores[played]),
            }
        )

    print(f"\nstockfish details: positions from ply {min_ply} on.")
    print(f"{'Rating':>10} {'Kept':>7} {'Set size':>9} {'Loss p50':>9} {'p90':>5}")
    for band, group in groups(rows):
        losses = [r["loss"] for r in group]
        print(
            f"{band:>10} {100 * statistics.mean(r['kept'] for r in group):>6.1f}% "
            f"{100 * statistics.mean(r['kept_share'] for r in group):>8.1f}% "
            f"{percentile(losses, 50):>9.0f} {percentile(losses, 90):>5.0f}"
        )
    print(
        f"Kept: played move within {threshold} cp of the best at low depth. "
        "Set size: share of legal moves kept. Loss: low-depth centipawn loss of the played move."
    )

    print("\nShare of played moves within each threshold:")
    thresholds = [0, 20, 50, 100, 150, 200, 300, 500]
    print(f"{'Rating':>10} " + " ".join(f"{t:>6}" for t in thresholds))
    for band, group in groups(rows):
        shares = [100 * sum(1 for r in group if r["loss"] <= t) / len(group) for t in thresholds]
        print(f"{band:>10} " + " ".join(f"{s:>5.1f}%" for s in shares))


# ---------------------------------------------------------------- main


def report_blunders(name, tasks, probs_by_id, truth, min_ply):
    """Probability a model puts on moves losing at least L centipawns, against
    how often the played move lost that much. Losses come from `truth`."""
    losses_levels = [100, 200, 300, 500]
    rows = []
    for task in tasks:
        if task["ply"] < min_ply:
            continue
        probs = probs_by_id.get(task["id"])
        scores = truth.get(task["id"], ({}, {}))[0]
        if not probs or not scores or task["played"] not in scores:
            continue
        best = max(scores.values())
        row = {"rating": task["rating"]}
        for level in losses_levels:
            row[("pred", level)] = sum(p for m, p in probs.items() if m in scores and best - scores[m] >= level)
            row[("real", level)] = best - scores[task["played"]] >= level
        rows.append(row)

    print()
    print(f"{name}: probability on moves losing at least L cp (predicted) against how often they were played (actual)")
    print(f"{'Rating':>10} " + " ".join(f"{'L=' + str(l):>17}" for l in losses_levels))
    print(f"{'':>10} " + " ".join(f"{'predicted actual':>17}" for _ in losses_levels))
    for band, group in groups(rows):
        cells = []
        for level in losses_levels:
            pred = 100 * statistics.mean(r[("pred", level)] for r in group)
            real = 100 * statistics.mean(r[("real", level)] for r in group)
            cells.append(f"{pred:>7.1f}% {real:>7.1f}%")
        print(f"{band:>10} " + " ".join(f"{c:>17}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games_file")
    parser.add_argument("--models", default="stockfish,maia",
                        help="comma-separated: stockfish, maia, maia3 (default stockfish,maia)")
    parser.add_argument("--calibrate", action="store_true",
                        help="also measure each Maia model with its probabilities reweighted by a low-depth evaluation")
    parser.add_argument("--depth", type=int, default=2,
                        help="Opponent Depth for the stockfish model and for calibration scores (default 2)")
    parser.add_argument("--threshold", type=int, default=200, help="Centipawn Threshold for stockfish (default 200)")
    parser.add_argument("--truth-depth", type=int, default=0,
                        help="depth of the scores used to check blunder probabilities (default 0: no check)")
    parser.add_argument("--min-prob", type=float, default=0.02, help="calibration: moves below this are dropped (default 0.02)")
    parser.add_argument("--scale-cp", type=float, default=200, help="calibration: loss scale in centipawns (default 200)")
    parser.add_argument("--max-rate", type=float, default=0.7, help="calibration: most of a move's probability moved (default 0.7)")
    parser.add_argument("--maia3-model", default="maia3-79m", help="Maia-3 checkpoint, e.g. maia3-79m, maia3-23m, maia3-5m (default maia3-79m)")
    parser.add_argument("--maia3-path", default=str(DEFAULT_MAIA3_PATH),
                        help="folder containing maia3_net.py (default: the StockfishTrapFinder project next to this one)")
    parser.add_argument("--games", type=int, default=200, help="games sampled per rating band (default 200)")
    parser.add_argument("--min-ply", type=int, default=6, help="first ply reported, to skip the opening (default 6)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2),
                        help="engine processes for the stockfish scores (default half the CPUs)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in ("stockfish", "maia", "maia3"):
            sys.exit(f"Unknown model: {m}")

    print(f"Sampling {args.games} games per rating band from {args.games_file} ...", flush=True)
    reservoirs, seen = sample_games(args.games_file, args.games, args.seed)
    for band in BANDS:
        print(f"  {band_name(band):>10}: {len(reservoirs[band])} of {seen[band]} games")
    tasks = positions_of([g for band in BANDS for g in reservoirs[band]])
    print(f"{len(tasks)} positions", flush=True)

    sf, errors = None, 0
    if "stockfish" in models or args.calibrate:
        print(f"Scoring with this engine's opponent model at depth {args.depth} ...", flush=True)
        shuffled = list(tasks)
        random.Random(args.seed).shuffle(shuffled)
        jobs = [(shuffled[i :: args.workers], args.depth, args.threshold) for i in range(args.workers)]
        with multiprocessing.Pool(args.workers) as pool:
            sf = {}
            for part, e in pool.map(stockfish_scores, jobs):
                sf.update(part)
                errors += e

    truth = None
    if args.truth_depth:
        print(f"Scoring at depth {args.truth_depth} for the blunder check ...", flush=True)
        shuffled = list(tasks)
        random.Random(args.seed).shuffle(shuffled)
        jobs = [(shuffled[i :: args.workers], args.truth_depth, args.threshold) for i in range(args.workers)]
        with multiprocessing.Pool(args.workers) as pool:
            truth = {}
            for part, e in pool.map(stockfish_scores, jobs):
                truth.update(part)
                errors += e

    probs = {}
    if "maia" in models:
        print("Scoring with Maia (Lc0) ...", flush=True)
        by_net = defaultdict(list)
        for task in tasks:
            by_net[maia_rating(task["rating"])].append(task)
        jobs = [(net_tasks, net) for net, net_tasks in sorted(by_net.items())]
        with multiprocessing.Pool(len(jobs)) as pool:
            probs["maia"] = {}
            for part in pool.map(maia_probs, jobs):
                probs["maia"].update(part)
    if "maia3" in models:
        print("Scoring with Maia-3 ...", flush=True)
        probs["maia3"] = maia3_probs(tasks, args.maia3_path, args.maia3_model)

    results, prob_sets = [], []
    if "stockfish" in models:
        results.append(("stockfish", records_for(tasks, {i: s for i, (s, _) in sf.items()}, False), False))
    for name, by_id in probs.items():
        results.append((name, records_for(tasks, by_id, True), True))
        prob_sets.append((name, by_id))
        if args.calibrate:
            calibrated = {
                i: calibrate(p, sf[i][0], args.min_prob, args.scale_cp, args.max_rate)
                for i, p in by_id.items() if p and i in sf and sf[i][0]
            }
            results.append((f"{name}+cal", records_for(tasks, calibrated, True), True))
            prob_sets.append((f"{name}+cal", calibrated))

    for name, records, has_probs in results:
        report(name, records, args.min_ply, has_probs)
    if "stockfish" in models:
        report_stockfish_details(tasks, sf, args.min_ply, args.threshold)
    if truth is not None:
        for name, by_id in prob_sets:
            report_blunders(name, tasks, by_id, truth, args.min_ply)
    if sf is not None or truth is not None:
        print(f"\nFast/exact disagreements reported by the engine: {errors}")


if __name__ == "__main__":
    main()
