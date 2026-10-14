"""Calibrates the closer look (OpponentDeepTactics) against real players: how
often a ~1500 Lichess blitz player makes a mistake, against how often the
opponent model would, in the same positions.

Positions come from full Lichess blitz games between players rated 1400 to
1600 (fetched with tools/fetch_lichess_games.py), one a game, at a random
half-move from the tenth on, so that they are not correlated. In each,
Stockfish at REFERENCE_DEPTH scores every legal move, and each move's loss is
the win chance it gives up against the best move, by Lichess's formula, as
Lichess judges mistakes: 10 points an inaccuracy, 20 a mistake, 30 a
blunder. A move in a lost position then counts only for the chances it
actually throws away.

The human's rate is how often the move played lost that much. The model's
rate is its expected rate: its probability of each move times whether the
move loses that much, averaged over the positions. The model is Maia-3
(23m, rating 1500 against 1500, as in self-play), shifted by the closer
look at each strength in STRENGTHS and each depth in DEPTHS: each move's
probability weighted by exp(-strength * loss / 300 cp), its loss by
Stockfish at that depth (see ShiftForCloserLook in
lc0-src/src/search/classic/opponent_model.cc). Strength 0 is Maia-3 alone.

The analysis of each position is saved as it goes, so an interrupted run
resumes. Files go to lc0/finetune/calibration/ (ignored by git): the games,
positions.jsonl and summary.txt.

Usage (about 25 minutes for 2000 positions with 14 Stockfish processes):
    .venv\\Scripts\\python tools\\fetch_lichess_games.py lc0\\finetune\\calibration\\blitz_1400_1600.tsv --games 2000 --min-rating 1400 --max-rating 1600 --speeds Blitz
    .venv\\Scripts\\python tools\\calibrate_closer_look.py
"""

import argparse
import json
import math
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "lc0" / "finetune" / "calibration"
GAMES = WORK / "blitz_1400_1600.tsv"
STOCKFISH = ROOT.parent / "StockfishTrapFinder" / "stockfish" / "stockfish.exe"
sys.path.insert(0, str(ROOT.parent / "StockfishTrapFinder"))

REFERENCE_DEPTH = 14
DEPTHS = (4, 10)
STRENGTHS = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
THRESHOLDS = (("inaccuracy", 10), ("mistake", 20), ("blunder", 30))
RATING = 1500
FIRST_PLY = 10
WORKERS = 14
SEED = 1


def win_chance(cp):
    """Lichess's win chance, in points, for a score in centipawns."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def sample_positions():
    random.seed(SEED)
    positions = []
    for line in GAMES.read_text().splitlines():
        moves = line.split("\t")[3].split()
        if len(moves) <= FIRST_PLY:
            continue
        ply = random.randrange(FIRST_PLY, len(moves))
        board = chess.Board()
        for san in moves[:ply]:
            board.push_san(san)
        if board.legal_moves.count() < 2:
            continue
        positions.append({"moves": [m.uci() for m in board.move_stack], "played": board.parse_san(moves[ply]).uci()})
    return positions


local = threading.local()
engines = []  # every thread's, closed at the end: open ones keep Python from exiting


def engine():
    if not hasattr(local, "engine"):
        local.engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
        local.engine.configure({"Threads": 1, "Hash": 32})
        engines.append(local.engine)
    return local.engine


def scores(board, depth):
    """Every legal move's score in centipawns for the side to move, as Lc0's
    closer look gets them: one search with a line per move, and -10000 for
    a move Stockfish does not report."""
    sf = engine()
    infos = sf.analyse(board, chess.engine.Limit(depth=depth), multipv=board.legal_moves.count(),
                       game=object())  # a new game each time, like Lc0's own requests
    reported = {i["pv"][0].uci(): i["score"].pov(board.turn).score(mate_score=10000) for i in infos}
    return {m.uci(): reported.get(m.uci(), -10000) for m in board.legal_moves}


def analyse(position):
    board = chess.Board()
    for uci in position["moves"]:
        board.push_uci(uci)
    result = dict(position)
    result["reference"] = scores(board, REFERENCE_DEPTH)
    for depth in DEPTHS:
        result[f"depth{depth}"] = scores(board, depth)
    return result


def summarize(rows, out):
    lines = []
    n = len(rows)
    lines.append(f"{n} positions from 1400-1600 blitz games, reference Stockfish depth {REFERENCE_DEPTH}")
    header = f"{'':28}" + "".join(f"{name:>12}" for name, _ in THRESHOLDS) + f"{'mean loss':>12}"
    lines.append(header)

    def losses(row):
        best = max(row["reference"].values())
        return {m: win_chance(best) - win_chance(s) for m, s in row["reference"].items()}

    all_losses = [losses(r) for r in rows]
    human = [l[r["played"]] for r, l in zip(rows, all_losses)]
    lines.append(f"{'Humans (the move played)':28}" + "".join(
        f"{100 * sum(x >= t for x in human) / n:11.1f}%" for _, t in THRESHOLDS) + f"{sum(human) / n:12.1f}")
    for depth in DEPTHS:
        for strength in STRENGTHS:
            if strength == 0 and depth != DEPTHS[0]:
                continue
            rates = [0.0] * len(THRESHOLDS)
            mean = 0.0
            for row, loss in zip(rows, all_losses):
                look = row[f"depth{depth}"]
                best = max(look.values())
                weights = {m: p * math.exp(-strength * (best - look[m]) / 300) for m, p in row["maia"].items()}
                total = sum(weights.values())
                for m, w in weights.items():
                    p = w / total
                    mean += p * loss[m]
                    for i, (_, t) in enumerate(THRESHOLDS):
                        rates[i] += p * (loss[m] >= t)
            name = "Maia-3 alone" if strength == 0 else f"Closer look d{depth}, strength {strength}"
            lines.append(f"{name:28}" + "".join(f"{100 * r / n:11.1f}%" for r in rates) + f"{mean / n:12.1f}")
    text = "\n".join(lines)
    print(text)
    out.write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--positions", type=int, default=2000, help="positions to analyse")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if not GAMES.exists():
        sys.exit(f"{GAMES} is missing: fetch the games first (see the usage above)")
    positions = sample_positions()[:args.positions]
    path = WORK / "positions.jsonl"
    done = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    todo = positions[len(done):]
    if todo:
        with ThreadPoolExecutor(WORKERS) as pool, path.open("a") as out:
            for row in tqdm(pool.map(analyse, todo), total=len(positions), initial=len(done),
                            desc="Stockfish", unit="pos"):
                out.write(json.dumps(row) + "\n")
                out.flush()
                done.append(row)
        for sf in engines:
            sf.quit()

    # Maia-3, in one pass, as self-play runs it
    from maia3_net import Maia3Net
    maia = Maia3Net("maia3-23m", device="cpu")
    maia.self_elo, maia.oppo_elo = RATING, RATING
    boards = []
    for row in done:
        board = chess.Board()
        for uci in row["moves"]:
            board.push_uci(uci)
        boards.append(board)
    for start in tqdm(range(0, len(boards), 64), desc="Maia-3", unit="batch"):
        policies, _ = maia.run(boards[start:start + 64])
        for row, policy in zip(done[start:start + 64], policies):
            row["maia"] = {m.uci(): p for m, p in policy.items()}
    summarize(done, WORK / "summary.txt")


if __name__ == "__main__":
    main()
