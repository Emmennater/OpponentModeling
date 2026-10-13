"""Generates training data for the engine's own policy: the opponent-model
search's choices in positions from real games.

Positions are drawn at random from lichess_games.tsv (the openings of Lichess
blitz and rapid games), from games between players averaging RATINGS, at a
random half-move from PLIES, without repeats. In each, the engine plays the
side to move and searches with the settings below: the defaults, Stockfish's
closer look, 12 root candidates and the minimum share below the root
(OpponentWideCandidates), so that it can find moves the main network rates
low. Every move's visits, prior and value at the root are recorded, with the
move the engine chose.

The report counts the discoveries: positions where the engine chose a move
its main network gave a prior under DISCOVERY_PRIOR. A clear discovery is
also worth at least CLEAR_GAIN more than the main network's top move, so that
near ties (a different move order, a choice between two equal openings) do
not count: those are the moves the policy should learn to try.

Positions are saved as they are searched, so an interrupted run resumes;
--report only summarizes the data so far. Files go to
lc0/finetune/policy_data/ (ignored by git): sample.json (the positions),
positions.jsonl (one searched position a line) and summary.txt.

Usage (about 1.5 s a position at 30k nodes, so 8000 positions take 3-4
hours, after a minute reading the games):
    .venv\\Scripts\\python tools\\generate_policy_data.py
    .venv\\Scripts\\python tools\\generate_policy_data.py --report
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import chess
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_value_head as fv  # noqa: E402
import play_vs_maia as pv  # noqa: E402

WORK = fv.SHARED / "policy_data"
GAMES = fv.ROOT / "lichess_games.tsv"
RATINGS = (1300, 1700)  # the players' average rating
PLIES = (6, 16)  # the position's half-move, from the first to the last
DISCOVERY_PRIOR = 0.03
CLEAR_GAIN = 0.03
SETTINGS = {  # the defaults otherwise
    "OpponentStockfishPath": str(pv.STOCKFISH),
    "OpponentRootCandidates": 12,
    "OpponentWideCandidates": 12,
}
ANNOTATION = re.compile(r"[?!]+$")


def sample_positions(count, seed):
    """Reservoir-samples games in the rating range, then takes a random
    half-move of each, skipping positions already taken."""
    rng = random.Random(seed)
    games, seen = [], 0
    with open(GAMES, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading games", unit=" games", unit_scale=True):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                average = (int(fields[0]) + int(fields[1])) // 2
            except ValueError:
                continue
            if not RATINGS[0] <= average <= RATINGS[1]:
                continue
            seen += 1
            if len(games) < 3 * count:
                games.append(fields[3])
            else:
                j = rng.randrange(seen)
                if j < 3 * count:
                    games[j] = fields[3]
    positions, fens = [], set()
    for moves in games:
        board = chess.Board()
        line = []
        target = rng.randint(*PLIES)
        for san in moves.split()[:target]:
            try:
                move = board.parse_san(ANNOTATION.sub("", san))
            except ValueError:
                break
            board.push(move)
            line.append(move.uci())
        if len(line) != target or board.is_game_over():
            continue
        fen = " ".join(board.fen().split()[:4])
        if fen in fens:
            continue
        fens.add(fen)
        positions.append(line)
        if len(positions) == count:
            break
    return positions


def chosen(record):
    """The chosen move's statistics. Lc0's statistics write castling as the
    king taking its rook, its best move as e1g1."""
    children = record["children"]
    if record["best"] in children:
        return children[record["best"]]
    board = chess.Board()
    for m in record["moves"]:
        board.push_uci(m)
    move = chess.Move.from_uci(record["best"])
    if board.is_castling(move):
        rook = chess.square(7 if chess.square_file(move.to_square) == 6 else 0, chess.square_rank(move.from_square))
        return children.get(chess.Move(move.from_square, rook).uci())
    return None


def report(path):
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    discoveries, clear = [], []
    for r in records:
        stats = chosen(r)
        if stats is None:
            continue
        _, prior, value = stats
        if prior >= DISCOVERY_PRIOR:
            continue
        discoveries.append(r)
        top = max(r["children"].values(), key=lambda v: v[1])  # the main network's top move
        if value - top[2] >= CLEAR_GAIN:
            clear.append((r, prior, value - top[2]))
    by_ply = Counter(len(r["moves"]) for r, _, _ in clear)
    lines = [f"{len(records)} positions searched",
             f"Discoveries (the engine's move had a prior under {100 * DISCOVERY_PRIOR:.0f}%): {len(discoveries)}"
             f" ({100 * len(discoveries) / max(len(records), 1):.1f}%)",
             f"Clear discoveries (also at least {CLEAR_GAIN} better than the network's top move): {len(clear)}"
             f" ({100 * len(clear) / max(len(records), 1):.1f}%)",
             "  by half-move: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_ply.items())),
             "Examples:"]
    for r, prior, gain in clear[:20]:
        board = chess.Board()
        for m in r["moves"]:
            board.push_uci(m)
        lines.append(f"  {board.fen()}  {board.san(pv.lc0_move(board, r['best']))}"
                     f" (prior {100 * prior:.1f}%, +{gain:.2f} over the network's top move)")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, default=8000)
    parser.add_argument("--nodes", type=int, default=30000, help="the engine's nodes per position")
    parser.add_argument("--report", action="store_true", help="only summarize the data so far")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "positions.jsonl"

    if not args.report:
        sample_path = WORK / "sample.json"
        if sample_path.exists():
            sample = json.loads(sample_path.read_text())
        else:
            sample = sample_positions(args.positions, seed=1)
            sample_path.write_text(json.dumps(sample))
        done = set()
        if path.exists():
            done = {json.loads(line)["id"] for line in path.read_text().splitlines()}
        todo = [i for i in range(min(args.positions, len(sample))) if i not in done]
        print(f"{len(todo)} positions to search ({len(done)} done), {args.nodes} nodes each;"
              f" settings: defaults with {SETTINGS}")
        lc0 = fv.Lc0(fv.NET, SETTINGS)
        with path.open("a") as f:
            for i in tqdm(todo, desc="Positions", unit="position"):
                root, children, best = lc0.search(" ".join(sample[i]), args.nodes)
                if best is None or not children:
                    continue
                f.write(json.dumps({"id": i, "moves": sample[i], "best": best,
                                    "root": list(root) if root else None,
                                    # move (castling as the king taking its
                                    # rook): [visits, prior, value]
                                    "children": {m: list(v) for m, v in children.items()}}) + "\n")
                f.flush()
        lc0.close()

    lines = report(path)
    (WORK / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSummary saved to {WORK / 'summary.txt'}")


if __name__ == "__main__":
    main()
