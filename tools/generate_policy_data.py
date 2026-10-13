"""Generates training data for the engine's own policy: the opponent-model
search's choices in games against Maia.

The engine plays games against Maia-3 (maia3-23m on the CPU), whose moves are
drawn from its probabilities for a player of OPPONENT_RATING, from the start
of the game, with each color in turn, up to MAX_PLIES half-moves. At each of
the engine's moves the search runs with the settings below: the defaults,
Stockfish's closer look, 12 root candidates and the minimum share below the
root (OpponentWideCandidates), so that it can find moves the main network
rates low. Every move's visits, prior and value at the root are recorded,
with the move the engine chose.

A "discovery" is a position where the engine chose a move its main network
gave a prior under DISCOVERY_PRIOR: the moves the policy should learn to try
against human opponents.

Games are saved as they finish, so an interrupted run resumes; --report only
summarizes the data so far. Files go to lc0/finetune/policy_data/ (ignored by
git): positions.jsonl (one engine move per line) and summary.txt.

Usage (about 20 seconds a game at 20k nodes a move, so 600 games take about
3-4 hours):
    .venv\\Scripts\\python tools\\generate_policy_data.py
    .venv\\Scripts\\python tools\\generate_policy_data.py --report
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_value_head as fv  # noqa: E402
import play_vs_maia as pv  # noqa: E402

WORK = fv.SHARED / "policy_data"
OPPONENT_RATING = 1500
MAX_PLIES = 40
DISCOVERY_PRIOR = 0.03
SETTINGS = {  # the defaults otherwise
    "OpponentStockfishPath": str(pv.STOCKFISH),
    "OpponentRootCandidates": 12,
    "OpponentWideCandidates": 12,
}


def play(lc0, maia, engine_white, seed, nodes):
    """Plays one game; returns the engine's positions with the search's
    statistics."""
    rng = random.Random(seed)
    board = chess.Board()
    records = []
    while len(board.move_stack) < MAX_PLIES and not board.is_game_over():
        if board.turn == (chess.WHITE if engine_white else chess.BLACK):
            moves = [m.uci() for m in board.move_stack]
            root, children, best = lc0.search(" ".join(moves), nodes)
            if best is None or not children:
                break
            records.append({
                "moves": moves,
                "best": best,
                "root": list(root) if root else None,
                # move (as Lc0 writes it, castling as the king taking its
                # rook): [visits, prior, value]
                "children": {m: list(v) for m, v in children.items()},
            })
            board.push(pv.lc0_move(board, best))
        else:
            policy, _ = maia.run([board])
            moves, probs = zip(*policy[0].items())
            board.push(rng.choices(moves, probs)[0])
    return records


def best_prior(record):
    """The prior of the move the engine chose. Lc0's statistics write
    castling as the king taking its rook, its best move as e1g1."""
    children = record["children"]
    if record["best"] in children:
        return children[record["best"]][1]
    board = chess.Board()
    for m in record["moves"]:
        board.push_uci(m)
    move = chess.Move.from_uci(record["best"])
    if board.is_castling(move):
        rook = chess.square(7 if chess.square_file(move.to_square) == 6 else 0, chess.square_rank(move.from_square))
        alt = chess.Move(move.from_square, rook).uci()
        if alt in children:
            return children[alt][1]
    return None


def report(path):
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    games = {r["game"] for r in records}
    priors = [(best_prior(r), r) for r in records]
    discoveries = [(p, r) for p, r in priors if p is not None and p < DISCOVERY_PRIOR]
    by_ply = Counter(len(r["moves"]) // 10 * 10 for _, r in discoveries)
    lines = [f"{len(records)} engine positions from {len(games)} games",
             f"Discoveries (the engine's move had a prior under {100 * DISCOVERY_PRIOR:.0f}%): {len(discoveries)}"
             f" ({100 * len(discoveries) / max(len(records), 1):.1f}%)",
             "  by half-move: " + ", ".join(f"{k}-{k + 9}: {v}" for k, v in sorted(by_ply.items())),
             "Examples:"]
    for p, r in discoveries[:15]:
        board = chess.Board()
        for m in r["moves"]:
            board.push_uci(m)
        lines.append(f"  {board.fen()}  {board.san(pv.lc0_move(board, r['best']))} (prior {100 * p:.1f}%)")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--nodes", type=int, default=20000, help="the engine's nodes per move")
    parser.add_argument("--report", action="store_true", help="only summarize the data so far")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "positions.jsonl"

    if not args.report:
        done = set()
        if path.exists():
            done = {json.loads(line)["game"] for line in path.read_text().splitlines()}
        todo = [i for i in range(args.games) if i not in done]
        print(f"{len(todo)} games to play ({len(done)} done), {args.nodes} nodes a move, against Maia"
              f" {OPPONENT_RATING}; settings: defaults with {SETTINGS}")
        maia = pv.Maia3Net("maia3-23m", device="cpu")
        maia.self_elo, maia.oppo_elo = OPPONENT_RATING, OPPONENT_RATING
        lc0 = fv.Lc0(fv.NET, SETTINGS)
        with tqdm(total=len(todo), desc="Games", unit="game") as bar:
            for i in todo:
                records = play(lc0, maia, i % 2 == 0, 5000 + i, args.nodes)
                with path.open("a") as f:
                    for r in records:
                        f.write(json.dumps({"game": i, **r}) + "\n")
                bar.update(1)
        lc0.close()

    lines = report(path)
    (WORK / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSummary saved to {WORK / 'summary.txt'}")


if __name__ == "__main__":
    main()
