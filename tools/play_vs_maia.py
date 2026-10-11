"""Plays the engine against Maia from the start of the game, and measures how
often and how quickly it reaches a winning position.

The engine plays one side with the playing settings below and each value of
an option; Maia-3 (maia3-23m, on the CPU) plays the other, each move drawn
from its probabilities for a player of OPPONENT_RATING, as a real player of
that rating might choose. After each move Stockfish scores the position. A
game stops when the engine is ahead by WINNING cp (reached) or behind by as
much (lost), or after MAX_PLIES half-moves.

Each game number uses the same random seed for every value, so the values
face the same draws of the opponent's choices (until the games diverge). The
engine plays White in half the games and Black in the other half.

Reported for each value: how often the engine reached +3, how often it fell
to -3 first, the median half-move at which it reached +3, and Stockfish's
average score at the end of each game.

Note that the engine's opponent model is the same Maia-3 network, so this
measures how well the engine exploits its own model; real players differ.

Everything is saved as it goes, so an interrupted run resumes. Files go to
lc0/finetune/<tag>/ (ignored by git); the summary to summary.txt there.

Usage:
    .venv\\Scripts\\python tools\\play_vs_maia.py --option OpponentWinValue --values 1 0.8
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_value_head as fv  # noqa: E402

sys.path.insert(0, str(fv.ROOT.parent / "StockfishTrapFinder"))
from maia3_net import Maia3Net  # noqa: E402

STOCKFISH = fv.ROOT.parent / "StockfishTrapFinder" / "stockfish" / "stockfish.exe"
PLAYING = {"OpponentModelRating": 1500, "OpponentModelOwnRating": 1500, "OpponentBestLean": 1.0,
           "OpponentDiscount": 0.995, "OpponentCaution": 0.15, "OpponentRefutationProbability": 0.05,
           "OpponentTactics": 0.5, "OpponentCautionAhead": 0.25, "CPuct": 1.377,
           "VerboseMoveStats": "false"}
OPPONENT_RATING = 1500
WINNING = 300  # centipawns for the engine
MAX_PLIES = 40


def lc0_move(board, uci):
    """Lc0 writes castling as the king taking its own rook."""
    move = chess.Move.from_uci(uci)
    if board.piece_type_at(move.from_square) == chess.KING and board.color_at(move.to_square) == board.turn:
        to_file = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        move = chess.Move(move.from_square, chess.square(to_file, chess.square_rank(move.from_square)))
    return move


def engine_cp(sf, board, engine_white):
    score = sf.analyse(board, chess.engine.Limit(depth=14))["score"].white().score(mate_score=10000)
    return score if engine_white else -score


def play(lc0, maia, sf, engine_white, seed, nodes):
    """Returns the game's moves, the engine's Stockfish score after each, and
    the half-move at which it reached +WINNING or -WINNING (or None)."""
    rng = random.Random(seed)
    board = chess.Board()
    scores, reached, lost = [], None, None
    while len(board.move_stack) < MAX_PLIES and not board.is_game_over():
        if board.turn == (chess.WHITE if engine_white else chess.BLACK):
            _, _, best = lc0.search(" ".join(m.uci() for m in board.move_stack), nodes)
            board.push(lc0_move(board, best))
        else:
            policy, _ = maia.run([board])
            moves, probs = zip(*policy[0].items())
            board.push(rng.choices(moves, probs)[0])
        cp = engine_cp(sf, board, engine_white)
        scores.append(cp)
        if cp >= WINNING:
            reached = len(board.move_stack)
            break
        if cp <= -WINNING:
            lost = len(board.move_stack)
            break
    return {"moves": [m.uci() for m in board.move_stack], "scores": scores, "reached": reached, "lost": lost}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--option", default="OpponentWinValue")
    parser.add_argument("--values", nargs="+", default=["1", "0.8"], help="the first is the baseline")
    parser.add_argument("--games", type=int, default=60, help="games per value, half with each color")
    parser.add_argument("--nodes", type=int, default=30000, help="the engine's nodes per move")
    parser.add_argument("--tag", default="", help="files go to lc0/finetune/<tag> (by default play_<option>)")
    args = parser.parse_args()
    work = fv.SHARED / (args.tag or f"play_{args.option}")
    work.mkdir(parents=True, exist_ok=True)
    path = work / "games.json"
    games = json.loads(path.read_text()) if path.exists() else {}

    print(f"{args.option} at {', '.join(args.values)}: {args.games} games each against Maia {OPPONENT_RATING},"
          f" {args.nodes} nodes per move, with {PLAYING}")
    maia = Maia3Net("maia3-23m", device="cpu")
    maia.self_elo, maia.oppo_elo = OPPONENT_RATING, PLAYING["OpponentModelOwnRating"]
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"Threads": 4, "Hash": 128})
    total = len(args.values) * args.games
    with tqdm(total=total, desc="Games", unit="game") as bar:
        for value in args.values:
            games.setdefault(value, {})
            todo = [i for i in range(args.games) if str(i) not in games[value]]
            bar.update(args.games - len(todo))
            if not todo:
                continue
            lc0 = fv.Lc0(fv.NET, {**PLAYING, args.option: value})
            for i in todo:
                games[value][str(i)] = play(lc0, maia, sf, i % 2 == 0, 1000 + i, args.nodes)
                path.write_text(json.dumps(games))
                bar.update(1)
            lc0.close()
    sf.quit()

    summary = [f"{args.option}: {args.games} games per value against Maia {OPPONENT_RATING} (sampled), {args.nodes}"
               f" nodes per move, settings {PLAYING}",
               f"Reached: the engine at +{WINNING / 100:.0f} or better by Stockfish within {MAX_PLIES} half-moves;"
               f" lost: at -{WINNING / 100:.0f} first.",
               f"  {'Value':<8} {'color':<6} {'reached':>8} {'lost':>6} {'median ply':>11} {'end score':>10}"]
    for value in args.values:
        for color, parity in (("White", 0), ("Black", 1), ("both", None)):
            gs = [g for i, g in games[value].items() if int(i) < args.games and (parity is None or int(i) % 2 == parity)]
            if not gs:
                continue
            reached = [g["reached"] for g in gs if g["reached"]]
            lost = sum(1 for g in gs if g["lost"])
            end = statistics.mean(max(-1000, min(1000, g["scores"][-1])) for g in gs if g["scores"])
            median = f"{statistics.median(reached):.0f}" if reached else "-"
            summary.append(f"  {value:<8} {color:<6} {len(reached):>4}/{len(gs):<3} {lost:>6} {median:>11}"
                           f" {end / 100:>+10.2f}")
    summary.append("  (median ply: the half-move at which +3 was reached, in the games that reached it; end score:"
                   " Stockfish at the game's last position, for the engine, capped at 10 pawns)")
    (work / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nSummary saved to {work / 'summary.txt'}")


if __name__ == "__main__":
    main()
