"""Generates training games for fine-tuning 791556 with Lc0's own pipeline:
the engine against the modeled opponent, in Lc0's self-play.

The engine (player 1) searches 800 visits a move with the self-play settings
of Lc0's training run 1 (test80, the residual-network run closest to 791556,
see https://training.lczero.org/training_runs), with smart pruning off as in
Lc0's 800-visit run, and one change: nobody resigns. The goal is to win
quickly by checkmate, so every game is played out, to mate or a draw; Lc0
drops the rare game that reaches its 450-ply limit. Each position's
plies_left in the training data is then the number of half-moves to the
mate.

The opponent (player 2) is Maia-3 with the playing settings' opponent model
(rating 1500 against 1500, the closer look at depth 4 and strength 0.5, no
other shifts), each move drawn from its probabilities (see OpponentPlayer in
lc0-src/src/selfplay/tournament.cc). Games start from the initial position;
the engine's temperature over its first moves varies its openings, as in
Lc0's runs.

Lc0 writes each game's training data (V6 chunks, which configs/791556.yaml
reads) to training/data/data-<id>/. This script appends a line per game to
training/data/games.tsv: the engine's color, its result, how the game ended
(mate or draw), the number of half-moves, the training file and the moves. A
rerun continues until there are --games games; an interrupted run loses only
the games in progress, whose partly written files are removed on the next
start.

--report summarizes games.tsv: the engine's results, how often it mates and
how quickly.

Usage (from Windows, with the repository's .venv; about 7,000 games an hour
at the default parallelism of 16, the fastest of 8 to 128 on an RTX 4080):
    .venv\\Scripts\\python training\\generate_games.py --games 100000
    .venv\\Scripts\\python training\\generate_games.py --report
"""

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

import chess
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
LC0 = ROOT / "lc0-src" / "build" / "lc0.exe"
NET = ROOT / "lc0" / "bin" / "791556.pb.gz"
MAIA = ROOT / "lc0" / "nets" / "maia3-23m-fp16.onnx"
STOCKFISH = ROOT.parent / "StockfishTrapFinder" / "stockfish" / "stockfish.exe"
DATA = ROOT / "training" / "data"
INDEX = DATA / "games.tsv"
COLUMNS = ["engine_color", "result", "ending", "plies", "file", "moves"]

VISITS = 800
# Lc0's training run 1 (test80), without its opening book and with
# resignation off (resign-percentage 0, so resign-playthrough and
# resign-wdlstyle do nothing); smart pruning off as in its 800-visit run 2.
ENGINE = {
    "cpuct": 1.20, "cpuct-at-root": 2.0, "root-has-own-cpuct-params": "true",
    "resign-percentage": 0, "temperature": 0.9, "temp-endgame": 0.60,
    "tempdecay-moves": 60, "tempdecay-delay-moves": 20, "temp-cutoff-move": 40,
    "temp-visit-offset": -0.8, "fpu-strategy": "reduction", "fpu-value": 0.49,
    "fpu-strategy-at-root": "absolute", "fpu-value-at-root": 1.0,
    "policy-softmax-temp": 1.45, "noise-epsilon": 0.1, "noise-alpha": 0.12,
    "sticky-endgames": "true", "moves-left-max-effect": 0.2,
    "moves-left-threshold": 0.0, "moves-left-slope": 0.007,
    "moves-left-quadratic-factor": 0.85, "moves-left-scaled-factor": 0.15,
    "moves-left-constant-factor": 0.0, "task-workers": 0,
    "smart-pruning-factor": 0.0,
}
OPPONENT = {
    "opponent-player": "true", "opponent-model-weights": MAIA,
    "opponent-model-rating": 1500, "opponent-model-own-rating": 1500,
    "opponent-model-temperature": 1.0, "opponent-engine-mix": 0,
    "opponent-tactics": 0, "opponent-min-probability": 0,
    "opponent-deep-tactics": 0.5, "opponent-deep-tactics-depth": 4,
    "opponent-stockfish-path": STOCKFISH,
}


def read_index():
    if not INDEX.exists():
        return []
    lines = INDEX.read_text().splitlines()[1:]
    return [dict(zip(COLUMNS, line.split("\t"))) for line in lines if line]


def remove_orphans(games):
    """Removes training files of games that never reached the index: those
    in progress when a run was stopped, possibly written in part."""
    listed = {g["file"] for g in games}
    removed = 0
    for path in DATA.glob("data-*/*.gz"):
        if path.relative_to(DATA).as_posix() not in listed:
            path.unlink()
            removed += 1
    if removed:
        print(f"Removed {removed} training files of unfinished games")


def lc0_move(board, uci):
    """Lc0 writes castling as the king taking its own rook."""
    move = chess.Move.from_uci(uci)
    if board.piece_type_at(move.from_square) == chess.KING and board.color_at(move.to_square) == board.turn:
        to_file = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        move = chess.Move(move.from_square, chess.square(to_file, chess.square_rank(move.from_square)))
    return move


def parse_game(line):
    """The index row for one of Lc0's gameready lines."""
    words = line.split()
    field = {words[i]: words[i + 1] for i in range(1, len(words) - 1) if words[i] in
             ("trainingfile", "player1", "result")}
    moves = line.split(" moves ", 1)[1].split()
    board = chess.Board()
    for uci in moves:
        board.push(lc0_move(board, uci))
    engine_white = field["player1"] == "white"
    if field["result"] == "draw":
        result = "draw"
    else:
        result = "win" if (field["result"] == "whitewon") == engine_white else "loss"
    ending = "mate" if board.is_checkmate() else ("draw" if result == "draw" else "other")
    return {"engine_color": field["player1"], "result": result, "ending": ending,
            "plies": str(len(moves)), "file": field["trainingfile"], "moves": " ".join(moves)}


def generate(target, parallelism):
    DATA.mkdir(parents=True, exist_ok=True)
    games = read_index()
    remove_orphans(games)
    if len(games) >= target:
        print(f"{len(games)} games already generated")
        return
    if not INDEX.exists():
        INDEX.write_text("\t".join(COLUMNS) + "\n")
    command = [str(LC0), "selfplay", f"--games={target - len(games)}",
               f"--parallelism={parallelism}", "--training=true",
               "--backend=multiplexing", "--backend-opts=backend=cuda-fp16",
               f"--weights={NET}", f"--player1.visits={VISITS}"]
    command += [f"--{name}={value}" for name, value in ENGINE.items()]
    command += [f"--player2.{name}={value}" for name, value in OPPONENT.items()]
    # Lc0 writes its training data under the current directory.
    proc = subprocess.Popen(command, cwd=DATA, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    bar = tqdm(total=target, initial=len(games), desc="Games", unit="game", smoothing=0.05)
    counts = {"mate": 0, "win": 0}
    try:
        with INDEX.open("a") as index:
            for line in proc.stdout:
                if not line.startswith("gameready"):
                    continue
                game = parse_game(line)
                index.write("\t".join(game[c] for c in COLUMNS) + "\n")
                index.flush()
                counts["win"] += game["result"] == "win"
                counts["mate"] += game["result"] == "win" and game["ending"] == "mate"
                bar.update(1)
                bar.set_postfix(wins=counts["win"], by_mate=counts["mate"])
    except KeyboardInterrupt:
        print("\nStopped; rerun to continue.")
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
        bar.close()
    report()


def quartiles(values):
    if len(values) < 2:
        return f"{values[0]}" if values else "-"
    q1, median, q3 = statistics.quantiles(values, n=4)
    return f"median {median:.0f} (quartiles {q1:.0f}-{q3:.0f})"


def report():
    games = read_index()
    if not games:
        print("No games yet")
        return
    n = len(games)
    print(f"{n} games")
    for color in ("white", "black", None):
        subset = [g for g in games if color is None or g["engine_color"] == color]
        if not subset:
            continue
        results = {r: sum(g["result"] == r for g in subset) for r in ("win", "draw", "loss")}
        mates = [int(g["plies"]) for g in subset if g["result"] == "win" and g["ending"] == "mate"]
        name = f"Engine as {color}" if color else "All"
        print(f"{name}: {len(subset)} games, +{results['win']} ={results['draw']} -{results['loss']}; "
              f"mates {len(mates)} ({100 * len(mates) / len(subset):.0f}%), "
              f"half-moves to mate {quartiles(mates)}")
    mates = sorted(int(g["plies"]) for g in games if g["result"] == "win" and g["ending"] == "mate")
    for limit in (40, 60, 80):
        print(f"  mate within {limit} half-moves: {sum(p <= limit for p in mates) / n:.0%} of games")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, default=100000, help="total games wanted")
    parser.add_argument("--parallelism", type=int, default=16, help="games played at once")
    parser.add_argument("--report", action="store_true", help="only summarize the games so far")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        generate(args.games, args.parallelism)


if __name__ == "__main__":
    main()
