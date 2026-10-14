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

The opponent (player 2) is Maia-3 with the playing settings' opponent model:
rating 1500 against 1500, the closer look at depth 4 and strength 2 (an
upper bound on what a 1500 player sees: tools/calibrate_closer_look.py
finds Maia-3 alone about as error-prone as such players), and no other
shifts. Each move is drawn from its probabilities (see OpponentPlayer in
lc0-src/src/selfplay/tournament.cc). Games start from the initial position;
the engine's temperature over its first moves varies its openings, as in
Lc0's runs.

Games come in generations, as in Lc0's runs: generation N is played by the
network trained on the generations before it, training/nets/net<N-1>.pb.gz
(generation 1 by 791556), or by --net. Lc0 writes each game's training data
(V6 chunks, which configs/791556.yaml reads) to
training/data/gen<N>/data-<id>/. This script appends a line per game to
gen<N>/games.tsv: the engine's color, its result, how the game ended (mate or
draw), the number of half-moves, the training file and the moves, and
records the network in gen<N>/net.txt, so that a generation is never played
by two. A rerun continues until the generation has --games games; an
interrupted run loses only the games in progress, whose partly written files
are removed on the next start.

--eval plays --net's games in training/data/eval/<network>/ instead, apart
from the generations and their training data, to measure a network.
--report compares the generations and evaluations: the engine's results,
how often it mates and how quickly. --pgn writes their games as PGN.

Usage (from Windows, with the repository's .venv; about 7,000 games an hour
at the default parallelism of 16, the fastest of 8 to 128 on an RTX 4080):
    .venv\\Scripts\\python training\\generate_games.py --generation 1
    .venv\\Scripts\\python training\\generate_games.py --eval --net training\\nets\\net01.pb.gz
    .venv\\Scripts\\python training\\generate_games.py --report
    .venv\\Scripts\\python training\\generate_games.py --generation 1 --pgn gen01.pgn
"""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
LC0 = ROOT / "lc0-src" / "build" / "lc0.exe"
FIRST_NET = ROOT / "lc0" / "bin" / "791556.pb.gz"
MAIA = ROOT / "lc0" / "nets" / "maia3-23m-fp16.onnx"
STOCKFISH = ROOT.parent / "StockfishTrapFinder" / "stockfish" / "stockfish.exe"
DATA = ROOT / "training" / "data"
NETS = ROOT / "training" / "nets"
COLUMNS = ["engine_color", "result", "ending", "plies", "file", "moves"]
GAMES = 15000  # per generation
EVAL_GAMES = 500  # for --eval: the median half-moves to mate within about 2

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
    "opponent-deep-tactics": 2.0, "opponent-deep-tactics-depth": 4,
    "opponent-stockfish-path": STOCKFISH,
}


def generation_dir(generation):
    return DATA / f"gen{generation:02d}"


def default_net(generation):
    """The network that plays a generation: the one trained after the
    previous generation, or 791556 for the first."""
    return FIRST_NET if generation == 1 else NETS / f"net{generation - 1:02d}.pb.gz"


def read_index(folder):
    index = folder / "games.tsv"
    if not index.exists():
        return []
    lines = index.read_text().splitlines()[1:]
    return [dict(zip(COLUMNS, line.split("\t"))) for line in lines if line]


def remove_orphans(folder, games):
    """Removes training files of games that never reached the index: those
    in progress when a run was stopped, possibly written in part."""
    listed = {g["file"] for g in games}
    removed = 0
    for path in folder.glob("data-*/*.gz"):
        if path.relative_to(folder).as_posix() not in listed:
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


def net_name(net):
    return Path(net).name.removesuffix(".pb.gz")


def label(folder):
    if folder.parent.name == "eval":
        return f"Evaluation of {folder.name}"
    return f"Generation {int(folder.name[3:])}"


def generate(folder, net, target, parallelism):
    net = Path(net).resolve()
    if not net.exists():
        sys.exit(f"{net} does not exist: train it first, or give --net")
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "net.txt"
    if record.exists() and Path(record.read_text().strip()) != net:
        sys.exit(f"{label(folder)} was played by {record.read_text().strip()}, not {net}")
    record.write_text(f"{net}\n")
    games = read_index(folder)
    remove_orphans(folder, games)
    print(f"{label(folder)}, played by {net.name}")
    if len(games) >= target:
        print(f"{len(games)} games already generated")
        return
    index_path = folder / "games.tsv"
    if not index_path.exists():
        index_path.write_text("\t".join(COLUMNS) + "\n")
    command = [str(LC0), "selfplay", f"--games={target - len(games)}",
               f"--parallelism={parallelism}", "--training=true",
               "--backend=multiplexing", "--backend-opts=backend=cuda-fp16",
               f"--weights={net}", f"--player1.visits={VISITS}"]
    command += [f"--{name}={value}" for name, value in ENGINE.items()]
    command += [f"--player2.{name}={value}" for name, value in OPPONENT.items()]
    # Lc0 writes its training data under the current directory.
    proc = subprocess.Popen(command, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    bar = tqdm(total=target, initial=len(games), desc="Games", unit="game", smoothing=0.05)
    counts = {"mate": 0, "win": 0}
    try:
        with index_path.open("a") as index:
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
    folders = sorted(DATA.glob("gen[0-9][0-9]")) + sorted(DATA.glob("eval/*"))
    if not folders:
        print("No games yet")
    for folder in folders:
        games = read_index(folder)
        if not games:
            continue
        net = Path((folder / "net.txt").read_text().strip()).name
        print(f"{label(folder)}, played by {net}: {len(games)} games")
        for color in ("white", "black", None):
            subset = [g for g in games if color is None or g["engine_color"] == color]
            if not subset:
                continue
            results = {r: sum(g["result"] == r for g in subset) for r in ("win", "draw", "loss")}
            mates = [int(g["plies"]) for g in subset if g["result"] == "win" and g["ending"] == "mate"]
            name = f"Engine as {color}" if color else "All"
            print(f"  {name}: +{results['win']} ={results['draw']} -{results['loss']}; "
                  f"mates {100 * len(mates) / len(subset):.0f}%, "
                  f"half-moves to mate {quartiles(mates)}")
        mates = [int(g["plies"]) for g in games if g["result"] == "win" and g["ending"] == "mate"]
        within = ", ".join(f"{limit}: {sum(p <= limit for p in mates) / len(games):.0%}"
                           for limit in (40, 60, 80))
        print(f"  Mate within so many half-moves, share of games: {within}")


def draw_reason(board):
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.can_claim_threefold_repetition():
        return "threefold repetition"
    if board.can_claim_fifty_moves():
        return "fifty-move rule"
    return "draw"


def write_pgn(folder, path):
    """Writes a generation's games as PGN, for looking through them."""
    games = read_index(folder)
    if not games:
        sys.exit(f"{label(folder)} has no games yet")
    engine = f"Lc0 {net_name((folder / 'net.txt').read_text().strip())} ({VISITS} visits)"
    opponent = (f"Maia-3 {OPPONENT['opponent-model-rating']} (closer look depth "
                f"{OPPONENT['opponent-deep-tactics-depth']}, strength {OPPONENT['opponent-deep-tactics']})")
    with open(path, "w") as out:
        for i, g in enumerate(games, 1):
            board = chess.Board()
            for uci in g["moves"].split():
                board.push(lc0_move(board, uci))
            pgn = chess.pgn.Game.from_board(board)
            engine_white = g["engine_color"] == "white"
            pgn.headers.update({
                "Event": label(folder), "Site": "Lc0 self-play", "Round": str(i),
                "White": engine if engine_white else opponent,
                "Black": opponent if engine_white else engine,
                "Termination": "checkmate" if g["ending"] == "mate" else draw_reason(board),
                "PlyCount": g["plies"]})
            pgn.headers.pop("Date", None)
            out.write(str(pgn) + "\n\n")
    print(f"Wrote {len(games)} games to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generation", type=int, default=1, help="generation to play")
    parser.add_argument("--net", help="network to play it with (default: see above)")
    parser.add_argument("--games", type=int, help=f"games wanted (default {GAMES} for a generation, {EVAL_GAMES} for --eval)")
    parser.add_argument("--parallelism", type=int, default=16, help="games played at once")
    parser.add_argument("--eval", action="store_true",
                        help="play --net's games apart from the generations, to evaluate it")
    parser.add_argument("--report", action="store_true", help="only compare the generations so far")
    parser.add_argument("--pgn", help="only write the generation's games to this PGN file")
    args = parser.parse_args()
    if args.eval:
        if not args.net:
            sys.exit("--eval needs --net")
        folder = DATA / "eval" / net_name(args.net)
    else:
        folder = generation_dir(args.generation)
    if args.report:
        report()
    elif args.pgn:
        write_pgn(folder, args.pgn)
    else:
        games = args.games or (EVAL_GAMES if args.eval else GAMES)
        generate(folder, args.net or default_net(args.generation), games, args.parallelism)


if __name__ == "__main__":
    main()
