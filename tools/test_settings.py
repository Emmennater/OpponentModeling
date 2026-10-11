"""Tests one of the engine's options over a few values: how often the engine,
when ahead, gives its advantage back, and its opening choices.

1. Positions: the real-game positions in lc0/finetune/positions.json, scored by
   Stockfish; the ones where the side to move is ahead by 1.5 to 10 pawns.
2. Moves: the engine's move in each, with the playing settings below and each
   value of the option, scored by Stockfish. A move that drops the evaluation
   by 2 pawns or more gives back much of the advantage.
3. Openings: the opening choices, the Belgrade with equal search for 5.Nd5 and
   5.Nxd4, three lines where the engine gambled, the Halloween positions and a
   mate in 2, for each value.

Everything is saved as it goes, so an interrupted run resumes. Files go to
lc0/finetune/<tag>/ (ignored by git); the summary to summary.txt there.

Usage:
    .venv\\Scripts\\python tools\\test_settings.py --option OpponentHorizon --values 0 6 10 16
    .venv\\Scripts\\python tools\\test_settings.py --option OpponentCautionAhead --values 0.0 0.25 0.5 --tag ahead
(the second, 200 positions and three values, took 28 minutes)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_value_head as fv  # noqa: E402

WORK = fv.SHARED  # lc0/finetune/<tag>, set in main()
OPTION = ""  # the option tested, set in main()
# StockfishTrapFinder's Stockfish 19: this project's own build in stockfish/src is
# about 10 times slower
STOCKFISH = fv.ROOT.parent / "StockfishTrapFinder" / "stockfish" / "stockfish.exe"
PLAYING = {"OpponentModelRating": 1400, "OpponentModelOwnRating": 1400, "OpponentBestLean": 1.0,
           "OpponentDiscount": 0.995, "OpponentCaution": 0.15, "OpponentRefutationProbability": 0.05,
           "OpponentTactics": 0.5, "OpponentCautionAhead": 0.5, "CPuct": 1.377}
AHEAD_MIN, AHEAD_MAX = 150, 1000  # the side to move's lead, in centipawns
GIVES_BACK = 200  # a drop of this many centipawns or more gives back much of the advantage
NF6_LINE = ("e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 d2d4 e5d4 c3d5 f6e4 d1e2 f7f5 f3g5 f8b4 e1d1 e8g8 e2h5 e4g5 "
            "c1g5 d8e8")
BE2_LINE = ("e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 d2d4 e5d4 c3d5 f6e4 d1e2 f7f5 f3g5 d7d6 g5e4 f5e4 e2e4 f8e7 "
            "c1g5 h7h6 g5h4 g7g5")
DXE5_LINE = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 c6a5 c4e2 f6d5 d2d4 f8e7"
OPENING_TESTS = [  # name, moves, nodes, wanted
    ("Start", "", 300000, "e2e4"),
    ("1.e4 e5", "e2e4 e7e5", 300000, ""),
    ("1.e4 e5 2.Nf3 Nc6", "e2e4 e7e5 g1f3 b8c6", 300000, ""),
    ("1.e4 c5", "e2e4 c7c5", 300000, ""),
    ("Belgrade, move 5", fv.BELGRADE, 100000, "c3d5"),
    ("Belgrade line, move 11", NF6_LINE, 300000, "h5e8"),
    ("Belgrade line, move 12", BE2_LINE, 300000, "h4g3"),
    ("Two Knights, move 8", DXE5_LINE, 300000, "not d4e5"),
    ("Halloween, move 8", fv.HALLOWEEN_8, 300000, "not c3d5"),
    ("Halloween, move 15", fv.HALLOWEEN_15, 300000, "not e1e7"),
    ("Mate in 2", fv.MATE_IN_2, 100000, "h1h6"),
]


def load(name, default):
    path = WORK / name
    return json.loads(path.read_text()) if path.exists() else default


def save(name, data):
    (WORK / name).write_text(json.dumps(data))


def board_of(moves):
    board = chess.Board()
    for m in moves.split():
        board.push_uci(m)
    return board


def mover_cp(sf, board, depth):
    """Stockfish's score for the side to move, in centipawns."""
    score = sf.analyse(board, chess.engine.Limit(depth=depth))["score"].relative
    return score.score(mate_score=10000)


def lc0_move(board, uci):
    """Lc0 writes castling as the king taking its own rook."""
    move = chess.Move.from_uci(uci)
    if board.piece_type_at(move.from_square) == chess.KING and board.color_at(move.to_square) == board.turn:
        to_file = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        move = chess.Move(move.from_square, chess.square(to_file, chess.square_rank(move.from_square)))
    return move


def positions(sf, count):
    print(f"Step 1: positions where the side to move is ahead by {AHEAD_MIN}-{AHEAD_MAX} cp")
    scored_path = fv.SHARED / "scored.json"  # shared by all experiments
    scored = json.loads(scored_path.read_text()) if scored_path.exists() else {}
    pool = json.loads((fv.SHARED / "positions.json").read_text())
    for p in tqdm(pool, desc="Scoring", unit="position"):
        key = " ".join(p["moves"])
        if key not in scored:
            board = board_of(key)
            scored[key] = mover_cp(sf, board, 12) if not board.is_game_over() else None
            if len(scored) % 100 == 0:
                scored_path.write_text(json.dumps(scored))
    scored_path.write_text(json.dumps(scored))
    ahead = sorted(k for k, cp in scored.items() if cp is not None and AHEAD_MIN <= cp <= AHEAD_MAX)
    random.Random(1).shuffle(ahead)
    print(f"  {len(ahead)} of {len(scored)} positions; using {min(count, len(ahead))}")
    return ahead[:count]


def moves(sf, keys, values, nodes):
    print(f"\nStep 2: the engine's moves at {nodes} nodes, scored by Stockfish")
    chosen = load("moves.json", {})
    for value in values:
        tag = str(value)
        chosen.setdefault(tag, {})
        todo = [k for k in keys if k not in chosen[tag]]
        if not todo:
            continue
        lc0 = fv.Lc0(fv.NET, {**PLAYING, OPTION: value, "VerboseMoveStats": "false"})
        for i, key in enumerate(tqdm(todo, desc=f"{OPTION} {value}", unit="position")):
            _, _, best = lc0.search(key, nodes)
            chosen[tag][key] = best
            if i % 10 == 9:
                save("moves.json", chosen)
        lc0.close()
        save("moves.json", chosen)

    scores = load("move_scores.json", {})
    jobs = [(k, chosen[str(v)][k]) for v in values for k in keys]
    for key, uci in tqdm(jobs, desc="Stockfish", unit="move"):
        if key not in scores:
            scores[key] = {"before": mover_cp(sf, board_of(key), 16)}
        if uci not in scores[key]:
            board = board_of(key)
            board.push(lc0_move(board, uci))
            scores[key][uci] = -mover_cp(sf, board, 16)
            save("move_scores.json", scores)
    save("move_scores.json", scores)

    summary = [f"Moves in {len(keys)} positions where the side to move is ahead by {AHEAD_MIN}-{AHEAD_MAX} cp "
               f"(Stockfish), {nodes} nodes, settings {PLAYING}:",
               f"  {'Value':<8} {'drop 100+':>10} {'drop 200+':>10} {'below +150':>11} {'lost half':>10}"
               f" {'changed':>8}"]
    base = chosen[str(values[0])]
    for value in values:
        after = {k: scores[k][chosen[str(value)][k]] for k in keys}
        drops = [scores[k]["before"] - after[k] for k in keys]
        count = lambda cp: sum(d >= cp for d in drops)  # noqa: E731
        below = sum(after[k] < 150 for k in keys if scores[k]["before"] >= 150)
        lost_half = sum(after[k] < scores[k]["before"] / 2 and after[k] < 300 for k in keys)
        changed = sum(chosen[str(value)][k] != base[k] for k in keys)
        summary.append(f"  {value:<8} {count(100):>10} {count(200):>10} {below:>11} {lost_half:>10} {changed:>8}")
    summary.append(f"  (numbers of positions. drop: Stockfish's score before minus after the move, for the side to"
                   f" move; below +150: ended below +150 from at least +150; lost half: lost over half the"
                   f" advantage, ending below +300; changed: a different move than at {values[0]})")
    print("\n".join(summary))
    return summary


def openings(values):
    print("\nStep 3: openings and the test lines")
    results = load("openings.json", {})
    with tqdm(total=len(values) * (len(OPENING_TESTS) + 4), desc="Openings", unit="search") as bar:
        for value in values:
            tag = str(value)
            if tag in results:
                bar.update(len(OPENING_TESTS) + 4)
                continue
            lc0 = fv.Lc0(fv.NET, {**PLAYING, OPTION: value})
            row = {}
            for nodes in (34000, 129000):
                q = {}
                for move in ("c3d5", "f3d4"):
                    _, children, _ = lc0.search(fv.BELGRADE, nodes, searchmoves=move)
                    q[move] = children.get(move, (0, 0, float("nan")))[2]
                    bar.update(1)
                row[f"gap {nodes}"] = q["c3d5"] - q["f3d4"]
            for name, line, nodes, _ in OPENING_TESTS:
                _, _, best = lc0.search(line, nodes)
                row[name] = best
                bar.update(1)
            lc0.close()
            results[tag] = row
            save("openings.json", results)

    summary = ["Belgrade, 5.Nd5 - 5.Nxd4 with equal search (about 34k and 129k visits each):"]
    for value in values:
        r = results[str(value)]
        summary.append(f"  {OPTION} {value:<5} {r['gap 34000']:+.3f}  {r['gap 129000']:+.3f}")
    summary.append("Choices:")
    summary.append(f"  {'Position':<24}" + "".join(f"{v:>10}" for v in values) + "   Wanted")
    for name, _, _, wanted in OPENING_TESTS:
        summary.append(f"  {name:<24}" + "".join(f"{results[str(v)][name]:>10}" for v in values)
                       + f"   {wanted}")
    print("\n".join(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--option", default="OpponentHorizon", help="the option to test")
    parser.add_argument("--values", nargs="+", default=["0", "6", "10", "16"],
                        help="its values; the first is the baseline")
    parser.add_argument("--tag", default="", help="experiment name: files go to lc0/finetune/<tag>"
                        " (by default the option's name)")
    parser.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                        help="other settings to change from the playing settings, such as OpponentModelOwnRating=1500")
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=50000)
    args = parser.parse_args()
    global WORK, OPTION
    OPTION = args.option
    WORK = fv.SHARED / (args.tag or args.option)
    WORK.mkdir(parents=True, exist_ok=True)
    for setting in args.set:
        name, value = setting.split("=", 1)
        PLAYING[name] = value
    print(f"Testing {OPTION} at {', '.join(args.values)}, with {PLAYING}")
    sf = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    sf.configure({"Threads": 4, "Hash": 256})
    keys = positions(sf, args.positions)
    summary = moves(sf, keys, args.values, args.nodes)
    sf.quit()
    summary += [""] + openings(args.values)
    (WORK / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nSummary saved to {WORK / 'summary.txt'}")


if __name__ == "__main__":
    main()
