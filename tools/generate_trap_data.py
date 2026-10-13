"""Finds the traps in real games, and the moves that set them, as training
data for the engine's own network.

For every position in the games where a human is to move (up to MAX_PLY),
Maia-3 (maia3-23m, on the CPU) gives the probability of each move, for the
players' ratings, and Stockfish (MultiPV, DEPTH half-moves) scores each move.
That gives the position's trap value for the human to move, a one-move
expectimax:

    V_trap = sum over moves r of P_Maia(r) * tanh(Stockfish(r) / 350)

against its objective value, tanh(Stockfish's best / 350), both as W - L for
the side to move. These are the targets for the value at positions where the
opponent is to move.

The report then finds the trap positions, where V_trap is at least
TRAP_MARGIN worse than the objective value for the human to move, and walks
back along the game to the move that set the trap: while the human's moves on
the way were likely (Maia at least LIKELY), the whole line stays likely
(REACH), and the trap setter's moves in between were natural (within NATURAL
of Stockfish's best), the earliest of the trap setter's moves starts it. A
start is kept if, in practice, it is worth at least the best move by
Stockfish:

    P(reaching the trap) * (trap value for the setter)
        + (1 - P) * tanh(Stockfish(start) / 350)  >=  tanh(Stockfish's best / 350)

These starts are the examples for the policy. They are written to
traps.pgn for review, each game up to the start, with the line to the trap.

Needs full games: lichess games with all their moves, from
tools/fetch_lichess_games.py, for example
    .venv\\Scripts\\python tools\\fetch_lichess_games.py full_games.tsv --games 3000 --min-rating 1300 --max-rating 1700

Games are saved as they are analysed, so an interrupted run resumes; --report
only analyses the data so far. Files go to lc0/finetune/trap_data/ (ignored by
git): positions.jsonl (one game a line), summary.txt and traps.pgn.

Usage (about 25 positions a second with 16 Stockfish processes, so 3000 games of
up to 40 half-moves take about 80 minutes):
    .venv\\Scripts\\python tools\\generate_trap_data.py full_games.tsv
    .venv\\Scripts\\python tools\\generate_trap_data.py full_games.tsv --report
"""

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_value_head as fv  # noqa: E402
import play_vs_maia as pv  # noqa: E402

WORK = fv.SHARED / "trap_data"
MAX_PLY = 40  # positions after at most this many half-moves
DEPTH = 8
PROCESSES = 16
CP_SCALE = 350.0  # W - L = tanh(cp / 350)
TRAP_MARGIN = 0.3
LIKELY = 0.4
REACH = 0.3
NATURAL = 100  # centipawns
MAX_SETTER_MOVES = 3
CHUNK = 10  # games analysed at once
ANNOTATION = re.compile(r"[?!]+$")


def wl(cp):
    return math.tanh(cp / CP_SCALE)


def read_games(path):
    games = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                white, black = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            board, moves = chess.Board(), []
            for san in fields[3].split()[:MAX_PLY + 1]:
                try:
                    move = board.parse_san(ANNOTATION.sub("", san))
                except ValueError:
                    break
                board.push(move)
                moves.append(move.uci())
            games.append({"white": white, "black": black, "moves": moves})
    return games


def score_moves(sf, board):
    """Stockfish's score of every legal move, in centipawns for the mover."""
    n = board.legal_moves.count()
    info = sf.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=n)
    return {i["pv"][0].uci(): i["score"].relative.score(mate_score=10000) for i in info if "pv" in i}


def positions_of(game):
    boards = []
    board = chess.Board()
    for ply in range(len(game["moves"])):
        if not board.is_game_over() and ply <= MAX_PLY:
            boards.append((ply, board.copy()))
        board.push_uci(game["moves"][ply])
    return boards


def maia_probs(game, boards, maia):
    """Maia's probabilities, by the mover's and the opponent's ratings."""
    probs = {}
    for white_to_move in (True, False):
        group = [(ply, b) for ply, b in boards if b.turn == white_to_move]
        if not group:
            continue
        own, other = (game["white"], game["black"]) if white_to_move else (game["black"], game["white"])
        maia.self_elo, maia.oppo_elo = own, other
        policies, _ = maia.run([b for _, b in group])
        for (ply, _), policy in zip(group, policies):
            probs[ply] = {m.uci(): p for m, p in policy.items()}
    return probs


def records_of(game, boards, probs, scores):
    """Each position: Maia's probabilities and Stockfish's scores of every
    move, with the trap value."""
    records = []
    for ply, b in boards:
        sc, pr = scores[ply], probs[ply]
        if not sc:
            continue
        best = max(sc.values())
        v_trap = sum(p * wl(sc.get(m, best)) for m, p in pr.items())
        played = game["moves"][ply] if ply < len(game["moves"]) else None
        records.append({"ply": ply, "best": best, "objective": wl(best), "v_trap": v_trap,
                        "played": played, "p_played": pr.get(played) if played else None,
                        "cp_played": sc.get(played) if played else None,
                        "top": sorted(pr.items(), key=lambda kv: -kv[1])[:5]})
    return records


def analyse_chunk(games, maia, pool, engines):
    """Analyses several games at once: Stockfish's jobs run in the pool while
    Maia runs here."""
    def job(item):
        ply, b = item
        sf = engines.get()
        try:
            return ply, score_moves(sf, b)
        finally:
            engines.put(sf)
    boards = [positions_of(g) for g in games]
    futures = [[pool.submit(job, item) for item in bs] for bs in boards]
    probs = [maia_probs(g, bs, maia) for g, bs in zip(games, boards)]
    return [records_of(g, bs, pr, dict(f.result() for f in fs))
            for g, bs, pr, fs in zip(games, boards, probs, futures)]


def find_starts(game):
    """The trap positions in a game and the moves that started them."""
    by_ply = {r["ply"]: r for r in game["records"]}
    starts = []
    for r in game["records"]:
        if r["objective"] - r["v_trap"] < TRAP_MARGIN:
            continue
        t = r["ply"]  # the human to move at t is the one to be trapped
        setter_value = -r["v_trap"]  # for the trap setter
        start, reach = t - 1, 1.0
        # Walk back: the human's move at start - 1, the setter's at start - 2.
        k = 1
        while k < MAX_SETTER_MOVES and start - 2 >= 0:
            human, setter = by_ply.get(start - 1), by_ply.get(start - 2)
            if not human or not setter or human["p_played"] is None:
                break
            setter_here = by_ply.get(start)
            natural = setter_here and setter_here["cp_played"] is not None and \
                setter_here["best"] - setter_here["cp_played"] <= NATURAL
            if human["p_played"] < LIKELY or reach * human["p_played"] < REACH or not natural:
                break
            reach *= human["p_played"]
            start -= 2
            k += 1
        s = by_ply.get(start)
        if not s or s["cp_played"] is None:
            continue
        practical = reach * setter_value + (1 - reach) * wl(s["cp_played"])
        if practical < s["objective"]:
            continue
        starts.append({"start": start, "trap": t, "reach": reach, "practical": practical,
                       "objective_best": s["objective"], "objective_start": wl(s["cp_played"]),
                       "trap_gap": r["objective"] - r["v_trap"]})
    # One start per starting move.
    unique = {}
    for s in starts:
        if s["start"] not in unique or s["practical"] > unique[s["start"]]["practical"]:
            unique[s["start"]] = s
    return list(unique.values())


def report(path):
    games = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    positions = sum(len(g["records"]) for g in games)
    traps = sum(1 for g in games for r in g["records"] if r["objective"] - r["v_trap"] >= TRAP_MARGIN)
    starts = [(g, s) for g in games for s in find_starts(g)]
    longer = sum(1 for _, s in starts if s["trap"] - s["start"] > 1)
    lines = [f"{len(games)} games, {positions} positions analysed",
             f"Trap positions (trap value at least {TRAP_MARGIN} below the objective value for the human to move):"
             f" {traps} ({100 * traps / max(positions, 1):.1f}%)",
             f"Moves that started a trap and pay off in practice: {len(starts)}, of which {longer} start it"
             f" a move or more before the trap"]
    starts.sort(key=lambda gs: -(gs[1]["practical"] - gs[1]["objective_best"]))
    with open(WORK / "traps.pgn", "w") as f:
        for g, s in starts:
            game = chess.pgn.Game()
            game.headers["White"] = str(g["white"])
            game.headers["Black"] = str(g["black"])
            game.headers["Event"] = (f"Trap start, {s['trap'] - s['start']} half-moves before the trap; reach"
                                     f" {100 * s['reach']:.0f}%, practical {s['practical']:+.2f} against the best"
                                     f" move's {s['objective_best']:+.2f}")
            node = game
            for ply, m in enumerate(g["moves"][:s["trap"]]):
                comment = ""
                if ply == s["start"] == s["trap"] - 1:
                    comment = (f"sets the trap: objective {s['objective_start']:+.2f}, best"
                               f" {s['objective_best']:+.2f}; ")
                elif ply == s["start"]:
                    comment = (f"starts the trap: objective {s['objective_start']:+.2f}, best"
                               f" {s['objective_best']:+.2f}")
                rec = next((r for r in g["records"] if r["ply"] == ply), None)
                if ply > s["start"] and rec and rec.get("p_played") is not None and (ply - s["start"]) % 2 == 1:
                    comment = f"Maia {100 * rec['p_played']:.0f}%"
                node = node.add_variation(chess.Move.from_uci(m), comment=comment)
            trap = next(r for r in g["records"] if r["ply"] == s["trap"])
            node.comment = node.comment + (f"the trap: objective {trap['objective']:+.2f} for the side to move, trap value"
                            f" {trap['v_trap']:+.2f}; Maia's likely moves: "
                            + ", ".join(f"{node.board().san(chess.Move.from_uci(m))} {100 * p:.0f}%"
                                        for m, p in trap["top"][:4]))
            print(game, file=f, end="\n\n")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", help="full games, as written by tools/fetch_lichess_games.py")
    parser.add_argument("--report", action="store_true", help="only analyse the data so far")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "positions.jsonl"

    if not args.report:
        games = read_games(args.games)
        done = set()
        if path.exists():
            done = {json.loads(line)["game"] for line in path.read_text().splitlines()}
        todo = [i for i in range(len(games)) if i not in done]
        print(f"{len(todo)} games to analyse ({len(done)} done); Stockfish depth {DEPTH} in {PROCESSES} processes")
        maia = pv.Maia3Net("maia3-23m", device="cpu")
        import queue
        engines = queue.Queue()
        for _ in range(PROCESSES):
            sf = chess.engine.SimpleEngine.popen_uci(str(pv.STOCKFISH))
            sf.configure({"Threads": 1, "Hash": 16})
            engines.put(sf)
        with ThreadPoolExecutor(PROCESSES) as pool, path.open("a") as f, \
                tqdm(total=len(todo), desc="Games", unit="game") as bar:
            for c in range(0, len(todo), CHUNK):
                chunk = todo[c:c + CHUNK]
                for i, records in zip(chunk, analyse_chunk([games[i] for i in chunk], maia, pool, engines)):
                    f.write(json.dumps({"game": i, **games[i], "records": records}) + "\n")
                f.flush()
                bar.update(len(chunk))
        while not engines.empty():
            engines.get().quit()

    lines = report(path)
    (WORK / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSummary saved to {WORK / 'summary.txt'}; the trap starts to {WORK / 'traps.pgn'}")


if __name__ == "__main__":
    main()
