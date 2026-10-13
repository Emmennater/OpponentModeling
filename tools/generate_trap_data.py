"""Finds the traps in real games, and the moves that set them, as training
data for the engine's own network.

For every position in the games where a human is to move (up to MAX_PLY),
Maia-3 (maia3-23m, on the CPU) gives the probability of each move, for the
players' ratings, and Stockfish (MultiPV, DEPTH half-moves) scores each move.
The probabilities are shifted toward Stockfish's best move as the engine's
OpponentDeepTactics does (DEEP_RATE): each move gives up
DEEP_RATE * (1 - exp(-loss / 300 cp)) of its probability to the best one, so
that a blunder a careful player would see counts for less. That gives the
position's trap value for the human to move, a one-move expectimax:

    V_trap = sum over moves r of P(r) * tanh(Stockfish(r) / 350)

against its objective value, tanh(Stockfish's best / 350), both as W - L for
the side to move. These are the targets for the value at positions where the
opponent is to move.

The report then finds the trap positions: where the human's most natural
move, Maia's most likely before the shift (at least NATURAL_P), loses
BLUNDER_CP or more against the best, and V_trap is at least TRAP_MARGIN worse
than the objective value. A position where the best move is the most likely
one is no trap, however many other moves would lose: the threat is plain to
see. And walks
back along the game to the move that set the trap: while the human's moves on
the way were likely (Maia at least LIKELY), the whole line stays likely
(REACH), and the trap setter's moves in between were natural (within NATURAL
of Stockfish's best), the earliest of the trap setter's moves starts it. A
start is kept if, in practice, it is worth about as much as the best move
by Stockfish or more:

    P(reaching the trap) * (trap value for the setter)
        + (1 - P) * tanh(Stockfish(start) / 350)
            >=  tanh(Stockfish's best / 350) - PRACTICAL_TOLERANCE

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
DEEP_RATE = 0.5  # as OpponentDeepTactics
TRAP_MARGIN = 0.3
NATURAL_P = 0.3
BLUNDER_CP = 200
KEEP_PROBABILITY = 0.005  # moves stored with their probability and score
LIKELY = 0.4
REACH = 0.3
NATURAL = 100  # centipawns
MAX_SETTER_MOVES = 3
PRACTICAL_TOLERANCE = 0.05  # for Stockfish's noise at DEPTH
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
    """Each position: the likely moves (and the best and the played one) with
    Maia's probability and Stockfish's score."""
    records = []
    for ply, b in boards:
        sc, pr = scores[ply], probs[ply]
        if not sc:
            continue
        best = max(sc.values())
        played = game["moves"][ply] if ply < len(game["moves"]) else None
        keep = {m for m, p in pr.items() if p >= KEEP_PROBABILITY}
        keep |= {max(sc, key=sc.get)} | ({played} if played else set())
        records.append({"ply": ply, "best": best, "played": played,
                        "cp_played": sc.get(played) if played else None,
                        "moves": [[m, pr.get(m, 0.0), sc[m]] for m in keep if m in sc]})
    return records


def evaluate(record, rate=DEEP_RATE):
    """The position's objective value, trap value and blunder probability for
    the side to move, with the probabilities shifted toward Stockfish's best
    move (see DEEP_RATE)."""
    moves = record["moves"]
    best = record["best"]
    total = sum(p for _, p, _ in moves) or 1.0
    probs = {m: p / total for m, p, _ in moves}
    scores = {m: cp for m, _, cp in moves}
    best_move = max(scores, key=scores.get)
    caught = 0.0
    for m in probs:
        rate_m = rate * (1.0 - math.exp(-(best - scores[m]) / 300.0))
        caught += probs[m] * rate_m
        probs[m] *= 1.0 - rate_m
    probs[best_move] += caught
    natural, natural_p = max(((m, p / total) for m, p, _ in moves), key=lambda mp: mp[1])
    return {"objective": wl(best),
            "v_trap": sum(p * wl(scores[m]) for m, p in probs.items()),
            "natural": natural, "natural_p": natural_p, "natural_loss": best - scores[natural],
            "probs": probs, "scores": scores}


def is_trap(e):
    """The human's most natural move loses, and the trap value shows it."""
    return (e["natural_p"] >= NATURAL_P and e["natural_loss"] >= BLUNDER_CP
            and e["objective"] - e["v_trap"] >= TRAP_MARGIN)


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
    evals = {r["ply"]: evaluate(r) for r in game["records"]}
    starts = []
    for r in game["records"]:
        e = evals[r["ply"]]
        if not is_trap(e):
            continue
        t = r["ply"]  # the human to move at t is the one to be trapped
        setter_value = -e["v_trap"]  # for the trap setter
        start, reach = t - 1, 1.0
        # Walk back: the human's move at start - 1, the setter's at start - 2.
        k = 1
        while k < MAX_SETTER_MOVES and start - 2 >= 0:
            human, setter = by_ply.get(start - 1), by_ply.get(start - 2)
            if not human or not setter or human["played"] is None:
                break
            p_human = evals[start - 1]["probs"].get(human["played"], 0.0)
            setter_here = by_ply.get(start)
            natural = setter_here and setter_here["cp_played"] is not None and \
                setter_here["best"] - setter_here["cp_played"] <= NATURAL
            if p_human < LIKELY or reach * p_human < REACH or not natural:
                break
            reach *= p_human
            start -= 2
            k += 1
        s = by_ply.get(start)
        if not s or s["cp_played"] is None:
            continue
        practical = reach * setter_value + (1 - reach) * wl(s["cp_played"])
        if practical < wl(s["best"]) - PRACTICAL_TOLERANCE:
            continue
        starts.append({"start": start, "trap": t, "reach": reach, "practical": practical,
                       "objective_best": wl(s["best"]), "objective_start": wl(s["cp_played"]),
                       "trap_gap": e["objective"] - e["v_trap"]})
    # One start per starting move.
    unique = {}
    for s in starts:
        if s["start"] not in unique or s["practical"] > unique[s["start"]]["practical"]:
            unique[s["start"]] = s
    return list(unique.values())


def report(path):
    games = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    positions = sum(len(g["records"]) for g in games)
    traps = sum(1 for g in games for r in g["records"] if is_trap(evaluate(r)))
    starts = [(g, s) for g in games for s in find_starts(g)]
    longer = sum(1 for _, s in starts if s["trap"] - s["start"] > 1)
    lines = [f"{len(games)} games, {positions} positions analysed",
             f"Trap positions (Maia's most likely move, at least {100 * NATURAL_P:.0f}%, loses {BLUNDER_CP} cp or more,"
             f" and the trap value, shifted at {DEEP_RATE}, is at least {TRAP_MARGIN} below the objective value):"
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
                if ply > s["start"] and rec and (ply - s["start"]) % 2 == 1:
                    comment = f"{100 * evaluate(rec)['probs'].get(m, 0.0):.0f}%"
                node = node.add_variation(chess.Move.from_uci(m), comment=comment)
            trap = next(r for r in g["records"] if r["ply"] == s["trap"])
            e = evaluate(trap)
            likely = sorted(trap["moves"], key=lambda x: -x[1])[:5]
            total = sum(p for _, p, _ in trap["moves"]) or 1.0
            node.comment = node.comment + (
                f"the trap: objective {e['objective']:+.2f} for the side to move, trap value {e['v_trap']:+.2f};"
                f" Maia's likely moves and Stockfish's loss in pawns: "
                + ", ".join(f"{node.board().san(chess.Move.from_uci(m))} {100 * p / total:.0f}%"
                            f" ({(trap['best'] - cp) / 100:.1f})" for m, p, cp in likely))
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
