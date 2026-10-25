"""Runs the training loop: generations of games against the modeled
opponent, each followed by training the network that plays the next, as in
Lc0's own runs.

For each generation N, from the first not yet finished:
1. Check (from the second generation on): the network about to play it,
   net<N-1>, against the one that played the generation before, both
   evaluated greedily as in real games (generate_games.py --eval --greedy,
   EVAL_GAMES games each, kept apart from the generations). If the new
   network is clearly worse at mating quickly, by the thresholds below, the
   loop stops there, for a look before it goes on (rerun with --no-check to
   go on anyway). With 500 games, the median half-moves to mate is known
   within about 1.2 and the shares within about 2 points.
2. Games: generation N is played by the newest network, 791556 for the
   first (generate_games.py, GAMES games, with exploration).
3. Training: net<N> from the network that played generation N, on the newest
   generations (train_net.sh in WSL, STEPS steps at learning rate LR).

The generation games explore (temperature over the first 40 moves, noise),
which their medians include; the greedy evaluations measure how the networks
play. Every step resumes where it stopped: a rerun picks up the games, the
evaluations, the training's checkpoints and the finished networks. Ctrl+C
stops it. --report compares the generations and evaluations so far.

With --search om, the loop runs the opponent-model branch instead
(generate_games.py --search om), from scratch: the games and the
evaluations search with the opponent model, and the networks are
net<N> with generate_games.py's OM_SUFFIX (net<N>-om2).

Usage (from Windows, with the repository's .venv, after setup_wsl.sh; a
generation of 5,000 games takes 1.5 to 2.5 hours, its training about 5
minutes, an evaluation about 5):
    .venv\\Scripts\\python training\\loop.py --generations 10
    .venv\\Scripts\\python training\\loop.py --generations 10 --search om
"""

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_games as gg  # noqa: E402

HERE = Path(__file__).resolve().parent
STEPS = 400
LR = 0.0001
# The new network is clearly worse if, in greedy play, against the previous:
WORSE_MEDIAN = 3  # its median half-moves to mate is this many more,
WORSE_MATES = 0.04  # or its share of games won by mate this much lower,
WORSE_LOSSES = 0.02  # or its share of games lost this much higher.


def wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def stats(folder):
    games = gg.read_index(folder)
    mates = [int(g["plies"]) for g in games if g["result"] == "win" and g["ending"] == "mate"]
    return {"games": len(games), "mates": len(mates) / len(games),
            "losses": sum(g["result"] == "loss" for g in games) / len(games),
            "median": statistics.median(mates) if mates else float("inf")}


def describe(s):
    return (f"{s['games']} games, {s['mates']:.1%} won by mate, {s['losses']:.1%} lost, "
            f"median {s['median']:.0f} half-moves to mate")


def evaluate(net, parallelism, search):
    """Plays `net`'s greedy evaluation; False if stopped before the end."""
    folder = gg.eval_dir(net, greedy=True, search=search)
    gg.generate(folder, net, gg.EVAL_GAMES, parallelism, greedy=True, search=search)
    return len(gg.read_index(folder)) >= gg.EVAL_GAMES


def check(before_net, now_net, search):
    """Whether `now_net` is not clearly worse than `before_net` in their
    greedy evaluations; prints the comparison."""
    before = stats(gg.eval_dir(before_net, greedy=True, search=search))
    now = stats(gg.eval_dir(now_net, greedy=True, search=search))
    print(f"{gg.net_name(before_net)}, greedy: {describe(before)}")
    print(f"{gg.net_name(now_net)}, greedy: {describe(now)}")
    problems = []
    if now["median"] > before["median"] + WORSE_MEDIAN:
        problems.append("slower mates")
    if now["mates"] < before["mates"] - WORSE_MATES:
        problems.append("fewer mates")
    if now["losses"] > before["losses"] + WORSE_LOSSES:
        problems.append("more losses")
    if problems:
        print(f"{gg.net_name(now_net)} looks worse ({', '.join(problems)}): stopping here. "
              f"Rerun with --no-check to go on anyway.")
    return not problems


def play(generation, parallelism, search):
    """Plays generation N; False if stopped before the end."""
    folder = gg.generation_dir(generation, search)
    gg.generate(folder, gg.default_net(generation, search), gg.GAMES, parallelism, search=search)
    return len(gg.read_index(folder)) >= gg.GAMES


def trained_net(generation, search):
    return gg.NETS / (f"net{generation:02d}" + (gg.OM_SUFFIX if search == "om" else "") + ".pb.gz")


def train(generation, steps, lr, search):
    net = trained_net(generation, search)
    if net.exists():
        return True
    command = ["wsl.exe", "-d", "Ubuntu", "--", "env", f"SEARCH={search}", "bash",
               wsl_path(HERE / "train_net.sh"), str(generation), str(steps), str(lr)]
    try:
        result = subprocess.run(command)
    except KeyboardInterrupt:
        print("\nStopped; rerun to continue.")
        return False
    return result.returncode == 0 and net.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generations", type=int, default=10, help="last generation to play")
    parser.add_argument("--parallelism", type=int, default=16, help="games played at once")
    parser.add_argument("--steps", type=int, default=STEPS, help="training steps per network")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--no-check", action="store_true", help="go on even if a network looks worse")
    parser.add_argument("--search", choices=("plain", "om"), default="plain",
                        help="the engine's search: plain Lc0's, or with the opponent model, "
                             "on a branch of its own (see generate_games.py)")
    parser.add_argument("--report", action="store_true", help="only compare the generations so far")
    args = parser.parse_args()
    if args.report:
        gg.report()
        return
    search = args.search
    for generation in range(1, args.generations + 1):
        if len(gg.read_index(gg.generation_dir(generation, search))) < gg.GAMES:
            if generation > 1 and not args.no_check:
                before, now = gg.default_net(generation - 1, search), gg.default_net(generation, search)
                if not (evaluate(before, args.parallelism, search) and evaluate(now, args.parallelism, search)):
                    return
                if not check(before, now, search):
                    return
            if not play(generation, args.parallelism, search):
                return
        if not train(generation, args.steps, args.lr, search):
            print(f"Training {trained_net(generation, search).name} did not finish; rerun to continue.")
            return
    last = trained_net(args.generations, search)
    if evaluate(last, args.parallelism, search):
        check(gg.default_net(args.generations, search), last, search)
    print(f"Generation {args.generations} is done; its games trained {last}.")


if __name__ == "__main__":
    main()
