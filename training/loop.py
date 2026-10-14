"""Runs the training loop: generations of games against the modeled
opponent, each followed by training the network that plays the next, as in
Lc0's own runs.

For each generation N, from the first not yet finished:
1. Games: generation N is played by the newest network, 791556 for the
   first (generate_games.py, GAMES games).
2. Check: generation N is the first test of the network that plays it. After
   its first CHECK_GAMES games, they are compared with the previous
   generation, played by the previous network. If the new network is clearly
   worse at what it is trained for, by the thresholds below, the loop stops
   there, for a look before it goes on (rerun with --no-check to go on
   anyway). With 500 games, the median half-moves to mate is known within
   about 1.2 and the shares within about 2 points.
3. Training: net<N> from the network that played generation N, on the newest
   generations (train_net.sh in WSL, STEPS steps at learning rate LR).

Every step resumes where it stopped: a rerun picks up the games, the
training's checkpoints and the finished networks. Ctrl+C stops it.
--report compares the generations so far.

Usage (from Windows, with the repository's .venv, after setup_wsl.sh; a
generation of 5,000 games takes 1.5 to 2.5 hours, its training about 5
minutes):
    .venv\\Scripts\\python training\\loop.py --generations 10
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
CHECK_GAMES = 500
# The new network is clearly worse if, against the previous generation:
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


def check(generation):
    """Whether generation N's network is not clearly worse than generation
    N-1's, from N's first games; prints the comparison."""
    before, now = stats(gg.generation_dir(generation - 1)), stats(gg.generation_dir(generation))
    print(f"Generation {generation - 1}: {describe(before)}")
    print(f"Generation {generation}: {describe(now)}")
    problems = []
    if now["median"] > before["median"] + WORSE_MEDIAN:
        problems.append("slower mates")
    if now["mates"] < before["mates"] - WORSE_MATES:
        problems.append("fewer mates")
    if now["losses"] > before["losses"] + WORSE_LOSSES:
        problems.append("more losses")
    if problems:
        print(f"The new network looks worse ({', '.join(problems)}): stopping here. "
              f"Rerun with --no-check to go on anyway.")
    return not problems


def play(generation, target, parallelism):
    """Plays generation N up to `target` games; False if stopped before."""
    folder = gg.generation_dir(generation)
    gg.generate(folder, gg.default_net(generation), target, parallelism)
    return len(gg.read_index(folder)) >= target


def train(generation, steps, lr):
    net = gg.NETS / f"net{generation:02d}.pb.gz"
    if net.exists():
        return True
    command = ["wsl.exe", "-d", "Ubuntu", "--", "bash", wsl_path(HERE / "train_net.sh"),
               str(generation), str(steps), str(lr)]
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
    parser.add_argument("--report", action="store_true", help="only compare the generations so far")
    args = parser.parse_args()
    if args.report:
        gg.report()
        return
    for generation in range(1, args.generations + 1):
        folder = gg.generation_dir(generation)
        if len(gg.read_index(folder)) < gg.GAMES:
            if generation > 1 and not args.no_check:
                if not play(generation, CHECK_GAMES, args.parallelism):
                    return
                if not check(generation):
                    return
            if not play(generation, gg.GAMES, args.parallelism):
                return
        if not train(generation, args.steps, args.lr):
            print(f"Training net{generation:02d} did not finish; rerun to continue.")
            return
    print(f"Generation {args.generations} is done; its games trained "
          f"training/nets/net{args.generations:02d}.pb.gz.")


if __name__ == "__main__":
    main()
