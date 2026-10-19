"""Plays engines against opponents, as a config file sets them, and compares
how quickly each engine mates each opponent.

It is generate_games.py's evaluation (Lc0's self-play, the engine against
Maia-3 with the opponent model) for every engine and opponent the config
(training/match.yaml by default, which describes the format) names: an
engine is a network, its visits a move and Lc0 settings, an opponent its
Maia-3 rating and closer look. Anything not given is as in generate_games.py.
An engine's policy-net and moves-left-net take those heads from other
networks (Lc0's PolicyWeights and MovesLeftWeights), to test which of a
network's heads matter.

Note that self-play's engine searches as plain Lc0: the opponent model only
picks the opponent's moves, so the search's opponent-model options
(OpponentCaution and the like) have no effect here and are refused.

Each pairing's games go to training/data/matches/<config>/<engine>-vs-<opponent>/,
as in generate_games.py (games.tsv, the training data and lc0.log), with its
settings in settings.json. A rerun resumes: finished pairings are skipped and
an unfinished one goes on; Ctrl+C stops. At the end, and with --report, a
table per opponent compares the engines, saved to summary.txt in the
config's folder.

Usage (from Windows, with the repository's .venv; 2,000 to 3,500 games an
hour on an RTX 4080 at parallelism 16, so match.yaml's 6 pairings of 500
games take about 1 to 1.5 hours):
    .venv\\Scripts\\python training\\match.py
    .venv\\Scripts\\python training\\match.py training\\my_test.yaml
    .venv\\Scripts\\python training\\match.py --report
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_games as gg  # noqa: E402

HERE = Path(__file__).resolve().parent
MATCHES = gg.DATA / "matches"
TOP_KEYS = {"games", "parallelism", "engine", "engines", "opponent", "opponents"}
HEAD_NETS = {"policy-net": "policy-weights", "moves-left-net": "moves-left-weights"}
ENGINE_KEYS = {"net", *HEAD_NETS, "visits", "greedy"}  # the rest of an engine's keys are Lc0 settings
NAME = re.compile(r"[A-Za-z0-9._+-]+")


def flag_value(value):
    """YAML's true and false as Lc0 writes them."""
    return str(value).lower() if isinstance(value, bool) else value


def resolve(path, key, value):
    """A network path from the config, relative to the repository."""
    net = Path(value)
    net = (net if net.is_absolute() else gg.ROOT / net).resolve()
    if not net.exists():
        sys.exit(f"{path}: {key} {net} does not exist")
    return net


def load_config(path):
    """The config's pairings: (engine name, opponent name, settings) each,
    with the settings in the form stored in settings.json."""
    config = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(config) - TOP_KEYS
    if unknown:
        sys.exit(f"{path}: unknown keys {', '.join(sorted(unknown))} (expected {', '.join(sorted(TOP_KEYS))})")
    for key in ("engines", "opponents"):
        if not config.get(key):
            sys.exit(f"{path}: no {key}")
        for name in config[key]:
            if not NAME.fullmatch(str(name)):
                sys.exit(f"{path}: {key[:-1]} name {name!r} must be letters, digits, '.', '_', '+' or '-'")

    engines = {}
    for name, own in config["engines"].items():
        merged = {"visits": gg.VISITS, "greedy": True, **(config.get("engine") or {}), **(own or {})}
        if "net" not in merged:
            sys.exit(f"{path}: engine {name} has no net")
        net = resolve(path, f"engine {name}'s network", merged["net"])
        settings = {k: flag_value(v) for k, v in merged.items() if k not in ENGINE_KEYS}
        for key, option in HEAD_NETS.items():
            if key not in merged:
                continue
            head_net = resolve(path, f"engine {name}'s {key}", merged[key])
            settings[option] = str(head_net)
            # All the networks get the identity plane if any was trained with
            # it: 791556's values move by at most 0.004 with it.
            if gg.uses_identity_plane(head_net) and not gg.uses_identity_plane(net):
                settings["player1.identity-plane"] = "true"
        ignored = [k for k in settings if k.startswith("opponent-")]
        if ignored:
            sys.exit(f"{path}: engine {name}: {', '.join(ignored)} would do nothing, since self-play's "
                     f"engine searches without the opponent model; set opponents' settings under opponent(s)")
        engines[name] = {"net": str(net), "visits": int(merged["visits"]),
                         "greedy": bool(merged["greedy"]), "settings": settings}
    opponents = {name: {k: flag_value(v) for k, v in {**(config.get("opponent") or {}), **(own or {})}.items()}
                 for name, own in config["opponents"].items()}
    return {"games": int(config.get("games", gg.EVAL_GAMES)),
            "parallelism": int(config.get("parallelism", 16)),
            "engines": engines, "opponents": opponents}


def recorded(engine, opponent):
    """What settings.json keeps of a pairing: all its settings, defaults
    included, so that a change to generate_games.py's counts too."""
    data = {**engine, "engine settings": {**gg.ENGINE, **engine["settings"]},
            "opponent settings": {**gg.OPPONENT, **opponent}}
    del data["settings"]
    return json.loads(json.dumps(data, default=str))


def folder_of(config_path, engine_name, opponent_name):
    return MATCHES / Path(config_path).stem / f"{engine_name}-vs-{opponent_name}"


def check_settings(folder, settings):
    """Records a pairing's settings, or stops if it was played with others."""
    path = folder / "settings.json"
    if path.exists():
        before = json.loads(path.read_text())
        if before != settings:
            changed = sorted(_differences(before, settings))
            sys.exit(f"{folder.name} was played with other settings ({', '.join(changed)}): rename the engine "
                     f"or opponent in the config to play it afresh, or remove {folder}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")


def _differences(before, now, prefix=""):
    for key in sorted(set(before) | set(now)):
        a, b = before.get(key), now.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            yield from _differences(a, b, f"{key}: ")
        elif a != b:
            yield f"{prefix}{key} {a} -> {b}"


def stats(games):
    mates = [int(g["plies"]) for g in games if g["result"] == "win" and g["ending"] == "mate"]
    share = lambda n: n / len(games)  # noqa: E731
    row = {"games": len(games),
           "won": share(sum(g["result"] == "win" for g in games)),
           "drawn": share(sum(g["result"] == "draw" for g in games)),
           "lost": share(sum(g["result"] == "loss" for g in games)),
           "mates": share(len(mates))}
    if len(mates) >= 2:
        row["q1"], row["median"], row["q3"] = statistics.quantiles(mates, n=4)
    elif mates:
        row["q1"] = row["median"] = row["q3"] = mates[0]
    for color in ("white", "black"):
        own = [int(g["plies"]) for g in games if g["engine_color"] == color
               and g["result"] == "win" and g["ending"] == "mate"]
        row[color] = statistics.median(own) if own else None
    for limit in (40, 60, 80):
        row[limit] = share(sum(p <= limit for p in mates))
    return row


def describe(settings):
    return ", ".join(f"{k} {v}" for k, v in settings.items()) or "as in generate_games.py"


def report(config_path, config):
    lines = [f"{Path(config_path).name}: {config['games']} games per pairing", "Engines:"]
    for name, e in config["engines"].items():
        settings = dict(e["settings"])
        settings.pop("player1.identity-plane", None)
        heads = [f"{gg.net_name(settings.pop(option))}'s {key.removesuffix('-net').replace('-', ' ')}"
                 for key, option in HEAD_NETS.items() if option in settings]
        net = gg.net_name(e["net"]) + (f" with {' and '.join(heads)}" if heads else "")
        lines.append(f"  {name}: {net}, {e['visits']} visits, "
                     f"{'greedy' if e['greedy'] else 'exploring'}; {describe(settings)}")
    width = max(len("Engine"), *(len(n) for n in config["engines"]))
    for opponent, settings in config["opponents"].items():
        lines += ["", f"Against {opponent} ({describe(settings)}):",
                  f"  {'Engine':<{width}}  {'games':>5}  {'won':>6} {'drawn':>6} {'lost':>6}  {'mates':>6}"
                  f"  {'median (quartiles)':>18}  {'W':>3} {'B':>3}  {'<=40':>5} {'<=60':>5} {'<=80':>5}"]
        for engine in config["engines"]:
            games = gg.read_index(folder_of(config_path, engine, opponent))
            if not games:
                lines.append(f"  {engine:<{width}}  {0:>5}  not played yet")
                continue
            s = stats(games)
            spread = (f"{s['median']:.0f} ({s['q1']:.0f}-{s['q3']:.0f})" if "median" in s else "-")
            by_color = " ".join(f"{s[c]:>3.0f}" if s[c] is not None else "  -" for c in ("white", "black"))
            lines.append(f"  {engine:<{width}}  {s['games']:>5}  {s['won']:>6.1%} {s['drawn']:>6.1%} "
                         f"{s['lost']:>6.1%}  {s['mates']:>6.1%}  {spread:>18}  {by_color}  "
                         f"{s[40]:>5.0%} {s[60]:>5.0%} {s[80]:>5.0%}")
    lines += ["", "(won, drawn, lost: shares of the engine's games; mates: share won by mate; median: half-moves",
              " to mate in those, overall and with the engine as White and Black; <=N: share of all games won",
              " by mate within N half-moves)"]
    print("\n".join(lines))
    summary = MATCHES / Path(config_path).stem / "summary.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {summary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", nargs="?", default=str(HERE / "match.yaml"), help="the config file")
    parser.add_argument("--report", action="store_true", help="only compare the games so far")
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.report:
        pairings = [(e, o) for e in config["engines"] for o in config["opponents"]]
        left = sum(max(0, config["games"] - len(gg.read_index(folder_of(args.config, e, o))))
                   for e, o in pairings)
        for engine_name, opponent_name in pairings:
            check_settings(folder_of(args.config, engine_name, opponent_name),
                           recorded(config["engines"][engine_name], config["opponents"][opponent_name]))
        print(f"{len(pairings)} pairings of {config['games']} games; {left} games to play")
        for i, (engine_name, opponent_name) in enumerate(pairings, 1):
            engine, opponent = config["engines"][engine_name], config["opponents"][opponent_name]
            folder = folder_of(args.config, engine_name, opponent_name)
            print(f"\n[{i}/{len(pairings)}] {engine_name} against {opponent_name}")
            status = gg.generate(folder, engine["net"], config["games"], config["parallelism"],
                                 greedy=engine["greedy"], overrides=engine["settings"], opponent=opponent,
                                 visits=engine["visits"], summary=False)
            if status != "done":
                return
        print()
    report(args.config, config)


if __name__ == "__main__":
    main()
