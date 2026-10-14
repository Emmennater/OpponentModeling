"""Checks lczero-training's data loader with the opponent-modeling options
(see lczero-training.patch) on finished games of a generation, for the
opponent's records (0 visits) handled one of two ways:

- skip (skip_unsearched): the loader yields exactly the engine's records;
- blank (blank_unsearched_policy): it yields every record, the opponent's
  with no policy target (every move -1) and the engine's with theirs.

Either way every position's value target must be the game's result for the
side to move, discounted by value_discount ** plies_left, the rest going to
the draw. Runs on the CPU, so it can run while games are generated.

Usage, from Windows:
    wsl -d Ubuntu -- ~/.venvs/lc0-training/bin/python training/check_loader.py 1 0.995 blank
"""

import gzip
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lczero-training" / "tf"))
from chunkparser import ChunkParser  # noqa: E402

generation, discount, mode = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
folder = HERE / "data" / f"gen{generation:02d}"
lines = (folder / "games.tsv").read_text().splitlines()[1:]
games = [dict(zip(["engine_color", "result", "ending", "plies", "file", "moves"], l.split("\t"))) for l in lines]
parser = ChunkParser([], 1, sample=1, workers=0, value_discount=discount,
                     skip_unsearched=mode == "skip", blank_unsearched_policy=mode == "blank")
inner = parser.inner
records = yielded = wrong = opponent_positions = 0
for g in games:
    path = folder / g["file"]
    data = gzip.open(path).read()
    visits = [struct.unpack_from("I", data, i + 8340)[0] for i in range(0, len(data), 8356)]
    records += len(visits)
    out = list(inner.single_file_gen(str(path)))
    yielded += len(out)
    expected_count = sum(v > 0 for v in visits) if mode == "skip" else len(visits)
    wrong += len(out) != expected_count
    for rec in out:
        _, probs, winner, _, plies = inner.convert_v6_to_tuple(rec)
        w, d, l = struct.unpack("fff", winner)
        n = struct.unpack("f", plies)[0]
        result_q = struct.unpack_from("f", rec, 8308)[0]
        kept = discount ** n
        expected = (kept, 1 - kept, 0) if result_q > 0 else (0, 1 - kept, kept) if result_q < 0 else (0, 1, 0)
        if max(abs(a - b) for a, b in zip((w, d, l), expected)) > 1e-5:
            wrong += 1
        policy = struct.unpack("1858f", probs)
        if struct.unpack_from("I", rec, 8340)[0] == 0:
            opponent_positions += 1
            wrong += any(p != -1.0 for p in policy)
        else:
            wrong += policy != struct.unpack_from("1858f", rec, 8)
print(f"{len(games)} games, {records} records, {yielded} yielded "
      f"({opponent_positions} the opponent's), {wrong} games or positions wrong")
