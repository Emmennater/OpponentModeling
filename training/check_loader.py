"""Checks lczero-training's data loader with the opponent-modeling options
(see lczero-training.patch) on finished games of a generation: with
skip_unsearched it yields exactly the engine's positions (visits above 0),
and with value_discount every position's value target is the game's result
discounted by value_discount ** plies_left, the rest going to the draw.
Runs on the CPU, so it can run while games are generated.

Usage, from Windows:
    wsl -d Ubuntu -- ~/.venvs/lc0-training/bin/python training/check_loader.py 1 0.995
"""

import gzip
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lczero-training" / "tf"))
from chunkparser import ChunkParser  # noqa: E402

generation, discount = int(sys.argv[1]), float(sys.argv[2])
folder = HERE / "data" / f"gen{generation:02d}"
lines = (folder / "games.tsv").read_text().splitlines()[1:]
games = [dict(zip(["engine_color", "result", "ending", "plies", "file", "moves"], l.split("\t"))) for l in lines]
parser = ChunkParser([], 1, sample=1, workers=0, value_discount=discount, skip_unsearched=True)
inner = parser.inner
records = yielded = wrong = 0
for g in games:
    path = folder / g["file"]
    data = gzip.open(path).read()
    visits = [struct.unpack_from("I", data, i + 8340)[0] for i in range(0, len(data), 8356)]
    records += len(visits)
    out = list(inner.single_file_gen(str(path)))
    yielded += len(out)
    wrong += len(out) != sum(v > 0 for v in visits)
    for rec in out:
        _, _, winner, _, plies = inner.convert_v6_to_tuple(rec)
        w, d, l = struct.unpack("fff", winner)
        n = struct.unpack("f", plies)[0]
        result_q = struct.unpack_from("f", rec, 8308)[0]
        kept = discount ** n
        expected = (kept, 1 - kept, 0) if result_q > 0 else (0, 1 - kept, kept) if result_q < 0 else (0, 1, 0)
        if max(abs(a - b) for a, b in zip((w, d, l), expected)) > 1e-5:
            wrong += 1
print(f"{len(games)} games, {records} records, {yielded} yielded (the engine's), "
      f"{wrong} games or positions wrong")
last = convert = inner.convert_v6_to_tuple(list(inner.single_file_gen(str(folder / games[0]["file"])))[0])
w, d, l = struct.unpack("fff", last[2])
print(f"First position of game 1 ({games[0]['result']} in {games[0]['plies']} half-moves): "
      f"plies_left {struct.unpack('f', last[4])[0]:.0f}, target W {w:.3f} D {d:.3f} L {l:.3f}")
