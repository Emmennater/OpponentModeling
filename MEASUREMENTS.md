# Measuring the Opponent Models Against Real Games
[tools/measure_model.py](tools/measure_model.py) replays games from a file of Lichess games and compares the model with the moves actually played. It needs Python with `python-chess`. Each line of the games file holds the white rating, black rating, result and the moves in SAN, separated by tabs. For example:

```
python tools/measure_model.py lichess_games.tsv --models stockfish,maia --games 300
python tools/measure_model.py lichess_games.tsv --models maia,maia3 --calibrate --truth-depth 4 --games 300
```

For each position, it records the rank of the move that was played among the model's moves, and whether it was the model's top move (its predicted move). For this engine's model (`--model stockfish`), it also records whether the model kept the played move, and the move's centipawn loss at low depth. From the loss, it reports which share of played moves each threshold would keep.

With `--model maia`, it measures the [Maia](https://github.com/CSSLab/maia-chess) networks instead. These networks are trained on Lichess games to predict the move a human of a given rating would play. Each position uses the network for the mover's rating (Maia-1100 to Maia-1900). This needs Lc0 in `lc0/bin` and the networks in `lc0/nets`; both are downloaded separately and not kept in git. The measurements below used the Lc0 v0.32.1 Windows CUDA 12 release.

First results, on 300 games per rating band, for positions from ply 6 to 15. The games file only has the first 16 plies of each game, so these are opening positions. Opponent Depth 2:

| Mover's rating | Kept at 200 cp | Predicted move played | Kept at 20 cp | Kept at 100 cp | Share of legal moves kept |
|---|---|---|---|---|---|
| Below 1200 | 84.7% | 27.0% | 41.6% | 71.4% | 51% |
| 1400–1599 | 94.8% | 34.7% | 56.4% | 87.7% | 57% |
| 1600–1799 | 96.7% | 35.9% | 59.6% | 91.0% | 60% |
| 2200 and above | 99.3% | 51.2% | 74.1% | 95.9% | 61% |
| All | 95.5% | 37.2% | 59.4% | 89.0% | 58% |

- **The 200 cp threshold keeps about 95% of the moves actually played.** It removes about 42% of the legal moves, and those are almost never played. Players below 1200 play a move more than 200 cp worse about 15% of the time.
- **A threshold of 20 cp would miss about 40% of the moves actually played.** The opponent would often play a move the engine never considered.
- **Players play the predicted move only about 37% of the time**, from 27% below 1200 to 51% at 2200 and above. Most of the time they play some other reasonable move.
- **Opponent Depth makes little difference.** Depths 1, 2 and 3 predict the played move about equally well (37.2%, 37.2% and 38.7% overall), so in the opening, depth 1 is as good a model as depth 2, at a fraction of the cost.

## Comparison with Maia
On the same 20,736 positions, Maia predicts the played move much better than this engine's model, in every rating band:

| Mover's rating | Maia top 1 | This model's top 1 (depth 2) | Maia top 3 | This model's top 3 | Maia top 5 | This model's top 5 |
|---|---|---|---|---|---|---|
| Below 1200 | 39.3% | 27.0% | 68.8% | 50.8% | 80.7% | 63.6% |
| 1400–1599 | 48.0% | 34.7% | 77.8% | 63.7% | 87.8% | 75.8% |
| 1600–1799 | 48.4% | 35.9% | 77.8% | 64.0% | 88.5% | 76.5% |
| 2200 and above | 55.0% | 51.2% | 82.2% | 79.5% | 90.5% | 88.4% |
| All | 48.8% | 37.2% | 78.0% | 65.6% | 88.0% | 77.4% |

The gap is largest for weaker players, which is where an opponent model matters most. Maia also gives each move a probability: on average it gave the played move 35.8%. That is the kind of input an expected-value search needs.

Two caveats. Maia was trained on Lichess games, and some of the games in this file may have been in its training data, which would flatter its results. And these are opening positions only.

## Maia-3
`--models maia3` measures [Maia-3](https://github.com/CSSLab/maia3), a transformer that takes both players' ratings as input and covers ratings from 600 to 3000 in one network. It runs in PyTorch, through the `Maia3Net` wrapper from the StockfishTrapFinder project (`--maia3-path`), so the tool must then run under a Python environment with PyTorch and the `maia3` package. On the same positions, all three sizes predict better than Maia-1:

| Model | Top 1, all | Top 1, 1200–1399 | Top 3, all | Log-likelihood, all |
|---|---|---|---|---|
| Maia-1 | 48.8% | 46.1% | 78.0% | −1.61 |
| Maia3-5M | 51.4% | 47.4% | 80.2% | −1.51 |
| Maia3-23M | 52.3% | 49.0% | 80.6% | −1.47 |
| Maia3-79M | 52.6% | 48.9% | 80.7% | −1.46 |

The 23M network is practically as good as the 79M one.

## Calibrating Maia with a low-depth evaluation
`--calibrate` also measures each Maia model with the reweighting from StockfishTrapFinder's `MaiaModel`: moves below 2% probability are dropped, the evaluation's best move is added, and each move gives up `0.7 × (1 − exp(−loss / 200))` of its probability to the evaluation's best move, where `loss` is how far its depth-2 score is below that move's. The aim is to stop the opponent model from counting on blunders a quick look would catch.

It makes the predictions slightly worse: Maia-3's top move matches the played move 50.9% of the time instead of 52.6%, and its log-likelihood drops from −1.46 to −1.91. Most of the log-likelihood drop comes from the 2% cut, which removes moves that are sometimes played. Without it (`--min-prob 0`) the log-likelihood is −1.50.

The more relevant question is whether the models put the right amount of probability on blunders. `--truth-depth 4` compares the probability each model puts on moves losing at least L centipawns, measured at depth 4, with how often players actually played such a move. For players rated 1200 to 1399:

| Model | L = 100 | L = 200 | L = 300 | L = 500 |
|---|---|---|---|---|
| Actually played | 13.6% | 5.2% | 3.0% | 1.0% |
| Maia-1 | 16.1% | 7.1% | 4.1% | 1.9% |
| Maia-1, calibrated | 9.1% | 3.1% | 1.4% | 0.5% |
| Maia3-79M | 14.8% | 6.3% | 3.7% | 1.6% |
| Maia3-79M, calibrated | 8.6% | 2.8% | 1.3% | 0.4% |

Raw Maia slightly overestimates how often players at this level blunder, and Maia-3 less than Maia-1. With these settings, the calibration overcorrects: it predicts less than half the blunders players actually make. A milder setting may land in between.

## Tuning the calibration
[tools/tune_calibration.py](tools/tune_calibration.py) collects the slow inputs once and saves them:
- Maia3-23M's probabilities;
- this engine's scores of every legal move at each calibration depth;
- for the blunder check, the score of every move from a normal Stockfish search (MultiPV) at depth 10.

It then scores a grid of settings from the saved data in seconds:

```
python tools/tune_calibration.py lichess_games.tsv --collect cal.pkl --games 500 --depths 1,2,3,4
python tools/tune_calibration.py --data cal.pkl --depths 1,2,3,4 --temperatures 0.9,1.0 --scales 200,800,3200 --rates 0.2,0.5,0.7
```

Collecting took about 4 minutes for 2,060 positions of players rated 1200–1399. Most of that was the depth-4 scores and the depth-10 truth scores. The results, losses measured at depth 10:

| Model | Log-likelihood | Top 1 | L = 100 | L = 200 | L = 300 | L = 500 |
|---|---|---|---|---|---|---|
| Actually played | | | 12.9% | 5.6% | 3.2% | 1.7% |
| Maia3-23M | −1.537 | 49.5% | 15.1% | 6.4% | 4.0% | 2.1% |
| Maia3-23M, temperature 0.9 | −1.544 | 49.5% | 13.9% | 5.7% | 3.5% | 1.8% |
| Calibrated: depth 1, scale 200, rate 0.3 | −1.543 | 49.9% | 13.4% | 5.3% | 3.2% | 1.6% |
| Calibrated: depth 2, scale 3200, rate 0.2, temperature 0.9 | −1.544 | 49.8% | 13.8% | 5.6% | 3.4% | 1.7% |
| Calibrated: depth 2, scale 200, rate 0.7, 2% cut (the old default) | −1.994 | 48.0% | 9.4% | 3.1% | 1.6% | 0.7% |

- **Raw Maia-3 overestimates blunders by about 20%** at every loss level. At L = 100 the gap is about three standard errors.
- **A mild calibration corrects it,** with a rate of 0.2 to 0.3, or with a large scale_cp. So does a temperature of 0.9 alone.
- **The old default removes about half the blunders players actually make.** Most of its drop in log-likelihood comes from the 2% cut.
- **The calibration depth hardly matters.** Depth 1 does as well as depth 4.
- **No setting predicts the played move noticeably better than raw Maia-3.** The differences in log-likelihood (±0.01) and top 1 (±0.5%) are within the noise of this sample.

These are opening positions only (plies 6–15), since the games file stops at ply 16. The full games below change the conclusion.

### Full games
[tools/fetch_lichess_games.py](tools/fetch_lichess_games.py) streams a monthly file from the Lichess database. It decompresses the file as it downloads and stops once it has enough games, so it downloads only a small part of the 30 GB file. It writes full games in the format of `lichess_games.tsv`.

The run used 3,000 rated blitz and rapid games from August 2026 with both players rated 1200–1399. That was 18 MB of the file, fetched in 4 seconds. From each game, `--per-phase 1` sampled one position from each phase that the game reached:
- opening: before ply 16;
- middlegame: plies 16–59;
- late: ply 60 on.

That gave 7,451 positions, and collecting their data took 10 minutes.

```
python tools/fetch_lichess_games.py full_games.tsv --month 2026-08 --games 3000
python tools/tune_calibration.py full_games.tsv --collect full.pkl --games 3000 --per-phase 1 --depths 1,2,3,4
python tools/tune_calibration.py --data full.pkl --phase middlegame
```

Probability on moves losing at least L centipawns (depth 10), against how often they were played:

| Phase | Model | Log-likelihood | L = 100 | L = 200 | L = 300 | L = 500 |
|---|---|---|---|---|---|---|
| Opening | Actually played | | 14.5% | 6.2% | 3.9% | 2.1% |
| | Maia3-23M | −1.533 | 15.3% | 6.5% | 3.9% | 2.2% |
| | Maia3-23M, temperature 0.9 | −1.540 | 14.1% | 5.9% | 3.4% | 1.9% |
| Middlegame | Actually played | | 29.3% | 15.7% | 10.7% | 5.6% |
| | Maia3-23M | −1.448 | 30.1% | 16.3% | 11.0% | 5.8% |
| | Maia3-23M, temperature 0.9 | −1.457 | 28.5% | 15.1% | 10.0% | 5.2% |
| Late | Actually played | | 25.8% | 16.8% | 13.4% | 9.9% |
| | Maia3-23M | −1.279 | 26.2% | 16.8% | 13.4% | 9.8% |
| | Maia3-23M, temperature 0.9 | −1.293 | 25.0% | 15.8% | 12.6% | 9.2% |

- **Blunders are about twice as common after the opening.** Moves losing 100 cp or more: 14.5% in the opening, 29.3% in the middlegame.
- **Raw Maia-3 matches the real blunder rates in every phase.** It is within 1 percentage point at every level, and closest late in the game.
- **A temperature of 0.9 makes the fit worse after the opening.** It underestimates blunders there, and it lowers the log-likelihood in every phase.
- **The calibration step adds nothing.** The best calibrated settings (rate 0.2–0.3, scale 3200) match the rates only marginally more closely. They lower the log-likelihood slightly, at any depth.
- **The earlier opening sample was different.** There, raw Maia-3 overestimated blunders by about 20%. That file's filtering differed: each game was assigned to a band by the players' average rating, and its time controls are unknown. Its opening blunder rate was also lower (12.9% at L = 100 against 14.5%).

So Lc0 should use Maia-3 as it is, with the default `OpponentModelTemperature` of 1.0 and no calibration.

### Tactics
On the same full-game positions, the move ahead of the 2nd-best by a clear margin at depth 10, for example a tactical shot, against how often real players found it and the probability Maia3-23M gave it:

| Best move's lead over the 2nd-best | Positions | Players found it | Maia-3's probability |
|---|---|---|---|
| 0–50 cp | 5,123 | 25.7% | 24.6% |
| 50–100 cp | 880 | 38.5% | 37.3% |
| 100–200 cp | 609 | 48.4% | 48.5% |
| 200–300 cp | 208 | 53.8% | 56.0% |
| 300–500 cp | 216 | 61.6% | 64.3% |

Maia-3 finds clear tactical shots about as often as players rated 1200–1399 do. Even so, the engine relies on the opponent missing them, which is what `OpponentEngineMix` in [LC0_PLAN.md](LC0_PLAN.md) is for.
