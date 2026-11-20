# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raina027/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raina027/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at the case target, as five quantiles (5/25/50/75/95%) per trophy. The rate is a share of tracked players, and a share saturates: each trophy's share is fitted with a saturating logistic in log-time, shrunk toward the last read by how much better the logistic explains the recent data than a flat line. Players at the target come from an ensemble of a log-log fit, a damped recent-growth extrapolation and a saturating adoption curve; the players forecast only widens the spread, never moves a median. Standard library only, deterministic, about 0.1 s per case.

On synthetic final-regime backtests (two days of history with growth and refresh dilution, a 4.5-day horizon, known answers) the model beats the house persistence baseline's pinball loss by about a quarter.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.
