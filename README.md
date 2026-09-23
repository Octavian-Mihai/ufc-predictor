# Local UFC Fight Predictor

Fully local Python app: official [UFC Stats](http://ufcstats.com/) career features, a calibrated scikit-learn win model, optional free [The Odds API](https://the-odds-api.com/) moneylines, and a Streamlit dashboard that ranks **value underdogs**.

No OpenAI, no cloud GPU, no paid hosting. The only optional account is a **free** Odds API key (500 credits/month). One cached `regions=us&markets=h2h` pull costs **1 credit**.

This is an **entertainment / analysis** tool, **not betting advice**. Fight outcomes are uncertain. A well-built UFC stats model typically lands around the mid-60s% on winner accuracy — an edge finder, not a lock machine.

## Screenshots

**Upcoming card** — model P(A)/P(B), market odds, edge, and EV per bout:

![Upcoming card dashboard](docs/screenshots/dashboard.png)

**Last 5 events** — recap grading the trained model against actual results:

![Last 5 events recap](docs/screenshots/last_events.png)

## What the model uses

UFC Stats does not publish “jabs” as a named column. Closest official fields, labeled honestly in the UI:

| Style | UFC Stats fields |
| --- | --- |
| Takedowns | TD landed/attempted, accuracy, opponent TD defense |
| Kicks | Significant strikes to the **leg** |
| Jabs | Significant strikes at **distance** (jab/boxing-volume proxy) |
| Under pressure | Strikes absorbed per minute, strike defense, control time against |
| Attacking | SLpM, control time for, knockdowns, ground strikes |

Plus age, reach, stance, days since last fight, last-3 form, and finish rates. Features are **red−blue differentials of career averages using only prior bouts** (no leakage from the fight being predicted). Debuts / missing UFC history show as **insufficient data**, not a fake pick.

## Setup (Mac)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: paste ODDS_API_KEY
python -m src.ingest.ufcstats --bootstrap
python -m src.model.train
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

Without an API key the odds client uses dummy cached JSON (and synthesizes moneylines for the current card) so the dashboard still shows model vs market and EV.

## Refresh before a card

```bash
python -m src.ingest.refresh
```

Pulls new completed UFC Stats fights, scrapes the upcoming card, rebuilds leak-free features, and performs **one** Odds API call if the 24h disk cache is cold.

Retrain when you want updated weights:

```bash
python -m src.model.train
```

Holdout protocol: train through 2022, calibrate on 2023, evaluate 2024–now (accuracy, Brier, log loss).

## Layout

```
src/ingest/ufcstats.py   seed CSVs + incremental scrape
src/ingest/odds.py       The Odds API, 24h SQLite cache, dummy JSON
src/ingest/refresh.py    before-card refresh
src/features/build.py    leak-free career features
src/model/train.py       HistGradientBoostingClassifier + calibration
src/model/predict.py     upcoming matchup → P(win)
src/value/ev.py          de-vig, EV, underdog filter
app.py                   Streamlit dashboard
```

Seed fight history comes from the maintained [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) CSVs, then new event pages are scraped incrementally from UFC Stats.

## Value math

American → decimal, multiplicative de-vig for fair market probability, then:

`EV = P_model * (decimal_odds - 1) - (1 - P_model)`

A fighter is flagged when they are the moneyline **underdog** and EV is above the sidebar slider (default ~5%).

Example: model 42% on a +180 dog → decimal 2.80 → EV ≈ +17.6%.

Name matching between UFC Stats and books uses `rapidfuzz` (e.g. Zhang Weili vs Weili Zhang).

## Out of scope

- No ChatGPT/Claude at prediction time
- No paid historical odds archive (live/upcoming only on the free key)
- No method-of-victory or prop markets in v1 (winner moneylines only)
