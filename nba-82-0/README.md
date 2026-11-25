# 🏀 82-0 NBA Challenge (2026-27 Season Edition)

A Grade 10 AI Literacy project: a roster-building game where you draft a starting five from 2026-27 NBA teams and an AI model predicts your 82-game regular-season record.

## Problem & User

The "82-0" challenge asks one question: *can you build a lineup good enough to never lose?* Many versions mix players from different decades. This app is a quick, modern version for **basketball fans and students** who want to play with **only 2026-27 players and teams**. An AI model judges the lineup, so the game also shows how a machine-learning model turns data into a prediction.

**How to play**
1. Click **Spin Team** to land on a random 2026-27 NBA team. A team can only come up once per game.
2. Pick **one player** from that team for an open slot (PG, SG, SF, PF, C).
3. Don't like the team? You get **2 Team Skips** per game.
4. Ratings stay **hidden** until all 5 slots are filled (blind draft).
5. The AI predicts your record (for example 64-18, or the elusive **82-0**) and gives a grade: S (82-0), A+, A, B, C or D. You also see each player's stats and a team stats chart.

## How the AI Model Works

**Dataset: `data/nba_2026_27_players.csv`**
- 180 players: 6 key players from each of the 30 NBA teams, with every position (PG, SG, SF, PF, C) covered on every team.
- Columns: `player_name, team, position, ppg, rpg, apg, spg, bpg`.
- It is a **static, saved file**, so the app never scrapes the web and can't crash because a website changed. The numbers are **estimated 2026-27 per-game stats** (a snapshot), not official NBA data.

**Training: `train_model.py`**
1. Simulates **20,000 random 5-man lineups** (one player per position). Some are fully random and some are star-heavy, so the model sees both weak teams and superteams.
2. Gives each lineup a "true" win total from 0 to 82 using a basketball formula:
   - all 5 stats (points, rebounds, assists, steals, blocks). Steals and blocks count more because they are rarer.
   - **positional balance**: guard playmaking (PG+SG assists), big-man rebounding and rim protection (PF+C rebounds and blocks), and a penalty for a lineup with a weak scorer.
   - random "luck", because the same team doesn't win the same number of games every season.
3. Trains a scikit-learn **`GradientBoostingRegressor`** on 9 lineup features (see `lineup_features.py`) and saves it to `model/win_predictor.joblib`.

Test results: about **2.7 wins average error**, R² ≈ 0.84 on held-out lineups.

**Sanity check (`python test_model.py`)**

| Lineup | Predicted |
|---|---|
| 5 role players (Gillespie, C. Williams, Wells, Hauser, Bitadze) | 31-51 |
| Average starters (D. Murray, Braun, Wiggins, K. Murray, Allen) | 58-24 |
| Superteam (SGA, Edwards, Kawhi, Giannis, Jokic) | 80-2 |
| Luka, Mitchell, Tatum, Giannis, Wembanyama | **82-0** |

## Responsible AI & Limitations (What it can't do)

- **AI can be wrong.** The predicted record is an estimate for fun and learning, not a real forecast.
- The model only looks at **5 box-score stats**: PPG, RPG, APG, SPG, BPG. It doesn't see shooting percentages, turnovers, minutes or advanced stats.
- It uses a **2026-27 season snapshot**. The stats are estimates, and rosters change through trades, signings and injuries during the season, so some players may be on different teams now.
- It **can't account for** team chemistry, coaching, injuries, bench depth, or off-ball defense.
- The model learned from **simulated** lineups scored by a formula that we designed. It copies our assumptions about what wins games, so it is only as good as that formula.

## How to Run Locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements.txt

python train_model.py      # optional: the trained model is already in model/
python test_model.py       # optional: check sample lineups
streamlit run app.py       # opens http://localhost:8501
```

## How to Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo**. Include `data/` and `model/win_predictor.joblib`, and don't include `.venv/` (it's already in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, then choose your repo, the branch (`main`) and the main file path `app.py`.
4. Under **Advanced settings**, choose **Python 3.12** or newer.
5. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a public link.

If you edit the CSV or the training script, run `python train_model.py` again and commit the new `model/win_predictor.joblib`.

## Project Files

```
app.py                      Streamlit game
train_model.py              Simulates lineups and trains the model
test_model.py               Checks weak, average and superteam lineups
lineup_features.py          Lineup -> model features (used by both scripts)
data/nba_2026_27_players.csv
model/win_predictor.joblib
requirements.txt
```
