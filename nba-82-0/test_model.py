"""Sanity-check the saved model with a weak, an average and a superteam lineup.

Run after train_model.py:  python test_model.py
"""
import joblib
import pandas as pd

from lineup_features import FEATURES, lineup_features

players = pd.read_csv("data/nba_2026_27_players.csv").set_index("player_name")
model = joblib.load("model/win_predictor.joblib")

LINEUPS = {
    "Role players": ["Collin Gillespie", "Cody Williams", "Jaylen Wells",
                     "Sam Hauser", "Goga Bitadze"],
    "Average starters": ["Dejounte Murray", "Christian Braun", "Andrew Wiggins",
                         "Keegan Murray", "Jarrett Allen"],
    "Superteam": ["Shai Gilgeous-Alexander", "Anthony Edwards", "Kawhi Leonard",
                  "Giannis Antetokounmpo", "Nikola Jokic"],
    "Best possible?": ["Luka Doncic", "Donovan Mitchell", "Jayson Tatum",
                       "Giannis Antetokounmpo", "Victor Wembanyama"],
}

for name, roster in LINEUPS.items():
    lineup = players.loc[roster].reset_index()
    X = pd.DataFrame([lineup_features(lineup)])[FEATURES]
    wins = int(round(min(max(model.predict(X)[0], 0), 82)))
    print(f"{name:18s} -> {wins}-{82 - wins}")
