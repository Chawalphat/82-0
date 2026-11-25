"""Train the 82-0 win predictor.

Steps:
1. Load the 2026-27 player CSV.
2. Simulate thousands of random 5-man lineups (one PG, SG, SF, PF, C).
3. Give each lineup a "true" win total (0-82) using a basketball formula:
   all 5 box-score stats + positional balance + some random luck.
4. Train a scikit-learn GradientBoostingRegressor to predict wins from
   the lineup's combined stats, then save it to model/win_predictor.joblib.

Run:  python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from lineup_features import FEATURES, POSITIONS, lineup_features

DATA_PATH = Path("data/nba_2026_27_players.csv")
MODEL_PATH = Path("model/win_predictor.joblib")
N_LINEUPS = 20_000
rng = np.random.default_rng(42)


def player_impact(df: pd.DataFrame) -> pd.Series:
    """One number per player. Defensive stats are rarer, so they count more."""
    return df.ppg + 1.2 * df.rpg + 1.5 * df.apg + 3.0 * df.spg + 3.0 * df.bpg


def true_wins(feats: dict, impact_sum: float) -> float:
    """The 'answer key' the model learns from (with random luck added)."""
    score = impact_sum
    # Positional balance: reward a real playmaker, rebounding, and rim protection.
    score += 2.0 * min(feats["guard_apg"] - 8, 6)
    score += 1.5 * min(feats["big_rpg"] - 14, 8)
    score += 3.0 * min(feats["big_bpg"] - 1.6, 2)
    # A lineup with a non-scorer is easier to defend.
    score += 1.0 * min(feats["min_ppg"] - 10, 0)
    luck = rng.normal(0, 0.3)  # same lineup won't win the exact same games every season
    # Squash the score into 0-82 wins with an S-curve. Stacked superteams
    # (score above 260) get an extra push so 82-0 is possible but rare.
    logit = (score - 119) / 53 + 0.045 * max(score - 260, 0) + luck
    return 82 / (1 + np.exp(-logit))


def simulate_lineups(players: pd.DataFrame) -> pd.DataFrame:
    players = players.assign(impact=player_impact(players))
    pools = {p: players[players.position == p] for p in POSITIONS}
    rows = []
    for i in range(N_LINEUPS):
        # Mix of fully random lineups and star-heavy lineups, so the model
        # sees both bad teams and superteams during training.
        star_bias = rng.choice([0, 2, 5])
        picks = []
        for pos in POSITIONS:
            pool = pools[pos]
            w = pool.impact.to_numpy() ** star_bias
            picks.append(pool.sample(1, weights=w, random_state=rng.integers(1e9)))
        lineup = pd.concat(picks)
        feats = lineup_features(lineup)
        feats["wins"] = true_wins(feats, lineup.impact.sum())
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    players = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(players)} players from {players.team.nunique()} teams")

    data = simulate_lineups(players)
    print(f"Simulated {len(data)} lineups. Wins: min {data.wins.min():.0f}, "
          f"median {data.wins.median():.0f}, max {data.wins.max():.0f}")

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data.wins, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Test MAE: {mean_absolute_error(y_test, pred):.1f} wins | "
          f"R^2: {r2_score(y_test, pred):.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
