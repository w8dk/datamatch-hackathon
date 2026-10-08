import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold

SHOT_EVENT_TYPES = {"goal", "shot_on_target", "shot_missed", "shot_blocked"}

FEATURE_COLS = [
    "dist",
    "angle",
    "x",
    "y_abs",
    "is_slot",
    "is_inner_slot",
    "n_home",
    "n_away",
    "manpower_diff",
    "is_pp",
    "is_pk",
    "rebound",
    "is_forward",
    "is_defense",
]


def extract_shot_features(events_df, games_df=None, players_df=None):
    """
    Extracts geometric and situational context features for all shot events.
    Excludes post-goal tracking artifacts and leaky temporal delays.
    """
    shots = events_df.loc[
        events_df["event_type"].isin(SHOT_EVENT_TYPES) & events_df["period"].gt(0)
    ].copy()
    shots["is_goal"] = (shots["event_type"] == "goal").astype(int)

    # Merge game metadata if available
    if games_df is not None:
        g_cols = [
            c
            for c in ["match_id", "rink_type", "home_team_id", "away_team_id"]
            if c in games_df.columns
        ]
        shots = shots.merge(games_df[g_cols], on="match_id", how="left")

    # Merge player metadata if available
    if players_df is not None:
        p_cols = [
            c for c in ["player_id", "position", "shoots"] if c in players_df.columns
        ]
        shots = shots.merge(
            players_df[p_cols], left_on="actor_id", right_on="player_id", how="left"
        )

    # Spatial features
    rink_type = shots["rink_type"] if "rink_type" in shots.columns else "european_60x30"
    shots["goal_x"] = np.where(rink_type == "nhl_61x26", 27.1, 26.5)

    shots["x"] = shots["x"].fillna(15.0)
    shots["y"] = shots["y"].fillna(0.0)
    shots["y_abs"] = shots["y"].abs()

    shots["dist"] = np.hypot(shots["goal_x"] - shots["x"], shots["y"])
    shots["angle"] = np.degrees(
        np.arctan2(shots["y_abs"], np.maximum(0.1, shots["goal_x"] - shots["x"]))
    )

    shots["is_slot"] = ((shots["x"] >= 18.0) & (shots["y_abs"] <= 4.5)).astype(int)
    shots["is_inner_slot"] = ((shots["x"] >= 21.0) & (shots["y_abs"] <= 2.5)).astype(
        int
    )

    # Context sequence: Rebound detection (shot within 3 seconds of prior shot by same team)
    shots = shots.sort_values(["match_id", "period", "seq"])
    prev_match = shots["match_id"].shift(1)
    prev_per = shots["period"].shift(1)
    prev_clock = shots["clock_s"].shift(1)
    prev_team = shots["actor_team_id"].shift(1)
    dt = shots["clock_s"] - prev_clock
    same_ctx = (
        (prev_match == shots["match_id"])
        & (prev_per == shots["period"])
        & (prev_team == shots["actor_team_id"])
    )
    shots["rebound"] = (same_ctx & (dt >= 0) & (dt <= 3.0)).astype(int)

    # Manpower situation
    n_h = (
        shots["n_home"].fillna(5)
        if "n_home" in shots.columns
        else pd.Series(5, index=shots.index)
    )
    n_a = (
        shots["n_away"].fillna(5)
        if "n_away" in shots.columns
        else pd.Series(5, index=shots.index)
    )
    shots["n_home"] = n_h
    shots["n_away"] = n_a

    if "home_team_id" in shots.columns:
        is_home = shots["actor_team_id"] == shots["home_team_id"]
        n_att = np.where(is_home, n_h, n_a)
        n_def = np.where(is_home, n_a, n_h)
        shots["manpower_diff"] = n_att - n_def
    else:
        shots["manpower_diff"] = 0

    shots["is_pp"] = (shots["manpower_diff"] > 0).astype(int)
    shots["is_pk"] = (shots["manpower_diff"] < 0).astype(int)

    # Position
    pos = (
        shots["position"].fillna("F")
        if "position" in shots.columns
        else pd.Series("F", index=shots.index)
    )
    shots["is_forward"] = pos.isin(["C", "LW", "RW"]).astype(int)
    shots["is_defense"] = (pos == "D").astype(int)

    return shots


def train_xg_model(shots_df, n_splits=5):
    """
    Trains LightGBM classifier using GroupKFold across matches to avoid cross-match leakage.
    Returns trained booster model and out-of-fold validation metrics.
    """
    X = shots_df[FEATURE_COLS].fillna(0)
    y = shots_df["is_goal"]
    matches = shots_df["match_id"].unique()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(shots_df))

    for train_idx, val_idx in kf.split(matches):
        tr_matches = set(matches[train_idx])
        va_matches = set(matches[val_idx])

        tr_mask = shots_df["match_id"].isin(tr_matches)
        va_mask = shots_df["match_id"].isin(va_matches)

        fold_clf = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=50,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        fold_clf.fit(X[tr_mask], y[tr_mask])
        va_probs = np.asarray(fold_clf.predict_proba(X[va_mask]))
        oof_preds[va_mask] = va_probs[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y, oof_preds)),
        "log_loss": float(log_loss(y, oof_preds)),
        "brier_score": float(brier_score_loss(y, oof_preds)),
        "mean_target": float(y.mean()),
        "mean_pred": float(oof_preds.mean()),
    }

    # Final model on full training data
    final_model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )
    final_model.fit(X, y)

    return final_model, metrics


def predict_xg(model, shots_df):
    """
    Generates calibrated xG predictions for shot events.
    """
    X = shots_df[FEATURE_COLS].fillna(0)
    probs = np.asarray(model.predict_proba(X))
    return probs[:, 1]
