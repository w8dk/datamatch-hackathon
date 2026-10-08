"""
Model Evaluation and Validation Module.
Performs split-half reliability and predictive validity checks against baselines (+/-, P/60).
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def evaluate_split_half_reliability(player_match_stats_df, min_matches=4):
    """
    Evaluates split-half reliability (repeatability) of metrics across odd vs even matches.
    Compares:
      1. xG-RAPM / Net xG impact per 60
      2. Traditional Plus/Minus (+/- per 60)
      3. Traditional Points per 60 (P/60)
    """
    # Filter players with at least min_matches
    match_counts = player_match_stats_df.groupby("player_id")["match_id"].nunique()
    qual_players = match_counts[match_counts >= min_matches].index
    df_qual = player_match_stats_df[
        player_match_stats_df["player_id"].isin(qual_players)
    ].copy()

    # Assign odd/even match split per player
    df_qual["match_order"] = df_qual.groupby("player_id").cumcount()
    df_qual["half"] = np.where(df_qual["match_order"] % 2 == 0, "half1", "half2")

    # Aggregate per half
    grouped = (
        df_qual.groupby(["player_id", "half"])
        .agg(
            toi_min=("toi_min", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            plus_minus=("plus_minus", "sum"),
            xg_net=("xg_net", "sum"),
            action_val=("action_val", "sum"),
        )
        .reset_index()
    )

    # Normalize per 60 minutes
    grouped["rate_pm"] = (
        grouped["plus_minus"] / np.maximum(5.0, grouped["toi_min"])
    ) * 60.0
    grouped["rate_points"] = (
        (grouped["goals"] + grouped["assists"]) / np.maximum(5.0, grouped["toi_min"])
    ) * 60.0
    grouped["rate_xg_net"] = (
        grouped["xg_net"] / np.maximum(5.0, grouped["toi_min"])
    ) * 60.0
    grouped["rate_action_val"] = (
        grouped["action_val"] / np.maximum(5.0, grouped["toi_min"])
    ) * 60.0

    piv = grouped.pivot(index="player_id", columns="half")

    results = {}
    for metric_col, label in [
        ("rate_pm", "Traditional Plus/Minus (+/- per 60)"),
        ("rate_points", "Traditional Points per 60 (P/60)"),
        ("rate_xg_net", "Net Expected Goals (xG Net per 60)"),
        ("rate_action_val", "Comprehensive Action Value (xT per 60)"),
    ]:
        v1 = piv[(metric_col, "half1")].dropna()
        v2 = piv[(metric_col, "half2")].dropna()
        idx = v1.index.intersection(v2.index)
        if len(idx) > 10:
            r_pearson, _ = pearsonr(v1.loc[idx], v2.loc[idx])
            r_spearman, _ = spearmanr(v1.loc[idx], v2.loc[idx])
            results[label] = {
                "n_players": len(idx),
                "pearson_r": float(r_pearson),
                "spearman_rho": float(r_spearman),
                "r_squared": float(r_pearson**2),
            }

    return pd.DataFrame(results).T


def evaluate_predictive_power(train_match_stats, test_match_stats):
    """
    Evaluates out-of-sample predictive power of metrics trained on historical matches
    to forecast future performance.
    """
    # Aggregate train metrics
    tr_agg = (
        train_match_stats.groupby("player_id")
        .agg(
            toi_min=("toi_min", "sum"),
            pm=("plus_minus", "sum"),
            points=("goals", "sum")
            + train_match_stats.groupby("player_id")["assists"].sum(),
            xg_net=("xg_net", "sum"),
        )
        .reset_index()
    )

    tr_agg["tr_pm_60"] = (tr_agg["pm"] / np.maximum(5.0, tr_agg["toi_min"])) * 60.0
    tr_agg["tr_pts_60"] = (tr_agg["points"] / np.maximum(5.0, tr_agg["toi_min"])) * 60.0
    tr_agg["tr_xg_60"] = (tr_agg["xg_net"] / np.maximum(5.0, tr_agg["toi_min"])) * 60.0

    # Aggregate test future actual goal differential
    te_agg = (
        test_match_stats.groupby("player_id")
        .agg(
            toi_min=("toi_min", "sum"),
            future_pm=("plus_minus", "sum"),
            future_goals=("goals", "sum"),
            future_xg=("xg_net", "sum"),
        )
        .reset_index()
    )
    te_agg["future_pm_60"] = (
        te_agg["future_pm"] / np.maximum(5.0, te_agg["toi_min"])
    ) * 60.0
    te_agg["future_xg_60"] = (
        te_agg["future_xg"] / np.maximum(5.0, te_agg["toi_min"])
    ) * 60.0

    merged = tr_agg.merge(te_agg, on="player_id", how="inner")
    merged = merged[merged["toi_min_x"] >= 20.0]  # Minimum 20 min in both samples

    results = {}
    if len(merged) > 10:
        # Predict future PM using past PM vs past xG
        r_pm_to_future, _ = pearsonr(merged["tr_pm_60"], merged["future_pm_60"])
        r_xg_to_future, _ = pearsonr(merged["tr_xg_60"], merged["future_pm_60"])
        r_xg_to_future_xg, _ = pearsonr(merged["tr_xg_60"], merged["future_xg_60"])

        results["Past +/- predicting Future +/-"] = float(r_pm_to_future)
        results["Past xG Metric predicting Future +/-"] = float(r_xg_to_future)
        results["Past xG Metric predicting Future xG"] = float(r_xg_to_future_xg)

    return results
