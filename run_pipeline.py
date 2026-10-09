import os
import time

import numpy as np
import pandas as pd

import src.action_value as av
import src.evaluate as ev
import src.xg_model as xg
from src import rapm, toi


def run():
    t_start = time.time()

    print("\n[1/6] Loading datasets & applying data quality filters")
    candidates = [
        "DataMatch_Аналитический_трек1/data",
        "data",
        "../DataMatch_Аналитический_трек1/data",
        "../data",
        ".",
    ]
    data_dir = next(
        (c for c in candidates if os.path.exists(os.path.join(c, "events.parquet"))),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError(
            f"File (events.parquet) not found. Move it to one of the paths: {candidates}"
        )
    print(f"Data path used: {data_dir}")

    events_train = pd.read_parquet(f"{data_dir}/events.parquet")
    games_train = pd.read_csv(f"{data_dir}/games.csv")
    players_train = pd.read_csv(f"{data_dir}/players.csv")

    events_test = pd.read_parquet(f"{data_dir}/test/events_test.parquet")
    games_test = pd.read_csv(f"{data_dir}/test/games_test.csv")
    players_new = pd.read_csv(f"{data_dir}/test/players_new.csv")
    player_seasons_test = pd.read_csv(f"{data_dir}/test/player_seasons_test.csv")

    all_players = pd.concat([players_train, players_new]).drop_duplicates(
        subset=["player_id"]
    )

    n_before = len(games_train)
    events_train = events_train[
        ~events_train["match_id"].isin(toi.EXCLUDED_MATCHES)
    ].copy()
    games_train = games_train[
        ~games_train["match_id"].isin(toi.EXCLUDED_MATCHES)
    ].copy()
    print(
        f"Training: {len(games_train)} matches (quarantined {n_before - len(games_train)} corrupted), {len(events_train):,} events"
    )
    print(f"Test: {len(games_test)} matches, {len(events_test):,} events")
    print(f"Players: {len(all_players):,} unique metadata profiles")

    # TOI reconstruction
    print("\n[2/6] Reconstructing Time on Ice from shift events")
    t0 = time.time()
    shifts_train = toi.extract_player_shifts(
        events_train, all_players, exclude_corrupted=True
    )
    shifts_test = toi.extract_player_shifts(
        events_test, all_players, exclude_corrupted=False
    )

    toi_train = toi.build_player_match_toi(shifts_train)
    toi_test = toi.build_player_match_toi(shifts_test)

    print(
        f"Extracted {len(shifts_train):,} training shifts and {len(shifts_test):,} test shifts in {time.time() - t0:.2f}s"
    )

    # xG model
    print("\n[3/6] Training expected goals (xG) model")
    t0 = time.time()
    shots_train = xg.extract_shot_features(events_train, games_train, all_players)
    print(
        f"Extracted {len(shots_train):,} shots from training data ({shots_train['is_goal'].sum()} goals, {shots_train['is_goal'].mean() * 100:.2f}%)"
    )

    xg_model, xg_metrics = xg.train_xg_model(shots_train, n_splits=5)
    print("xG model GroupKFold CV metrics:")
    print(f"ROC-AUC: {xg_metrics['roc_auc']:.4f}")
    print(f"Log Loss: {xg_metrics['log_loss']:.4f}")
    print(f"Brier Score: {xg_metrics['brier_score']:.4f}")

    shots_test = xg.extract_shot_features(events_test, games_test, all_players)
    shots_test["xg"] = xg.predict_xg(xg_model, shots_test)
    shots_train["xg"] = xg.predict_xg(xg_model, shots_train)
    print(
        f"Scored test shots: {len(shots_test):,} shots, total xG = {shots_test['xg'].sum():.1f} vs actual {shots_test['is_goal'].sum()} goals in {time.time() - t0:.2f}s"
    )

    print("\n[4/6] Fitting spatial expected threat (xT) grid and scoring actions")
    t0 = time.time()
    xt_model = av.ExpectedThreatModel(nx=12, ny=8)
    xt_model.fit(events_train, shots_train["xg"])
    print(
        f"xT spatial grid fitted. Threat values range from {xt_model.grid.min():.4f} to {xt_model.grid.max():.4f}"
    )

    actions_test = av.score_player_actions(events_test, xt_model)
    print(
        f"Evaluated {len(actions_test):,} non-shot actions in test dataset in {time.time() - t0:.2f}s"
    )

    print("\n[5/6] Building game stints and computing RAPM with bootstrap CIs")
    t0 = time.time()

    shot_xg_dict = {}
    for r in shots_test[
        ["match_id", "period", "clock_s", "actor_team_id", "xg"]
    ].itertuples(index=False):
        mid, per, clk, tid, val = r
        key = (mid, per)
        if key not in shot_xg_dict:
            shot_xg_dict[key] = []
        shot_xg_dict[key].append((clk, tid, val))

    test_stints_by_match = {}

    for _, g_row in games_test.iterrows():
        mid = g_row["match_id"]
        ht = g_row["home_team_id"]
        at = g_row["away_team_id"]

        m_events = events_test[events_test["match_id"] == mid]
        m_shifts = shifts_test[shifts_test["match_id"] == mid]

        stints_list = toi.build_stints_for_match(m_events, m_shifts, ht, at)

        for s in stints_list:
            per = s["period"]
            t0_c, t1_c = s["start_clock"], s["end_clock"]
            cand_shots = shot_xg_dict.get((mid, per), [])

            xg_h, xg_a = 0.0, 0.0
            for clk, tid, val in cand_shots:
                if t0_c <= clk <= t1_c:
                    if tid == ht:
                        xg_h += val
                    elif tid == at:
                        xg_a += val
            s["xg_home"] = xg_h
            s["xg_away"] = xg_a
            s["is_home_pp"] = int(s["n_home_skaters"] > s["n_away_skaters"])
            s["is_away_pp"] = int(s["n_away_skaters"] > s["n_home_skaters"])

        test_stints_by_match[mid] = stints_list

    total_test_stints = sum(len(st) for st in test_stints_by_match.values())
    print(f"  Constructed {total_test_stints:,} test stints across 84 matches")

    test_skater_ids = sorted(set(toi_test["player_id"]))
    print(
        f"Fitting RAPM across {len(test_skater_ids)} test skaters with 30 bootstrap resamples"
    )

    df_ratings = rapm.fit_rapm_with_bootstrap(
        test_stints_by_match, test_skater_ids, n_bootstraps=30, lambda_reg=80.0
    )

    p_meta = all_players[["player_id", "position", "shoots"]].drop_duplicates(
        "player_id"
    )
    df_ratings = df_ratings.merge(p_meta, on="player_id", how="left")

    p_toi_summary = (
        toi_test.groupby("player_id")
        .agg(matches_played=("match_id", "nunique"), toi_total_min=("toi_min", "sum"))
        .reset_index()
    )
    p_toi_summary["toi_per_game"] = (
        p_toi_summary["toi_total_min"] / p_toi_summary["matches_played"]
    )

    df_ratings = df_ratings.merge(p_toi_summary, on="player_id", how="left")

    # primary team
    p_teams = player_seasons_test[["player_id", "team_id"]].drop_duplicates("player_id")
    df_ratings = df_ratings.merge(p_teams, on="player_id", how="left")

    df_ratings = df_ratings.sort_values("rating", ascending=False).reset_index(
        drop=True
    )

    print(f"RAPM fitted in {time.time() - t0:.2f}s")
    print("\nTop 10 skaters by net rating per 60 min:")
    print(
        df_ratings[
            [
                "player_id",
                "position",
                "team_id",
                "matches_played",
                "toi_per_game",
                "rating",
                "ci_lower",
                "ci_upper",
                "rating_off",
                "rating_def",
            ]
        ]
        .head(10)
        .to_string()
    )

    print("\n[6/6] Exporting test_ratings.csv")

    all_test_players = sorted(set(player_seasons_test["player_id"]))
    existing_pids = set(df_ratings["player_id"])
    missing_goalies = [p for p in all_test_players if p not in existing_pids]

    if missing_goalies:
        print(
            f"Adding {len(missing_goalies)} goalies to ratings table (baseline 0.0 with CI)"
        )
        goalie_rows = []
        for g_id in missing_goalies:
            pos_val = all_players.loc[all_players.player_id == g_id, "position"].values
            pos = pos_val[0] if len(pos_val) > 0 and pd.notna(pos_val[0]) else "G"
            t_val = player_seasons_test.loc[
                player_seasons_test.player_id == g_id, "team_id"
            ].values
            tid = t_val[0] if len(t_val) > 0 else "UNKNOWN"
            goalie_rows.append(
                {
                    "player_id": g_id,
                    "rating": 0.0,
                    "ci_lower": -0.15,
                    "ci_upper": 0.15,
                    "rating_off": 0.0,
                    "rating_def": 0.0,
                    "position": pos,
                    "shoots": np.nan,
                    "matches_played": 0,
                    "toi_total_min": 0.0,
                    "toi_per_game": 0.0,
                    "team_id": tid,
                }
            )
        df_ratings = pd.concat(
            [df_ratings, pd.DataFrame(goalie_rows)], ignore_index=True
        )

    export_cols = [
        "player_id",
        "rating",
        "ci_lower",
        "ci_upper",
        "position",
        "team_id",
        "matches_played",
        "toi_total_min",
        "rating_off",
        "rating_def",
    ]

    df_ratings[export_cols].to_csv("test_ratings.csv", index=False)
    df_ratings[export_cols].to_csv(
        "DataMatch_Аналитический_трек1/test_ratings.csv", index=False
    )
    print(f"Successfully saved test_ratings.csv ({len(df_ratings)} rows)")

    # Валидация

    val_m = set(games_train["match_id"].head(350))
    skaters_sub = shifts_train[
        shifts_train.match_id.isin(val_m) & (~shifts_train.is_goalie)
    ].copy()
    events_sub = events_train[events_train.match_id.isin(val_m)].copy()
    shots_sub = shots_train[shots_train.match_id.isin(val_m)].copy()
    toi_sub = toi_train[toi_train.match_id.isin(val_m)].copy()

    goals = events_sub[events_sub.event_type == "goal"][
        ["match_id", "period", "clock_s", "actor_team_id"]
    ]
    iv_grouped = {k: g for k, g in skaters_sub.groupby(["match_id", "period"])}

    pm_records = []
    for r in goals.itertuples(index=False):
        mid, per, clk, g_team = r
        if (mid, per) not in iv_grouped:
            continue
        active = iv_grouped[(mid, per)]
        on_ice = active[(active.start_clock <= clk) & (active.end_clock >= clk)]
        for row in on_ice.itertuples(index=False):
            pm_records.append((mid, row.player_id, 1 if row.team_id == g_team else -1))

    df_pm = (
        pd.DataFrame(pm_records, columns=["match_id", "player_id", "pm"])
        .groupby(["match_id", "player_id"])["pm"]
        .sum()
        .reset_index()
    )

    xg_records = []
    for r in shots_sub[
        ["match_id", "period", "clock_s", "actor_team_id", "xg"]
    ].itertuples(index=False):
        mid, per, clk, s_team, val = r
        if (mid, per) not in iv_grouped:
            continue
        active = iv_grouped[(mid, per)]
        on_ice = active[(active.start_clock <= clk) & (active.end_clock >= clk)]
        for row in on_ice.itertuples(index=False):
            xg_records.append(
                (mid, row.player_id, val if row.team_id == s_team else -val)
            )

    df_xg_onice = (
        pd.DataFrame(xg_records, columns=["match_id", "player_id", "xg_net"])
        .groupby(["match_id", "player_id"])["xg_net"]
        .sum()
        .reset_index()
    )

    pts_ev = events_sub[
        events_sub.event_type.isin(["goal", "assist_primary", "assist_secondary"])
    ]
    p_goals = (
        pts_ev[pts_ev.event_type == "goal"]
        .groupby(["match_id", "actor_id"])
        .size()
        .rename("goals")
    )
    p_assists = (
        pts_ev[pts_ev.event_type.str.startswith("assist_")]
        .groupby(["match_id", "actor_id"])
        .size()
        .rename("assists")
    )
    df_pts = (
        pd.concat([p_goals, p_assists], axis=1)
        .fillna(0)
        .reset_index()
        .rename(columns={"actor_id": "player_id"})
    )

    act_ev = events_sub[
        events_sub.event_type.isin(
            [
                "pass_complete",
                "key_pass_complete",
                "steal",
                "entry_success",
                "exit_success",
            ]
        )
    ]
    df_acts = (
        act_ev.groupby(["match_id", "actor_id"])
        .size()
        .rename("action_val")
        .reset_index()
        .rename(columns={"actor_id": "player_id"})
    )

    m_stats = toi_sub.merge(df_pm, on=["match_id", "player_id"], how="left").fillna(
        {"pm": 0}
    )
    m_stats = m_stats.merge(
        df_xg_onice, on=["match_id", "player_id"], how="left"
    ).fillna({"xg_net": 0})
    m_stats = m_stats.merge(df_pts, on=["match_id", "player_id"], how="left").fillna(
        {"goals": 0, "assists": 0}
    )
    m_stats = m_stats.merge(df_acts, on=["match_id", "player_id"], how="left").fillna(
        {"action_val": 0}
    )
    m_stats["plus_minus"] = m_stats["pm"]

    split_half_df = ev.evaluate_split_half_reliability(m_stats, min_matches=4)
    print("\nSplit-Half Reliability Comparison:")
    print(split_half_df.to_string())

    print(f"\nPipeline finished in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    run()
