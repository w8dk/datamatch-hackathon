"""
Time on Ice (TOI) and Stint Reconstruction Module.
Extracts shift intervals and constructs constant-skater stints.
Incorporates lineup-based goalie disambiguation and data quality auditing.
"""

import pandas as pd

EXCLUDED_MATCHES = {
    "G0072": "Team ID conflict: games table lists T316, events record T096",
    "G0190": "Missing clock_s for all 300 shift events in period 1",
    "G0352": "Unrecorded skater shifts during first 559.8s of period 2 (feed_break)",
}


def resolve_player_roles(events_df, players_df=None):
    """
    Resolves player roles (goalie vs skater) per match using match lineup events.
    Lineup events (lineup_G_*) take precedence over static player database positions,
    resolving 1,067+ documented discrepancies (e.g. pulled goalies, emergency designations).
    """
    lineup = events_df.loc[
        events_df["event_type"].str.startswith("lineup_", na=False)
        & events_df["actor_id"].notna()
    ].copy()
    lineup["is_g"] = lineup["event_type"].str.startswith("lineup_G_")
    lineup_roles = lineup.groupby(["match_id", "actor_id"])["is_g"].agg("min").to_dict()

    pos_map = {}
    if players_df is not None:
        pos_map = players_df.set_index("player_id")["position"].to_dict()

    return lineup_roles, pos_map


def extract_player_shifts(events_df, players_df=None, exclude_corrupted=False):
    """
    Extracts shift intervals [start, end, duration] for each player across matches.
    Tracks state by player_id rather than shift slot for robust line-change resolution.
    """
    df = events_df
    if exclude_corrupted:
        df = df[~df["match_id"].isin(EXCLUDED_MATCHES)].copy()

    lineup_roles, pos_map = resolve_player_roles(df, players_df)

    shifts = df[df["event_type"].str.startswith("shift_", na=False)].sort_values(
        ["match_id", "period", "seq"]
    )

    intervals = []

    for (mid, per), group in shifts.groupby(["match_id", "period"], sort=False):
        if per == 0:  # Shootouts excluded from regulation TOI
            continue

        active = {}  # player_id -> (start_clock, team_id, role)

        cols = [
            "clock_s",
            "actor_id",
            "actor_team_id",
            "other_id",
            "other_team_id",
            "event_type",
        ]
        for clock_s, aid, atid, oid, otid, etype in group[cols].itertuples(
            index=False, name=None
        ):
            if pd.isna(clock_s):
                continue
            t = float(clock_s)

            # Exiting player
            if pd.notna(oid) and oid in active:
                start, team, role = active.pop(oid)
                if t > start:
                    is_g = lineup_roles.get((mid, oid), False) or (
                        pos_map.get(oid) == "G" and role == "shift_G"
                    )
                    intervals.append(
                        (mid, oid, team, int(per), start, t, t - start, is_g)
                    )

            # Entering player
            if pd.notna(aid) and aid not in active:
                active[aid] = (t, atid, etype)

        # Close unclosed active shifts at regulation period boundary (1200s)
        for pid, (start, team, role) in active.items():
            end_t = 1200.0 if per <= 3 else max(start, 300.0)
            if end_t > start:
                is_g = lineup_roles.get((mid, pid), False) or (
                    pos_map.get(pid) == "G" and role == "shift_G"
                )
                intervals.append(
                    (mid, pid, team, int(per), start, end_t, end_t - start, is_g)
                )

    shifts_df = pd.DataFrame(
        intervals,
        columns=[
            "match_id",
            "player_id",
            "team_id",
            "period",
            "start_clock",
            "end_clock",
            "duration_s",
            "is_goalie",
        ],
    )
    return shifts_df


def build_player_match_toi(shifts_df):
    """
    Computes total TOI per player per match for skaters only.
    """
    skaters = shifts_df[~shifts_df["is_goalie"]].copy()
    toi = (
        skaters.groupby(["match_id", "player_id", "team_id"])
        .agg(toi_s=("duration_s", "sum"), shifts_count=("duration_s", "count"))
        .reset_index()
    )
    toi["toi_min"] = toi["toi_s"] / 60.0
    return toi


def build_stints_for_match(events_match, shifts_match, home_team, away_team):
    """
    Constructs stints for a single match: intervals with constant skater lineups.
    """
    # Exclude goalies from skater line-up analysis
    skater_shifts = shifts_match[~shifts_match["is_goalie"]]

    stints = []

    for per in sorted(events_match["period"].unique()):
        if per == 0:
            continue
        p_shifts = skater_shifts[skater_shifts["period"] == per]
        p_ev = events_match[events_match["period"] == per]

        max_clock = p_ev["clock_s"].max() if len(p_ev) > 0 else 1200.0
        max_clock = max(1200.0, max_clock)

        # Cutpoints at shift transitions
        cuts = sorted(
            set(
                [0.0, max_clock]
                + list(p_shifts["start_clock"].unique())
                + list(p_shifts["end_clock"].unique())
            )
        )

        for i in range(len(cuts) - 1):
            t0, t1 = cuts[i], cuts[i + 1]
            dur = t1 - t0
            if dur <= 0.0:
                continue

            # Skaters active at interval midpoint
            t_mid = (t0 + t1) / 2.0
            active = p_shifts[
                (p_shifts["start_clock"] <= t_mid) & (p_shifts["end_clock"] >= t_mid)
            ]

            home_skaters = set(active[active["team_id"] == home_team]["player_id"])
            away_skaters = set(active[active["team_id"] == away_team]["player_id"])

            stints.append(
                {
                    "period": per,
                    "start_clock": t0,
                    "end_clock": t1,
                    "duration_s": dur,
                    "n_home_skaters": len(home_skaters),
                    "n_away_skaters": len(away_skaters),
                    "home_skaters": home_skaters,
                    "away_skaters": away_skaters,
                }
            )

    return stints
