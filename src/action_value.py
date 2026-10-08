import numpy as np
import pandas as pd


class ExpectedThreatModel:
    def __init__(self, nx=12, ny=8):
        self.nx = nx
        self.ny = ny
        self.x_bins = np.linspace(-30.5, 30.5, nx + 1)
        self.y_bins = np.linspace(-15.0, 15.0, ny + 1)
        self.grid = np.zeros((nx, ny))

    def fit(self, events_df, xg_series=None):
        """
        Fits spatial threat grid based on historical scoring chances and xG values.
        """
        shots = events_df[
            events_df["event_type"].isin(
                ["goal", "shot_on_target", "shot_missed", "shot_blocked"]
            )
            & events_df["period"].gt(0)
        ].copy()

        # Target: xG if provided, otherwise goal indicator
        if xg_series is not None and len(xg_series) == len(shots):
            shots["val"] = xg_series.values
        else:
            shots["val"] = (shots["event_type"] == "goal").astype(float)

        x_idx = np.clip(
            np.digitize(shots["x"].fillna(0), self.x_bins) - 1, 0, self.nx - 1
        )
        y_idx = np.clip(
            np.digitize(shots["y"].fillna(0), self.y_bins) - 1, 0, self.ny - 1
        )

        # Accumulate shot threat
        weight_grid = np.zeros((self.nx, self.ny))
        count_grid = np.zeros((self.nx, self.ny))

        for xi, yi, val in zip(x_idx, y_idx, shots["val"]):
            weight_grid[xi, yi] += val
            count_grid[xi, yi] += 1

        # Threat is smoothed empirical expected goal value in cell
        base_threat = np.zeros((self.nx, self.ny))
        for xi in range(self.nx):
            for yi in range(self.ny):
                cnt = count_grid[xi, yi]
                if cnt > 10:
                    base_threat[xi, yi] = weight_grid[xi, yi] / cnt
                else:
                    # Fallback proportional to offensive distance
                    center_x = (self.x_bins[xi] + self.x_bins[xi + 1]) / 2.0
                    center_y = (self.y_bins[yi] + self.y_bins[yi + 1]) / 2.0
                    dist = np.sqrt((26.5 - center_x) ** 2 + center_y**2)
                    base_threat[xi, yi] = 0.15 * np.exp(-dist / 8.0)

        # Apply slight spatial Gaussian smoothing
        from scipy.ndimage import gaussian_filter

        self.grid = gaussian_filter(base_threat, sigma=0.8)
        return self

    def get_threat(self, x, y):
        """
        Vectorized lookup of threat for coordinates (x, y).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xi = np.clip(
            np.digitize(np.nan_to_num(x, nan=0.0), self.x_bins) - 1, 0, self.nx - 1
        )
        yi = np.clip(
            np.digitize(np.nan_to_num(y, nan=0.0), self.y_bins) - 1, 0, self.ny - 1
        )
        return self.grid[xi, yi]


def score_player_actions(events_df, xt_model, shots_xg_map=None):
    """
    Computes individual action values for each player:
    - Shooting Value (xG and finishing)
    - Passing / Playmaking Value (xT delta and shot assists)
    - Transition Value (entries, exits)
    - Defensive Disruption (steals, blocked shots, duels)
    - Error Penalties (turnovers)
    """
    records = []

    # 1. Completed passes
    passes = events_df[events_df["event_type"] == "pass_complete"].copy()
    if len(passes) > 0:
        t_start = xt_model.get_threat(passes["x"], passes["y"])
        t_end = xt_model.get_threat(passes["x2"], passes["y2"])
        delta_xt = np.maximum(-0.05, t_end - t_start)
        for pid, val in zip(passes["actor_id"], delta_xt):
            if pd.notna(pid):
                records.append((pid, "passing", val))

    # 2. Zone entries
    entries = events_df[events_df["event_type"] == "entry_success"].copy()
    if len(entries) > 0:
        val_entry = xt_model.get_threat(
            entries["x"].fillna(10.0), entries["y"].fillna(0.0)
        )
        for pid, val in zip(entries["actor_id"], val_entry):
            if pd.notna(pid):
                records.append((pid, "entry", val * 0.5))

    # 3. Zone exits
    exits = events_df[events_df["event_type"] == "exit_success"].copy()
    if len(exits) > 0:
        for pid in exits["actor_id"]:
            if pd.notna(pid):
                records.append((pid, "exit", 0.015))  # Relieving defensive pressure

    # 4. Steals / Takeaways
    steals = events_df[events_df["event_type"] == "steal"].copy()
    if len(steals) > 0:
        t_rec = xt_model.get_threat(steals["x"], steals["y"])
        for pid, val in zip(steals["actor_id"], t_rec):
            if pd.notna(pid):
                records.append((pid, "takeaway", 0.02 + val * 0.8))

    # 5. Shot blocks
    blocks = events_df[events_df["event_type"] == "shot_blocked"].copy()
    if len(blocks) > 0:
        for pid in blocks[
            "other_id"
        ]:  # other_id is typically the blocking player or goalie
            if pd.notna(pid):
                records.append((pid, "block", 0.035))

    # 6. Turnovers
    turnovers = events_df[events_df["event_type"] == "turnover"].copy()
    if len(turnovers) > 0:
        t_loss = xt_model.get_threat(turnovers["x"], turnovers["y"])
        opp_threat = xt_model.get_threat(
            -turnovers["x"].fillna(0), -turnovers["y"].fillna(0)
        )
        total_loss = -(0.015 + t_loss * 0.5 + opp_threat * 0.5)
        for pid, val in zip(turnovers["actor_id"], total_loss):
            if pd.notna(pid):
                records.append((pid, "turnover", val))

    # 7. Duels won
    duels = events_df[events_df["event_type"] == "duel"].copy()
    if len(duels) > 0:
        for pid in duels["actor_id"]:
            if pd.notna(pid):
                records.append((pid, "duel", 0.008))

    # 8. Shots and Goals (xG)
    if shots_xg_map is not None and len(shots_xg_map) > 0:
        for pid, xg_val, is_goal in shots_xg_map:
            if pd.notna(pid):
                records.append((pid, "shooting_xg", xg_val))
                if is_goal:
                    records.append((pid, "finishing_bonus", max(0.0, 1.0 - xg_val)))

    df_actions = pd.DataFrame(
        records, columns=["player_id", "action_category", "value"]
    )
    return df_actions
