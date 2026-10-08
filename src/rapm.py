import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_rapm_matrices(stints_records, player_list, lambda_reg=100.0):
    """
    Constructs weighted sparse design matrix X and target y for offensive & defensive RAPM.

    stints_records: list of dicts with keys:
      'duration_s': float
      'home_skaters': set of player_id
      'away_skaters': set of player_id
      'xg_home': float
      'xg_away': float
      'is_home_pp': int
      'is_away_pp': int
    """
    p2idx = {p: i for i, p in enumerate(player_list)}
    n_players = len(player_list)

    # Feature columns:
    # [0 .. n_players-1]: Offensive coefficients
    # [n_players .. 2*n_players-1]: Defensive coefficients
    # 2*n_players: Intercept
    # 2*n_players + 1: Home Advantage
    # 2*n_players + 2: Powerplay advantage
    # 2*n_players + 3: Penalty kill disadvantage
    n_features = 2 * n_players + 4
    intercept_col = 2 * n_players
    home_adv_col = 2 * n_players + 1
    pp_col = 2 * n_players + 2
    pk_col = 2 * n_players + 3

    rows, cols, vals = [], [], []
    y_vals, weights = [], []

    row_idx = 0

    for s in stints_records:
        dur = s["duration_s"]
        if dur < 2.0:  # Skip trivial sub-2-second noise stints
            continue

        dur_h = dur / 3600.0  # Duration in fractions of 60 minutes
        w = np.sqrt(dur)  # Weighting by sqrt(duration) for WLS

        # Home Team Offense
        rate_h = s["xg_home"] / dur_h
        y_vals.append(rate_h * w)
        weights.append(w)

        # Intercept & Home Ice Advantage
        rows.append(row_idx)
        cols.append(intercept_col)
        vals.append(1.0 * w)
        rows.append(row_idx)
        cols.append(home_adv_col)
        vals.append(1.0 * w)
        if s.get("is_home_pp", 0):
            rows.append(row_idx)
            cols.append(pp_col)
            vals.append(1.0 * w)
        if s.get("is_away_pp", 0):
            rows.append(row_idx)
            cols.append(pk_col)
            vals.append(-1.0 * w)

        # Home skaters attacking (+ in Offense block)
        for p in s["home_skaters"]:
            if p in p2idx:
                rows.append(row_idx)
                cols.append(p2idx[p])
                vals.append(1.0 * w)
        # Away skaters defending (- in Defense block, so +beta_def suppresses)
        for p in s["away_skaters"]:
            if p in p2idx:
                rows.append(row_idx)
                cols.append(n_players + p2idx[p])
                vals.append(-1.0 * w)

        row_idx += 1

        # Away Team Offense
        rate_a = s["xg_away"] / dur_h
        y_vals.append(rate_a * w)
        weights.append(w)

        # Intercept
        rows.append(row_idx)
        cols.append(intercept_col)
        vals.append(1.0 * w)
        if s.get("is_away_pp", 0):
            rows.append(row_idx)
            cols.append(pp_col)
            vals.append(1.0 * w)
        if s.get("is_home_pp", 0):
            rows.append(row_idx)
            cols.append(pk_col)
            vals.append(-1.0 * w)

        # Away skaters attacking (+ in Offense block)
        for p in s["away_skaters"]:
            if p in p2idx:
                rows.append(row_idx)
                cols.append(p2idx[p])
                vals.append(1.0 * w)
        # Home skaters defending (- in Defense block)
        for p in s["home_skaters"]:
            if p in p2idx:
                rows.append(row_idx)
                cols.append(n_players + p2idx[p])
                vals.append(-1.0 * w)

        row_idx += 1

    X_data = sp.csr_matrix(
        (vals, (rows, cols)), shape=(row_idx, n_features), dtype=np.float32
    )
    y_data = np.array(y_vals, dtype=np.float32)

    return X_data, y_data, p2idx


def fit_rapm(X, y, n_players, lambda_reg=100.0, iter_lim=100):
    """
    Fits Ridge RAPM using regularized sparse least-squares.
    Returns (beta_off, beta_def, intercept, controls).
    """
    # Augment X with ridge penalty rows for player coefficients only
    # lambda * I on player features (first 2*n_players)
    n_reg = 2 * n_players
    reg_rows = np.arange(n_reg)
    reg_cols = np.arange(n_reg)
    reg_vals = np.full(n_reg, np.sqrt(lambda_reg), dtype=np.float32)

    R = sp.csr_matrix(
        (reg_vals, (reg_rows, reg_cols)), shape=(n_reg, X.shape[1]), dtype=np.float32
    )

    X_aug = sp.vstack([X, R]).tocsr()
    y_aug = np.concatenate([y, np.zeros(n_reg, dtype=np.float32)])

    sol = spla.lsqr(X_aug, y_aug, damp=0.0, iter_lim=iter_lim)
    beta = sol[0]

    beta_off = beta[:n_players]
    beta_def = beta[n_players : 2 * n_players]

    return beta_off, beta_def, beta[2 * n_players :]


def fit_rapm_with_bootstrap(
    stints_by_match, player_list, n_bootstraps=30, lambda_reg=80.0
):
    """
    Fits primary RAPM model and computes empirical bootstrap confidence intervals
    by resampling matches with replacement.
    Returns DataFrame with player_id, rating, ci_lower, ci_upper, rating_off, rating_def.
    """
    all_matches = list(stints_by_match.keys())

    # Main model fit on all data
    all_stints = [s for mid in all_matches for s in stints_by_match[mid]]
    X, y, _ = build_rapm_matrices(all_stints, player_list, lambda_reg)
    n_p = len(player_list)
    b_off, b_def, _ = fit_rapm(X, y, n_p, lambda_reg=lambda_reg)

    point_ratings = b_off + b_def

    # Bootstrap across matches
    rng = np.random.RandomState(42)
    boot_ratings = np.zeros((n_bootstraps, n_p), dtype=np.float32)

    for b in range(n_bootstraps):
        sampled_matches = rng.choice(all_matches, size=len(all_matches), replace=True)
        boot_stints = [s for mid in sampled_matches for s in stints_by_match[mid]]
        X_b, y_b, _ = build_rapm_matrices(boot_stints, player_list, lambda_reg)
        b_off_b, b_def_b, _ = fit_rapm(
            X_b, y_b, n_p, lambda_reg=lambda_reg, iter_lim=60
        )
        boot_ratings[b, :] = b_off_b + b_def_b

    ci_lower = np.percentile(boot_ratings, 5.0, axis=0)
    ci_upper = np.percentile(boot_ratings, 95.0, axis=0)

    df_results = pd.DataFrame(
        {
            "player_id": player_list,
            "rating": point_ratings,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "rating_off": b_off,
            "rating_def": b_def,
        }
    )

    return df_results
