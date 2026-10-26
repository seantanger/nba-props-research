"""Probability-of-over models. Each returns P(actual > line) for every row of `pairs_test`.

All models are fit on data strictly before `cutoff` (a date) and predict pairs on/after it.
  - proj_lgbm : LightGBM mean + dispersion on game logs (never sees prices) -> NegBin/Normal -> P(over)
  - eb_rate   : empirical-Bayes per-minute rate x projected minutes -> NegBin -> P(over)
  - clf_lgbm  : LightGBM classifier on offer rows (features + line + market implied prob)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from src.common import feature_cols, MARKETS

LGB_PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=31, min_data_in_leaf=80, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=0)


def nb_sf(line, mu, var):
    """P(X > line) for a count X with mean mu and variance var (NegBin if var>mu else Poisson). line is x.5"""
    mu = np.clip(mu, 1e-3, None)
    var = np.maximum(var, mu * 1.05)
    k = np.floor(line)  # P(X >= k+1) = sf(k)
    r = mu ** 2 / (var - mu)
    pnb = mu / var
    return stats.nbinom.sf(k, r, pnb)


def _fit_mean_disp(train, target, cols, rounds=600):
    """Mean model (poisson deviance) + dispersion model on log((y-mu)^2) -> returns predictor(X)->(mu,var)."""
    dtr = lgb.Dataset(train[cols], train[target])
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds)
    mu_tr = m.predict(train[cols])
    # dispersion: model excess variance ratio phi = var/mu as a function of features (log-link)
    r2 = (train[target].values - mu_tr) ** 2
    d = lgb.train(dict(LGB_PARAMS, objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200),
                  lgb.Dataset(train[cols], np.log(r2 + 0.25)), num_boost_round=300)
    # calibrate scale of dispersion model: E[r2] vs exp(pred) on train
    sc = r2.mean() / np.exp(d.predict(train[cols])).mean()

    def predict(X):
        mu = m.predict(X[cols])
        var = np.exp(d.predict(X[cols])) * sc
        return mu, var
    return predict, m


def proj_lgbm(pg, pairs_test, cutoff, rounds=600, return_models=False):
    """pg: full player_games frame. Trains one model per market on logs with GAME_DATE < cutoff."""
    cols = feature_cols(pg)
    train = pg[(pg.GAME_DATE < cutoff) & (pg.MIN > 0)]
    X = pairs_test.merge(pg[["PLAYER_ID", "GAME_ID"] + cols], left_on=["player_id", "game_id"], right_on=["PLAYER_ID", "GAME_ID"], how="left")
    out = np.full(len(pairs_test), np.nan)
    mus = np.full(len(pairs_test), np.nan); vars_ = np.full(len(pairs_test), np.nan)
    models = {}
    for mk in MARKETS:
        pred, model = _fit_mean_disp(train, mk, cols, rounds)
        sel = (X.market == mk).values & X.PLAYER_ID.notna().values
        if sel.sum() == 0:
            continue
        mu, var = pred(X[sel])
        out[sel] = nb_sf(X.line.values[sel], mu, var)
        mus[sel] = mu; vars_[sel] = var
        models[mk] = (pred, model)
    if return_models:
        return out, mus, vars_, models
    return out, mus, vars_


def eb_rate(pg, pairs_test, cutoff, halflife=8, prior_games=6):
    """Empirical Bayes: per-minute rate shrunk toward last-season/position prior; minutes = EWM of minutes.
    Variance: player-specific residual variance (20g) floored at NegBin with market dispersion."""
    hist = pg[(pg.GAME_DATE < cutoff) & (pg.MIN > 0)]
    X = pairs_test.merge(pg[["PLAYER_ID", "GAME_ID", "MIN_ewm", "MIN_m20"] + [f"{m}_per_min20" for m in MARKETS] + [f"{m}_s20" for m in MARKETS] +
                            [f"{m}_m20" for m in MARKETS] + [f"ps_{m}" for m in ["PTS", "REB", "AST", "FG3M"]] + ["ps_MIN", "games_prior", "pos_G", "pos_F", "pos_C"]],
                         left_on=["player_id", "game_id"], right_on=["PLAYER_ID", "GAME_ID"], how="left")
    out = np.full(len(pairs_test), np.nan)
    # league dispersion per market from history: var = mu + a*mu^2
    for mk in MARKETS:
        h = hist[hist[f"{mk}_m20"].notna()]
        resid = h[mk] - h[f"{mk}_m20"]
        a = max(((resid ** 2 - h[f"{mk}_m20"]) / (h[f"{mk}_m20"] ** 2 + 1e-9)).clip(-1, 5).median(), 0.02)
        sel = (X.market == mk).values & X.PLAYER_ID.notna().values
        x = X[sel]
        rate = x[f"{mk}_per_min20"]
        if mk == "PRA":
            prior_rate = (x.ps_PTS + x.ps_REB + x.ps_AST) / x.ps_MIN
        else:
            prior_rate = x[f"ps_{mk}"] / x.ps_MIN
        prior_rate = prior_rate.fillna(rate)
        n = x.games_prior.clip(upper=20).fillna(0)
        w = n / (n + prior_games)
        rate_hat = (w * rate.fillna(prior_rate) + (1 - w) * prior_rate).fillna(rate)
        minutes = x.MIN_ewm.fillna(x.MIN_m20)
        mu = (rate_hat * minutes).values
        nbvar = mu + a * mu ** 2
        s20 = x[f"{mk}_s20"].values
        s20 = np.where(np.isnan(s20), np.sqrt(nbvar), s20)
        var = np.maximum(nbvar, s20 ** 2 * 0.8 + 0.2 * nbvar)
        out[sel] = nb_sf(x.line.values, mu, var)
    return out


CLF_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.6,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)


def clf_features(pairs, pg, extra=None):
    cols = feature_cols(pg)
    X = pairs.merge(pg[["PLAYER_ID", "GAME_ID"] + cols], left_on=["player_id", "game_id"], right_on=["PLAYER_ID", "GAME_ID"], how="left")
    X["mk"] = X.market.map({m: i for i, m in enumerate(MARKETS)})
    X["line_f"] = X.line
    X["imp"] = X.p_over_imp
    X["vig_f"] = X.vig
    # line relative to the player's recent output in the same stat
    for mk in MARKETS:
        s = X.market == mk
        for suffix in ["m5", "m10", "m20", "ewm", "std_season", "s20"]:
            X.loc[s, f"line_rel_{suffix}"] = X.loc[s, f"{mk}_{suffix}"] - X.loc[s, "line"]
        X.loc[s, "line_z"] = (X.loc[s, f"{mk}_m20"] - X.loc[s, "line"]) / X.loc[s, f"{mk}_s20"].replace(0, np.nan)
    feats = cols + ["mk", "line_f", "imp", "vig_f", "line_z"] + [f"line_rel_{s}" for s in ["m5", "m10", "m20", "ewm", "std_season", "s20"]]
    if extra:
        for e in extra:
            feats.append(e)
    return X, feats


def clf_lgbm(pg, pairs_train, pairs_test, rounds=400, extra_cols=None):
    Xtr, feats = clf_features(pairs_train, pg, extra_cols)
    Xte, _ = clf_features(pairs_test, pg, extra_cols)
    y = (Xtr.res_over == "win").astype(int)
    ok = Xtr.res_over.isin(["win", "loss"])
    m = lgb.train(CLF_PARAMS, lgb.Dataset(Xtr.loc[ok, feats], y[ok], categorical_feature=["mk"]), num_boost_round=rounds)
    return m.predict(Xte[feats]), m, feats


def proj_dist(pg, keys, cutoff, markets, rounds=600, params=None):
    """Fit one mean+dispersion model per market on logs with GAME_DATE < cutoff; return mu/var for the requested
    (PLAYER_ID, GAME_ID, market) keys (a DataFrame with those columns)."""
    cols = feature_cols(pg)
    train = pg[(pg.GAME_DATE < cutoff) & (pg.MIN > 0)]
    out = []
    for mk in markets:
        k = keys[keys.market == mk][["PLAYER_ID", "GAME_ID"]].drop_duplicates()
        if len(k) == 0:
            continue
        X = k.merge(pg[["PLAYER_ID", "GAME_ID"] + cols], on=["PLAYER_ID", "GAME_ID"], how="inner")
        pred, model = _fit_mean_disp(train, mk, cols, rounds)
        mu, var = pred(X)
        out.append(pd.DataFrame({"PLAYER_ID": X.PLAYER_ID.values, "GAME_ID": X.GAME_ID.values, "market": mk, "mu": mu, "var": var}))
    return pd.concat(out, ignore_index=True)
