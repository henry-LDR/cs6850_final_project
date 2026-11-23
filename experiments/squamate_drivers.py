"""Back out environmental drivers of body-form evolution from the tree, then run what-ifs.

An environment (from SquamBase's major biome) is added as a fifth binary trait that
evolves along the tree with the four body traits:
    desert        "Deserts & Xeric Shrublands"                      (aridification)
    moist_forest  "Tropical & Subtropical Moist Broadleaf Forests"  (deforestation)
The model fits, besides the body traits' gain rates and shared loss rate, the
environment's own gain and loss rates and couplings in BOTH directions:
    env -> trait   living in the environment multiplies the trait's gain rate by e^Gamma
    trait -> env   having the trait multiplies the rate of moving into the environment
so "deserts drive elongation" is separated from "elongate lizards move into deserts".
Couplings among the body traits are held at their values in the 4-trait fit.

Error bars come from a numerical Hessian (normal approximation), propagated with
parameter draws.

What-ifs (all from the fitted rates, nothing set by hand):
  - start: mean time from an ordinary lizard to a snake-like body, for a lineage
    starting inside vs outside the environment;
  - exposure: the same averaged over lineages, as the share of lineages starting in
    the environment moves from today's share to 0, 2x today's share, or 1;
  - shift: the environment's entry rate doubled (aridification for desert) or its
    exit rate doubled (habitat loss for moist forest).
  - species network: the biased walk of experiments/species_walk.py, with each step that
    gains trait t from a species in the environment also multiplied by e^Gamma[env -> t];
    in the scenario, 20% of the other species are moved into the environment.

Usage: python experiments/squamate_drivers.py desert|moist_forest
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt

from evograph import fit, walks

here = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sq = module("squamate_fit")
unc = module("squamate_uncertainty")
sw = unc.sw
sn = sw.sn
OUT = Path("results")
BIOMES = {"desert": "Deserts & Xeric Shrublands", "moist_forest": "Tropical & Subtropical Moist Broadleaf Forests"}
K = 5
N_DRAWS = 200


def valid(x):
    return not ((x >> 2) & 1 and not (x >> 1) & 1)


VALID5 = [x for x in range(2**K) if valid(x)]
KEEP5 = np.array([[valid(x ^ (1 << i)) and valid(x) for i in range(K)] for x in range(2**K)])


def environment(tips_names, driver):
    sb = pd.read_excel(sq.RAW / "squambase_1.xlsx", usecols=["Species name (Binomial)", "Major Biome (GARD 1.7)"])
    biome = dict(zip(sb["Species name (Binomial)"].str.replace(" ", "_"), sb["Major Biome (GARD 1.7)"]))
    return np.array([2 if not isinstance(biome.get(t), str) else int(biome[t] == BIOMES[driver]) for t in tips_names])


def build(theta, G4):
    """theta: log gains of 4 body traits, log body loss, log env gain, log env loss, Gamma env->body (4), Gamma body->env (4)."""
    G = np.zeros((K, K))
    G[:4, :4] = G4
    G[4, :4] = theta[7:11]
    G[:4, 4] = theta[11:15]
    gain = np.exp(np.r_[theta[:4], theta[5]])
    loss = np.r_[np.full(4, np.exp(theta[4])), np.exp(theta[6])]
    return walks.TraitModel(gain, loss, gamma=1.0, D=G, keep=KEEP5)


def restricted(model):
    Q = model.generator().toarray()[np.ix_(VALID5, VALID5)]
    lam = 2 * (-np.diag(Q)).max()
    return Q, np.eye(len(VALID5)) + Q / lam, lam


def time_to_snake(model, start_env_share):
    Q, P, lam = restricted(model)
    target = [i for i, x in enumerate(VALID5) if (x & 0b101) == 0b101]
    h = walks.hitting_times(P, target) / lam * 100
    out_i, in_i = VALID5.index(0), VALID5.index(1 << 4)
    return float((1 - start_env_share) * h[out_i] + start_env_share * h[in_i])


def what_ifs(theta, G4, share_now):
    m = build(theta, G4)
    base = time_to_snake(m, share_now)
    r = {"time_now_Myr": base,
         "time_start_outside_Myr": time_to_snake(m, 0.0), "time_start_inside_Myr": time_to_snake(m, 1.0),
         "exposure_none_change": time_to_snake(m, 0.0) / base - 1,
         "exposure_double_change": time_to_snake(m, min(1.0, 2 * share_now)) / base - 1,
         "exposure_all_change": time_to_snake(m, 1.0) / base - 1}
    th = theta.copy()
    th[5] += np.log(2)  # entry into the environment twice as fast
    r["entry_rate_doubled_change"] = time_to_snake(build(th, G4), share_now) / base - 1
    th = theta.copy()
    th[6] += np.log(2)  # leaving the environment twice as fast (habitat loss)
    r["exit_rate_doubled_change"] = time_to_snake(build(th, G4), share_now) / base - 1
    return r


def network_walk(theta_draws, driver, rho):
    d, age, form = sn.load()
    env = environment(d.tip, driver)
    net_rng = np.random.default_rng(6850)
    R = sn.knn_graph(age, sn.K_NN, net_rng)
    L = sn.knn_graph(form, sn.K_NN, net_rng)
    G = ((R + L) > 0).astype(float)
    snake = np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")["snake"]
    X = sw.species_states(d)
    ordinary = np.flatnonzero(X.sum(1) == 0)
    base = sw.walk_matrix(G, X, rho)
    gains = np.clip(X[None, :, :] - X[:, None, :], 0, None)  # traits gained on a step i -> j
    others = np.flatnonzero(env == 0)
    move = np.random.default_rng(1).choice(others, size=int(0.2 * len(others)), replace=False)
    out = {"now": [], "scenario": []}
    for th in theta_draws:
        g = th[7:11]
        for name, e in (("now", env == 1), ("scenario", (env == 1) | np.isin(np.arange(len(env)), move))):
            tilt = np.exp(gains @ g) * e[:, None] + (~e)[:, None]
            W = base * tilt
            P = W / W.sum(1, keepdims=True)
            out[name].append(sw.analyze_walk(P, snake, ordinary)["mean_steps_ordinary_to_snake_like"])
    now, sc = np.array(out["now"]), np.array(out["scenario"])
    return {"species_in_environment": int((env == 1).sum()), "species_known": int((env != 2).sum()),
            "steps_now": unc.pct(now), "steps_with_20pct_more_species_in_environment": unc.pct(sc),
            "change": unc.pct(sc / now - 1)}


def main(driver):
    parent, length, obs, d, X, _ = sq.load()
    env = environment(d.tip, driver)
    tips = {v: fit.tip_vector(list(o) + [a], K) for (v, o), a in zip(obs.items(), env)}
    known = env != 2
    share_now = float(env[known].mean())
    p4 = json.loads((OUT / f"squamate_fit_params{sq.SUFFIX}.json").read_text())["P_pairwise"]
    G4 = np.array(p4["Gamma"])
    nll = lambda th: -fit.log_likelihood(build(th, G4), parent, length, tips)
    x0 = np.r_[np.log(p4["gain"]), np.log(p4["loss"]), np.log(0.1), np.log(0.3), np.zeros(8)]
    bounds = [(-8, 6)] * 7 + [(-6, 6)] * 8
    cache = OUT / f"squamate_driver_{driver}_fit.json"
    if cache.exists():
        res_x, res_fun = np.array(json.loads(cache.read_text())["theta"]), json.loads(cache.read_text())["nll"]
    else:
        res = opt.minimize(lambda th: (lambda v: 1e12 if not np.isfinite(v) else v)(nll(th)), x0, method="L-BFGS-B", bounds=bounds)
        res_x, res_fun = res.x, float(res.fun)
        cache.write_text(json.dumps({"theta": res_x.tolist(), "nll": res_fun, "converged": bool(res.success)}))
    th0 = res_x.copy()
    th0[7:15] = 0  # no environment couplings: likelihood-ratio test
    res0 = opt.minimize(lambda th: nll(np.r_[th, np.zeros(8)]), res_x[:7], method="L-BFGS-B", bounds=bounds[:7])
    lr = 2 * (res0.fun - res_fun)
    from scipy.stats import chi2

    at_bound = [i for i in range(15) if (i < 7 and (res_x[i] <= -7.9 or res_x[i] >= 5.9)) or (i >= 7 and abs(res_x[i]) >= 5.9)]
    free = [i for i in range(15) if i not in at_bound]
    H = unc.hessian(nll, res_x, free)
    w, V = np.linalg.eigh(H)
    cov = V @ np.diag(1 / np.maximum(w, 1e-6)) @ V.T
    se = np.zeros(15)
    se[free] = np.sqrt(np.diag(cov))
    labels = [f"{driver} -> {t}" for t in sq.TRAITS] + [f"{t} -> {driver}" for t in sq.TRAITS]
    couplings = {}
    for k, lab in enumerate(labels):
        i = 7 + k
        if i in at_bound:
            couplings[lab] = {"multiplier": float(np.exp(res_x[i])), "at_bound": True}
        else:
            couplings[lab] = {"multiplier": round(float(np.exp(res_x[i])), 3),
                              "ci95": [round(float(np.exp(res_x[i] - 1.96 * se[i])), 3), round(float(np.exp(res_x[i] + 1.96 * se[i])), 3)]}
    rng = np.random.default_rng(6850)
    draws = np.tile(res_x, (N_DRAWS, 1))
    draws[:, free] = rng.multivariate_normal(res_x[free], cov, size=N_DRAWS)
    wi = [what_ifs(th, G4, share_now) for th in draws]
    point = what_ifs(res_x, G4, share_now)
    what = {k: {"estimate": round(point[k], 4), "ci95": unc.pct([w_[k] for w_ in wi])} for k in point}
    rho = np.array([v["gain_to_loss_ratio"] for v in json.loads((OUT / f"squamate_time{sq.SUFFIX}.json").read_text())["trait_changes_per_history"].values()])
    walk = network_walk(draws[:50], driver, rho)
    summary = {"driver": driver, "biome": BIOMES[driver], "share_of_species_now": share_now,
               "loglik_with_couplings": -res_fun, "loglik_without": -float(res0.fun),
               "likelihood_ratio": float(lr), "p_value_df8": float(chi2.sf(lr, 8)),
               "couplings": couplings, "what_ifs": what, "species_network_walk": walk}
    (OUT / f"squamate_driver_{driver}{sq.SUFFIX}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
