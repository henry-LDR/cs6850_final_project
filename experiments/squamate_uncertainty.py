"""Error bars for the squamate trait model and everything computed from it.

Parameter uncertainty: a numerical Hessian of the log-likelihood at the fitted pairwise
model (experiments/squamate_fit.py) gives an approximate covariance for the free
parameters (log gain rates, log loss rate, couplings). Couplings at the optimizer's bound
(6, a 400-fold multiplier) are held there: their intervals are one-sided, "at least 400x".

Propagation: we draw N_DRAWS parameter vectors from that normal approximation and, for
each, recompute
  - the mean time from the ancestral lizard to a snake-like body and the route shares,
  - MAPS_PER_DRAW stochastic maps on the dated tree (time at each step, origins,
    gain-to-loss ratios), so the interval covers both parameter and mapping uncertainty,
  - the biased random walk on the species network (steps to the snake-like community).
We report 2.5%, 50% and 97.5% percentiles.

Run from the repository root after experiments/squamate_fit.py and experiments/species_network.py.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from evograph import fit, walks

here = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sq = module("squamate_fit")
st = module("squamate_time")
sw = module("species_walk")
sn = sw.sn
OUT = Path("results")
N_DRAWS = 60
MAPS_PER_DRAW = 10
STEP = 1e-3


def model_from_theta(theta):
    k, free = sq.K, sq.FREE
    G = np.zeros((k, k))
    G[free] = theta[k + 1 : k + 1 + int(free.sum())]
    return walks.TraitModel(np.exp(theta[:k]), np.full(k, np.exp(theta[k])), gamma=1.0, D=G, keep=sq.KEEP)


def hessian(f, x, idx, h=STEP):
    """Central-difference Hessian of f over the coordinates idx."""
    n = len(idx)
    H = np.zeros((n, n))
    f0 = f(x)
    def at(*moves):
        y = x.copy()
        for i, s in moves:
            y[idx[i]] += s * h
        return f(y)
    for i in range(n):
        H[i, i] = (at((i, 1)) - 2 * f0 + at((i, -1))) / h**2
        for j in range(i):
            H[i, j] = H[j, i] = (at((i, 1), (j, 1)) - at((i, 1), (j, -1)) - at((i, -1), (j, 1)) + at((i, -1), (j, -1))) / (4 * h * h)
    return H


def history_stats(parent, length, depth, obs, model, rng, maps):
    Q = model.generator().toarray()
    n = len(parent)
    informative = np.zeros(n, bool)
    for v in range(n - 1, -1, -1):
        if v in obs:
            informative[v] = True
        if informative[v] and parent[v] >= 0:
            informative[parent[v]] = True
    nodes = [v for v in range(1, n) if informative[v]]
    P = dict(zip(nodes, sla.expm(Q[None] * length[nodes][:, None, None])))
    L = np.ones((n, 2**sq.K))
    for v in range(n - 1, 0, -1):
        if informative[v]:
            if v in obs:
                L[v] *= fit.tip_vector(obs[v], sq.K)
            L[v] /= L[v].max()
            L[parent[v]] *= P[v] @ L[v]
    occ, exits = np.zeros(2**sq.K), np.zeros(2**sq.K)
    gains, losses = np.zeros(sq.K), np.zeros(sq.K)
    origins, journey, elong_first = [], [], []
    for _ in range(maps):
        state, events, _ = st.sample_history(parent, length, nodes, P, L, Q, rng)
        js = st.journeys(parent, depth, state, events)
        origins.append(len(js))
        journey += [sum(t for _, t in j["stays"]) * 100 for j in js]
        elong_first += [next((bool(x & 1) for x, _ in j["stays"] if x & 0b111), True) for j in js]
        for v, path in events.items():
            prev, t0 = state[parent[v]], 0.0
            for s, y in path:
                occ[prev] += s - t0
                exits[prev] += 1
                i = (prev ^ y).bit_length() - 1
                (gains if y >> i & 1 else losses)[i] += 1
                prev, t0 = y, s
            occ[prev] += length[v] - t0
    return {"ancestor_mean_stay_Myr": occ[0] / max(exits[0], 1) * 100, "journey_median_Myr": float(np.median(journey)),
            "origins": float(np.mean(origins)), "elongate_before_digit_loss": float(np.mean(elong_first)),
            "rho": gains / np.maximum(losses, 1)}


def pct(values):
    v = np.asarray(values, float)
    return [round(float(q), 4) for q in np.percentile(v, [2.5, 50, 97.5])]


def main():
    parent, length, obs, *_ = sq.load()
    depth = np.zeros(len(parent))
    for v in range(1, len(parent)):
        depth[v] = depth[parent[v]] + length[v]
    tips = {v: fit.tip_vector(o, sq.K) for v, o in obs.items()}
    fitted = json.loads((OUT / f"squamate_fit_params{sq.SUFFIX}.json").read_text())["P_pairwise"]
    theta = np.array(fitted["theta"], float)
    nll = lambda th: -fit.log_likelihood(model_from_theta(th), parent, length, tips)
    pos = sq.K + 1
    at_bound = [i for i in range(len(theta)) if (i < pos and (theta[i] <= -7.9 or theta[i] >= 5.9)) or (i >= pos and abs(theta[i]) >= 5.9)]
    free_idx = [i for i in range(len(theta)) if i not in at_bound]
    H = hessian(nll, theta, free_idx)
    w, V = np.linalg.eigh(H)
    w_clip = np.maximum(w, 1e-6)
    cov = V @ np.diag(1 / w_clip) @ V.T
    se = np.sqrt(np.diag(cov))

    names = [f"log gain {t}" for t in sq.TRAITS] + ["log loss"] + [f"Gamma {sq.TRAITS[j]} -> {sq.TRAITS[i]}" for j in range(sq.K) for i in range(sq.K) if sq.FREE[j, i]]
    params = {}
    for i, name in enumerate(names):
        if i in at_bound:
            params[name] = {"estimate": float(theta[i]), "at_bound": True}
        else:
            s = se[free_idx.index(i)]
            params[name] = {"estimate": round(float(theta[i]), 3), "se": round(float(s), 3),
                            "ci95": [round(float(theta[i] - 1.96 * s), 3), round(float(theta[i] + 1.96 * s), 3)]}
    multipliers = {n.replace("Gamma ", ""): {"multiplier": round(float(np.exp(v["estimate"])), 2),
                                             "ci95": [round(float(np.exp(c)), 2) for c in v["ci95"]] if "ci95" in v else "at least 400 (at bound)"}
                   for n, v in params.items() if n.startswith("Gamma")}

    rng = np.random.default_rng(6850)
    draws = rng.multivariate_normal(theta[free_idx], cov, size=N_DRAWS)
    d, age, form = sn.load()
    net_rng = np.random.default_rng(6850)  # same tie-breaks as experiments/species_walk.py
    R = sn.knn_graph(age, sn.K_NN, net_rng)
    Lk = sn.knn_graph(form, sn.K_NN, net_rng)
    G = ((R + Lk) > 0).astype(float)
    snake = np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")["snake"]
    X = sw.species_states(d)
    ordinary = np.flatnonzero(X.sum(1) == 0)
    unbiased = sw.analyze_walk(sw.walk_matrix(G, X, None), snake, ordinary)["mean_steps_ordinary_to_snake_like"]
    rec = {k: [] for k in ("time_to_snake_like_Myr", "elongate_first_route_share", "ancestor_mean_stay_Myr", "journey_median_Myr",
                           "origins", "elongate_before_digit_loss", "biased_walk_steps", "walk_speedup")}
    rho_all = []
    for n_draw, x in enumerate(draws):
        th = theta.copy()
        th[free_idx] = x
        model = model_from_theta(th)
        Q, P, lam = sq.restricted_chain(model)
        target = [i for i, s in enumerate(sq.VALID) if (s & 0b101) == 0b101]
        rec["time_to_snake_like_Myr"].append(walks.hitting_times(P, target)[0] / lam * 100)
        order, _ = sq.routes_to_snake(Q, runs=4000, seed=n_draw)
        rec["elongate_first_route_share"].append(sum(v for k, v in order.items() if k.startswith("elongate")))
        hs = history_stats(parent, length, depth, obs, model, rng, MAPS_PER_DRAW)
        for k in ("ancestor_mean_stay_Myr", "journey_median_Myr", "origins", "elongate_before_digit_loss"):
            rec[k].append(hs[k])
        rho_all.append(hs["rho"])
        steps = sw.analyze_walk(sw.walk_matrix(G, X, hs["rho"]), snake, ordinary)["mean_steps_ordinary_to_snake_like"]
        rec["biased_walk_steps"].append(steps)
        rec["walk_speedup"].append(unbiased / steps)
        print(n_draw, {k: round(v[-1], 2) for k, v in rec.items()}, flush=True)
    rho_all = np.array(rho_all)
    summary = {
        "method": f"Hessian normal approximation, {N_DRAWS} parameter draws x {MAPS_PER_DRAW} stochastic maps each",
        "coupling_multipliers": multipliers,
        "parameters": params,
        "derived_95_intervals": {k: pct(v) for k, v in rec.items()},
        "gain_to_loss_ratio_95_intervals": {sq.TRAITS[i]: pct(rho_all[:, i]) for i in range(sq.K)},
        "unbiased_walk_steps": unbiased,
    }
    (OUT / f"squamate_uncertainty{sq.SUFFIX}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("coupling_multipliers", "derived_95_intervals", "gain_to_loss_ratio_95_intervals")}, indent=2))


if __name__ == "__main__":
    main()
