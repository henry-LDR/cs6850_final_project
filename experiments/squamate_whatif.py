"""Empirical what-ifs for burrowing, the driver the tree supports.

Burrowing's effects are fitted, not assumed: in the pairwise model (squamate_fit.py) a
burrowing lineage gains elongation and digit reduction about 16x faster (95% intervals in
results/squamate_uncertainty.json). The what-ifs change only how often lineages take up
burrowing, as habitat change might (surface habitat loss pushing species underground, or
soil degradation and pollution pushing them out), and let the fitted couplings carry the
consequences:

  trait model   burrowing entry rate x2 or x0.5 -> mean time from an ordinary lizard to a
                snake-like body, and the share of routes that start by burrowing
  network walk  the biased walk of species_walk.py with each gain step from a burrowing
                species multiplied by its fitted e^Gamma[burrowing -> trait]; the scenario
                moves 20% of the non-burrowing lizards into burrowing

Error bars: 200 draws from the Hessian normal approximation (as in squamate_uncertainty.py).
Run from the repository root after squamate_fit.py, squamate_time.py and species_network.py.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

from evograph import fit, walks

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("unc", here / "squamate_uncertainty.py")
unc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unc)
sq, sw, sn = unc.sq, unc.sw, unc.sn
OUT = Path("results")
N_DRAWS = 200
F = 3  # burrowing


def scenario_time(theta, factor):
    th = theta.copy()
    th[F] += np.log(factor)
    m = unc.model_from_theta(th)
    Q, P, lam = sq.restricted_chain(m)
    target = [i for i, s in enumerate(sq.VALID) if (s & 0b101) == 0b101]
    return walks.hitting_times(P, target)[0] / lam * 100, Q


def main():
    parent, length, obs, *_ = sq.load()
    tips = {v: fit.tip_vector(o, sq.K) for v, o in obs.items()}
    theta = np.array(json.loads((OUT / f"squamate_fit_params{sq.SUFFIX}.json").read_text())["P_pairwise"]["theta"])
    nll = lambda th: -fit.log_likelihood(unc.model_from_theta(th), parent, length, tips)
    pos = sq.K + 1
    at_bound = [i for i in range(len(theta)) if (i < pos and (theta[i] <= -7.9 or theta[i] >= 5.9)) or (i >= pos and abs(theta[i]) >= 5.9)]
    free = [i for i in range(len(theta)) if i not in at_bound]
    H = unc.hessian(nll, theta, free)
    w, V = np.linalg.eigh(H)
    cov = V @ np.diag(1 / np.maximum(w, 1e-6)) @ V.T
    rng = np.random.default_rng(6850)
    draws = np.tile(theta, (N_DRAWS, 1))
    draws[:, free] = rng.multivariate_normal(theta[free], cov, size=N_DRAWS)

    def run(th, seed):
        base, Qb = scenario_time(th, 1.0)
        up, Qu = scenario_time(th, 2.0)
        down, Qd = scenario_time(th, 0.5)
        share = lambda Q: sq.routes_to_snake(Q, runs=3000, seed=seed)[1].get("fossorial first", 0.0)
        return {"time_now_Myr": base, "burrowing_x2_change": up / base - 1, "burrowing_half_change": down / base - 1,
                "burrow_first_now": share(Qb), "burrow_first_x2": share(Qu), "burrow_first_half": share(Qd)}

    point = run(theta, 0)
    ds = [run(th, i + 1) for i, th in enumerate(draws)]
    trait = {k: {"estimate": round(point[k], 4), "ci95": unc.pct([d[k] for d in ds])} for k in point}

    # species-network walk: burrowing species' gain steps carry the fitted burrowing multipliers
    d, age, form = sn.load()
    net_rng = np.random.default_rng(6850)
    R = sn.knn_graph(age, sn.K_NN, net_rng)
    L = sn.knn_graph(form, sn.K_NN, net_rng)
    G = ((R + L) > 0).astype(float)
    snake = np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")["snake"]
    X = sw.species_states(d)
    ordinary = np.flatnonzero(X.sum(1) == 0)
    rho = np.array([v["gain_to_loss_ratio"] for v in json.loads((OUT / f"squamate_time{sq.SUFFIX}.json").read_text())["trait_changes_per_history"].values()])
    base = sw.walk_matrix(G, X, rho)
    gains = np.clip(X[None, :, :] - X[:, None, :], 0, None)
    burrow = X[:, F] == 1
    others = np.flatnonzero(~burrow)
    move = np.random.default_rng(1).choice(others, size=int(0.2 * len(others)), replace=False)
    burrow_sc = burrow | np.isin(np.arange(len(d)), move)
    k_free = sq.FREE
    steps = {"now": [], "scenario": []}
    for th in draws[:60]:
        Gm = np.zeros((sq.K, sq.K))
        Gm[k_free] = th[pos : pos + int(k_free.sum())]
        g = Gm[F]  # burrowing -> each trait
        for name, b in (("now", burrow), ("scenario", burrow_sc)):
            W = base * (np.exp(gains @ g) * b[:, None] + (~b)[:, None])
            P = W / W.sum(1, keepdims=True)
            steps[name].append(sw.analyze_walk(P, snake, ordinary)["mean_steps_ordinary_to_snake_like"])
    now, sc = np.array(steps["now"]), np.array(steps["scenario"])
    summary = {"trait_model": trait,
               "species_network_walk": {"burrowing_lizards_now": int(burrow.sum()), "moved_into_burrowing": int(len(move)),
                                        "steps_now": unc.pct(now), "steps_scenario": unc.pct(sc), "change": unc.pct(sc / now - 1)}}
    (OUT / f"squamate_whatif{sq.SUFFIX}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
