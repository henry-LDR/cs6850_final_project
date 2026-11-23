"""Known-truth check of the stochastic-mapping pipeline (needs no raw data).

We simulate the fitted trait model on random 2,000-tip trees as deep as the squamate
tree (206 Myr), keeping the full history, so the true number of snake-like origins,
the true journey times and the true ordinary-lizard stay are known. We then hand only
the tip states to the mapping code used in experiments/squamate_time.py and
squamate_uncertainty.py and compare what it reports with the truth.

Run from the repository root: python experiments/mapping_validation.py
Writes results/mapping_validation.json.
"""
import importlib.util
import json

import numpy as np

from evograph import trees, walks
def module(name):
    spec = importlib.util.spec_from_file_location(name, f"experiments/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
un = module("squamate_uncertainty"); sq, st = un.sq, un.st
p = json.load(open("results/squamate_fit_params.json"))["P_pairwise"]
model = walks.TraitModel(p["gain"], np.full(4, p["loss"]), gamma=1.0, D=np.array(p["Gamma"]), keep=sq.KEEP)
Q = model.generator().toarray(); ex = -np.diag(Q)
out = {}
for seed in (1, 2, 3, 4):
    rng = np.random.default_rng(seed)
    parent, length = trees.yule_tree(2000, rng=seed)
    parent = np.asarray(parent); length = trees.rescale(np.asarray(length, float), parent, height=2.06)
    n = len(parent); depth = np.zeros(n)
    for v in range(1, n): depth[v] = depth[parent[v]] + length[v]
    state, events = {0: 0}, {}
    for v in range(1, n):
        x, t, path = state[parent[v]], 0.0, []
        while ex[x] > 0:
            t += rng.exponential(1 / ex[x])
            if t > length[v]: break
            x = int(rng.choice(16, p=np.clip(Q[x], 0, None) / ex[x])); path.append((t, x))
        state[v], events[v] = x, path
    js = st.journeys(parent, depth, state, events)
    occ = exits = 0.0
    for v, path in events.items():
        prev, t0 = state[parent[v]], 0.0
        for s, y in path:
            if prev == 0: occ += s - t0; exits += 1
            prev, t0 = y, s
        if prev == 0: occ += length[v] - t0
    tips = trees.tips(parent)
    obs = {int(v): [(state[v] >> i) & 1 for i in range(4)] for v in tips}
    truth = {"origins": len(js), "journey_median_Myr": round(float(np.median([sum(t for _, t in j["stays"]) * 100 for j in js])), 1),
             "journey_mean_Myr": round(float(np.mean([sum(t for _, t in j["stays"]) * 100 for j in js])), 1),
             "ancestor_mean_stay_Myr": round(float(occ / exits * 100), 1), "tips_snake_like": int(sum((state[v] & 5) == 5 for v in tips)),
             "tips_ordinary": int(sum(state[v] == 0 for v in tips))}
    hs = un.history_stats(parent, length, depth, obs, model, np.random.default_rng(seed + 10), 20)
    mapped = {k: round(float(hs[k]), 1) for k in ("origins", "journey_median_Myr", "ancestor_mean_stay_Myr")}
    out[f"tree_{seed}"] = {"truth": truth, "recovered_from_tips": mapped}
    print(seed, out[f"tree_{seed}"], flush=True)
json.dump(out, open("results/mapping_validation.json", "w"), indent=2)
