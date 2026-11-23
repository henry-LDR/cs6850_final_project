"""Time bottlenecks: how long squamate lineages spend at each step on the way to a snake-like body.

Stochastic character mapping (Nielsen 2002; Huelsenbeck, Nielsen and Bollback 2003):
using the 4-trait model fitted by experiments/squamate_fit.py, we sample complete
histories of body-plan change on the dated Zheng and Wiens (2016) tree, each
consistent with the traits of the 1,369 species at the tips.

  1. Pruning gives, for every node, the likelihood of the data below it for each state.
  2. Going from the root (an ordinary lizard body, state 0) toward the tips, each node's
     state is drawn given its parent's state and the data below it.
  3. Along each branch, a path of changes between the two end states is drawn by
     simulating the model forward and keeping paths that end in the right state.

From each sampled history we record every origin of a snake-like body (a lineage
becoming both elongate and digitless), and trace that lineage back to the last
time it was an ordinary lizard. That gives each journey's departure time and how
long it spent in every intermediate body plan. A step where lineages spend a
long time is a bottleneck in time.

Run from the repository root after experiments/squamate_fit.py.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

from evograph import fit, trees, walks

spec = importlib.util.spec_from_file_location("sq", Path(__file__).with_name("squamate_fit.py"))
sq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sq)

N_MAPS = 100
SNAKE = 0b101  # elongate and digitless
OUT = Path("results")


def snake(x):
    return (x & SNAKE) == SNAKE


def prepare():
    parent, length, obs, *_ = sq.load()
    p = json.loads((OUT / f"squamate_fit_params{sq.SUFFIX}.json").read_text())["P_pairwise"]
    model = walks.TraitModel(p["gain"], np.full(sq.K, p["loss"]), gamma=1.0, D=np.array(p["Gamma"]), keep=sq.KEEP)
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
    tipvec = {v: fit.tip_vector(o, sq.K) for v, o in obs.items()}
    L = np.ones((n, 2**sq.K))
    for v in range(n - 1, 0, -1):
        if not informative[v]:
            continue
        if v in tipvec:
            L[v] *= tipvec[v]
        L[v] /= L[v].max()
        L[parent[v]] *= P[v] @ L[v]
    depth = np.zeros(n)
    for v in range(1, n):
        depth[v] = depth[parent[v]] + length[v]
    return parent, length, depth, nodes, P, L, Q


def sample_history(parent, length, nodes, P, L, Q, rng):
    exit_rate = -np.diag(Q)
    jump = np.where(exit_rate[:, None] > 0, np.clip(Q, 0, None) / np.where(exit_rate > 0, exit_rate, 1)[:, None], 0)
    cum = np.cumsum(jump, axis=1)
    state = {0: 0}
    events, failures = {}, 0
    for v in nodes:
        a = state[parent[v]]
        w = P[v][a] * L[v]
        b = int(rng.choice(len(w), p=w / w.sum()))
        state[v] = b
        t, path = length[v], None
        for _ in range(5000):
            x, s, trial = a, 0.0, []
            if a != b:  # condition on at least one change: truncated exponential first wait
                r = exit_rate[a]
                s = -np.log(1 - rng.random() * (1 - np.exp(-r * t))) / r
                x = int(np.searchsorted(cum[a], rng.random() * cum[a, -1]))
                trial.append((s, x))
            while exit_rate[x] > 0:
                s += rng.exponential(1 / exit_rate[x])
                if s > t:
                    break
                x = int(np.searchsorted(cum[x], rng.random() * cum[x, -1]))
                trial.append((s, x))
            if x == b:
                path = trial
                break
        if path is None:  # keep the endpoints and put one direct change mid-branch
            failures += 1
            path = [(t / 2, b)] if a != b else []
        events[v] = path
    return state, events, failures


def journeys(parent, depth, state, events):
    """For each first-time snake-like origin (no snake-like ancestor), trace the lineage back
    to the last time it was an ordinary lizard and record the time spent in each state."""
    out, ever = [], {0: False}
    for v, path in events.items():  # events are in index order: parents before children
        been = ever[parent[v]] or snake(state[parent[v]])
        for s, y in path:
            if snake(y) and not been:
                out.append(trace_back(parent, depth, state, events, v, s))
                been = True
        ever[v] = been or snake(state[v])
    return out


def trace_back(parent, depth, state, events, v, s):
    """Walk back from time s on branch v to the last time the lineage was in state 0."""
    stays, end, node, cutoff = [], depth[parent[v]] + s, v, s
    while True:
        start_state = state[parent[node]]
        path = [(t, y) for t, y in events.get(node, []) if t < cutoff]
        times = [depth[parent[node]] + t for t, _ in path]
        segments = list(zip([depth[parent[node]]] + times, [start_state] + [y for _, y in path]))
        for t0, x in reversed(segments):
            if x == 0:
                return {"departure_depth": end, "ordinary_since": t0, "stays": stays[::-1]}
            stays.append((x, end - t0))
            end = t0
        node = parent[node]
        if node <= 0:
            return {"departure_depth": 0.0, "stays": stays[::-1], "never_ancestral": True}
        cutoff = np.inf


def main():
    OUT.mkdir(exist_ok=True)
    parent, length, depth, nodes, P, L, Q = prepare()
    H = depth.max()
    rng = np.random.default_rng(6850)
    origins, journeys_all, failures = [], [], 0
    occupancy = np.zeros(2**sq.K)
    exits = np.zeros(2**sq.K)
    gains, losses = np.zeros(sq.K), np.zeros(sq.K)  # trait changes summed over all sampled histories
    for m in range(N_MAPS):
        state, events, f = sample_history(parent, length, nodes, P, L, Q, rng)
        failures += f
        js = journeys(parent, depth, state, events)
        origins.append(len(js))
        journeys_all += js
        for v, path in events.items():  # empirical time in each state and number of departures
            prev, t0 = state[parent[v]], 0.0
            for s, y in path:
                occupancy[prev] += s - t0
                exits[prev] += 1
                i = (prev ^ y).bit_length() - 1
                (gains if y >> i & 1 else losses)[i] += 1
                prev, t0 = y, s
            occupancy[prev] += length[v] - t0
    to_myr = 100.0
    stay = {sq.label(x): float(occupancy[x] / exits[x] * to_myr) for x in sq.VALID if exits[x] > 0}
    per_state = {}
    for j in journeys_all:
        for x, t in j["stays"]:
            per_state.setdefault(sq.label(x), []).append(t * to_myr)
    total = [sum(t for _, t in j["stays"]) * to_myr for j in journeys_all]
    departure = [(H - j["departure_depth"]) * to_myr for j in journeys_all]
    elong_first = np.mean([next((bool(x & 1) for x, _ in j["stays"] if x & 0b111), True) for j in journeys_all])
    digitless_first = np.mean([any((x & 0b100) and not (x & 1) for x, _ in j["stays"]) for j in journeys_all])
    summary = {
        "maps": N_MAPS,
        "branch_paths_that_fell_back": failures,
        "independent_snake_like_origins_per_history": {"median": float(np.median(origins)), "range": [int(min(origins)), int(max(origins))]},
        "journey_from_ordinary_lizard_to_snake_like_Myr": {"median": float(np.median(total)), "quartiles": np.round(np.quantile(total, [0.25, 0.75]), 1).tolist()},
        "departure_Myr_before_present": {"median": float(np.median(departure)), "quartiles": np.round(np.quantile(departure, [0.25, 0.75]), 1).tolist()},
        "time_in_each_state_on_journeys_Myr": {k: {"median": round(float(np.median(v)), 2), "mean": round(float(np.mean(v)), 2), "visits": len(v)}
                                               for k, v in sorted(per_state.items(), key=lambda kv: -np.median(kv[1]))},
        "share_of_journeys_elongate_before_any_digit_loss": float(elong_first),
        "share_of_journeys_digitless_before_elongate": float(digitless_first),
        "trait_changes_per_history": {sq.TRAITS[i]: {"gains": float(gains[i] / N_MAPS), "losses": float(losses[i] / N_MAPS),
                                                      "gain_to_loss_ratio": float(gains[i] / max(losses[i], 1))} for i in range(sq.K)},
        "mean_stay_all_branches_Myr": {k: round(v, 1) for k, v in sorted(stay.items(), key=lambda kv: -kv[1])},
    }
    (OUT / f"squamate_time{sq.SUFFIX}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
