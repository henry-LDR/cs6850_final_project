# How Traversable Is the Space of Body Plans?

CS/INFO 6850 (Fall 2026) project by Henry Ji and Calvin Zhang.

We model evolution as a biased random walk on a state graph whose nodes are body plans (combinations of k binary traits) and whose edges are single trait changes, so the base graph is the hypercube H_k. Developmental constraints delete edges at random, selection biases the walk, and a phylogeny records how lineages' walks branched. We ask how traversable this space is through the three parts of the course: random graphs and small worlds, cascades, and spectral analysis with random walks. The full plan is in [docs/roadmap.pdf](docs/roadmap.pdf).

## How the course readings map to the code

| Reading | Results we use | Module |
| --- | --- | --- |
| Random Graphs and Expansion | G(n, p); configuration model; expansion α(G) as the minimum surface-to-volume ratio; expansion implies paths of length O((d/α) log n); hypercubes have expansion exactly 1 | `evograph/statespace.py` |
| Spectral Analysis of Graphs | L = D − A; λ₂ as a Rayleigh quotient; Claim 2.2, 2α ≥ λ₂ ≥ α²/(2Δ) | `evograph/spectral.py` |
| Random Walks | lazy walk M′ = I/2 + M/2; mixing in (2d/λ₂)(1.5 ln n + ln 1/δ) steps; irreducible chains, stationary distribution, return time 1/σ | `evograph/walks.py` |
| Cascading Behavior in Networks (AGT ch. 24) | threshold rule from the coordination game; contagion threshold ≤ 1/2; Linear Threshold and Independent Cascade models; greedy seed selection within 1 − 1/e under submodularity | `evograph/cascades.py` |

`tests/test_readings.py` checks that the code reproduces results stated in these readings, for example that α(H₄) = 1, that H_k's Laplacian has eigenvalues 2j with multiplicity C(k, j), and that the lazy walk on H₆ mixes within its bound.

The biased evolutionary walk is a continuous-time chain turned into a lazy discrete-time chain by uniformization, P = I + Q / (2 · maximum exit rate). With equal rates and no constraints, P is exactly the lazy walk M′ on H_k, so every tool above applies. For the biased chain we use conductance (sets weighted by the stationary distribution) in place of expansion.

## Layout

| Path | Contents |
| --- | --- |
| `evograph/statespace.py` | Trait hypercube, constraint masks, G(n, p), configuration model, distances, exact expansion |
| `evograph/spectral.py` | Laplacian, Fiedler vector, sweep cuts, expansion bounds |
| `evograph/walks.py` | Lazy walk, mixing bound, `TraitModel` (gain and loss rates, coupling γ, dependency graph D), stationary distribution, hitting and return times, integration index |
| `evograph/cascades.py` | Threshold, Linear Threshold and Independent Cascade models; greedy seed selection |
| `evograph/trees.py` | Yule trees, simulation of a `TraitModel` along a tree, counting independent origins |
| `experiments/percolation.py` | Giant component of the constrained hypercube near p = 1/k |
| `results/` | Figures and tables from the experiments |
| `data/README.md` | Datasets and where to get them |
| `docs/` | Roadmap (LaTeX and PDF) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python experiments/percolation.py
```

## Quick start

```python
import numpy as np
from evograph import spectral, walks

# Four crab traits: wide carapace, folded abdomen, reinforced shell, sideways gait.
D = np.ones((4, 4)) - np.eye(4)            # each trait helps the others
model = walks.TraitModel(r_gain=[0.5] * 4, r_loss=[1.0] * 4, gamma=0.8, D=D)
P = model.transition_matrix()
sigma = walks.stationary(P)

print("integration index:", walks.integration_index(model))         # > 1: linked traits attract
print("steps to evolve the crab form from 0000:", walks.hitting_times(P, [0b1111])[0])
S, phi = spectral.sweep_cut(walks.flow_graph(P, sigma), measure=sigma)
print("bottleneck conductance:", phi)
```

## Milestones

| Date | Deliverable |
| --- | --- |
| Nov 16 | Roadmap ([docs/roadmap.pdf](docs/roadmap.pdf)) |
| Nov 23 | Slides: simulations and a first squamate fit |
| Week of Nov 30 | Lightning talk and check-in |
| Dec 15 | Final write-up |

## Next steps

- [ ] Part 1: expansion and diameter of constrained hypercubes against G(n, p) and configuration-model nulls with the same degrees
- [ ] Part 1: hitting times to adaptive states as k and p vary
- [ ] Part 2: fit thresholds on trait-dependency graphs from trait acquisition order (C4, squamates)
- [ ] Part 2: Independent Cascade on a bacterial species network for gene transfer; greedy seed selection
- [ ] Part 3: conductance bottlenecks and choke-point states for fitted models
- [ ] Data: load the squamate tree and traits (see `data/README.md`) and fit `TraitModel` by likelihood
- [ ] Validation: recover planted rates, γ and bottlenecks from trees simulated with `evograph.trees`

Course readings are not committed (one is a book chapter). Keep local copies in `readings/`, which is git-ignored.
