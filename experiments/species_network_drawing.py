"""Draw the squamate species network for the slides: relatives and look-alike links,
coloured by the Cheeger cut found in experiments/species_network.py.

Run from the repository root after experiments/species_network.py.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

spec = importlib.util.spec_from_file_location("sn", Path(__file__).with_name("species_network.py"))
sn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sn)
OUT = Path("results")


def main():
    rng = np.random.default_rng(6850)
    d, age, form = sn.load()
    R, L = sn.knn_graph(age, sn.K_NN, rng), sn.knn_graph(form, sn.K_NN, rng)
    cut = np.load(OUT / f"species_network_cut{sn.SUFFIX}.npz")
    snake, bridge = set(cut["snake"].tolist()), set(cut["bridge"].tolist())
    derived = ((d.Presacrals >= 35) | (d.Digits < 10)).to_numpy()
    G = nx.Graph()
    G.add_nodes_from(range(len(d)))
    rel = sp.triu(R, 1).tocoo()
    look = sp.triu(L, 1).tocoo()
    G.add_edges_from(zip(rel.row, rel.col), kind="relative")
    conv = []
    for i, j in zip(look.row, look.col):
        kind = "convergence" if age[i, j] > sn.CONVERGENT_AGE and derived[i] and derived[j] else "look-alike"
        if not G.has_edge(i, j):
            G.add_edge(i, j, kind=kind)
        if kind == "convergence":
            conv.append((i, j))
    pos = nx.spring_layout(G, seed=3, iterations=200, k=0.9 / np.sqrt(len(d)))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    fig.patch.set_facecolor("#fbfbf8")
    ax.set_facecolor("#fbfbf8")
    other = [e for e in G.edges if G.edges[e]["kind"] != "convergence"]
    nx.draw_networkx_edges(G, pos, edgelist=other, edge_color="#d9d8d3", width=0.35, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=conv, edge_color="#2a78d6", width=0.6, alpha=0.7, ax=ax)
    limbed = [v for v in G if v not in snake]
    nx.draw_networkx_nodes(G, pos, nodelist=limbed, node_size=7, node_color="#a8a7a1", linewidths=0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(snake - bridge), node_size=9, node_color="#2a78d6", linewidths=0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(bridge), node_size=26, node_color="#e34948", linewidths=0, ax=ax)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / f"species_network_drawing{sn.SUFFIX}.png", facecolor=fig.get_facecolor())
    print("nodes", G.number_of_nodes(), "edges", G.number_of_edges(), "convergence edges", len(conv))


if __name__ == "__main__":
    main()
