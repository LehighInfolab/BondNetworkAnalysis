import getopt
import os
import math
import sys
import argparse
import shutil
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

import grakel

from Bio.PDB import Structure
import Bio.PDB

from proteininterface import ProteinInterface


def get_graph_edge_coord(graphs):
    """Gets the center of the bond as a point.

    Args:
        graphs (_type_): _description_

    Returns:
        a list of dictionaries where each dictionary matches an edge to [coordinate of point, bond type]
    """
    graph_edge_coord = []
    for g in range(len(graphs)):
        e = {}
        for i in list(graphs[g].edges):
            bond_type = graphs[g].edges[i]["bond_type"]
            # weight = graphs[g].edges[i]["weight"]
            c1 = graphs[g].nodes[i[0]]["coord"]
            c2 = graphs[g].nodes[i[1]]["coord"]
            c_f = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2]
            e[i] = [c_f, bond_type]

        graph_edge_coord.append(e)

    return graph_edge_coord


def reformat_graph_edge(graph_edge):
    """Reformat graph_edge dictionary to be compatible with has_a_perfect_match(list1,list2)"""
    l = []
    for i in graph_edge:
        l.append(
            {
                "edge": i,
                "bond_type": graph_edge[i][1],
                "x": graph_edge[i][0][0],
                "y": graph_edge[i][0][1],
                "z": graph_edge[i][0][2],
                # "w": graph_edge[i][2],
            }
        )
    return l


def get_bond_dist_match(g_e, d):
    graph_edge_match = {}

    counter = 0
    for i in g_e[0]:
        graph1_edges = []
        c0 = g_e[0][i][0]
        x0 = c0[0]
        y0 = c0[1]
        z0 = c0[2]
        bond_type0 = g_e[0][i][1]
        for j in g_e[1]:
            c1 = g_e[1][j][0]
            x1 = c1[0]
            y1 = c1[1]
            z1 = c1[2]
            bond_type1 = g_e[1][j][1]

            if bond_type0 == bond_type1:
                if math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2) < d:
                    counter = counter + 1
                    graph1_edges.append(j)
        graph_edge_match[i] = graph1_edges

    # for k, v in graph_edge_match.items():
    #     print(k, "->", v)


def check_dist(a, b, d):
    """Check if distance of coordinates from one point is less than distance of coordinates from other point"""
    if a[2] == b[2]:
        x0 = a[3]
        y0 = a[4]
        z0 = a[5]

        x1 = b[3]
        y1 = b[4]
        z1 = b[5]

        if math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2) < d:
            return True
    return False


def has_a_perfect_match(list1, list2):

    g = nx.Graph()

    l = [("l", d["edge"], d["bond_type"], d["x"], d["y"], d["z"]) for d in list1]
    r = [("r", d["edge"], d["bond_type"], d["x"], d["y"], d["z"]) for d in list2]

    g.add_nodes_from(l, bipartite=0)
    g.add_nodes_from(r, bipartite=1)

    edges = [(a, b) for a in l for b in r if check_dist(a, b, 5)]
    g.add_edges_from(edges)

    no_degree_list = (node for node, degree in g.degree() if degree == 0)
    for i in no_degree_list:
        print(i)

    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))

    m = nx.bipartite.maximum_matching(g, l)

    colors = ["blue" if m.get(a) == b else "grey" for a, b in edges]
    match_count = colors.count("blue")
    nonmatch_count = colors.count("grey")

    nx.draw_networkx(
        g,
        pos=pos,
        arrows=False,
        labels={n: "%s\n%s" % (n[1], n[2]) for n in g.nodes()},
        edge_color=colors,
    )
    plt.axis("off")
    plt.show()

    return m, match_count, nonmatch_count


def main():
    # i_list, m_list, k_hops, verbose, output = parseArg()
    i_list = [
        "dataset\\1brs_dataset\\00107",
        "dataset\\1brs_dataset\\00108",
    ]

    # i_list = [
    #     "dataset\\Result_1brs_barnase_A+h_1brs_barstar_D+h",
    #     "dataset\\Result_1brs_barnase_A+h_1brs_barstar_D+h",
    # ]
    m_list = ["i", "h"]
    k_hops = 3
    verbose = False

    interface = ProteinInterface(i_list, m_list, k_hops, verbose=verbose)

    graph_edges = get_graph_edge_coord(interface.graphs)
    ge0 = reformat_graph_edge(graph_edges[0])
    ge1 = reformat_graph_edge(graph_edges[1])

    has_a_perfect_match(ge0, ge1)


if __name__ == "__main__":
    main()
