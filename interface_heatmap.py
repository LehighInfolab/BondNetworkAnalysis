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

sys.path.insert(0, "src")
from proteininterface import ProteinInterface

# Parse arguments from command line
def parseArg():
    parser = argparse.ArgumentParser(
        description="Aligns 2 structures to each other given graphs of the interface."
    )

    parser.add_argument(
        "-i",
        nargs=2,
        required=True,
        metavar="InputResultFolder",
        help="Input result folders from DiffBond containing graphs and PDB file. First input will be the reference structure used for alignment and second input will be the aligned sample structure.",
    )

    parser.add_argument("-o", nargs="?", metavar="Output", help="Output folder name")

    parser.add_argument(
        "-m",
        nargs="+",
        required=True,
        metavar="mode",
        help="Search mode can be multiple combinations of the following options. Must include at least 1 option. Contact = c, Ionic bond = i, Hydrogen bond = h, Salt bridge = S, Cation pi = p",
    )

    parser.add_argument(
        "-k",
        metavar="k-hops",
        type=int,
        default=2,
        help="Size of neighborhood in which to align based on number of k-hops from primary amino acid. With k-hops = 2, the list of amino acids in the neighborhood will be: 1 primary AA, 2 AA within 2 hops of primary AA on each side of sequence = 4 AA, 1 AA on opposite interface, 2 AA within 2 hops of opposite AA on each side of sequence = 4 AA. Total = 10AA.",
    )

    parser.add_argument(
        "-v",
        action="store_true",
        help="Print out additional information when running. Default = False",
    )

    # parse list of args
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print(parser.print_help())

    args = vars(args)
    i_list = args["i"]
    if not i_list:
        raise argparse.ArgumentTypeError("-i requires two inputs.")

    m_list = args["m"]

    k_hops = args["k"]
    verbose = args["v"]
    output = args["o"]

    if output is None:
        output = str(i_list[0]) + "_" + str(i_list[1])

    return i_list, m_list, k_hops, verbose, output


def get_graph_edge_coord(interface):
    graph_edge_coord = []
    for g in range(len(interface.graphs)):
        e = {}
        for i in list(interface.graphs[g].edges):
            bond_type = interface.graphs[g].edges[i]["bond_type"]
            c1 = interface.graphs[g].nodes[i[0]]["coord"]
            c2 = interface.graphs[g].nodes[i[1]]["coord"]
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
            }
        )
    # print(l)
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


def get_coord_dist(a, b, d):
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
    # if len(list1) != len(list2):
    #     return False

    g = nx.Graph()

    l = [("l", d["edge"], d["bond_type"], d["x"], d["y"], d["z"]) for d in list1]
    r = [("r", d["edge"], d["bond_type"], d["x"], d["y"], d["z"]) for d in list2]

    g.add_nodes_from(l, bipartite=0)
    g.add_nodes_from(r, bipartite=1)

    edges = [(a, b) for a in l for b in r if get_coord_dist(a, b, 5)]
    g.add_edges_from(edges)

    no_degree_list = (node for node, degree in g.degree() if degree == 0)
    for i in no_degree_list:
        print(i)

    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))

    m = nx.bipartite.maximum_matching(g, l)

    colors = ["blue" if m.get(a) == b else "grey" for a, b in edges]

    nx.draw_networkx(
        g,
        pos=pos,
        arrows=False,
        labels={n: "%s\n%s" % (n[1], n[2]) for n in g.nodes()},
        edge_color=colors,
    )
    plt.axis("off")
    plt.show()

    return len(m) // 2 == len(list1)


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

    graph_edges = get_graph_edge_coord(interface)
    ge0 = reformat_graph_edge(graph_edges[0])
    ge1 = reformat_graph_edge(graph_edges[1])

    has_a_perfect_match(ge0, ge1)


if __name__ == "__main__":
    main()
