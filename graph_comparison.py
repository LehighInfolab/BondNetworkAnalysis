import getopt
import os
import math
import sys
import argparse
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt

import itertools
from itertools import combinations

import networkx as nx
from networkx import isomorphism
from gklearn.kernels import commonWalkKernel

import grakel
from grakel.utils import graph_from_networkx
from grakel import Graph
from grakel import GraphKernel

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


def parse_graphs(path):
    """
    parse_graphs() reads all folders in a path and extracts the hbond, ionic bond, and adj bond graphs from the folder. The folders are expected to have been outputted by DiffBond_v2.py
    """
    G_hb = []
    G_ionic = []
    G_adj = []
    print("### READING FOLDERS ###")
    for i in os.listdir(path):
        # print("Dir:", i)
        if i == "graph_comparison.py" or i == "Results":
            continue
        f = open(path + "/" + i + "/hbond.gml", "rb")
        graph = nx.read_gml(f)
        G_hb.append(graph)

        f = open(path + "/" + i + "/ionic_bonds.gml", "rb")
        graph = nx.read_gml(f)
        G_ionic.append(graph)

        f = open(path + "/" + i + "/adj_bonds.gml", "rb")
        graph = nx.read_gml(f)
        G_adj.append(graph)

    return G_hb, G_ionic, G_adj


def get_grakel_graphs(graphs):
    """
    this function converts graphs to networkx objects
    """
    G = graph_from_networkx(
        graphs,
        node_labels_tag="AA",
        edge_labels_tag="bond_type",
        edge_weight_tag="weight",
    )
    return G


def compose_graphs(G1, G2):
    """
    this function takes multiple graphs and combines them. This is used for combining the hbond, ionic bond, and adj graphs
    """
    graphs = []
    for i in range(len(G1)):
        C = nx.compose(G1[i], G2[i])
        graphs.append(C)
    return graphs


def visualize_graph(G, count):
    """
    Util function for visualizing graph
    """
    print(G)
    ## set up drawing of graphs
    plt.subplot(121)
    nx.draw(G, with_labels=True, pos=nx.circular_layout(G))

    edge_labels = nx.get_edge_attributes(G, "bond_type")
    nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=edge_labels)

    plt.savefig("results/graph" + str(count) + ".png")


def jaccard(g1, g2):
    U = nx.union(g1, g2, rename=("G1_", "G2_"))
    jaccard = nx.jaccard_coefficient(U)
    for u, v, p in jaccard:
        if u.split("_")[0] != v.split("_")[0]:
            print("Jaccard:", f"({u}, {v}) -> {p:.8f}")


def graph_edit_distance(graphs):
    # graph_combs = list(itertools.combinations(graphs, 2))
    dist = nx.graph_edit_distance(graphs[0], graphs[1])
    print("Graph edit distance:", dist)


def gklearn_tests(graphs):
    walks = commonWalkKernel.find_all_walks(graphs[1], 2)
    print("GK - Common Walk Kernel:", walks)


def graph_isomorphism(graphs):
    GM = isomorphism.GraphMatcher(graphs[0], graphs[0])
    print("Are graphs isomorphic?", GM.subgraph_is_isomorphic())


def shortest_path_kernel(G):
    gk = grakel.ShortestPath(normalize=True, with_labels=True)
    K = gk.fit_transform(G)
    return K


def wl_kernel(G):
    gk = grakel.WeisfeilerLehman(
        normalize=True, n_iter=5, base_graph_kernel=grakel.VertexHistogram
    )
    K = gk.fit_transform(G)
    return K


def hadamard(G):
    gk = grakel.HadamardCode(
        normalize=True, n_iter=5, base_graph_kernel=grakel.VertexHistogram
    )
    K = gk.fit_transform(G)
    return K


def subgraph_matching(G):
    gk = grakel.SubgraphMatching(normalize=True, k=5)
    K = gk.fit_transform(G)
    return K


def graph_difference(graph1, graph2):
    # print(graph1.nodes)
    # print(graph2.nodes)

    # nodes_del = graph1.nodes - graph2.nodes
    # nodes_add = graph2.nodes - graph1.nodes
    # edges_del = graph1.edges - graph2.edges
    # edges_add = graph2.edges - graph1.edges

    # print("Nodes added:", nodes_add)
    # print("Nodes deleted:", nodes_del)
    # print("Edges added:", edges_add)
    # print("Edges deleted:", edges_del)

    g1_AA = set(nx.get_node_attributes(graph1, "AA").items())
    g2_AA = set(nx.get_node_attributes(graph2, "AA").items())
    # print(g1_AA)
    # print(g2_AA)
    AA_diff = g1_AA ^ g2_AA

    g1_bondtype = set(nx.get_edge_attributes(graph1, "bond_type").items())
    g2_bondtype = set(nx.get_edge_attributes(graph2, "bond_type").items())
    # print(g1_bondtype)
    # print(g2_bondtype)
    bondtype_diff = g1_bondtype ^ g2_bondtype

    return AA_diff, bondtype_diff


def make_neighborhood_subgraph(graph):
    for node in list(graph.nodes):
        print(node)
        neighbors = list(nx.neighbors(graph, node))
        print(neighbors)


def main():
    G_hb, G_ionic, G_adj = parse_graphs("dataset")
    graphs = compose_graphs(G_hb, G_ionic)
    graphs = compose_graphs(graphs, G_adj)

    # count = 0
    # for i in graphs:
    #     print(i)
    #     visualize_graph(i, count)
    #     count = count + 1
    #     print(nx.get_node_attributes(i, "AA"))
    #     print(nx.get_edge_attributes(i, "bond_type"))

    # jaccard(graphs[1], graphs[1])
    # gklearn_tests(graphs)
    # graph_edit_distance(graphs)
    # graph_isomorphism(graphs)

    # G_gk = get_grakel_graphs(graphs)
    # print(shortest_path_kernel(G_gk))

    # G_gk = get_grakel_graphs(graphs)
    # print(wl_kernel(G_gk))

    # G_gk = get_grakel_graphs(graphs)
    # print(hadamard(G_gk))

    G_gk = get_grakel_graphs(graphs)
    # print(subgraph_matching(G_gk))

    # print(graph_difference(graphs[0], graphs[1]))

    ## Find neighborhood nodes of current node and compare neighborhood to find if similar
    # print(graphs[0].nodes["104"]["AA"])
    make_neighborhood_subgraph(graphs[0])


if __name__ == "__main__":
    main()
