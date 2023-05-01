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


# Declared Constants
verbose = True


def parse_graphs(path, bond_options):
    """
    parse_graphs() reads all folders in a path and extracts the hbond, ionic bond, and adj bond graphs from the folder. The folders are expected to have been outputted by DiffBond_v2.py
    """
    G_hb = []
    G_ionic = []
    G_adj = []
    G_contact = []
    print("### READING FOLDERS ###")
    for i in os.listdir(path):
        # print("Dir:", i)
        if i == "graph_comparison.py" or i == "Results":
            continue
        if "h" in bond_options:
            f = open(path + "/" + i + "/hbond.gml", "rb")
            graph = nx.read_gml(f)
            G_hb.append(graph)

        if "i" in bond_options:
            f = open(path + "/" + i + "/ionic_bonds.gml", "rb")
            graph = nx.read_gml(f)
            G_ionic.append(graph)

        if "a" in bond_options:
            f = open(path + "/" + i + "/adj_bonds.gml", "rb")
            graph = nx.read_gml(f)
            G_adj.append(graph)

        if "c" in bond_options:
            f = open(path + "/" + i + "/contact_bonds.gml", "rb")
            graph = nx.read_gml(f)
            G_contact.append(graph)

    return G_hb, G_ionic, G_adj, G_contact


def get_grakel_graphs(graphs):
    """
    this function converts graphs to networkx objects
    """
    G = grakel.graph_from_networkx(
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


def combine_PDB_structures(paths):
    """Combines 2 pdb structures into one PDB with 2 separate models. Models are named 0 and 1.

    Args:
        paths (list of two strings): the list contains the two strings for the 2 pdb files to combine.

    Returns:
        Bio.PDB.Structure : a structure with both models in it
    """
    parser = Bio.PDB.PDBParser()
    structures = []
    count = 0
    for path in paths:
        structures.append(parser.get_structure(count, path))
        count = count + 1

    final_model = Bio.PDB.Structure.Structure("master")
    count = 0
    for structure in structures:
        for model in list(structure):
            new_model = model.copy()
            new_model.id = count
            new_model.serial_num = count + 1
            count = count + 1
            final_model.add(new_model)

    # for model in final_model.get_models():
    #     print(model)
    #     for chain in model.get_chains():
    #         print(chain)
    #         for residue in chain.get_residues():
    #             print(residue)

    return final_model


def main():
    G_hb, G_ionic, G_adj, G_contact = parse_graphs(
        "PDB_dataset/trypsin-BPTi/DiffBond_results", ["i", "c", "h", "a"]
    )
    graphs = compose_graphs(G_hb, G_ionic)

    graph1 = graphs[0]
    graph2 = graphs[1]

    print("1st graph:", graph1)
    print("2nd graph:", graph2)
