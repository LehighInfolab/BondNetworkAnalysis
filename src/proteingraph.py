import os
import math
import time
import networkx as nx


class ProteinGraph:
    """Class to create a ProteinGraph object with graph util functions"""

    def __init__(self, i, m_list, verbose=False):
        """Create graph

        Args:
            i (string): string of a graph file path outputted by DiffBond
            m_list (string[]): graph modes that can be composed
            verbose (bool, optional): If true, prints additional information. Defaults to False.
        """
        # parsing each graph based on results directories given in i_list
        self.graph = self.parse_graph(i, m_list, verbose)

    def parse_graph_files(self, path, bond_options):
        """
        parse_graphs() reads all folders in a path and extracts the hbond, ionic bond, and adj bond graphs from the folder. The folders are expected to have been outputted by DiffBond_v2.py
        """
        graphs = []
        print("--- Parsing current folder path:", path, "---")

        if "i" in bond_options:
            try:
                f = open(path + "/ionic_bonds.gml", "rb")
            except:
                raise Exception("Missing ionic_bonds.gml file")
            graph = nx.read_gml(f)
            graphs.append(graph)
            # graphs["i"] = graph

        if "h" in bond_options:
            try:
                f = open(path + "/hbonds.gml", "rb")
            except:
                raise Exception("Missing hbond.gml file")
            graph = nx.read_gml(f)
            graphs.append(graph)
            # graphs["h"] = graph

        if "a" in bond_options:
            try:
                f = open(path + "/adj_bonds.gml", "rb")
            except:
                raise Exception("Missing adj_bonds.gml file")
            graph = nx.read_gml(f)
            graphs.append(graph)
            # graphs["a"] = graph

        if "c" in bond_options:
            try:
                f = open(path + "/contact_bonds.gml", "rb")
            except:
                raise Exception("Missing contact_bonds.gml file")
            graph = nx.read_gml(f)
            graphs.append(graph)
            # graphs["c"] = graph

        return graphs

    def compose_graphs(self, G1, G2):
        """
        this function takes 2 graphs and combines them. This is used for combining the hbond, ionic bond, and adj graphs
        """
        C = nx.compose(G1, G2)
        return C

    def parse_graph(self, i, m_list, verbose):
        """_summary_

        Args:
            i_list (string[]): List of 2 input graphs to parse
            m_list (string[]): List of graph modes to compose into one graph. Options are currently i, h, a, c
            verbose (bool): Print additional information if true

        Returns:
            _type_: _description_
        """
        g_list = self.parse_graph_files(i, m_list)
        G_composed = g_list[0]
        for j in range(1, len(g_list)):
            G_composed = self.compose_graphs(G_composed, g_list[j])

            if verbose:
                print(
                    "Combining different bond graphs...currently adding in",
                    m_list[j],
                    "graph to",
                    m_list[0],
                    "graph",
                )

        return G_composed
