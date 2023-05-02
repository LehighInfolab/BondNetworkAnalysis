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

# Parse arguments from command line
def parseArg():
    parser = argparse.ArgumentParser(
        description="Aligns 2 structures to each other given graphs of the interface."
    )

    parser.add_argument(
        "-i",
        nargs=2,
        metavar="InputResultFolder",
        help="Input result folders from DiffBond containing graphs and PDB file. First input will be the reference structure used for alignment and second input will be the aligned sample structure.",
    )

    parser.add_argument("-o", nargs="?", metavar="Output", help="Output folder name")

    parser.add_argument(
        "-m",
        nargs="+",
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


def parse_graphs(path, bond_options):
    """
    parse_graphs() reads all folders in a path and extracts the hbond, ionic bond, and adj bond graphs from the folder. The folders are expected to have been outputted by DiffBond_v2.py
    """
    # graphs = {}
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
            f = open(path + "/hbond.gml", "rb")
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
    this function takes 2 graphs and combines them. This is used for combining the hbond, ionic bond, and adj graphs
    """
    C = nx.compose(G1, G2)
    return C


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


def get_adjacent_residues(residue, chains, num_hops):
    """helper function used in set_graph_pdb_pair() for getting the neighboring residues

    Args:
        residue (Bio.PDB.Residue): Current working residue
        chains (Bio.PDB.Chains): Chain for current working residue
        num_hops (int): Number of hops on one side of the residue. If hop=4, then 4 residues on each side for one residue totalling 9 residues.

    Returns:
        prev_res_list: a list of residues before current working residue. Prev list is reversed so that it goes in start to end order.
        next_res_list: a list of residues after current working residue
    """
    prev_res_list = []
    next_res_list = []
    for hops in range(1, num_hops + 1):
        prev_res_idx = int(residue.id[1]) - hops
        try:
            # print(chains.__getitem__(prev_res_idx))
            prev_res = chains.__getitem__(prev_res_idx)
        except:
            print(
                "Out of bounds at beginning of protein at",
                prev_res_idx,
                "from",
                residue.id[1],
            )
            prev_res = None

        next_res_idx = int(residue.id[1]) + hops
        try:
            # print(chains.__getitem__(next_res_idx))
            next_res = chains.__getitem__(next_res_idx)
        except:
            print(
                "Out of bounds at end of protein at",
                next_res_idx,
                "from",
                residue.id[1],
            )
            next_res = None

        prev_res_list.append(prev_res)
        next_res_list.append(next_res)

    prev_res_list.reverse()
    return prev_res_list, next_res_list


def set_graph_PDB_pair(graph, pdb):
    """Function for creating dictionary pairs of graph nodes to PDB residues and neighbors.

    Args:
        graph (networkx.graph): Current graph to parse nodes and get dictionary of.
        pdb (Bio.PDB.Structure): PDB with combined halves.
        hops (int): Number of hops on one side of the residue. If hop=4, then 4 residues on each side for one residue totalling 9 residues.

    Returns:
        _type_: _description_
    """
    dict = {}
    combination_dict = {}
    # Loop through all nodes in graph
    for node in graph.nodes():
        # Loop through all chains to get the same chain as graph node chain
        for chains in pdb.get_chains():
            if str(graph.nodes[node]["chain"]) == str(chains.id):
                # Loop through all residues to find same residue seq id in PDB as graph node
                for residue in chains.get_residues():
                    # If node is the same as residue, we found a node - residue match and can create a node-residue dict and a residue-neighbors dict as well.
                    if str(node) == str(residue.get_id()[1]):
                        # print(node, graph.nodes[node])
                        # print(residue)
                        dict[str(graph.nodes[node])] = residue

                        prev_res_list, next_res_list = get_adjacent_residues(
                            residue, chains, hops
                        )
                        if None in prev_res_list:
                            continue
                        if None in next_res_list:
                            continue
                        combination_dict[residue] = (
                            prev_res_list + [residue] + next_res_list
                        )

    # Loop through all nodes in graph
    for node in graph.nodes():

        # min variable to store the closest neighboring residues to the current working node
        min = [float("inf"), None]
        # print(graph.nodes[node]["coord"])

        # Saving coordinate of nodes for comparison later.
        node_coord = graph.nodes[node]["coord"]
        x1 = node_coord[0]
        y1 = node_coord[1]
        z1 = node_coord[2]

        # Loop through all chains to get the opposite chain as graph node chain
        for chains in pdb.get_chains():
            if str(graph.nodes[node]["chain"]) != str(chains.id):

                # Loop through all alpha carbons in chain and find the closest alpha carbon to current node on opposite chain.
                try:
                    for residue in chains.get_residues():
                        vector = residue["CA"].get_vector()
                        x2 = vector[0]
                        y2 = vector[1]
                        z2 = vector[2]

                        # Calculate distance. If dist is less than min distance so far, we update min with dist and current residue.
                        dist = math.sqrt(
                            math.pow(x2 - x1, 2)
                            + math.pow(y2 - y1, 2)
                            + math.pow(z2 - z1, 2) * 1.0
                        )
                        if dist < min[0]:
                            min = [dist, residue]

                    # After getting min distance residue on opposite chain, we get the adjacent residue lists.
                    prev_res_list, next_res_list = get_adjacent_residues(
                        min[1], chains, hops
                    )
                    if None in prev_res_list:
                        continue
                    if None in next_res_list:
                        continue
                    try:
                        # Update dictionaries to include matched residues on opposite side.
                        res = dict[str(graph.nodes[node])]
                        combination_dict[res] = (
                            combination_dict[res]
                            + prev_res_list
                            + [min[1]]
                            + next_res_list
                        )
                    except:
                        continue
                except:
                    continue
    # print(list(combination_dict)[0])
    # print(combination_dict[list(combination_dict)[0]])
    return dict, combination_dict


def compile_backbone_atoms(combination_dict):
    atoms_list = []
    for c in combination_dict:
        atoms = []
        for res in combination_dict[c]:
            if res == None:
                continue
            else:
                # print(ref_res.get_list())
                # print(ref_res["CA"])
                atoms.append(res["CA"])
            # print(atoms)
        atoms_list.append(atoms)

    return atoms_list


def get_atom_list_combinations(atom_list):
    middle_idx = int(len(atom_list) / 2)
    chain1 = atom_list[:middle_idx]
    chain2 = atom_list[middle_idx:]

    list_comb = [atom_list]
    list_comb.append(chain1 + list(reversed(chain2)))
    list_comb.append(list(reversed(chain1)) + chain2)
    list_comb.append(list(reversed(chain1)) + list(reversed(chain2)))

    # for i in list_comb:
    #     for j in i:
    #         # print(j)
    #         p = j.get_parent()
    #         print(p.get_parent(), p)

    return list_comb


def get_rmsd_list(ref_atom_list, sample_atom_list, pdb1, pdb2):

    start = time.time()

    min_idx = [float("inf"), -1, -1]
    zeroes_idx = []
    rmsd_list = []
    for i in range(len(ref_atom_list)):
        if len(ref_atom_list[i]) < hops * 4 + 2:
            continue
        for j in range(len(sample_atom_list)):
            if len(sample_atom_list[j]) < hops * 4 + 2:
                continue
            try:
                # Need to try all combinations of ref_atom_list and sample_atom_list for specific alignment

                ref_list_comb = get_atom_list_combinations(ref_atom_list[i])
                sample_list_comb = get_atom_list_combinations(sample_atom_list[j])

                for x in range(len(ref_list_comb)):
                    for y in range(len(sample_list_comb)):
                        super_imposer = Bio.PDB.Superimposer()
                        super_imposer.set_atoms(ref_list_comb[x], sample_list_comb[y])
                        super_imposer.apply(pdb2.get_atoms())
                        # print(super_imposer.rms)
                        rmsd_list.append(
                            [super_imposer.rms, ref_list_comb[x], sample_list_comb[y]]
                        )
                        if super_imposer.rms < min_idx[0] and super_imposer.rms != 0:
                            min_idx = [
                                super_imposer.rms,
                                i,
                                j,
                                ref_list_comb[x],
                                sample_list_comb[y],
                            ]
                        if super_imposer.rms == 0:
                            zeroes_idx.append(
                                [
                                    super_imposer.rms,
                                    i,
                                    j,
                                    ref_list_comb[x],
                                    sample_list_comb[y],
                                ]
                            )
                        # print(
                        #     ref_atom_list[i][0].get_parent(),
                        #     sample_atom_list[j][0].get_parent(),
                        # )
            except np.linalg.LinAlgError as error:
                # print(error)
                continue
    end = time.time()
    print("------------- RMSD calculated for all combinations -------------")
    print("Number of RMSD calculated:", len(rmsd_list))
    print("Time to get RMSD:", end - start)

    return rmsd_list, min_idx, zeroes_idx


def super_imposer_helper(l1, l2, pdb, counter, output_folder):
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(l1, l2)
    super_imposer.apply(pdb.get_atoms())

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb)
    io.save("results/" + output_folder + "/pdb_" + str(counter) + "_aligned.pdb")


def main():
    i_list, m_list, k_hops, verbose, output = parseArg()

    graphs = []
    for i in i_list:
        g_list = parse_graphs(i, m_list)
        G_composed = g_list[0]
        for index in range(1, len(g_list)):
            G_composed = compose_graphs(G_composed, g_list[index])

            if verbose:
                print(
                    "Combining different bond graphs...currently adding in graph",
                    index,
                    "for",
                    i,
                )
        graphs.append(G_composed)

    G_1 = graphs[0]
    G_2 = graphs[1]

    print("1st graph:", G_1)
    print("2nd graph:", G_2)

    pdbs = []
    for i in i_list:
        pdb = combine_PDB_structures(i)

    # # paths = [
    # #     "PDB_dataset/1brs_muts/00104/H1-A/final_half1.pdb",
    # #     "PDB_dataset/1brs_muts/00104/H2-B/final_half2.pdb",
    # # ]

    # paths = [
    #     "PDB_dataset/trypsin-BPTi/1eaw+h-A.pdb",
    #     "PDB_dataset/trypsin-BPTi/1eaw+h-B.pdb",
    # ]

    # # paths2 = [
    # #     "PDB_dataset/trypsin-BPTi/3btg+h-E.pdb",
    # #     "PDB_dataset/trypsin-BPTi/3btg+h-I.pdb",
    # # ]

    # paths2 = [
    #     "PDB_dataset/trypsin-BPTi/1eaw+h-A.pdb",
    #     "PDB_dataset/trypsin-BPTi/1eaw+h-B.pdb",
    # ]

    # # paths2 = [
    # #     "PDB_dataset/1brs_muts/00104/H1-A/final_half1.pdb",
    # #     "PDB_dataset/1brs_muts/00104/H2-B/final_half2.pdb",
    # # ]

    # start = time.time()

    # dict1, combination_dict1 = set_graph_PDB_pair(graph1, pdb1)
    # dict2, combination_dict2 = set_graph_PDB_pair(graph2, pdb2)

    # ref_atom_list = compile_backbone_atoms(combination_dict1)
    # sample_atom_list = compile_backbone_atoms(combination_dict2)

    # end = time.time()
    # print("Time to get graph pairings and compile backbone atoms:", end - start)

    # rmsd_list, min_idx, zeroes_idx = get_rmsd_list(
    #     ref_atom_list, sample_atom_list, pdb1, pdb2
    # )

    # # Increase threshold incrementally to get lowest relevant RMSDs
    # threshold = 0.1
    # rmsd_filtered = []
    # while len(rmsd_filtered) < 3 and threshold < 3:
    #     rmsd_filtered = [i for i in rmsd_list if i[0] < threshold]
    #     threshold = threshold + 0.05

    # print(
    #     "Number of alignments with RMSD less than", threshold, ":", len(rmsd_filtered)
    # )
    # # print(rmsd_filtered)
    # # print(min_idx)

    # print("Min rmsd:", min_idx[0])
    # # print("Set of res on first pdb:")
    # # for x in ref_atom_list[min_idx[1]]:
    # #     p = x.get_parent()
    # #     print(p.get_parent(), p)
    # # print("Set of res on second pdb:")
    # # for x in sample_atom_list[min_idx[2]]:
    # #     p = x.get_parent()
    # #     print(p.get_parent(), p)
    # print("Pairs with rmsd = 0 (These are errors):", zeroes_idx)

    # counter = 0
    # for i in rmsd_filtered:
    #     super_imposer_helper(i[1], i[2], pdb2, counter, output)
    #     counter = counter + 1


if __name__ == "__main__":
    main()
