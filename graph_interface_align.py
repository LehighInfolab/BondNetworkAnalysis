import getopt
import os
import math
import sys
import argparse
import shutil
import os


import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

import grakel

from Bio.PDB import Structure
import Bio.PDB


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

    # print(prev_res_list)
    # print(next_res_list)
    prev_res_list.reverse()
    return prev_res_list, next_res_list


def set_graph_PDB_pair(graph, pdb, hops):
    dict = {}
    combination_dict = {}
    # Loop through all nodes in graph
    for node in graph.nodes():
        # Loop through all chains to get the same chain as graph node chain
        for chains in pdb.get_chains():
            if str(graph.nodes[node]["chain"]) == str(chains.id):
                # Loop through all residues to find same residue seq id in PDB as graph node
                for residue in chains.get_residues():
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

    # print(dict)
    # print(combination_dict)

    # Loop through all nodes in graph
    for node in graph.nodes():
        # Loop through all chains to get the same chain as graph node chain
        # print(graph.nodes[node]["coord"])
        min = [float("inf"), None]
        node_coord = graph.nodes[node]["coord"]
        x1 = node_coord[0]
        y1 = node_coord[1]
        z1 = node_coord[2]
        for chains in pdb.get_chains():
            if str(graph.nodes[node]["chain"]) != str(chains.id):
                # Loop through all residues to find same residue seq id in PDB as graph node
                for residue in chains.get_residues():
                    vector = residue["CA"].get_vector()
                    x2 = vector[0]
                    y2 = vector[1]
                    z2 = vector[2]

                    dist = math.sqrt(
                        math.pow(x2 - x1, 2)
                        + math.pow(y2 - y1, 2)
                        + math.pow(z2 - z1, 2) * 1.0
                    )
                    if dist < min[0]:
                        min = [dist, residue]

                prev_res_list, next_res_list = get_adjacent_residues(
                    min[1], chains, hops
                )
                if None in prev_res_list:
                    continue
                if None in next_res_list:
                    continue
                try:
                    res = dict[str(graph.nodes[node])]
                    combination_dict[res] = (
                        combination_dict[res] + prev_res_list + [min[1]] + next_res_list
                    )
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


def main():
    G_hb, G_ionic, G_adj, G_contact = parse_graphs("1brs_dataset", "c")
    hops = 1
    # graphs = compose_graphs(G_hb, G_ionic)
    # graphs = compose_graphs(graphs, G_adj)
    # print(G_hb, G_ionic, G_adj)
    # print(G_contact)

    # graph = graphs[0]
    print(G_contact[0])

    paths = [
        "PDB_dataset/1brs_muts/00104/H1-A/final_half1.pdb",
        "PDB_dataset/1brs_muts/00104/H2-B/final_half2.pdb",
    ]

    paths2 = [
        "PDB_dataset/1brs_muts/00105/H1-A/final_half1.pdb",
        "PDB_dataset/1brs_muts/00105/H2-D/final_half2.pdb",
    ]

    pdb = combine_PDB_structures(paths)
    pdb2 = combine_PDB_structures(paths2)

    dict1, combination_dict1 = set_graph_PDB_pair(G_contact[0], pdb, hops)
    dict2, combination_dict2 = set_graph_PDB_pair(G_contact[1], pdb2, hops)

    # add_opposite_nodes(G_contact[0], pdb, dict1, combination_dict1)

    # print(combination_dict1)
    # for i in combination_dict1:
    #     print(i)

    ref_atom_list = compile_backbone_atoms(combination_dict1)
    sample_atom_list = compile_backbone_atoms(combination_dict2)

    # for i in ref_atom_list:
    #     print(len(i))
    # for i in sample_atom_list:
    #     print(len(i))

    # print(sample_atom_list)
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
                # sample_pdb = pdb2.copy()
                super_imposer = Bio.PDB.Superimposer()
                super_imposer.set_atoms(ref_atom_list[i], sample_atom_list[j])
                super_imposer.apply(pdb2.get_atoms())
                # print(super_imposer.rms)
                rmsd_list.append(super_imposer.rms)
                if super_imposer.rms < min_idx[0] and super_imposer.rms != 0:
                    min_idx = [super_imposer.rms, i, j]
                if super_imposer.rms == 0:
                    zeroes_idx.append([super_imposer.rms, i, j])
                    # print(
                    #     ref_atom_list[i][0].get_parent(),
                    #     sample_atom_list[j][0].get_parent(),
                    # )
            except np.linalg.LinAlgError as error:
                # print(error)
                continue
    print(rmsd_list)
    print(len(rmsd_list))
    # print(min(rmsd_list))

    print("Min rmsd:", min_idx[0])
    print("Set of res on first pdb:")
    for x in ref_atom_list[min_idx[1]]:
        p = x.get_parent()
        print(p.get_parent(), p)
    print("Set of res on second pdb:")
    for x in sample_atom_list[min_idx[2]]:
        p = x.get_parent()
        print(p.get_parent(), p)
    print("Pairs with rmsd = 0 (errored):", zeroes_idx)

    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atom_list[min_idx[1]], sample_atom_list[min_idx[2]])
    super_imposer.apply(pdb2.get_atoms())
    # print(super_imposer.rms)

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb2)
    io.save("pdb2_aligned.pdb")


if __name__ == "__main__":
    main()
