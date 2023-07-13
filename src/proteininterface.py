import os
import math
import time

import networkx as nx

from Bio.PDB import Structure
import Bio.PDB

"""Class for a ProteinInterface Object. Requires input of DiffBond results folders, list of modes to incorporate into the final full graph, and number of neighbor nodes to include.

Raises:
    Exception: _description_
    Exception: _description_
    Exception: _description_
    Exception: _description_

Returns:
    _type_: _description_
"""


class ProteinInterface:
    def __init__(self, i_list, m_list, k_hops, verbose=False):
        # parsing each graph based on results directories given in i_list
        self.graphs = self.parse_graphs(i_list, m_list, verbose)
        self.pdbs = self.parse_pdbs(i_list)

        start = time.time()
        self.graph_pdb_dicts, self.permutation_dicts = self.set_graph_PDB_pairs(
            self.graphs, self.pdbs, k_hops
        )
        end = time.time()
        if verbose:
            print("------------------------------------------")
            print("Time to get graph pairings: {0:.2f} s".format(end - start))
            print("------------------------------------------")

        self.ref_atom_list = self.compile_backbone_atoms(self.permutation_dicts[0])
        self.sample_atom_list = self.compile_backbone_atoms(self.permutation_dicts[1])

    def combine_PDB_structures(self, path):
        """Combines 2 pdb structures into one PDB with 2 separate models. Models are named 0 and 1.

        Args:
                                        paths (list of two strings): the list contains the two strings for the 2 pdb files to combine.

        Returns:
                                        Bio.PDB.Structure : a structure with both models in it
        """
        parser = Bio.PDB.PDBParser()
        structures = []
        count = 0
        for f in os.listdir(path + "/pdb"):
            structures.append(parser.get_structure(count, path + "/pdb/" + f))
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

        return final_model

    def parse_pdbs(self, i_list):
        pdbs = []
        for i in i_list:
            # get and combine pdb structures from pdb folder in i_list
            pdb = self.combine_PDB_structures(i)
            pdbs.append(pdb)

        return pdbs

    def get_adjacent_residues(self, residue, chains, num_hops):
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

    # get graph to PDB amino acid pairings as graph_pdb_dicts
    def parse_graph_PDB_pair(self, graph, pdb, k_hops):
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

                            prev_res_list, next_res_list = self.get_adjacent_residues(
                                residue, chains, k_hops
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
                        prev_res_list, next_res_list = self.get_adjacent_residues(
                            min[1], chains, k_hops
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

    # Get the alpha carbon atoms for ref_atom_list and sample_atom_list
    def set_graph_PDB_pairs(self, graph, pdb, k_hops):
        graph_pdb_dicts = []
        perm_dicts = []
        for i in range(2):
            G_pdb_dict, perm_dict = self.parse_graph_PDB_pair(
                self.graphs[i], self.pdbs[i], k_hops
            )
            graph_pdb_dicts.append(G_pdb_dict)
            perm_dicts.append(perm_dict)

        return graph_pdb_dicts, perm_dicts

    # Get the alpha carbon atoms for ref_atom_list and sample_atom_list
    def compile_backbone_atoms(self, perm_dict):
        atoms_list = []
        for c in perm_dict:
            atoms = []
            for res in perm_dict[c]:
                if res == None:
                    continue
                else:
                    # print(ref_res.get_list())
                    # print(ref_res["CA"])
                    atoms.append(res["CA"])
                # print(atoms)
            atoms_list.append(atoms)

        return atoms_list
