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


def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.3+
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


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


def get_atom_list_combinations(atom_list):
    middle_idx = int(len(atom_list) / 2)
    chain1 = atom_list[:middle_idx]
    chain2 = atom_list[middle_idx:]

    list_comb = [atom_list]
    # list_comb.append(chain1 + list(reversed(chain2)))
    # list_comb.append(list(reversed(chain1)) + chain2)
    # list_comb.append(list(reversed(chain1)) + list(reversed(chain2)))

    return list_comb


def get_rmsd_list(ref_atom_list, sample_atom_list, pdb1, pdb2, k_hops):
    """calculates the rmsd for every combination of ref_atom_list and sample_atom_list.
    It takes each list of atoms, get 4 orientations of each list by flipping the directions of the atoms, and then calculates the rmsd of all combinations in ref_atom_list to sample_atom_list.

    Args:
        ref_atom_list (List of lists): Each embedded list contains a set of atoms which is the neighborhood to be used for comparison with sample_atom_list.
        sample_atom_list (List of lists): Each embedded list contains a set of atoms which is the neighborhood to be used for comparison with ref_atom_list.
        pdb1 (Bio.PDB.Structure): The full PDB associated with ref_atom_list. This is used as the reference when finding the rotation/translation matrix of RMSD.
        pdb2 (Bio.PDB.Structure): The full PDB associated with sample_atom_list. This is used as the sample to rotate and translate for the rotation/translation matrix.
        k_hops (int): Number of hops from primary atom to include in the atom_list neighborhood.

    Returns:
        rmsd_list (List of floats): Contains all rmsd calculated for every combination of atom list, and the
        min_idx (List): A list containing associated values for the atom list combination with the lowest RMSD.
        zeroes_idx: RMSD combinations that gave an RMSD = 0. If there are any values in this index, an error has occurred.
    """

    min_idx = [float("inf"), -1, -1]
    zeroes_idx = []
    rmsd_list = []

    # num_computes to set progress bar range
    num_computes = len(ref_atom_list)

    # Loop through ref_atom_list and sample_atom_list to do rmsd matches
    for i in progressbar(range(num_computes)):
        # Check for error in atom list
        if len(ref_atom_list[i]) < k_hops * 4 + 2:
            continue

        # Get multiple combinations of the atom list
        ref_list_comb = get_atom_list_combinations(ref_atom_list[i])
        for j in range(len(sample_atom_list)):
            # Check for error in atom list
            if len(sample_atom_list[j]) < k_hops * 4 + 2:
                continue

            # Get multiple combinations of the atom list
            sample_list_comb = get_atom_list_combinations(sample_atom_list[j])
            try:
                # Need to try all combinations of ref_atom_list and sample_atom_list for specific alignment
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

    return rmsd_list, min_idx, zeroes_idx


def print_results(
    interface, rmsd_list, time, threshold, rmsd_filtered, min_idx, zeroes_idx, verbose
):
    print("------------- RMSD calculated for all combinations -------------")
    print("Number of RMSD calculated:", len(rmsd_list))
    print("Time to get RMSD: {0:.2f} s".format(time))

    print(
        "Number of alignments with RMSD less than", threshold, ":", len(rmsd_filtered)
    )

    print("Min rmsd:", min_idx[0])

    print(
        "------------- RMSD of all",
        len(rmsd_filtered),
        "calculated alignments: -------------",
    )
    for i in rmsd_filtered:
        print(i[0])

    if verbose:
        print("------------------------------------------")
        print("------------- Verbose logs -------------")
        print("------------------------------------------")
        print("Set of res on first pdb for minimum RMSD:")
        for x in interface.ref_atom_list[min_idx[1]]:
            p = x.get_parent()
            print(p.get_parent(), p)
        print("------------------------------------------")
        print("Set of res on second pdb for minimum RMSD:")
        for x in interface.sample_atom_list[min_idx[2]]:
            p = x.get_parent()
            print(p.get_parent(), p)
        print("------------------------------------------")
        print("Pairs with rmsd = 0 (These are errors):", zeroes_idx)
    print("------------------------------------------")


def super_imposer_helper(l1, l2, pdb, counter, output_folder):
    """applies super imposer to pdb based off of l1 alignment to l2. Will also generate the results folder

    Args:
        l1 (_type_): _description_
        l2 (_type_): _description_
        pdb (_type_): _description_
        counter (_type_): _description_
        output_folder (_type_): _description_
    """
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(l1, l2)
    super_imposer.apply(pdb.get_atoms())

    io = Bio.PDB.PDBIO()
    io.set_structure(pdb)
    io.save("results/" + output_folder + "/pdb_" + str(counter) + "_aligned.pdb")


def copy_all_PDB(i_list, dir):
    for i in i_list:
        pdbs = os.listdir(i + "/pdb")
        # shutil.copy(i + "/pdb/", dir)
        for pdb in pdbs:
            shutil.copy(i + "/pdb/" + pdb, dir)


def main():
    i_list, m_list, k_hops, verbose, output = parseArg()

    interface = ProteinInterface(i_list, m_list, k_hops, verbose=verbose)

    # graph details
    G_1 = interface.graphs[0]
    G_2 = interface.graphs[1]
    print("-----------------------------------------------------")
    print("1st graph:", G_1)
    print("2nd graph:", G_2)
    print("-----------------------------------------------------")

    # Calculate the rmsd for all combinations of ref_atom_list and sample_atom_list.
    # This step will take the longest.
    start = time.time()
    rmsd_list, min_idx, zeroes_idx = get_rmsd_list(
        interface.ref_atom_list,
        interface.sample_atom_list,
        interface.pdbs[0],
        interface.pdbs[1],
        k_hops,
    )
    end = time.time()

    # Increase threshold incrementally to get lowest relevant RMSDs
    threshold = 0.1
    rmsd_filtered = []
    while len(rmsd_filtered) < 5 and threshold < 3.0:
        rmsd_filtered = [i for i in rmsd_list if i[0] < threshold]
        threshold = round(threshold + 0.02, 3)

    print_results(
        interface,
        rmsd_list,
        end - start,
        threshold,
        rmsd_filtered,
        min_idx,
        zeroes_idx,
        verbose,
    )

    counter = 0
    if os.path.exists("results/" + output):
        print(
            "Output directory already exists. Adding files to",
            "'results/",
            output,
            "'",
            "directory...",
        )
    else:
        os.makedirs("results/" + output)
        print("New output directory created.")

    copy_all_PDB(i_list, "results/" + output)

    if len(rmsd_filtered) != 0:
        for i in progressbar(rmsd_filtered):
            super_imposer_helper(i[1], i[2], interface.pdbs[1], counter, output)
            counter = counter + 1
        print("All files added to dir: results/" + output)


if __name__ == "__main__":
    main()
