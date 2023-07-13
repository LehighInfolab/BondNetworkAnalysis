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
from proteingraph import ProteinGraph
from rmsdlist import RMSDlist


def parseArg():
    """Parse arguments from command line

    Raises:
        argparse.ArgumentTypeError: _description_

    Returns:
        _type_: _description_
    """
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


def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.3+
    """Prints out a progress bar in stdout in for loops

    Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    it (_type_): _description_
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    prefix (str, optional): _description_. Defaults to "".
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    size (int, optional): _description_. Defaults to 60.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    out (_type_, optional): _description_. Defaults to sys.stdout.

    Yields:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    _type_: _description_
    """
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


def main():
    i_list, m_list, k_hops, verbose, output = parseArg()

    # g1 = ProteinGraph(i_list[0], m_list)
    # g2 = ProteinGraph(i_list[1], m_list)
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

    rmsd_list = RMSDlist(interface)
    rmsd_list.print()

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

    if len(rmsd_list.filtered_rmsd) != 0:
        for i in progressbar(rmsd_list.filtered_rmsd):
            super_imposer_helper(i[1], i[2], interface.pdbs[1], counter, output)
            counter = counter + 1
        print("All files added to dir: results/" + output)


if __name__ == "__main__":
    main()
