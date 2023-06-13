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


def main():
    # i_list, m_list, k_hops, verbose, output = parseArg()
    i_list = [
        "dataset\\Result_1brs_barnase_A+h_1brs_barstar_D+h",
        "dataset\\Result_1brs_barnase_A+h_1brs_barstar_D+h",
    ]
    m_list = ["i", "h"]
    k_hops = 3
    verbose = True

    interface = ProteinInterface(i_list, m_list, k_hops, verbose=verbose)


if __name__ == "__main__":
    main()
