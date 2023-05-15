import getopt
import os
import math
import sys
import argparse
import shutil
import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

import grakel

from Bio.PDB import Structure
import Bio.PDB

sys.path.insert(1, "./src")
import read_pdb
import write_pdb


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


def fit_rms(ref_c, c):
    # move geometric center to the origin
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    C = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(C)

    # compute sign (remove mirroring)
    if np.linalg.det(C) < 0:
        r2[2, :] *= -1.0
    U = np.dot(r1, r2)
    return (c_trans, U, ref_trans)


class RMSDcalculator:
    def __init__(self, atoms1, atoms2, name=None):
        xyz1 = self.get_xyz(atoms1, name=name)
        xyz2 = self.get_xyz(atoms2, name=name)
        self.set_rmsd(xyz1, xyz2)

    def get_xyz(self, atoms, name=None):
        xyz = []
        for atom in atoms:
            if name:
                if atom.name != name:
                    continue
            xyz.append([atom.x, atom.y, atom.z])
        return np.array(xyz)

    def set_rmsd(self, c1, c2):
        self.rmsd = 0.0
        self.c_trans, self.U, self.ref_trans = fit_rms(c1, c2)
        new_c2 = np.dot(c2 - self.c_trans, self.U) + self.ref_trans
        self.rmsd = np.sqrt(np.average(np.sum((c1 - new_c2) ** 2, axis=1)))

    def get_aligned_coord(self, atoms, name=None):
        new_c2 = copy.deepcopy(atoms)
        for atom in new_c2:
            atom.x, atom.y, atom.z = (
                np.dot(np.array([atom.x, atom.y, atom.z]) - self.c_trans, self.U)
                + self.ref_trans
            )
        return new_c2


def main():
    pdbf1 = "pdb_0_aligned.pdb"
    pdbf2 = "pdb_1_aligned.pdb"
    pdb1 = write_pdb.PDBio(pdbf1)
    pdb2 = write_pdb.PDBio(pdbf2)
    atoms1 = pdb1.get_atoms(to_dict=False)
    print(atoms1)
    atoms2 = pdb2.get_atoms(to_dict=False)

    RMSD_calculator = RMSDcalculator(atoms1, atoms2, name="CA")
    rmsd = RMSD_calculator.rmsd
    new_atoms = RMSD_calculator.get_aligned_coord(atoms2)
    pdb2.write_pdb("aligned_%s" % pdbf2, new_atoms)
    print("RMSD : %8.3f" % rmsd)
    print("New structure file: ", "aligned_%s" % pdbf2)


if __name__ == "__main__":
    main()
