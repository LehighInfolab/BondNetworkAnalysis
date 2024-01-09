import os
import math
import time
import sys
import numpy as np
import re

import networkx as nx

from proteininterface import ProteinInterface
from scipy.spatial.transform import Rotation

from Bio.PDB import Structure
import Bio.PDB


class RMSDlist:
    """Generates a list of rmsds based off pairs of atoms in protein interface"""

    def __init__(self, interface: ProteinInterface, verbose=False):
        self.interface = interface

        start = time.time()
        self.rmsd_list, self.min_idx, self.zeroes_idx = get_rmsd_list(
            interface.ref_atom_list,
            interface.sample_atom_list,
            interface.pdbs[0],
            interface.pdbs[1],
            interface.k_hops,
        )
        self.threshold = None
        self.filtered_rmsd = None
        self.filtered_ska = None

        end = time.time()

        self.time = end - start

    def filter_rmsd(self, verbose=False):
        """filter the list of rmsds with a specific threshold. The threshold is set to max at 2.0 rmsd and max_num of 20

        Args:
            verbose (bool, optional): _description_. Defaults to False.
        """
        self.filtered_rmsd, self.threshold = filter_rmsd(
            self.rmsd_list, verbose=verbose
        )
        print(
            "------------- RMSD of all",
            len(self.filtered_rmsd),
            "calculated alignments with threshold less than",
            self.threshold,
            " -------------",
        )
        for index, i in enumerate(self.filtered_rmsd):
            print(index, ":", i[0])

    def filter_ska(self, rmsd, trans, angle_threshold, verbose):
        """filter the list of rmsds with a specific threshold. The threshold is set to max at 2.0 rmsd and max_num of 20
        """
        self.filtered_ska = filter_ska_rotran(
            self.rmsd_list, rmsd=rmsd, trans=trans, angle_threshold=angle_threshold
        )
        print(
            "------------- RMSD of all",
            len(self.filtered_ska),
            "calculated alignments that are within ska angle threshold of",
            angle_threshold,
            "and less than 3.0" " -------------",
        )
        for index, i in enumerate(self.filtered_ska):
            print(index, ":", i[0])

    def print(self):
        print_results(
            self.rmsd_list,
            self.time,
            self.threshold,
            self.min_idx,
            self.zeroes_idx,
            self.interface,
            verbose=False,
        )


def get_rmsd_list(ref_atom_list, sample_atom_list, pdb1, pdb2, k_hops):
    """calculates the rmsd for every combination of ref_atom_list and sample_atom_list.
    It takes each list of atoms, get 4 orientations of each list by flipping the directions of the atoms, and then calculates the rmsd of all combinations in ref_atom_list to sample_atom_list.

    Args:
    #    ref_atom_list (List of lists): Each embedded list contains a set of atoms which is the neighborhood to be used for comparison with sample_atom_list.
    #    sample_atom_list (List of lists): Each embedded list contains a set of atoms which is the neighborhood to be used for comparison with ref_atom_list.
    #    pdb1 (Bio.PDB.Structure): The full PDB associated with ref_atom_list. This is used as the reference when finding the rotation/translation matrix of RMSD.
    #   pdb2 (Bio.PDB.Structure): The full PDB associated with sample_atom_list. This is used as the sample to rotate and translate for the rotation/translation matrix.
    #  k_hops (int): Number of hops from primary atom to include in the atom_list neighborhood.

    Returns:
    #rmsd_list (List of floats): Contains all rmsd calculated for every combination of atom list, and the
    #min_idx (List): A list containing associated values for the atom list combination with the lowest RMSD.
    #zeroes_idx: RMSD combinations that gave an RMSD = 0. If there are any values in this index, an error has occurred."""

    def get_atom_list_combinations(atom_list, reverse=False):
        middle_idx = len(atom_list)
        chain1 = atom_list[:middle_idx]
        chain2 = atom_list[middle_idx:]

        list_comb = [atom_list]

        if reverse == True:
            list_comb.append(chain1 + list(reversed(chain2)))
            list_comb.append(list(reversed(chain1)) + chain2)
            list_comb.append(list(reversed(chain1)) + list(reversed(chain2)))

        return list_comb

    min_idx = [float("inf"), -1, -1]
    zeroes_idx = []
    rmsd_list = []

    # num_computes to set progress bar range
    num_computes = len(ref_atom_list)
    # Loop through ref_atom_list and sample_atom_list to do rmsd matches
    for i in range(len(ref_atom_list)):
        # Check for error in atom list
        if len(ref_atom_list[0]) < k_hops * 4 + 2:

            print("Not the right size. Continuing...")
            continue

        # Get multiple combinations of the atom list
        ref_list_comb = get_atom_list_combinations(ref_atom_list[i])
        for j in range(len(sample_atom_list)):
            # Check for error in atom list
            if len(sample_atom_list[0]) < k_hops * 4 + 2:
                print("Not the right size. Continuing...")
                continue

            # Get multiple combinations of the atom list
            sample_list_comb = get_atom_list_combinations(sample_atom_list[j])
            try:
                # Need to try all combinations of ref_atom_list and sample_atom_list for specific alignment
                for ref_comb in ref_list_comb:
                    for samp_comb in sample_list_comb:
                        try:
                            super_imposer = Bio.PDB.Superimposer()
                            super_imposer.set_atoms(ref_comb, samp_comb)
                            super_imposer.apply(pdb2.get_atoms())
                        except:
                            print(
                                "Error superimposing atoms. Moving to next combination..."
                            )
                            continue
                        rmsd_list.append(
                            [
                                super_imposer.rms,
                                ref_comb,
                                samp_comb,
                                super_imposer.rotran,
                            ]
                        )
                        # print("Super imposer rotran:", super_imposer.rotran)
                        if super_imposer.rms < min_idx[0] and super_imposer.rms != 0:
                            min_idx = [super_imposer.rms, i, j, ref_comb, samp_comb]
                        if super_imposer.rms == 0:
                            zeroes_idx.append(
                                [super_imposer.rms, i, j, ref_comb, samp_comb]
                            )
                        # print(
                        #     ref_atom_list[i][0].get_parent(),
                        #     sample_atom_list[j][0].get_parent(),
                        # )
            except:
                print("Error has occurred while getting rmsd list.")
                continue
            
    return rmsd_list, min_idx, zeroes_idx


def filter_rmsd(
    rmsd_list, interval=0.01, threshold=0.1, max=2.0, max_num=20, verbose=False
):
    """Filters rmsd by adding elements with rmsd starting at < 0.1 and then increment and keep adding elements 
    until either max is reached (2.0) or max num is reached (20)

    Args:


    Returns:
        _type_: _description_
    """
    if len(rmsd_list) == None:
        return [], threshold

    # Increase threshold incrementally to get lowest relevant RMSDs
    filtered_rmsd = []

    if verbose:
        print("Full RMSD list len:", len(rmsd_list), "\n")
        print("Filtering RMSD list using the following parameters:")
        print("Starting threshold -", threshold)
        print("Max -", max)
        print("Max number of RMSD -", max_num)
        print("Interval -", interval)

    first_pass_rmsd_list = [i for i in rmsd_list if i[0] < max]

    while len(filtered_rmsd) < max_num and threshold < max:
        filtered_rmsd = [i for i in first_pass_rmsd_list if i[0] < threshold]
        threshold = round(threshold + interval, 3)

    return filtered_rmsd, threshold

def filter_ska_rotran(rmsd_list, rmsd, trans, angle_threshold=30, max=3.0):
    """Filters rmsd by adding elements with rmsd starting at < 0.1 and then increment and keep adding elements 
    until either max is reached (2.0) or max num is reached (20).
    This also uses ska rotation values as a threshold, making sure that angle is +-30 on each side of rotation which gives a total of 60 degree range
    """
    # Increase threshold incrementally to get lowest relevant RMSDs
    filtered_rmsd = []
    print("Angle threshold:", angle_threshold)

    rmsd = float(rmsd.split(":")[-1])
    print("RMSD:", rmsd)

    if len(rmsd_list) == 0:
        return []

    ska = []
    for i in trans:
        floats_list = re.findall(r"[+-]?\d+\.\d+", i)
        floats_list = [float(i) for i in floats_list]
        ska.append(floats_list)
    print("Ska transformation matrix:", ska)

    ska_rot = ska[:3]
    ska_trans = ska[-1]
    r = Rotation.from_matrix(ska_rot)
    angles = r.as_euler("zyx", degrees=True)
    print("Ska rotation angles:", angles)

    for i in rmsd_list:
        # print("Current alignment RMSD:", i[0])
        rot = i[3][0]
        diff = np.linalg.inv(rot) * ska_rot
        r = Rotation.from_matrix(diff)
        angles = r.as_euler("zyx", degrees=True)

        if (
            -angle_threshold <= angles[0] <= angle_threshold
            and -angle_threshold <= angles[1] <= angle_threshold
            and -angle_threshold <= angles[2] <= angle_threshold
        ):
            # print("Angle in zyx:", angles)
            filtered_rmsd.append(i)

        tran = i[3][1]

    filtered_rmsd = [i for i in filtered_rmsd if i[0] < max]

    return filtered_rmsd


def print_results(
    rmsd_list,
    time,
    threshold,
    min_idx,
    zeroes_idx,
    interface,
    verbose=False,
):
    print("------------- RMSD calculated for all combinations -------------")
    print("Number of RMSD calculated:", len(rmsd_list))
    print("Time to get RMSD: {0:.2f} s".format(time))

    print("Min rmsd:", min_idx[0])

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
