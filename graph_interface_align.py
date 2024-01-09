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
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
import Bio.PDB

sys.path.insert(0, "src")
from proteininterface import ProteinInterface
from proteingraph import ProteinGraph
from rmsdlist import RMSDlist
import bond_matching
import ska_wrapper as ska

import pickle


def parseArg():
    """Parse arguments from command line

    Raises:
        argparse.ArgumentTypeError

    Returns:
        i_list: -i two input files
        
        m_list: -m modes allows the following modes: c, i, h, s, (p not implemented)
        
        k_hops: -k number of hops
        
        verbose: -v verbose tag
        
        output: -o output file name
        
        ska: -s ska alignment tag
        
    """
    parser = argparse.ArgumentParser(
        description="Aligns 2 structures to each other given graphs of the interface."
    )

    parser.add_argument(
        "-i",
        "-input",
        nargs=2,
        required=True,
        metavar="InputResultFolder",
        help="Input result folders from DiffBond containing graphs and PDB file. First input will be the reference structure used for alignment and second input will be the aligned sample structure.",
    )

    parser.add_argument("-o", nargs="?", metavar="Output", help="Output folder name")

    parser.add_argument(
        "-m",
        "-mode",
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
        "-verbose",
        action="store_true",
        help="Print out additional information when running. Default = False",
    )

    parser.add_argument(
        "-s",
        "-ska",
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
    ska = args["s"]
    output = args["o"]

    if output is None:
        output = "output"

    return i_list, m_list, k_hops, verbose, output, ska



def get_grakel_graphs(graphs):
    """
    Helper function converts graphs to networkx objects
    """
    G = grakel.graph_from_networkx(
        graphs,
        node_labels_tag="AA",
        edge_labels_tag="bond_type",
        edge_weight_tag="weight",
    )
    return G

def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.3+
    """Prints out a progress bar in stdout in for loops
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



def run_ska(c1, c2, ska_path, reckless=False):
    """Enables ska to be run for alignment before rmsd alignments.
    Helper function included to convert PDB halves to one Bio.PDB structure.

    Args:
        path (string): _description_
        
        interface (ProteinInterface): 
        
        ska_path (string): path to ska executable
        
        reckless (bool): Default=False. Allows ska to use the reckless tag.
        
    """

    def convert_to_one_structure(pdb):
        """Convert two PDB halves to one Bio.PDB structure
        """
        index = 0
        l_s = list(pdb)
        main_structure = Structure("n_s")
        new_model = Model("n_m")

        main_structure.add(new_model)

        for structure in l_s:
            for chain in structure:
                chain.id = str(index)
                index += 1
                chain.detach_parent()
                new_model.add(chain)

        return main_structure

    temp_struct = convert_to_one_structure(c1)
    temp_struct_convert = ska.convert_atom_res_and_chain_to_A(temp_struct)
    final_struct1 = ska.save_structure(temp_struct_convert, "f1")

    temp_struct = convert_to_one_structure(c2)
    temp_struct_convert = ska.convert_atom_res_and_chain_to_A(temp_struct)
    final_struct2 = ska.save_structure(temp_struct_convert, "f2")

    ska.run_ska(final_struct1, final_struct2, ska_path, reckless=reckless)

    rmsd_lines = ska.extract_ska_rmsd("ska_output.txt")
    transformation_lines = ska.extract_ska_transformation("ska_output.pdb")

    return rmsd_lines, transformation_lines

def super_imposer_helper(l1, l2, pdb):
    """Applies super imposer to pdb based off of l1 alignment to l2

    Args:
        l1 (list): List 1 of atoms to superimpose. This is the fixed atoms.
        
        l2 (list): List 2 of atoms to superimpose. This is the moving atoms.
        
        pdb (Bio.PDB): PDB model
    
    Returns:
		pdb
    """
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(l1, l2)
    super_imposer.apply(pdb.get_atoms())

    return pdb

def get_graph_edge_coord(interface):
    """create a dictionary of graph edges which includes attributes like bond_type, coordinates, and bond centroid

    Args:
        interface (_type_): _description_

    Returns:
        graph_edge_coord (list): list of all relevant graph edge information
    """
    graph_edge_coord = []
    for g in range(len(interface.graphs)):
        e = {}
        for i in list(interface.graphs[g].edges):
            bond_type = interface.graphs[g].edges[i]["bond_type"]
            c1 = interface.graphs[g].nodes[i[0]]["coord"]
            c2 = interface.graphs[g].nodes[i[1]]["coord"]
            c_f = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2]
            e[i] = [c_f, bond_type]

        graph_edge_coord.append(e)

    return graph_edge_coord



def copy_all_PDB(i_list, dir):
    """copy PDBS into results directory

    Args:
        i_list (_type_): list of input PDB files
        dir (_type_): directory name
    """
    for index, i in enumerate(i_list):
        pdbs = os.listdir(i + "/pdb")
        # shutil.copy(i + "/pdb/", dir)
        for pdb in pdbs:
            try:
                shutil.copy(
                    i + "/pdb/" + pdb, dir + "/original" + str(index) + "_" + pdb
                )
            except shutil.SameFileError:
                print("File already exists.")
            except PermissionError:
                print("Permission denied.")
                perm = os.stat(i + "/pdb/" + pdb).st_mode
                print("File Permission mode:", perm, "\n")
            except:
                print("Error occurred while copying file.")


def main():
    """Main function runs the whole graph_interface_align.py pipeline.
    Pipeline:
		1. First reads in a ProteinInterface object.
		2. Calculate a list of RMSDs using bond-bond correspondence.
		3. 
    """
    i_list, m_list, k_hops, verbose, output, ska = parseArg()

    interface = ProteinInterface(i_list, m_list, k_hops, verbose=verbose)

    ## graph details
    G_1 = interface.graphs[0]
    G_2 = interface.graphs[1]
    print("-----------------------------------------------------")
    print("1st graph:", G_1)
    print("2nd graph:", G_2)
    print("-----------------------------------------------------")
    if nx.is_empty(G_1) or nx.is_empty(G_2):
        raise Exception(
            "One of the graphs has no edges. Cannot calculate graph alignment."
        )

    ## Convert to undirected if directed
    if nx.is_directed(G_1):
        G_1 = nx.to_undirected(G_1)
    if nx.is_directed(G_2):
        G_2 = nx.to_undirected(G_2)

    ## Calculate the rmsd for all combinations of ref_atom_list and sample_atom_list.
    ## This step will take the longest.
    ## Most of the RMSD functions occur when RMSDlist object is created
    if verbose:
        print("Calculating RMSD")
    rmsd_list = RMSDlist(interface, verbose=verbose)
    rmsd_list.print()

    ## If ska enabled, this will run to filter using ska RMSD and rotran features
    if ska:
        print("################################################################")
        print("####################     RUNNING SKA     #######################")
        print("################################################################")
        c1 = interface.pdbs[0]
        c2 = interface.pdbs[1]
        ska_rmsd, ska_transformation = run_ska(c1, c2, ska_path="/src")
        if ska_rmsd == None or ska_transformation == None:
            print("WARNING: Cannot filter using SKA. No transformation found from SKA.")
        else:
            rmsd_list.filter_ska(ska_rmsd, ska_transformation, 30, verbose)
    else:
        print("################################################################")
        print(
            "####################     GETTING FILTERED RMSD     #######################"
        )
        print("################################################################")
        rmsd_list.filter_rmsd(verbose)
        

    ## Setup output directory
    counter = 0
    while os.path.exists("results/" + output + "_" + str(counter)):
        counter = counter + 1
    output = output + "_" + str(counter)

    os.makedirs("results/" + output)
    print("New output directory created:", output)
    
    ## Copy all PDBs to results directory and save interface object as a pickle file to results directory
    copy_all_PDB(i_list, "results/" + output)
    with open("results/" + output + "/interface.pkl", "wb") as outp:
        pickle.dump(interface, outp, pickle.HIGHEST_PROTOCOL)

    counter = 0
    ## Go through filtered list to save all PDBs to file
    if rmsd_list.filtered_rmsd != None and rmsd_list.filtered_rmsd != []:
        print("Printing filtered rmsd...")
        if len(rmsd_list.filtered_rmsd) != 0:
            for i in progressbar(rmsd_list.filtered_rmsd):
                super_imposed_pdb = super_imposer_helper(i[1], i[2], interface.pdbs[1])
                io = Bio.PDB.PDBIO()
                io.set_structure(super_imposed_pdb)
                io.save("results/" + output + "/pdb_" + str(counter) + "_aligned.pdb")

                f = open(
                    "results/" + output + "/pdb_" + str(counter) + "_aligned.pdb", "a+"
                )
                f.seek(0)
                rot_x = " ".join([str(elem) for elem in i[3][0][0]])
                rot_y = " ".join([str(elem) for elem in i[3][0][1]])
                rot_z = " ".join([str(elem) for elem in i[3][0][2]])
                tran = " ".join([str(elem) for elem in i[3][1]])
                f.write("REMARK " + rot_x + "\n")
                f.write("REMARK " + rot_y + "\n")
                f.write("REMARK " + rot_z + "\n")
                f.write("REMARK " + tran + "\n")
                f.close()

                counter = counter + 1

            print("All files added to dir: results/" + output)

    if rmsd_list.filtered_ska != None and rmsd_list.filtered_ska != []:
        print("Printing filtered ska...")
        if len(rmsd_list.filtered_ska) != 0:
            for i in progressbar(rmsd_list.filtered_ska):
                super_imposed_pdb = super_imposer_helper(i[1], i[2], interface.pdbs[1])
                io = Bio.PDB.PDBIO()
                io.set_structure(super_imposed_pdb)
                io.save("results/" + output + "/pdb_" + str(counter) + "_aligned.pdb")
                counter = counter + 1
            print("All files added to dir: results/" + output)


if __name__ == "__main__":
    main()
