import getopt
import math
import sys
import optparse
import os


from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB import Structure
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain


def parse_PDB_to_struct(path: str, id: str):
    """Parses a file from [path] into a Bio.PDB Structure with [id]"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(id, path)
    return structure


def convert_atom_res_and_chain_to_A(structure):
    """Converts all residues to chain A and renumbers the atoms not from chain A so that each subsequent chain starts at the next nearest 100s.
    Ex. Chain A has 120 residues. Chain B with 260 residues will start at 200 and go up to 460.

    Args:
            structure (Bio.PDB Structure): A PDB structure with a full hierarchy -> model is a child of structure, chain is a child of model, atom is a child of chain.

    Returns:
            Bio.PDB Structure: Protein structure after having all atoms and chains converted.
    """
    new_structure = Structure("n_s")
    new_model = Model("n_m")
    new_chain = Chain("A")

    new_structure.add(new_model)
    new_model.add(new_chain)

    for model in structure:
        ## Get last residue number (resseq) of atoms in chain A
        max = 0
        last = 0
        try:
            chainA = model.__getitem__("A")
            for atom in chainA:
                new_chain.add(atom)
                res_id = list(atom.id)
                if last < res_id[1]:
                    last = res_id[1]
            print("Last res num in chain A:", last)
            ## Round up last number to nearest 100.
            max = 100 * math.ceil(last / 100)
        except:
            print("No chain A found.")

        ## For all other chains besides A, increment all res_ids by the total residues so far, rounded up to the nearest 100.
        for chain in model:
            if chain.id == "A":
                continue
            else:
                try:
                    print(
                        "Converting residues from", chain, "--> Starting at index", max
                    )
                    last = 0
                except KeyError as error:
                    print("ERROR:", error)
                    print("No other chains found. Exiting.")
                    break

                for atom in chain:
                    res_id = list(atom.id)
                    res_id[1] = max + res_id[1]
                    atom.detach_parent()
                    atom.id = tuple(res_id)

                    ## Get last id in current chain
                    if last < res_id[1]:
                        last = res_id[1]

                    ## Set new parent for atom
                    new_chain.add(atom)

                max = 100 * math.ceil(last / 100)
    print("################################################################")
    return new_structure


def save_structure(structure, name):
    name = name.split("/")[-1]
    name = name.split(".")[0]
    save_name = "{}_edited_model.pdb".format(name)

    io = PDBIO()
    io.set_structure(structure)
    io.save(save_name, preserve_atom_numbering=True)

    return save_name


def run_ska(s1, s2, ska_path, reckless=False):
    print("Running SKA...")
    if reckless:
        command = (
            "."
            + ska_path
            + "/ska "
            + s1
            + " "
            + s2
            + " -reckless -o ska_output.pdb > ska_output.txt"
        )

    else:
        command = (
            "."
            + ska_path
            + "/ska "
            + s1
            + " "
            + s2
            + " -o ska_output.pdb > ska_output.txt"
        )
    try:
        os.system(command)
    except:
        print("Error occurred while running SKA.")


def extract_ska_rmsd(ska_output):
    if not os.path.exists(ska_output):
        return None

    if os.path.getsize(ska_output) == 0:
        return None

    rmsd_lines = []
    with open(ska_output, "r") as file:
        for line in file:
            if line.startswith("RMSD"):
                rmsd_lines.append(line.strip())
    return rmsd_lines[0]


def extract_ska_transformation(ska_output):
    if not os.path.exists(ska_output):
        return None

    if os.path.getsize(ska_output) == 0:
        return None

    remark_lines = []
    with open(ska_output, "r") as file:
        for line in file:
            if line.startswith("REMARK"):
                remark_lines.append(line.strip().replace("REMARK", ""))
    transformation_lines = remark_lines[3:]

    return transformation_lines


def main():
    # path = "ska_src/5co5.pdb"
    # path = "ska_src/1eaw.pdb"
    path1 = "ska_src/3nps.pdb"
    path2 = "ska_src/3bn9.pdb"

    struct = parse_PDB_to_struct(path1, "s1")
    print("------------------Converting 1st structure------------------")
    new_structure = convert_atom_res_and_chain_to_A(struct)
    s1 = save_structure(new_structure, path1)

    struct = parse_PDB_to_struct(path2, "s2")
    print("------------------Converting 2nd structure------------------")
    new_structure = convert_atom_res_and_chain_to_A(struct)
    s2 = save_structure(new_structure, path2)

    run_ska(s1, s2, "", True)

    rmsd_lines = extract_ska_rmsd("ska_output.txt")
    transformation_lines = extract_ska_transformation("ska_output.pdb")

    print(rmsd_lines)
    print(transformation_lines)


if __name__ == "__main__":
    main()
