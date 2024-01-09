import getopt
import math
import sys
import argparse
import os
import shutil

# Parse arguments from command line
def parseArg():
    parser = argparse.ArgumentParser(
        description="Aligns 2 structures to each other given graphs of the interface."
    )

    parser.add_argument(
        "-i",
        nargs="+",
        required=True,
        metavar="InputResultFolder",
        help="Input result folders from DiffBond containing graphs and PDB file. First input will be the reference structure used for alignment and second input will be the aligned sample structure.",
    )

    parser.add_argument(
        "-s", nargs=1, metavar="Src folder", help="Include src folder path"
    )

    parser.add_argument("-o", nargs="?", metavar="Output", help="Output folder name")

    # parse list of args
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        print(parser.print_help())

    args = vars(args)
    i_list = args["i"]
    if not i_list:
        raise argparse.ArgumentTypeError("-i requires at least one input.")

    src = args["s"]
    output = args["o"]

    if output is None:
        output = "output"
        for i in i_list:
            output = output + "_" + i

    return i_list, src, output


def check_path(path):
    if not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return "file"
    elif os.path.isdir(path):
        return "dir"


def make_temp_dir_for_hydrogen_cleaning(src, output):
    try:
        os.mkdir("temp_hydrogen_cleaning")
        print("Temp directory created.")
    except:
        print("Temp directory already exists. Moving to hydrogen cleaning step...")

    try:
        os.mkdir(output)
        print("Output directory created: ", output)
    except:
        print(
            "Output directory already exists:"
            + output
            + "  . Moving to hydrogen cleaning step..."
        )

    try:
        os.system("cp " + src + "striph" + " temp_hydrogen_cleaning/")
        print("Copied striph to temp dir")
    except:
        print("striph already exists in dir.")

    try:
        os.system("cp " + src + "reduce_linux225" + " temp_hydrogen_cleaning/")
        print("Copied reduce_linux225 to temp dir")
    except:
        print("striph already exists in dir.")

    try:
        os.system("cp " + src + "ska" + " temp_hydrogen_cleaning/")
        print("Copied ska to temp dir")
    except:
        print("striph already exists in dir.")


def remove_temp_dir():
    shutil.rmtree(
        "temp_hydrogen_cleaning/",
    )
    print("Cleaned up temp directory.")


def clean_hydrogens(file, output):
    os.system("cp " + file + " temp_hydrogen_cleaning/")
    os.chdir("temp_hydrogen_cleaning/")
    list = os.listdir("./")
    for i in list:
        if i != "striph" and i != "reduce_linux225" and i != "ska":
            file = i

    print("------------ RUNNING striph ------------")
    os.system("./striph " + file + " > " + " h-_" + file)
    os.system("rm -rf " + file)
    print("------------ RUNNING reduce_linux225 ------------")
    os.system(
        "./reduce_linux225 -BUILD h-_" + file + " > " + "h+_" + file + " 2> ../log.txt"
    )
    print("reduce_linux output sent to log.txt.")
    os.system("rm -rf h-_" + file)

    print("------------ MOVING h+_" + file + "------------")
    os.system("mv h+_" + file + " ../" + output)
    os.chdir("../")


def main():
    # i_list, output = parseArg()
    # print(i_list)
    # i_list = ["../04-filteredPDBs/"]
    i_list = ["../04-filteredPDBs/1BRS.pdb"]
    src = "./"
    output = "output/"

    option = ""
    if len(i_list) == 2:
        if check_path(i_list[0]) == "file" and check_path(i_list[1] == "file"):
            option = "two_files"
    if len(i_list) == 1:
        option = check_path(i_list[0])

    make_temp_dir_for_hydrogen_cleaning(src, output)

    if option == None:
        print("Error")
    elif option == "two_files":
        clean_hydrogens(i_list[0], output)
        clean_hydrogens(i_list[1], output)
    elif option == "file":
        clean_hydrogens(i_list[0], output)
    elif option == "dir":
        list = os.listdir(i_list[0])
        for i in list:
            clean_hydrogens(i_list[0] + i, output)

    remove_temp_dir()


if __name__ == "__main__":
    main()
