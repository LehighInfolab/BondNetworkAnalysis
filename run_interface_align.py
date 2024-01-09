import getopt
import os
import math
import sys
import argparse
import shutil
import os
import csv
import logging

# Configure logging to save exceptions to a log file
log_filename = 'exceptions.log'
logging.basicConfig(filename=log_filename, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def run_align(file1, file2, output):
	# # python.exe graph_interface_align.py -i dataset/1brs_dataset/00105 dataset/1brs_dataset/00106 -m i -v
	os.system("python.exe graph_interface_align.py" + " -i " + file1 + " " + file2 + " -m h -o " + output)


def run_align_skempi(wt_path, mt_path, output):
	wt_folders = os.listdir(wt_path)
	
	index = 0
	for wt_f in wt_folders:
		print(wt_f)
		# if index > 2:
		# 	break
		# index += 1

		try:
			mt_folders = os.listdir(mt_path + "/" + wt_f)
			if not mt_folders:
				logging.error("Mt_folders %s is empty", str(mt_folders))
			for mt_f in mt_folders:
				run_align(wt_path + "/" + wt_f, mt_path + "/" + wt_f + "/" + mt_f, output + "/" + wt_f + "_h")
		except Exception as e:
			logging.error("An error occurred: %s", str(e))
		
		# for mt_f in mt_folders:
			

def main():
	# base0 = "../../SKEMPI_dataset_download/base-0"
	# reader = parse_skempi(base0)
	# base_diffbond_calc(reader, base0)
 
	base_0_output = "base-0"

	wt_path = "../../2023_DiffBond_Github/DiffBond/Results/wt"
	base_path = "../../2023_DiffBond_Github/DiffBond/Results/base-0"
	run_align_skempi(wt_path, base_path, base_0_output)
	




if __name__ == "__main__":
	main()
