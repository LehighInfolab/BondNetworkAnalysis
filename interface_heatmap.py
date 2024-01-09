import getopt
import os
import math
import sys
import argparse
import shutil
import os
import time
import csv

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation

import networkx as nx

import grakel

from Bio.PDB import Structure
import Bio.PDB
import pickle

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


def open_pickle_files(path):
	"""loads in all interface files as pickle files from the results directory

	Args:
		path (_type_): path to results directory

	Returns:
		list: all data loaded in as a list of tuples (interface object, alignment values from all aligned pdbs)
	"""
	all_data = []
	index = 0
	for i in os.listdir(path):
		# index += 1
		# if index > 3:
		#     break
		file_path = path + f"/" + i + "/interface.pkl"
		try:
			with open(file_path, "rb") as file:
				data = pickle.load(file)
				# Do something with the loaded data, e.g., print it
				print(f"Loaded data from {file_path}: {data}")
				aligned_list = read_indexed_files(path + f"/" + i)
				# print(aligned_list)
				all_data.append([data, aligned_list])
		except FileNotFoundError:
			print(f"File not found: {file_path}")
		except Exception as e:
			print(f"Error loading data from {file_path}: {e}")

	return all_data

def read_indexed_files(path):
	"""reading in the indexed aligned pdb files generated from structure alignment using rmsd. Helper function used in open_pickle_files()
	Each PDB file has the aligned structure as well as the rotation and translation written as remarks at the bottom of the file (x,y,z,trans)

	Args:
		path (_type_): path to results directory

	Returns:
		_type_: list of alignments (rot, trans) for each aligned pdb for an interface
	"""
	index = 0
	aligned_list = []
	while True:
		align = []
		file_name = "pdb_" + str(index) + "_aligned.pdb"
		file_path = path + "/" + file_name

		if not os.path.exists(file_path):
			# print(f"No file found for index {index}, stopping.")
			break

		with open(file_path, "r") as file:
			for line in file.readlines()[-4:]:
				add_line = line.replace("REMARK", "").split()
				align.append(add_line)

		aligned_list.append(align)
		index += 1
	return aligned_list

# def extract_rot(directory_path):
	
#     all_rot = []

#     # Loop through all files in the specified directory
#     for filename in os.listdir(directory_path):
#         # Check if the file ends with "_aligned.pdb"
#         if filename.endswith("_aligned.pdb"):
#             # Construct the full file path
#             filepath = os.path.join(directory_path, filename)
#             print(filename)

#             # Open and read the file
#             with open(filepath, "r") as file:
#                 # Read all lines from the file
#                 lines = file.readlines()
#                 rot = []
#                 # Iterate over the lines in reverse to get the last 3 lines
#                 for line in reversed(lines[-4:]):
#                     # Check if the line starts with "REMARK"
#                     if line.startswith("REMARK"):
#                         # Print the extracted line along with the filename
#                         elements = line.split()[1:]
#                         rot.append(elements)
#                         rot.reverse()
#                 all_rot.append(rot)
#     return all_rot


## Helper functions for converting graph data to usable data in bond matching -> check_perfect_matches() function
def get_graph_edge_coord(graphs):
	"""Gets the centroid of the bond as a point.

	Args:
		graphs (_type_): _description_

	Returns:
		a list of dictionaries where each dictionary matches an edge to [coordinate of point, bond type]
	"""
	graph_edge_coord = []
	for g in range(len(graphs)):
		e = {}
		for i in list(graphs[g].edges):
			bond_type = graphs[g].edges[i]["bond_type"]
			# weight = graphs[g].edges[i]["weight"]
			c1 = graphs[g].nodes[i[0]]["coord"]
			c2 = graphs[g].nodes[i[1]]["coord"]
			c_f = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2]
			e[i] = [c_f, bond_type]

		graph_edge_coord.append(e)

	return graph_edge_coord

def reformat_graph_edge(graph_edge):
	"""Reformat graph_edge dictionary to be compatible with has_a_perfect_match(list1,list2)"""
	l = []
	for i in graph_edge:
		l.append(
			{
				"edge": i,
				"bond_type": graph_edge[i][1],
				"x": graph_edge[i][0][0],
				"y": graph_edge[i][0][1],
				"z": graph_edge[i][0][2],
				# "w": graph_edge[i][2],
			}
		)
	return l



## Functions to get matches
def check_perfect_matches(interface, aligned_list):
	"""loop through the associated aligned_list (rot,trans) data for an interface to find the best edge match. Uses the has_a_perfect_match() function.

	Args:
		interface (_type_): _description_
		aligned_list (_type_): _description_

	Returns:
		_type_: Returns the best matched (rot,trans)
	"""
	max_match_data = []
	max = -1
	for align in aligned_list:
		
		## Get transformation as np array
		rot = np.array(align[:3]).astype("float")
		# print("Rotation Matrix:", rot)
		trans = np.array(align[3]).astype("float")
		# print("Translation Vector:", trans)
		
		## Keep track of original coordinates in temp
		g_align = interface.graphs[1]
		
		for node in g_align.nodes():
		
			# r = Rotation.from_euler("zyx", rot, degrees=True)
			# r = Rotation.from_matrix(rot).inv()
			# print(r.as_matrix())
			
			## Get coord as np array and apply rotation
			c = np.array(g_align.nodes[node]["coord"]).astype("float")
			# print("Point", c)
			
			rot_point = np.dot(c, rot)
			# print("After rot", rot_point)
			
			trans_point = rot_point + trans
			# print("After translation:",trans_point)
			
			## Set graph coordinate to new coordinates
			g_align.nodes[node]["coord"] = trans_point
			
		# for node in interface.graphs[0].nodes():
		#     print(interface.graphs[0].nodes[node]["coord"])

		## Format data to be read in perfect_match function
		graph_edges = get_graph_edge_coord([interface.graphs[0], g_align])
		ge0 = reformat_graph_edge(graph_edges[0])
		ge1 = reformat_graph_edge(graph_edges[1])

		## Perfect match function returns count of max 1-to-1 correspondences
		match_count, matched_graph = has_a_perfect_match(ge0, ge1)
		
		## If match is higher than or equal to max, keep track of the interface and the rotation
		if match_count > max:
			max = match_count
			max_match_data=[interface, align, match_count, matched_graph]

	print("### Finished checking interface", interface, "###")
	return max_match_data

def has_a_perfect_match(list1, list2, display=False):
	"""Converts graph1 and graph2 in interface to lists which are then converted to a bipartite graph with maximum matching

	Args:
		list1 (_type_): graph1 edges and its attributes
		list2 (_type_): graph2 edges and its attributes

	Returns:
		count: the number of maximum matches
		m: the bipartite graph with maximum edges
	"""
	# if len(list1) != len(list2):
	#     return False

	g = nx.Graph()

	l = [("l", d["edge"], d["bond_type"], d["x"], d["y"], d["z"]) for d in list1]
	r = [("r", d["edge"], d["bond_type"], d["x"], d["y"], d["z"]) for d in list2]

	# print("#########", l)
	# print("#########", r)
	g.add_nodes_from(l, bipartite=0)
	g.add_nodes_from(r, bipartite=1)

	edges = [(a, b) for a in l for b in r if check_dist(a, b, 10)]
	g.add_edges_from(edges)
	# print(g.edges())

	no_degree_list = (node for node, degree in g.degree() if degree == 0)
	# for i in no_degree_list:
	#     print("No degrees (no bonds formed):", i)

	pos = {}
	pos.update((node, (1, index)) for index, node in enumerate(l))
	pos.update((node, (2, index)) for index, node in enumerate(r))

	m = nx.bipartite.maximum_matching(g, l)
	colors = ["blue" if m.get(a) == b else "grey" for a, b in edges]
	
	count = sum(1 if m.get(a)==b else 0 for a, b in edges)
	# print("Count", count)

	if display:
		nx.draw_networkx(
			g,
			pos=pos,
			arrows=False,
			labels={n: "%s\n%s" % (n[1], n[2]) for n in g.nodes()},
			edge_color=colors,
		)
		plt.axis("off")
		plt.show()

	return count, m

def check_dist(a, b, d):
	"""Check if distance of coordinates from one point is less than distance of coordinates from other point"""
	if a[2] == b[2]:
		x0 = a[3]
		y0 = a[4]
		z0 = a[5]

		x1 = b[3]
		y1 = b[4]
		z1 = b[5]

		if math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2) < d:
			return True
	return False

def test_rotation():
	# data = open_pickle_files("results/1brs_i")
	# max_match_data = check_perfect_matches(data)
	# print(max_match_data)
	
	# data = open_pickle_files("results/1eaw_i")
	# max_match_data = check_perfect_matches(data)
	# print(max_match_data)
	
	data = open_pickle_files("results/lynx_pairwise_h")
	max_match_data = check_perfect_matches(data)
	print(max_match_data)


## After getting maximum match, apply matches directly to interface and get correspondence to be displayed as heatmaps
def apply_perfect_matches(interface, max_match):
	"""Apply the maximum matched (rot, trans) to the interface itself.

	Args:
		interface (_type_): _description_
		max_match (_type_): _description_

	Returns:
		interface: returns the interface after manually changing the coordinates based on (rot, trans)
	"""
	if max_match == []:
		return interface
	rot = np.array(max_match[1][:3]).astype("float")
	trans = np.array(max_match[1][3]).astype("float")
		
	for node in interface.graphs[1].nodes():
		 ## Get coord as np array and apply rotation
		c = np.array(interface.graphs[1].nodes[node]["coord"]).astype("float")
		rot_point = np.dot(c, rot)
		trans_point = rot_point + trans
		
		## Set graph coordinate to new coordinates
		interface.graphs[1].nodes[node]["coord"] = trans_point
		
	return interface

def set_correspondence(interface, max_match):
	if max_match == []:
		return interface
	m = max_match[3]
	# print(m)

	# for edge in interface.graphs[1].edges():
	#     print(edge)
	
	temp_dict = {}
	# print("New m graph", m)
	# colors = ["blue" if m.get(a) == b else "grey" for a, b in edges]
	
	for k,v in m.items():
		e1 = k[1]
		e2 = v[1]
		# print("Both edges:",e1, e2)
		# for edge in interface.graphs[1].edges():
		#     if edge == e2:
		#         edge = e1
				
			
		
## Functions to get graph differences for heatmap analysis
def find_different_edges(graph1, graph2):
	# Get the sets of edges from both graphs
	edges1 = set(graph1.edges())
	edges2 = set(graph2.edges())

	# Find the edges that are in one graph but not the other
	different_edges1 = edges1 - edges2
	different_edges2 = edges2 - edges1

	return different_edges1, different_edges2

def get_weight(edge, edge_counts):
	"""Get edge weights by counting the number of edges that have changed from the original edge list

	Args:
		edge (_type_): _description_
		edge_counts (_type_): _description_

	Returns:
		total_count: count of changes for edge
	"""
	total_count = 0
	for diff_edge, count in edge_counts.items():
		if edge == diff_edge:
			total_count += count
		if (edge[0], edge[1]) == (diff_edge[1], diff_edge[0]):
			total_count += count

	return total_count


# Find difference in graphs



def make_heatmatrix(G_orig):
	# Create a figure and axis for plotting
	fig, ax = plt.subplots(figsize=(8, 8))

	first_node = next(iter(G_orig.nodes()))
	first_chain = G_orig.nodes(data=True)[first_node]['attribute']['chain']	
 
	l = [
		node
		for node, attrs in G_orig.nodes(data=True)
		if "chain" in attrs["attribute"] and attrs["attribute"]["chain"] == first_chain
	]
	r = [
		node
		for node, attrs in G_orig.nodes(data=True)
		if "chain" in attrs["attribute"] and attrs["attribute"]["chain"] != first_chain
	]
	G = nx.Graph()
	G.add_nodes_from(l, bipartite=0)
	G.add_nodes_from(r, bipartite=1)
	G.add_weighted_edges_from([u, v, G_orig[u][v]["weight"]] for u, v in G_orig.edges())

	matrix = np.zeros((len(l), len(r)))
	for edge in G.edges(data=True):
		try:
			l_idx = l.index(edge[0])
		except ValueError as error:
			continue
		try:
			r_idx = r.index(edge[1])
		except ValueError as error:
			continue
		weight = edge[2]["weight"]
		matrix[l_idx][r_idx] = weight
	# Step 6: Visualize the matrix using Matplotlib
	plt.imshow(matrix, cmap="Greys", interpolation="none", aspect="auto")
	sns.heatmap(matrix, annot=True, cmap="Greys")
	# plt.colorbar(label="Edge Weight")
	plt.xticks(np.arange(len(r)) + 0.5, r)
	plt.yticks(np.arange(len(l)) + 0.5, l)
	plt.xlabel("Barstar")
	plt.ylabel("Barnase")
	plt.title("Bipartite Graph Matrix")
	plt.show()

def make_heatmap(G_orig):
	# Extract nodes with "chain" attribute equal to "A" and store them in a list
	first_node = next(iter(G_orig.nodes()))
	first_chain = G_orig.nodes(data=True)[first_node]['attribute']['chain']
	# print("--------------------Chain------------------------")
	# print(first_chain)
	
	l = [
		node
		for node, attrs in G_orig.nodes(data=True)
		if "chain" in attrs["attribute"] and attrs["attribute"]["chain"] == first_chain
	]
	r = [
		node
		for node, attrs in G_orig.nodes(data=True)
		if "chain" in attrs["attribute"] and attrs["attribute"]["chain"] != first_chain
	]

	# Extract edge weights into a list
	edge_weights = [G_orig[u][v]["weight"] for u, v in G_orig.edges()]

	# Define a colormap from green (weight=1) to red (higher weights)
	cmap = plt.cm.get_cmap("Greys", max(edge_weights) + 1)

	# Create a figure and axis for plotting
	fig, ax = plt.subplots(figsize=(8, 8))

	G = nx.Graph()
	G.add_nodes_from(l, bipartite=0)
	G.add_nodes_from(r, bipartite=1)
	G.add_weighted_edges_from((u, v, G_orig[u][v]["weight"]) for u, v in G_orig.edges())

	top = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}

	pos = nx.bipartite_layout(G, top)

	nx.draw(
		G,
		pos,
		with_labels=True,
		node_size=500,
		node_color="lightblue",
		ax=ax,
		edge_color=edge_weights,
		width=2,
		edge_cmap=cmap,
	)

	# nx.draw_networkx_edge_labels(
	#     G, pos, edge_labels=nx.get_edge_attributes(G, "weight")
	# )

	# Add a colorbar to show the mapping of edge weights to colors
	sm = plt.cm.ScalarMappable(
		cmap=cmap, norm=plt.Normalize(vmin=1, vmax=max(edge_weights))
	)
	sm.set_array([])
	cbar = plt.colorbar(sm)

	# Show the plot
	plt.show()


def main():
	# data = open_pickle_files("results/1brs_i")
	data = open_pickle_files("results/base-0/1BRS")
	print(data[0])

	# test_rotation()

	## Initialize an empty Counter to keep track of edge counts
	edge_counts = Counter()
	aggregated_graph = nx.Graph()
	csv_file_name = "edge_difference.csv"
	difference_data = []
	
	## Get maximum matching from aligned data
	max_match_list = []
	for interface, aligned_list in data:
		max_match = check_perfect_matches(interface, aligned_list)
		max_match_list.append(max_match)
	# print(len(max_match_list))
	# print(max_match_list)
	
	# Track how many graphs have a difference of a certain number of bonds
	diff_tracker = [0,0,0,0]
 
	count = 0
	max_diff_count = -1
	max_diff_set = {}
	max_diff_index = -1
	for interface, aligned_list in data:
		## Find graph with most difference in edges
		temp, X = find_different_edges(interface.graphs[0], interface.graphs[1])
		difference_data.append((count, temp))
		if len(temp)>=max_diff_count:
			max_diff_count = len(temp)
			max_diff_set = temp
			max_diff_index = count
		
		if len(temp) == 0:
			diff_tracker[0] +=1
		elif len(temp) == 1:
			diff_tracker[1] +=1
			print("Mutant Index Count:", count)
			print(temp)
		elif len(temp) == 2:
			diff_tracker[2] +=1
		elif len(temp) == 3:
			diff_tracker[3] +=1
		
		
		## Apply maximum match (rot, trans) to interface
		interface = apply_perfect_matches(interface, max_match_list[count])
		set_correspondence(interface, max_match_list[count])
		count +=1
		
		## Add the nodes and edges from graph1 and graph2 to one graph, aggregating both graphs together
		aggregated_graph.add_nodes_from(
			[
				(node, {"attribute": attr})
				for (node, attr) in interface.graphs[0].nodes.items()
			]
		)
		aggregated_graph.add_nodes_from(
			[
				(node, {"attribute": attr})
				for (node, attr) in interface.graphs[1].nodes.items()
			]
		)

		# Aggregate all edges that appear into one graph
		aggregated_graph.add_edges_from(interface.graphs[0].edges())
		aggregated_graph.add_edges_from(interface.graphs[1].edges())

		d1, d2 = find_different_edges(interface.graphs[0], interface.graphs[1])

		# Iterate through each set of edges and update the counts
		edge_counts.update(d1)
		edge_counts.update(d2)

	with open(csv_file_name, 'w', newline='') as csv_file:
		csv_writer = csv.writer(csv_file)
		for item in difference_data:
			index, edges = item
			for edge in edges:
				csv_writer.writerow([index, edge])
 
	for edge, count in edge_counts.items():
		print(f"Edge {edge}: {count} times in all sets")
	
	
	print("Diff_tracker:",diff_tracker)

	print("Max # of different edges:", max_diff_count)
	print("Mutation set index:", count)
	print("Set of nodes for different edges:", max_diff_set)
	nx.draw(nx.Graph(max_diff_set), with_labels = True)
	plt.show()

	print("Aggregated Graph Edges:", aggregated_graph.edges())

	for edge in aggregated_graph.edges():
		weight = get_weight(edge, edge_counts)
		nx.set_edge_attributes(aggregated_graph, {edge: {"weight": weight}})

	## print(aggregated_graph.nodes(data=True))
	make_heatmap(aggregated_graph)

	make_heatmatrix(aggregated_graph)


if __name__ == "__main__":
	main()
