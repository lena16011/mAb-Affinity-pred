'''
Script to generate the sequence network graphs of the 26 selected CDR3 sequences;

- Calculating of normalized LD distance matrix
- Creating ebuches (iterable 3-tuples, that defines nodes between seq1 and seq2; weihght is the similarity (0-1) between the
two sequences) --> eb = (seq1_id, seq2_id, weight)
- Create network plots
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stringdist
import os
import networkx as nx
from utils import NW_functions as NW


# set seed
np.random.seed(123)
### set paths
abs_path = 'D:/Dokumente/Masterarbeit/Lena/VDJ_Sequence_Selection'

in_file_path = abs_path + '/data/Clustering/Summary/overlap_filt_clustering_simfilt80.txt'
output_path = abs_path + '/data/Plots/CDR3_Selection/NW_plots/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load in the data file of the filtering (80%) files
data_raw = pd.read_csv(in_file_path, sep='\t')


###
# NOTEs:
# - need a list of 3-tuples, that store the sequences and their respective distance measure;
# - nodes are named according to the index here; so Index and sequence identity is stored in this data set!
# - target sequence has node nr. 5;
###

data_tup_all = NW.ebunch_norm(list(data_raw.CDR3_AA))


###### SET UP NETWORK GRAPH BASED ON ALL SEQUENCES

# set number of nodes
n_nodes = len(data_raw)
# set colors for the nodes
colors = ['red']*5 + ['green'] + ['red']*20

# set k for retrieving positions of the nodes using Fruchterman-Reingold force-directed algorithm
# Optimal distance between nodes; default here 0.196116 (1/sqrt(n))
k1=0.4
# set up plot
plt.plot()
G1 = nx.Graph()
# Add the nodes
G1.add_nodes_from(range(n_nodes))
# Then we add on the edges, that are within the threshold we applied;
G1.add_weighted_edges_from(data_tup_all)
pos = nx.spring_layout(G1, k=k1, seed=3)
nx.draw(G1, pos=pos, node_color=colors, font_size=8, node_size=100, with_labels=True)
plt.title('Network plot with all sequence edges k=' + str(k1))
plt.savefig(output_path + 'NW_plot_allseq.pdf')
plt.show()



########### NETWORK GRAPH WITH EDGES according to set "SIMILARITY" THRESHOLD (edges are only shown, when similarity between
# the sequences is > the threshold)

# set similarity threshold
thres = 0.9
data_filt = data_tup_all.copy()
# filter the edges that are under the threshold
#  we apply the distance threshold from which we assume they are not connected anymore;
[data_filt.remove(tup) for tup in data_tup_all if tup[2] < thres]
n_tar_edges = len([tup for tup in data_filt if (tup[0]==5 or tup[1]==5)])


print(str('Number of edges in the unfiltered graph: ' + str(len(data_tup_all))))
print(str('Number of edges in the filtered graph (threshold: ' + str(thres) + '): ' + str(len(data_filt))))
print(str('Number of edges connected to our target node: ' + str(n_tar_edges)))

# set up the network graph
k = 0.1
plt.plot()
G2 = nx.Graph()
# Add the nodes
G2.add_nodes_from(range(n_nodes))
# Then we add on the edges, that are within the threshold we applied;
G2.add_weighted_edges_from(data_filt)
pos = nx.spring_layout(G2, k=k, iterations=1000, seed=163)
nx.draw(G2, pos=pos, node_color=colors, font_size=8, node_size=100, with_labels=True)
plt.title(str('Network plot with k=' + str(k) + ' and norm_LD-threshold =' + str(thres) +
              '\nnumber of edges connected to target node: ' + str(n_tar_edges)))
plt.savefig(output_path + 'NW_plot_thresh'+str(thres)[2:]+'.pdf')
plt.show()




#### NETWORK PLOT based on CDR3 LD(n) similarity layer (= based on Levenshtein Distance)

# create LD ebunches
data_tup2 = NW.ebunch_LD(list(data_raw.CDR3_AA))

print('LDs of the target sequence vs all', set([tup[2] for tup in data_tup2 if (tup[0]==5 or tup[1]==5)]))
# our target sequence has LDs in the range of 1-3 to all the other sequences
print('number of sequences that have LD 1: ', len([tup[2] for tup in data_tup2 if tup[2]==1]))
print('number of sequences that have LD 2: ', len([tup[2] for tup in data_tup2 if tup[2]==2]))
print('number of sequences that have LD 3: ', len([tup[2] for tup in data_tup2 if tup[2]==3]))
print('number of sequences that have LD 4: ', len([tup[2] for tup in data_tup2 if tup[2]==4]))
print('number of sequences that have LD 5: ', len([tup[2] for tup in data_tup2 if tup[2]==5]))


### Extract the tuples that have LD=x (check out how many edges are gonna be in the plot and connected to the target sequence
LD = 1
data_tup_LD = [tup for tup in data_tup2 if tup[2]==LD]
print(set([tup[0] for tup in data_tup2] + [tup[1] for tup in data_tup2 if tup[2]==LD]))
# see the sequnces that appear to have edges with LD=2
print(len(data_tup_LD))
# 148 edges at LD=2, 35 at LD=1

# show the edges to or from the target sequence
target_seq_id = 5
n_tar_edges = len([tup for tup in data_tup_LD if (tup[0]==target_seq_id or tup[1]==target_seq_id)])
print(str('Number of edges connected to our target node: ' + str(n_tar_edges)))
# 8 nodes bei LD2

# set up the network graph
k3 = 0.5

plt.plot()
G3 = nx.Graph()
# Add the nodes
G3.add_nodes_from(range(n_nodes))
# Then we add on the edges, that are within the threshold we applied;
G3.add_weighted_edges_from(data_tup_LD)
pos = nx.spring_layout(G3, k=k3, seed=13)#iterations=1000, )
nx.draw(G3, pos=pos, node_color=colors, font_size=8, node_size=100, with_labels=True)
plt.title(str('Network plot of sequences with LD=' + str(LD) + ' with k = ' + str(k3)) +
          '\nnumber of edges connected to target node: ' + str(n_tar_edges))
plt.savefig(output_path+'/NW_plot_LD'+str(LD)+'.pdf')
plt.show()

