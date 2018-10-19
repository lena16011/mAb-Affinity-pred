import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import stringdist
import math
import networkx as nx

# implement this network graph for the 26 sequences
# we need a list of 3-tuples, that holds the sequences and their respective distance
# measure;
in_file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_filt_clustering_simfilt80.txt'
# Load in the data file of the filtering (80%) files
data_raw = pd.read_csv(in_file_path, sep='\t')
###
# NOTE: we call the nodes according to the index here; so Index and sequence identity is stored in this data set!
# our target sequence has node nr. 5;
###

# now calculate the distance between the sequences to use this measure later as weights for the nodes;
col=range(len(data_raw.CDR3_AA))
dist_matrix = pd.DataFrame(columns=col)
data_ser = pd.DataFrame(columns=col)
for i in range(len(data_raw.CDR3_AA)):
    data_ser = data_raw.CDR3_AA[:i+1].apply(stringdist.levenshtein_norm,
                                              args=(data_raw.CDR3_AA[i],))
    dist_matrix.loc[i] = data_ser.T

# now we can store the 3-tuples with the distances as weights (ebunch for the network graph)
data_tup = []
for i in range(len(data_raw)):
    for j in range(i+1, len(data_raw)):
        val = (i, j, 1 - dist_matrix.iloc[j, i])
        data_tup.append(val)

# make a copy of all the nodes & edges
data_tup_all = data_tup.copy()

###### SET UP NETWORK GRAPH BASED ON ALL SEQUENCES

# set number of nodes
n_nodes = len(data_raw)
# set colors for the nodes
colors = ['red']*5 + ['green'] + ['red']*20
# set k for retrieving positions of the nodes using Fruchterman-Reingold force-directed algorithm
# Optimal distance between nodes; default here 0.196116 (1/sqrt(n))
k1=0.4

# show the network graph
plt.plot()
G1 = nx.Graph()
# # Add the nodes
G1.add_nodes_from(range(n_nodes))
# # Then we add on the edges, that are within the threshold we applied;
G1.add_weighted_edges_from(data_tup_all)
pos = nx.spring_layout(G1, k=k1, fixed=[5])
nx.draw(G1, pos=pos, node_color=colors, font_size=8, node_size=100, with_labels=True)
plt.title(str('Network plot with all sequence edges k=' + str(k1)))
# plt.savefig('/media/lena/LENOVO/Dokumente/Masterarbeit/data/NW_plot_allseq.pdf')
plt.show()

########### NETWORK GRAPH WITH "SIMILARITY"  BASED FILTERING

# add the graph with the filtered edges with the set threshhold
# set threshold
thres = 0.85
data_filt = data_tup.copy()
# filter the edges that are under the threshold
#  we apply the distance threshold from which we assume they are not connected anymore;
[data_filt.remove(tup) for tup in data_tup_all if tup[2] < thres]
n_tar_edges = len([tup for tup in data_filt if (tup[0]==5 or tup[1]==5)])

print(str('Number of edges in the unfiltered node: ' + str(len(data_tup_all))))
print(str('Number of edges in the filtered node (threshold: ' + str(thres) + '): ' + str(len(data_filt))))
print(str('Number of edges connected to our target node: ' + str(n_tar_edges)))

# set up the network graph
k = 0.6
plt.plot()
G2 = nx.Graph()
# Add the nodes
G2.add_nodes_from(range(n_nodes))
# Then we add on the edges, that are within the threshold we applied;
G2.add_weighted_edges_from(data_filt)
pos = nx.spring_layout(G2, k=k, iterations=1000)
nx.draw(G2, node_color=colors, font_size=8, node_size=100, with_labels=True)
plt.title(str('Network plot with k=' + str(k) + ' and norm_LD-threshold =' + str(thres) +
              '\nnumber of edges connected to target node: ' + str(n_tar_edges)))
#plt.savefig(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/NW_plot_thresh'+str(thres)[2:]+'.pdf'))
plt.show()


#### NETWORK PLOT based on CDR3 LD(n) similarity layer (= based on Levenshtein Distance)

# Calculate the Levenshtein distance
col=range(len(data_raw.CDR3_AA))
dist_matrix_lev = pd.DataFrame(columns=col)
data_ser = pd.DataFrame(columns=col)
for i in range(len(data_raw.CDR3_AA)):
    data_ser = data_raw.CDR3_AA[:i+1].apply(stringdist.levenshtein, args=(data_raw.CDR3_AA[i],))
    dist_matrix_lev.loc[i] = data_ser.T

# now we can store the 3-tuples with the distances as weights (ebunch for the network graph)
data_tup2 = []
for i in range(len(data_raw)):
    for j in range(i+1, len(data_raw)):
        val = (i, j, dist_matrix_lev.iloc[j, i])
        data_tup2.append(val)

print('LD of the target sequence vs all', set([tup[2] for tup in data_tup2 if (tup[0]==5 or tup[1]==5)]))
# our target sequence has LDs in the range of 1-3 to all the other sequences
print('number of sequences that have LD 1: ', len([tup[2] for tup in data_tup2 if tup[2]==1]))
print('number of sequences that have LD 2: ', len([tup[2] for tup in data_tup2 if tup[2]==2]))
print('number of sequences that have LD 3: ', len([tup[2] for tup in data_tup2 if tup[2]==3]))
print('number of sequences that have LD 4: ', len([tup[2] for tup in data_tup2 if tup[2]==4]))
print('number of sequences that have LD 5: ', len([tup[2] for tup in data_tup2 if tup[2]==5]))

### Extract the tuples that have LD=x with the target sequence
LD = 5
data_tup_LD = [tup for tup in data_tup2 if tup[2]==LD]
print(set([tup[0] for tup in data_tup2] + [tup[1] for tup in data_tup2 if tup[2]==LD]))
# see the sequnces that appear to have edges with LD=2 or 2
print(len(data_tup_LD))
# 148 edges at LD=2, 35 at LD=1

n_tar_edges = len([tup for tup in data_tup_LD if (tup[0]==5 or tup[1]==5)])
print(str('Number of edges connected to our target node: ' + str(n_tar_edges)))
# 16 nodes bei LD1

# set up the network graph
k3 = 0.8
plt.plot()
G3 = nx.Graph()
# Add the nodes
G3.add_nodes_from(range(n_nodes))
# Then we add on the edges, that are within the threshold we applied;
G3.add_weighted_edges_from(data_tup_LD)
pos = nx.spring_layout(G3, k=k3)
nx.draw(G3, node_color=colors, font_size=8, node_size=100, with_labels=True)
plt.title(str('Network plot of sequences with LD=' + str(LD) + ' with k = ' + str(k3)) +
          '\nnumber of edges connected to target node: ' + str(n_tar_edges))
plt.savefig(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/NW_plot_LD'+str(LD)+'.pdf'))
plt.show()

