'''
Script to visualize the VDJ sequences selected/filtered in VDJ_selection_from_Annot_Table.py script
that occur with our selected CDR3s; Network plots will be created with:

 - similarity layer approach: Edges between the sequences will be drawn between all sequence pairs
                            that have a Levenshtein distance (LD) of 1, 2, ... n

 - similarity threshhold: (Edges between the sequences will be drawn if the sequences'
                        similarity is above a former defined threshhold)

As we only have distances between the sequences/nodes, Fruchtermann-Reingold force-directed algorithm
(spring_layout()) is used to position the nodes in the plot with the given distances.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stringdist
import math
import networkx as nx
import random
from utils import NW_functions as NW
import os

####### Input data are the VDJs selected from the overlap of affinity propagation and similarity filtering

# set the input path and file
abs_path = os.getcwd()
in_file_path = os.path.join(abs_path, 'VDJ_Sequence_Selection/data/VDJ_Selection/VDJ_final_data/VDJs_Selection_with_CDR3.txt')
in_file_freqs = os.path.join(abs_path, 'VDJ_Sequence_Selection/data/VDJ_Selection/VDJ_final_data/VDJ_Selection_with_CDR3_Suppl_Table.txt')
in_file_50 = os.path.join(abs_path, 'VDJ_Sequence_Selection/data/VDJ_Selection/VDJ_final_data/final_50_selection/final_50_selection_data_SUPPL_TBL_KDs.txt')

# create folder, if it doesn't exist
if not os.path.exists(os.path.join(abs_path, 'VDJ_Sequence_Selection/data/Plots/VDJ_selection/')):
    os.makedirs(os.path.join(abs_path, 'VDJ_Sequence_Selection/data/Plots/VDJ_selection/'))


# set target sequences
t_cdr = 'CTRDYYGSNYLAWFAYW'
t_VDJ = 'GAELVRPGASVTLSCKASGYTFTDYEMHWVKQTPVHGLEWIGDIDPETGGTAYNQNFKGKATLTADKSSSTAYMEFRSLTSEDSAVYYCTRDYYGSNYLAWFAYWGQGTLVTVSA'
random.seed(123)

# Load in the data
data_raw = pd.read_csv(in_file_path, sep='\t')

#########
# NOTE: we call the nodes according to the index here; so Index and sequence identity is stored in this data set!
#########

# create ebunches
data_tup_all = NW.ebunch_norm(list(data_raw.VDJ_AA))

# set the position of the sequences with the target CDR3
tar_idx = list(data_raw.index.values[data_raw.CDR3_AA == t_cdr])

# set number of nodes
n_nodes = len(data_raw)

# set colors for the nodes
colors = ['red']*int(len(data_raw))
for x in tar_idx:
    colors[x] = 'green'
t_idx = int(data_raw.index.values[data_raw.VDJ_AA == t_VDJ])
colors[t_idx] = 'blue'


####### include frequencies doesn't look good ##########
# set node sizes from the frequencies
# load the frequencies
# initialize a list for the counts to store
# count_seqs = []
# # open file
# with open(in_file_freqs, 'r') as f_in:
#     for line in f_in:
#         cols = line.rstrip().split('\t')
#
#         # iterate through the VDJs and extract the corresponding frequencies
#         for i, VDJ in enumerate(data_raw.VDJ_AA):
#             # check if the VDJs are in the column of the current line
#             if VDJ == cols[3] and cols[7] != 0 and cols[4]!=0:
#                 count_seqs = count_seqs + [int(float(cols[7])) + int(float(cols[4]))]
#                 continue
#             if VDJ == cols[3] and cols[4] != 0:
#                 count_seqs = count_seqs + [int(float(cols[4]))]
#             if VDJ == cols[3] and cols[7] != 0:
#                 count_seqs = count_seqs + [int(float(cols[7]))]
#
# ######### TRY to log the sizes
# count_seqs_log = [math.log(x) for x in count_seqs]
# count_seqs_log_mul = count_seqs*10
########################################################################################################################


######################### NETWORK GRAPH WITH "SIMILARITY"  BASED FILTERING

# set similarity threshold
thres = 0.99

# filter the edges that are under the threshold
#  we apply the distance threshold from which we assume they are not connected anymore;
data_filt = data_tup_all.copy()
[data_filt.remove(tup) for tup in data_tup_all if tup[2] < thres]
n_tar_edges = len([tup for tup in data_filt if (tup[0]==tar_idx or tup[1]==tar_idx)])


print(str('Number of edges in the unfiltered node: ' + str(len(data_tup_all))))
print(str('Number of edges in the filtered node (threshold: ' + str(thres) + '): ' + str(len(data_filt))))
print(str('Number of edges connected to our target node: ' + str(len([tup for tup in data_filt if (tup[0]==tar_idx or tup[1]==tar_idx)]))))


# set k (optimal distance between nodes) as a parameter for the Fruchtermann-Reingold algorithm
k = 0.6

# set up the network graph
fig = plt.figure(figsize=(23,20))
ax = fig.add_subplot(111)
G2 = nx.Graph()

# Add the nodes
G2.add_nodes_from(range(n_nodes))

# Then we add on the edges, that are within the threshold we applied;
G2.add_weighted_edges_from(data_filt)
pos = nx.spring_layout(G2, k=k, scale=5, seed=123)
nx.draw(G2, pos=pos, node_color=colors, font_size=7, node_size=150, with_labels=True)

# set title for the plot
ax.set_title(label=str('Network plot with k=' + str(k) + ' and norm_LD-threshold =' + str(thres) +
              '\nnumber of edges connected to target node: ' + str(n_tar_edges)),
             fontdict={'fontsize': 20})
#fig.savefig(abs_path + '/data/Plots/VDJ_selection/NW_plot_VDJ_thresh'+str(thres)[2:]+'.pdf'))
fig.show()




#### NETWORK PLOT based on VDJ LD(n) similarity layer (= based on Levenshtein Distance)
data_tup2 = NW.ebunch_LD(list(data_raw.VDJ_AA))

print('LD of the target sequence vs all', set([tup[2] for tup in data_tup2 if (tup[0]==tar_idx or tup[1]==tar_idx)]))
print('All occuring LDs:', set([tup[2] for tup in data_tup2]))
# our target sequence has LDs in the range of 1-3 to all the other sequences
print('number of sequences that have LD 1: ', len([tup[2] for tup in data_tup2 if tup[2]==1]))
print('number of sequences that have LD 2: ', len([tup[2] for tup in data_tup2 if tup[2]<=2]))
print('number of sequences that have LD 3: ', len([tup[2] for tup in data_tup2 if tup[2]<=3]))
print('number of sequences that have LD 10: ', len([tup[2] for tup in data_tup2 if tup[2]<=10]))
print('number of sequences that have LD 20: ', len([tup[2] for tup in data_tup2 if tup[2]<=20]))
print('number of sequences that have LD 27: ', len([tup[2] for tup in data_tup2 if tup[2]<=27]))

### Extract the tuples that have LD=x with the target sequence
LD = 1
data_tup_LD = [tup for tup in data_tup2 if tup[2]<=LD]
print(set([tup[0] for tup in data_tup2] + [tup[1] for tup in data_tup2 if tup[2]<=LD]))
# see the sequnces that appear to have edges with LD=2 or 2
print(len(data_tup_LD))



print(str('Number of edges connected to our target node: ' + str(len([tup for tup in data_tup_LD if (tup[0]==t_idx or tup[1]==t_idx)]))))
# 16 nodes bei LD1

k3 = 0.19
fig = plt.figure(figsize=(16,14))
ax = fig.add_subplot(111)
G3 = nx.Graph()
# Add the nodes
G3.add_nodes_from(range(n_nodes))
# Then we add on the edges, that are within the threshold we applied;
G3.add_weighted_edges_from(data_tup_LD)
pos = nx.spring_layout(G3, k=k3, scale=5, seed=139)
nx.draw(G3, node_color=colors, font_size=5, node_size=150
        , with_labels=True, pos = pos)
ax.set_title(label=str('Network plot of sequences with LD=' + str(LD) + ' with k = ' + str(k3)) +
                   '\nnumber of edges connected to target node: ' + str(n_tar_edges), fontdict={'fontsize': 20})
# plt.savefig(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Plots/VDJ_selection/INTERM_NW_plot_VDJs_FREQ'+str(LD)+'.pdf'))
plt.show()






#### NETWORK PLOT based on 50 selected VDJ LD(n) similarity layer (= based on Levenshtein Distance)
data_raw = pd.read_csv(in_file_50, sep='\t')
# reset the index to the
# sequences were adjusted with a different alignment
t_VDJ = "QVQLQQSGAELVRPGASVTLSCKASGYTFTDYEMHWVKQTPVHGLEWIGDIDPETGGTAYNQNFKGKATLTADKSSSTAYMEFRSLTSEDSAVYYCTRDYYGSNYLAWFAYWGQGTLVTVSA"
# reset index from R
data_raw.reset_index(drop=True, inplace=True)
data_raw["KD"] = data_raw["KD_nM"]
data_raw["KD_nM"] = data_raw["KD"]*1e9

data_tup2 = NW.ebunch_LD(list(data_raw.VDJ_AA))

# set target index as the initial binder sequence

# set colors for the nodes
colors = ['red']*int(len(data_raw))
t_idx = int(data_raw.index.values[data_raw.VDJ_AA == t_VDJ])
colors[t_idx] = 'green'

#
# print('LD of the target sequence vs all', set([tup[2] for tup in data_tup2 if (tup[0]==t_idx or tup[1]==t_idx)]))
# print('All occuring LDs:', set([tup[2] for tup in data_tup2]))
# # our target sequence has LDs in the range of 1-3 to all the other sequences
# print('number of sequences that have LD 1: ', len([tup[2] for tup in data_tup2 if tup[2]==1]))
# print('number of sequences that have LD 2: ', len([tup[2] for tup in data_tup2 if tup[2]<=2]))
# print('number of sequences that have LD 3: ', len([tup[2] for tup in data_tup2 if tup[2]<=3]))
# print('number of sequences that have LD 10: ', len([tup[2] for tup in data_tup2 if tup[2]<=10]))
# print('number of sequences that have LD 20: ', len([tup[2] for tup in data_tup2 if tup[2]<=20]))
# print('number of sequences that have LD 27: ', len([tup[2] for tup in data_tup2 if tup[2]<=27]))

### Extract the tuples that have LD=x with the target sequence
LD = 2
data_tup_LD = [tup for tup in data_tup2 if tup[2]<=LD]
print(set([tup[0] for tup in data_tup2] + [tup[1] for tup in data_tup2 if tup[2]<=LD]))
# see the sequnces that appear to have edges with LD=2 or 2
print(len(data_tup_LD))


k3 = 1.1
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
G3 = nx.Graph()
# Add the nodes
G3.add_nodes_from(range(len(data_raw)))
# Then we add on the edges, that are within the threshold we applied;
G3.add_weighted_edges_from(data_tup_LD)
pos = nx.spring_layout(G3, k=k3,# scale=5,
                       seed=16)
nx.draw(G3, node_color=colors, font_size=12, node_size=330
        , with_labels=True, pos = pos)

ax.set_title(label=str('Network plot of 50 selected sequences with LD=' + str(LD) + ' with k = ' + str(k3)) +
                   '\nnumber of edges connected to target node: ' + str(len([tup for tup in data_tup_LD if (tup[0]==t_idx or tup[1]==t_idx)])), fontdict={'fontsize': 15})
#fig.savefig(os.path.join(abs_path ,'VDJ_Sequence_Selection/data/Plots/VDJ_selection/NW_plot_50_VDJ_LD'+str(LD)+'2.pdf'))
plt.show()





####### NETWORK PLOT WITH KD VALUES
# # set colors for the nodes
# data_raw["KD"] = data_raw["KD_nM"]
# data_raw["KD_nM"] = data_raw["KD"]*1e9
# data_raw["colors"] = np.where(((data_raw["KD_nM"] < 1) & ~(data_raw["KD_nM"].isna())), "green", "red")
# data_raw.loc[data_raw["KD_nM"].isna(), "colors"] = 'grey'

# set colors for the nodes
data_raw.loc[data_raw["KD_nM"] < 1, "colors"] = '#1874cd' # dodger blue
data_raw.loc[(data_raw["KD_nM"] < 3) & (data_raw["KD_nM"] >= 1), "colors"] = '#228b22' # forest green
data_raw.loc[data_raw["KD_nM"] >= 3, "colors"] = '#cd0000' # red 3
data_raw.loc[data_raw["KD_nM"].isna(), "colors"] = '#C0C0C0' # grey



k3 = 1.1
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
G3 = nx.Graph()
# Add the nodes
G3.add_nodes_from(range(len(data_raw)))
# Then we add on the edges, that are within the threshold we applied;
G3.add_weighted_edges_from(data_tup_LD)
pos = nx.spring_layout(G3, k=k3,# scale=5,
                       seed=16)
nx.draw(G3, node_color=data_raw["colors"], font_size=12, node_size=330
        , with_labels=False, pos = pos)
labels = {n:lab[2:] for n,lab in data_raw.HC_ID.items() if n in pos}
nx.draw_networkx_labels(G3, pos, labels)#, font_size=22, font_color="whitesmoke")
ax.set_title(label=str('Network plot of 50 selected sequences with LD=' + str(LD) + ' with k = ' + str(k3)) +
                   '\nnumber of edges connected to target node: ' + str(len([tup for tup in data_tup_LD if (tup[0]==t_idx or tup[1]==t_idx)])), fontdict={'fontsize': 15})
fig.savefig(os.path.join(abs_path ,'VDJ_Sequence_Selection/data/Plots/VDJ_selection/NW_plot_50_VDJ_LD'+str(LD)+'_KDs.pdf'))
plt.show()
