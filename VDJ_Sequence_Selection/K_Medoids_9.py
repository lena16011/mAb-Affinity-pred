'''
Implementation of a k-medoid clustering to choose the sequences that
represent the data as well as possible.

The steps for the implementation of a PAM (Partitioning around Medoids)
algorithm are as follows:
- Initialize: randomly select k medoids of the data points
- associate each data point to the closest medoid
- while the cost of configuration decreases:
    - for each medoid m and non-medoid o:
        - swap m and o, reassign datapoints to closest medoids and recalculate cost
              if cost increases: undo swap
'''


import numpy as np
import stringdist
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
from utils import NW_functions as NW


# find k-Medoids
def kMedoids(D, k, tmax=100):
    '''
    Cal
    :param D:
    :param k:
    :param tmax:
    :return:
    '''
    
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

# network plot function
def network_plot(data_tup, n_nodes,tar_idx, k, colors, scale_factor=5, seed=123, figsize = (16,14),
                 save=False, out_file = None):
    '''
    :param data_tup: ebunch, list of 3-tuple generated with the ebunch() function; each entry of
                    the list is (node1, node2, weight);
    :param n_nodes: number of nodes to add to the network plot
    :param tar_idx: index of our target node to print the number of edges connected to the node
    :param k: is the hyperparameter for the Fruchtermann-Reingold algorithm [0-1]; the higher the
             value the higher the further away the
    :param scale_factor: another factor to scale the distance between the nodes
    :param colors: list of colors for the nodes
    :param figsize: tuple of the colors
     '''
    # set number of edges connected to the target node
    n_tar_edges = len([tup for tup in data_tup if (tup[0]==tar_idx or tup[1]==tar_idx)])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # set up the network graph
    G = nx.Graph()
    # Add the nodes
    G.add_nodes_from(range(n_nodes))
    # Then add the edges from the 3-tuple
    G.add_weighted_edges_from(data_tup)
    # calculate the positions of the nodes with the Fruchtermann-Reingold force-directed algorithm
    pos = nx.spring_layout(G, k=k, scale=scale_factor, seed=seed)
    nx.draw(G, node_color=colors, font_size=7, node_size=150, with_labels=True, pos = pos)
    ax.set_title(label=str('Network plot of sequences with LD=' + str(data_tup[0][2]) + ' with k = ' + str(k)) +
              '\nnumber of edges connected to target node: ' + str(n_tar_edges), fontdict={'fontsize': 20})
    if save == True:
        plt.savefig(out_file)
    plt.show()



# Load the data with the sequences as a list

abs_path = 'D:/Dokumente/Masterarbeit/Lena/VDJ_Sequence_Selection'


in_seq_file = abs_path + '/data/VDJ_selection/original_data/uniq_VDJs_from_Ann_Table_data_AP_simfilt80.txt'
with open(in_seq_file, 'r') as f_in:
    seqs_lst = []
    # skip header
    next(f_in)
    for line in f_in:
        # read line and take
        parts = line.rstrip().split("\t")
        seqs_lst.append(parts[4])


# Calculate the distance matrix or load it
distance_matrix = NW.calculate_norm_dist_matrix(seqs_lst)

# k-Medoids clustering with 50 clusters
medoids, clusters = kMedoids(distance_matrix, 50)

# save the medoid sequences
med_seqs = [seqs_lst[med] for med in medoids]

seqs_df = pd.read_csv(in_seq_file, sep='\t')

medoid_data = pd.DataFrame(columns=seqs_df.columns)
for i in range(len(seqs_df.VDJ_AA)):
    if seqs_df.VDJ_AA[i] in med_seqs:
        medoid_data = medoid_data.append(seqs_df.iloc[i,:])
# reset index
medoid_data.reset_index(drop=True, inplace=True)

# Save the (intermediated) dataset
out_path = abs_path + '/data/VDJ_selection/VDJ_final_data/final_50_selection/not_edited/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

medoid_data.to_csv(out_path + 'final_50_selection_data_115AA.txt',
                   sep='\t')

### Analyze data
# load data
file_path_sel = abs_path + '/data/VDJ_selection/VDJ_final_data/final_50_selection/not_edited/final_50_selection_data_115AA.txt'
sel_50 = pd.read_csv(file_path_sel, sep='\t')
sel_50_lst = list(sel_50.VDJ_AA)

# calculate LD
sel_dist_matrix = NW.calculate_LD_dist_matrix(sel_50_lst)



#### Network plots

# set number of nodes
n_nodes = len(seqs_lst)

# set the indices of the Medoids
med_idx = [x for x in medoids]

# set target VDJ
t_VDJ = 'GAELVRPGASVTLSCKASGYTFTDYEMHWVKQTPVHGLEWIGDIDPETGGTAYNQNFKGKATLTADKSSSTAYMEFRSLTSEDSAVYYCTRDYYGSNYLAWFAYWGQGTLVTVSA'

# set colors for the nodes where blue nodes are the medoids and the green one is the
colors = ['red']*int(len(seqs_lst))
for x in med_idx:
    colors[x] = 'blue'

target_idx = seqs_lst.index(t_VDJ)
colors[target_idx] = 'green'

# create ebunches from sequence list
nodes_LD1 = NW.ebunch_LD(seqs_lst, 1)
nodes_all = NW.ebunch_LD(seqs_lst)

# set k for the Fruchter Reingard algorithm
k = 0.19
# define filepath for the network plot
file_path = abs_path + '/data/Plots/VDJ_selection/final_50_selection/'
if not os.path.exists(file_path):
    os.makedirs(file_path)

# generate the networkplot
network_plot(data_tup= nodes_LD1, n_nodes= n_nodes, tar_idx= target_idx, k=k,
             seed=1487, colors= colors, figsize = (20,18), scale_factor=6,
             save=False, out_file=file_path + 'Network_LD_1.pdf')



