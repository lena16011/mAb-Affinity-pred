#!/usr/bin/env python
import pandas as pd
import numpy as np
import sklearn.cluster as clust

#### Affinity propagation try with unique length distance matrix

# set absolute path
abs_path = 'D:/Dokumente/Masterarbeit/Lena/VDJ_Sequence_Selection'

# Load distance matrix
file_name = abs_path+'/data/Clustering/Clustering_uniq_length_dist_matrix_20range/uniq_length_dist_matrix.txt'
dist_matrix_CDR3 = pd.read_csv(file_name, sep='\t', dtype=np.float16, index_col=0,
                                    low_memory=True)
dist_matrix = dist_matrix_CDR3.iloc[:500, :500]

# save distance matrix for the testing of the script
dist_matrix_df = pd.DataFrame(dist_matrix)
# dist_matrix.to_csv(abs_path+'/data/Clustering/', sep='/t')

# convert the distance matrix to a similarity matrix
sim_matrix = 1 - dist_matrix

# make the matrix symmetric
sim_matrix = sim_matrix.fillna(value=0).as_matrix()
sim_matrix = np.tril(sim_matrix, -1) + np.tril(sim_matrix).T


# Perform affinity propagation clustering
af = clust.AffinityPropagation(affinity='precomputed')
cluster_labels = af.fit_predict(sim_matrix)
print(np.unique(cluster_labels))
# here we have 4 clusters

# save cluster labels
cluster_labels_df = pd.DataFrame(cluster_labels)
cluster_labels_df.to_csv(abs_path+'/data/Clustering/Affinity_propagation/aff_prop_cluster_labels.txt',
                      header=['cluster_nr'], index=True, sep='\t')


# save the sequences that are in the same cluster as the target sequence; tested with 300;
target_index = 300
seq_number = len(cluster_labels)
# get the sequence indices of the sequences that are in the target cluster
for i in range(seq_number):
    tar_seq_indices = [x for x in range(seq_number) if cluster_labels[x]==cluster_labels[target_index]]

# write index to files
tar_seq_indices = np.asarray(tar_seq_indices)
tar_seq_indices.tofile(abs_path+'/data/Clustering/Affinity_propagation/target_cluster_seq_indices.txt',
                       sep='\t')

# write summary file with the number of clusters found and a
out_string = str('Number of found clusters: ' + str(np.argmax(np.unique(cluster_labels))) + ' Target cluster: ' +
                str(cluster_labels[target_index]))
f= open(abs_path+'/data/Clustering/not_important/Summary_AP_clustering.txt',"w+")
f.write(out_string)
f.close()