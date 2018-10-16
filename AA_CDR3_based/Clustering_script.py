#!/usr/bin/env python
import pandas as pd
import numpy as np
FilePath = 'Cluster_labels/'
# Load the distance matrix
file_name = 'uniqCDR3_DistMatrix.txt'
dist_matrix_CDR3 = pd.read_csv(file_name, sep='\t', dtype=np.float16, index_col=0,
                                    low_memory=True)
# make the matrix symmetric
dist_matrix_CDR3 = dist_matrix_CDR3.fillna(value='0').as_matrix()
dist_matrix_CDR3 = np.tril(dist_matrix_CDR3) + np.tril(dist_matrix_CDR3, 1).T

### Agglomerative Clustering
import sklearn.cluster as clust
n_clust = range(50, 1000, 50)
clust_names = ['cluster_'+str(i) for i in range(50, 1000, 50)]
j=0
for i in n_clust:
    model = clust.AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                          linkage='average')
    clust_fit = model.fit(dist_matrix_CDR3)
    cluster_labels = clust_fit.fit_predict(dist_matrix_CDR3)
    locals()[clust_names[i]] = cluster_labels
    j=j+1

# write the cluster labels to files
for i in range(len(clust_names)):
    filesave = pd.DataFrame(locals()[clust_names[i]])
    filesave.to_csv(str(FilePath+'labels_'+str(clust_names[i])+'.txt'),
                    sep = '\t')
# get the index of the sequences, which are in the same cluster with the target sequence
tar_clust_names = ['tar_cluster_'+str(i) for i in range(50, 1000, 50)]
for j in range(len(clust_names)):
    labels = locals()[clust_names[j]]
    seq_number = len(labels)
    for i in range(seq_number):
        listind = [x for x in range(seq_number) if labels[x]==labels[0]]
    # write index to files
    listind = np.asarray(listind)
    listind.tofile(str(FilePath+'sequences_'+str(tar_clust_names[j])+'.txt'), sep='\t')



