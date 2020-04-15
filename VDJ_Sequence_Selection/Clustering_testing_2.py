'''
This script is just made to test and elaborate the hierarchical clustering script that was
then run on the Euler cluster
'''


import pandas as pd
import numpy as np


## 1. Load the WHOLE distance matrix
#dir_name = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/uniqCDR3_DistMatrix.txt'
#dist_matrix_CDR3 = pd.read_csv(dir_name, sep='\t', dtype=np.float16, index_col=0,
#                                    low_memory=True)

# take a subset of the distance matrix (500 dist measures)
# try_dist = dist_matrix_CDR3_load.iloc[:100,:100]
# Make the matrix symmetric
#dist_matrix_CDR3_load = dist_matrix_CDR3_load.as_matrix()
#dist_matrix_CDR3_load = np.tril(dist_matrix_CDR3_load) + np.triu(dist_matrix_CDR3_load).T

# 2. OR Read a part of the distance matrix (entries of 1000 sequence)
file = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Clustering_dist_matrix/'
try_dist = pd.read_csv(str(file+'uniqCDR3_DistMatrix.txt'), nrows= 1001, index_col=0,
                      usecols=range(1002), sep='\t', dtype=np.float16, low_memory=True)

## 3. OR read in a part of the distance matrix and include the target sequence (with the entry in row 17358
# cols = range(17359,17460)
# try_dist = pd.read_csv(file, nrows= 101, skiprows= 17358, usecols=cols, sep='\t', dtype=np.float16, low_memory=True)
# # target sequence is now in the first row/column
# # set the indices
# idx = [i for i in range(17358, 17459)]
# try_dist.index = idx
# try_dist.columns = idx

# make a symmetric matrix
try_dist = try_dist.fillna(value='0').as_matrix()
try_dist = np.tril(try_dist) + np.tril(try_dist).T

### (1) Agglomerative Clustering (WITH FIXED CLUSTER NUMBERS)
# import sklearn.cluster as clust
# n_clust = [10, 20, 30, 40, 50]
# clust_names = ['cluster_10', 'cluster_20', 'cluster_30', 'cluster_40', 'cluster_50']
#
# for i in range(len(n_clust)):
#     model = clust.AgglomerativeClustering(n_clusters=n_clust[i], affinity='precomputed',
#                                           linkage='average')
#     clust_fit = model.fit(try_dist)
#     cluster_labels = clust_fit.fit_predict(try_dist)
#     locals()[clust_names[i]] = cluster_labels
#
# # write the clusterlabels to files
# FilePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
# for i in range(len(clust_names)):
#     locals()[clust_names[i]].tofile(str(FilePath+'Cluster_labels/'+clust_names[i]+'.txt'),
#                                     sep='\t')
#
# # get the sequences, which are in the same cluster like the target sequence
# tar_clust_names = ['tar_clust10_ind', 'tar_clust20_ind', 'tar_clust30_ind', 'tar_clust40_ind',
#                    'tar_clust50_ind']
# for j in range(len(tar_clust_names)):
#     labels = locals()[clust_names[j]]
#     seq_number = len(labels)
#     for i in range(seq_number):
#         listind = [x for x in range(seq_number) if labels[x]==labels[0]]
#     locals()[tar_clust_names[j]] = listind
#     # write index to files
#     listind = np.asarray(listind)
#     listind.tofile(str(FilePath+"sequences_cluster"+str(j+1)+'0.txt'), sep='\t')

# now we have sequences in cluster 2 (target cluster)
# listind = [x for x in range(len(tar_clust10)) if tar_clust10[x] =='True']
#listind = [x+17538 for x in listind]
#listind holds the index numbers, of the sequences that are in same cluster


#### (2) Agglomerative Clustering (WITH RANGE OF CLUSTERING)
import sklearn.cluster as clust
n_clust = range(20, 100, 20)
clust_names = ['cluster_'+str(i) for i in range(20, 100, 20)]
j=0
for i in n_clust:
   model = clust.AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                         linkage='average')
    clust_fit = model.fit(try_dist)
    cluster_labels = clust_fit.fit_predict(try_dist)
    locals()[clust_names[j]] = cluster_labels
    j = j + 1

# write the cluster labels to files
for i in range(len(clust_names)):
    filesave = pd.DataFrame(locals()[clust_names[i]])
    filesave.to_csv(str(FilePath+'/Cluster_labels/labels_'+str(clust_names[i])+'.txt'),
                                    sep = '\t')

# get the index of the sequences, which are in the same cluster with the target sequence
tar_clust_names = ['tar_cluster_'+str(i) for i in range(20, 100, 20)]
for j in range(len(clust_names)):
    labels = locals()[clust_names[j]]
    seq_number = len(labels)
    for i in range(seq_number):
        listind = [x for x in range(seq_number) if labels[x]==labels[0]] # WRONG!! target sequence is
    locals()[tar_clust_names[j]] = listind                               # not in entry [0]!!
    # write index to files
    listind = np.asarray(listind)
    listind.tofile(str(FilePath+"/Cluster_labels/sequences_"+str(tar_clust_names[j])+'.txt'), sep='\t')

# # implement silhouette plots to evaluate the chosen number of cluster
# from sklearn.metrics import silhouette_samples, silhouette_score
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
#
# range_n_cluster = n_clust
# silhouette_s_names = ['silhouette_s_20', 'silhouette_s_30', 'silhouette_s_40', 'silhouette_s_50']
# silhouette_a_names = ['silhouette_avg_20', 'silhouette_avg_30', 'silhouette_avg_40', 'silhouette_avg_50']
# # compute silhouette average and silhouette score for each sample
# for n_clusters in range_n_cluster:
#     model = clust.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
#                                           linkage='average')
#     clust_fit = model.fit(try_dist)
#     cluster_labels = clust_fit.fit_predict(try_dist)
#     silhouette_avg= silhouette_score(try_dist, cluster_labels)
#     sample_silhouette_values = silhouette_samples(try_dist, cluster_labels)
#
#     # set up the plot
#     plt.plot
#     # x axis is the silhouette score that ranges between -1 and 1;
#     #plt.xlim([-0.2, 0.5])
#     #plt.ylim([0, len(try_dist) + (n_clusters + 1) * 10])
#
#     y_lower = 10
#
#     # get scores for each sample belonging to cluster i and sort the values
#     for i in range(n_clusters):
#         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#         #sort the cluster values
#         ith_cluster_silhouette_values.sort()
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#         color = cm.nipy_spectral(float(i) / n_clusters)
#
#         plt.fill_betweenx(np.arange(y_lower, y_upper),
#                       0, ith_cluster_silhouette_values,
#                       facecolor=color, edgecolor=color, alpha=0.7)
#         # label plot with cluster number
#         plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#         #Compute the new y_lower for next plot
#         y_lower = y_upper + 2  # 10 for the 0 samples
#
#     # add line for the average silhouette score
#     plt.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     plt.show()

##### (2) Agglomerative Clustering (WITH RANGE OF CLUSTERING); and loop over ALL linkage methods

###  Clustering
import sklearn.cluster as clust
clust_names = []
for linkage in ('average', 'complete'):
    n_clust = range(50, 1000, 50)
    names = ['cluster_' + str(linkage) + str(i) for i in range(50, 1000, 50)]
    clust_names.extend(names)
    j=0
    for i in n_clust:
        model = clust.AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                              linkage=linkage)
        clust_fit = model.fit(dist_matrix_CDR3)
        cluster_labels = clust_fit.fit_predict(dist_matrix_CDR3)
        locals()[clust_names[i+j]] = cluster_labels
        # write the cluster labels to files
        cluster_labels = pd.DataFrame(cluster_labels)
        cluster_labels.to_csv(str(file_path + 'labels_' + str(clust_names[j]) + '.txt'), sep='\t')
        j = j + 1

    # get the index of the sequences, which are in the same cluster with the target sequence
    tar_clust_names = ['tar_cluster_' + str(linkage) + str(i) for i in range(50, 1000, 50)]
    for j in range(len(clust_names)):
        labels = locals()[clust_names[j]]
        seq_number = len(labels)
        for i in range(seq_number):
            listind = [x for x in range(seq_number) if labels[x]==labels[17358]]
        # write index to files
        listind = np.asarray(listind)
        listind.tofile(str(file_path+'sequences_'+str(tar_clust_names[j])+'.txt'), sep='\t')