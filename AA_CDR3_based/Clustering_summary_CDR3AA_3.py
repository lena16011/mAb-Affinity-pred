import pandas as pd
import numpy as np

############################## NON FILTERED DATA

#  (1) Load the clustering labels (average linkage) in a dataframe
label_data_all = ['labels_cluster_average' + str(i) for i in range(50, 1000, 50)]
cols = ['ReadID']+['Dataset']+['Boost']+['ClusterNr_'+str(i) for i in range(50, 1000, 50)]+['CDR3_AA']
labels_all_average = pd.DataFrame(index=range(27077), columns=cols, dtype=np.int32)
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Clustering_dist_matrix/'
data = pd.DataFrame()
i=3
for file in label_data_all:
    data = pd.read_csv(str(file_path + 'Cluster_labels_2/' + file + '.txt'), index_col=0, sep='\t', low_memory=True,
                       dtype=np.int32)
    labels_all_average[labels_all_average.columns[i]] = data
    i=i+1

# in labels we have a dataframe with all the Sequence indices and the according clusters
# we now want to add a column as the ReadIDs, Dataset and boost
# therefore we read the columns of the file, where all the unique/united sequences are

file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_all_average['ReadID'] = pd.read_csv(str(file_path2+'data_uniqCDR3.txt'), sep = '\t', usecols=['ReadID'])
labels_all_average['Dataset'] = pd.read_csv(str(file_path2+'data_uniqCDR3.txt'), sep = '\t', usecols=['Dataset'])
labels_all_average['Boost'] = pd.read_csv(str(file_path2+'data_uniqCDR3.txt'), sep = '\t', usecols=['Boost'])
labels_all_average['CDR3_AA'] = pd.read_csv(str(file_path2+'data_uniqCDR3.txt'), sep = '\t', usecols=['CDR3_AA'])
# save the file with the clusterlabels
#labels_all_average.to_csv(str(file_path+'labels_summary_all_average.txt'), sep = '\t')

# Filter for the sequences in the cluster of the target sequence
tar_cluster_names_all_av = ['tar_cluster_all_average_'+str(i) for i in range(50, 1000, 50)]

j = range(50, 1000, 50)
for i in range(len(tar_cluster_names_all_av)):
    locals()[tar_cluster_names_all_av[i]] = pd.DataFrame(columns=cols)
    tar_cluster = labels_all_average[labels_all_average[str('ClusterNr_'+str(j[i]))] == labels_all_average.iloc[17358, i+3]]
    locals()[tar_cluster_names_all_av[i]] = tar_cluster
# print number of sequences in the target cluster
print('Nr. of sequences in target cluster, all CDR3s average linkage (from 50-950 clusters):')
for i in range(len(tar_cluster_names_all_av)):
    print(len(locals()[tar_cluster_names_all_av[i]]))

# save certain data of interest;
# tar_cluster_all_average_550.to_csv(str(file_path2 + '/Summary/average_550_nonfilt/average_labels_550.txt'), sep = '\t')


# (2) Load the clustering labels (complete linkage) in a dataframe
label_data_all = ['labels_cluster_complete' + str(i) for i in range(50, 1000, 50)]
cols = ['ReadID']+['Dataset']+['Boost']+['ClusterNr_'+str(i) for i in range(50, 1000, 50)]+['CDR3_AA']
labels_all_complete = pd.DataFrame(index=range(27077), columns=cols, dtype=np.int32)
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Clustering_dist_matrix/'
data = pd.DataFrame()
i=3
for file in label_data_all:
    data = pd.read_csv(str(file_path + 'Cluster_labels_2/' + file + '.txt'), index_col=0, sep='\t', low_memory=True,
                       dtype=np.int32)
    labels_all_complete[labels_all_complete.columns[i]] = data
    i=i+1

# in labels we have a dataframe with all the Sequence indices and the according clusters
# we now want to add a column as the ReadIDs, Dataset and boost
# therefore we read the columns of the file, where all the unique/united sequences are

file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_all_complete['ReadID'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['ReadID'])
labels_all_complete['Dataset'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Dataset'])
labels_all_complete['Boost'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Boost'])
labels_all_complete['CDR3_AA'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['CDR3_AA'])
# save the file with the clusterlabels
#labels_all_complete.to_csv(str(file_path+'labels_summary_all_complete.txt'), sep = '\t')

tar_cluster_names_all_com = ['tar_cluster_all_complete_'+str(i) for i in range(50, 1000, 50)]

j = range(50, 1000, 50)
for i in range(len(tar_cluster_names_all_com)):
    locals()[tar_cluster_names_all_com[i]] = pd.DataFrame(columns=cols)
    tar_cluster = labels_all_complete[labels_all_complete[str('ClusterNr_'+str(j[i]))] == labels_all_complete.iloc[17358, i+3]]
    locals()[tar_cluster_names_all_com[i]] = tar_cluster
# print number of sequences in the target cluster
print('Nr. of sequences in target cluster, all CDR3s complete linkage (from 50-950 clusters):')
for i in range(len(tar_cluster_names_all_com)):
    print(len(locals()[tar_cluster_names_all_com[i]]))

# save certain data of interest;
# tar_cluster_all_complete_100.to_csv(str(file_path2 + 'Summary/complete_100_nonfilt/complete_labels_100.txt'), sep = '\t')


############################################### CDR3 LENGTH FILTERED DATA ## new range (20, 300, 200)

# (1) Load the clustering labels (average linkage) in a dataframe
label_data_ur = ['labels_cluster_average' + str(i) for i in range(20, 300, 20)]
cols = ['ReadID']+['Dataset']+['Boost']+['ClusterNr_'+str(i) for i in range(20, 300, 20)]+['CDR3_AA']
labels_ur_average = pd.DataFrame(index=range(1760), columns=cols, dtype=np.int32)
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Clustering_uniq_length_dist_matrix_20range/'
data = pd.DataFrame()
i=3
for file in label_data_ur:
    data = pd.read_csv(str(file_path + 'Cluster_labels/' + file + '.txt'), index_col=0, sep='\t', low_memory=True,
                       dtype=np.int32)
    labels_ur_average[labels_ur_average.columns[i]] = data
    i=i+1

# in labels we have a dataframe with all the Sequence indices and the according clusters
# we now want to add a column as the ReadIDs, Dataset and boost
# therefore we read the columns of the file, where all the unique/united sequences are

file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_ur_average['ReadID'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['ReadID'])
labels_ur_average['Dataset'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Dataset'])
labels_ur_average['Boost'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Boost'])
labels_ur_average['CDR3_AA'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['CDR3_AA'])
# save a file with the clusterlabels
# labels_ur_average.to_csv(str(file_path+'labels_summary_unlength_20range_average.txt'), sep = '\t')

# Filter for the sequences in the cluster of the target sequence
tar_cluster_names_ur_av = ['tar_cluster_average_'+str(i) for i in range(20, 300, 20)]

j = range(20, 300, 20)
for i in range(len(tar_cluster_names_ur_av)):
    locals()[tar_cluster_names_ur_av[i]] = pd.DataFrame(columns=cols)
    tar_cluster = labels_ur_average[labels_ur_average[str('ClusterNr_'+str(j[i]))] == labels_ur_average.iloc[1069, i+3]]
    locals()[tar_cluster_names_ur_av[i]] = tar_cluster
# print number of sequences in the target cluster
print('Nr. of sequences in target cluster, length filtered average linkage (from 20-280 clusters):')
for i in range(len(tar_cluster_names_ur_av)):
    print(len(locals()[tar_cluster_names_ur_av[i]]))

# save certain data of the cluster of interest
# tar_cluster_average_60.to_csv(str(file_path + 'average_labels_60.txt'), sep = '\t')


# (2) Load the clustering labels (complete linkage) in a dataframe
label_data_ur = ['labels_cluster_complete' + str(i) for i in range(20, 300, 20)]
cols = ['ReadID']+['Dataset']+['Boost']+['ClusterNr_'+str(i) for i in range(20, 300, 20)]+['CDR3_AA']
labels_ur_complete = pd.DataFrame(index=range(1760), columns=cols, dtype=np.int32)
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Clustering_uniq_length_dist_matrix_20range/'
data = pd.DataFrame()
i=3
for file in label_data_ur:
    data = pd.read_csv(str(file_path + 'Cluster_labels/' + file + '.txt'), index_col=0, sep='\t', low_memory=True,
                       dtype=np.int32)
    labels_ur_complete[labels_ur_complete.columns[i]] = data
    i=i+1

# in labels we have a dataframe with all the Sequence indices and the according clusters
# we now want to add a column as the ReadIDs, Dataset and boost
# therefore we read the columns of the file, where all the unique/united sequences are

file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_ur_complete['ReadID'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['ReadID'])
labels_ur_complete['Dataset'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Dataset'])
labels_ur_complete['Boost'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Boost'])
labels_ur_complete['CDR3_AA'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['CDR3_AA'])
# save the file with the clusterlabels
# labels_ur_complete.to_csv(str(file_path+'labels_summary_unlength_20range_complete.txt'), sep = '\t')

tar_cluster_names_ur_com = ['tar_cluster_complete_'+str(i) for i in range(20, 300, 20)]

j = range(20, 300, 20)
for i in range(len(tar_cluster_names_ur_com)):
    locals()[tar_cluster_names_ur_com[i]] = pd.DataFrame(columns=cols)
    tar_cluster = labels_ur_complete[labels_ur_complete[str('ClusterNr_'+str(j[i]))] == labels_ur_complete.iloc[1069, i+3]]
    locals()[tar_cluster_names_ur_com[i]] = tar_cluster
# print number of sequences in the target cluster
print('Nr. of sequences in target cluster,length filtered complete linkage (from 20-280 clusters):')
for i in range(len(tar_cluster_names_ur_com)):
    print(len(locals()[tar_cluster_names_ur_com[i]]))

# save certain data of interest;
# tar_cluster_complete_280.to_csv(str(file_path + 'complete_labels_280.txt'), sep = '\t')

### make a document with the number of sequences written to it
cols = [str(x) for x in range(50,1000,50)] + ['CDR3_len', 'linkage']
sum_data_all = pd.DataFrame(columns=cols)
for i in range(len(tar_cluster_names_all_av)):
    sum_data_all.loc[0, str(cols[i])] = len(locals()[tar_cluster_names_all_av[i]])
sum_data_all['linkage'][0] = 'average'
sum_data_all['CDR3_len'][0] = 'nonfilt'
for i in range(len(tar_cluster_names_all_com)):
    sum_data_all.loc[1, str(cols[i])] = len(locals()[tar_cluster_names_all_com[i]])
sum_data_all['linkage'][1] = 'complete'
sum_data_all['CDR3_len'][1] = 'nonfilt'
# save the file
sum_data_all.to_csv(str(file_path2) + 'Summary/Nr_seq_nonfilt_clustering.txt', sep='\t', index='linkage')

cols_2 = [str(x) for x in range(20,300,20)] + ['CDR3_len', 'linkage']
sum_data_filt = pd.DataFrame(columns=cols_2)
for i in range(len(tar_cluster_names_ur_av)):
    sum_data_filt.loc[0, str(cols_2[i])] = len(locals()[tar_cluster_names_ur_av[i]])
sum_data_filt['linkage'][0] = 'average'
sum_data_filt['CDR3_len'][0] = 'filt'
for i in range(len(tar_cluster_names_ur_com)):
    sum_data_filt.loc[1, str(cols_2[i])] = len(locals()[tar_cluster_names_ur_com[i]])
sum_data_filt['linkage'][1] = 'complete'
sum_data_filt['CDR3_len'][1] = 'filt'
# save the file
sum_data_filt.to_csv(str(file_path2) + 'Summary/Nr_seq_filt_clustering.txt', sep='\t', index='linkage')


############################## AFFINITY PROPAGATION DATA

#  (1) Load the clustering labels in a dataframe
cols = ['ReadID']+['Dataset']+['Boost']+['Cluster_label']+['CDR3_AA']
file_path_AP = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Affinity_propagation/'
labels_AP = pd.DataFrame(columns=cols, dtype=np.int32)
labs = pd.read_csv(str(file_path_AP + 'Cluster_labels/aff_prop_cluster_labels.txt'), index_col=0,
                        sep='\t', dtype=np.int32)
labels_AP.Cluster_label = labs.cluster_nr
target_seq = 'CTRDYYGSNYLAWFAYW'
# in labels we have a dataframe with all the Sequence indices and the according clusters
# we now want to add a column as the ReadIDs, Dataset and boost
# therefore we read the columns of the file, where all the unique/united sequences are

file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_AP['ReadID'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['ReadID'])
labels_AP['Dataset'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Dataset'])
labels_AP['Boost'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['Boost'])
labels_AP['CDR3_AA'] = pd.read_csv(str(file_path2+'data_uniq_length_CDR3.txt'), sep = '\t', usecols=['CDR3_AA'])
# save the file with the clusterlabels
labels_AP.to_csv(str(file_path2+'/Summary/affinity_propagation/labels_and_data_AP.txt'), sep = '\t')

# Filter for the sequences in the cluster of the target sequence by comparing CDR3s with target CDR3
labels_AP[labels_AP.CDR3_AA == target_seq]
#      ReadID Dataset  Boost  Cluster_label            CDR3_AA
#1069      34       A      3            118  CTRDYYGSNYLAWFAYW

# set the target cluster
n_tar_cluster = int(labels_AP.Cluster_label[labels_AP.CDR3_AA == target_seq])

tar_cluster_AP = pd.DataFrame(columns=cols)
tar_cluster_AP = labels_AP[labels_AP['Cluster_label'] == n_tar_cluster]

# print number of sequences in the target cluster  and in the other clusters
print('Nr. of sequences in target cluster, affinity propagation:' +
      str(len(tar_cluster_AP)))
# 549 sequences in the target cluster

tot_n_cluster = np.argmax(np.unique(labels_AP.Cluster_label))

for i in range(tot_n_cluster):
    print('Cluster: ' + str(i) + ' Nr. of sequences ' +
          str(len(labels_AP[labels_AP['Cluster_label']==i])))

# save the data
tar_cluster_AP.to_csv(str(file_path2 + '/Affinity_propagation/target_clust_data_AP.txt'), sep = '\t')
