# to see how the similartiy filering and the Clustering performed, and how much
# 'overlap" we can see between the two
# methods, we want to compare the similarity filtering (80%, about 300 sequences) and the clustering with the length
# filtered files average linkage (cluster number: 60, 179 sequences) and complete linkage (cluster number: 20,
# 199 sequences);
## EDIT: same is done with the affinity propagation clustering;
# Then we can visualize the overlap analysis in a Venn diagramm;

import pandas as pd
import numpy as np

###### AGGLOMERATIVE CLUSTERING VS. SIMILARITY FILTERING
# Load the data of the cluster labels
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_complete_20 = pd.read_csv(str(file_path + 'Summary/complete_20_filt/complete_labels_20.txt')
                                 , index_col=0, sep='\t', low_memory=True)
labels_average_60 = pd.read_csv(str(file_path + 'Summary/average_60_filt/average_labels_60.txt')
                                , index_col=0, sep='\t', low_memory=True)
data_uniq_length = pd.read_csv(str(file_path + 'data_uniq_length_CDR3.txt'), index_col=0, sep='\t', low_memory=True)

# filter the original file for the sequences in the target cluster
data_complete_20 = data_uniq_length.loc[data_uniq_length['CDR3_AA'].isin(labels_complete_20.CDR3_AA)]
data_average_60 = data_uniq_length.loc[data_uniq_length['CDR3_AA'].isin(labels_average_60.CDR3_AA)]

# save it
data_complete_20.to_csv(str(file_path + 'Summary/complete_20_filt/clust_data_complete_20.txt'),
                        sep = '\t', index=False)
data_average_60.to_csv(str(file_path + 'Summary/average_60_filt/clust_data_average_60.txt'),
                        sep = '\t', index=False)

# count the overlapping data between complete and average linkage;
print(len(data_complete_20))
print(len(data_average_60))
print(len(data_complete_20.loc[data_complete_20['CDR3_AA'].isin(data_average_60.CDR3_AA)]))
#142 sequences (of 199/179 are overlapping)

###### (1) Load in the data from the similarity filtering (80%)
# as only in the datasets 2B, 2C and 3A sequences are present, we only need to load these files;
filt80_data_2B = pd.read_csv('/media/lena/LENOVO/Dokumente/Masterarbeit/data/filtered_files/similarity80_FilesAA/simfilt80DataAA2B.txt',
                      sep='\t')
filt80_data_2C = pd.read_csv('/media/lena/LENOVO/Dokumente/Masterarbeit/data/filtered_files/similarity80_FilesAA/simfilt80DataAA2C.txt',
                      sep='\t')
filt80_data_3A = pd.read_csv('/media/lena/LENOVO/Dokumente/Masterarbeit/data/filtered_files/similarity80_FilesAA/simfilt80DataAA3A.txt',
                      sep='\t')

# add columns for the dataset and boost;
for i in range(len(filt80_data_2B)):
    filt80_data_2B = filt80_data_2B.assign(Dataset=['B']*len(filt80_data_2B), Boost=['2']*len(filt80_data_2B))
for i in range(len(filt80_data_2C)):
    filt80_data_2C = filt80_data_2C.assign(Dataset=['C']*len(filt80_data_2C), Boost=['2']*len(filt80_data_2C))
for i in range(len(filt80_data_3A)):
    filt80_data_3A = filt80_data_3A.assign(Dataset=['A']*len(filt80_data_3A), Boost=['3']*len(filt80_data_3A))
# merge the datasets
filt80_data_all = filt80_data_2B.append(filt80_data_2C)
filt80_data_all = filt80_data_all.append(filt80_data_3A)
# filter for unique sequences
filt80_data_all = filt80_data_all.drop_duplicates(['CDR3_AA'])
# now we only have 29 sequences left!
# save the file for further analysis
filt80_data_all.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/filteredFiles/similarity80_FilesAA/simfilt80_data_all.txt'),
                         sep='\t', index=False)

print(len(data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]))
print(len(data_average_60.loc[data_average_60['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]))
# they both share only 26 sequences in common

# print the number of sequences that have different length compared to the target sequence; just for interest
bools = []
for seq in filt80_data_all.CDR3_AA:
    if len(seq) == 17:
        bools.append('TRUE')
print(len(bools))
# 26 sequences have the same length as the target sequence
# due to the small amount of sequences, see how many sequences remain in the 70% filtering!

###### (2) Load in the data from the similarity filtering (70%)
file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/filtered_files/similarity70_FilesAA/'
files = []
list = ['A', 'B', 'C']
i=0
for i in range(4):
    for j in range(3):
    files.append(str(file_path2 + 'simfilt70DataAA' + str(i) + list[j] + '.txt'))

dataNames = []
for i in range(4):
    for j in range(3):
        dataNames.append(str("filt70_data" + str(i) + list[j]))

for i in range(12):
    locals()[dataNames[i]] = pd.read_csv(files[i], sep='\t', header=None, names=['ReadID', 'Functionality',
                                                'CDR3_AA', 'Table_CDR3', 'VDJ_AA', 'VDJ_NT'], skiprows=1)

# Add Dataset and Boost columns
list_boost = ['0', '1', '2', '3']

for i in range(len(dataNames)):
    if dataNames[i][12] == "A":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Dataset = ['A']*len(locals()[dataNames[i]]))
    elif dataNames[i][12] == "B":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Dataset=['B']*len(locals()[dataNames[i]]))
    elif dataNames[i][12] == "C":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Dataset=['C']*len(locals()[dataNames[i]]))

for i in range(len(dataNames)):
    if dataNames[i][11] == "0":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['0']*len(locals()[dataNames[i]]))
    elif dataNames[i][11] == "1":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['1']*len(locals()[dataNames[i]]))
    elif dataNames[i][11] == "2":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['2']*len(locals()[dataNames[i]]))
    elif dataNames[i][11] == "3":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['3']*len(locals()[dataNames[i]]))

# merge the filtered files
mFiles = ['filt70_data0', 'filt70_data1', 'filt70_data2', 'filt70_data3']
count = 0
for i in range(len(mFiles)):
    fmerge1 = locals()[dataNames[count]]
    fmerge2 = locals()[dataNames[count+1]]
    fmerge3 = locals()[dataNames[count+2]]
    fmerge = fmerge1.append(fmerge2, ignore_index=True)
    fmerge = fmerge.append(fmerge3, ignore_index=True)
    locals()[mFiles[i]] = fmerge
    count = count + 3

# merge the data of all mice
filt70_data_all = filt70_data0.append(filt70_data1, ignore_index=True)
filt70_data_all = filt70_data_all.append(filt70_data2, ignore_index=True)
filt70_data_all = filt70_data_all.append(filt70_data3, ignore_index=True)

# drop the duplicate sequences
filt70_data_all = filt70_data_all.drop_duplicates(['CDR3_AA'])
#data_uniqVDJ = data_all.drop_duplicates(['VDJ_AA'])
#reset the index of the unique entries
filt70_data_all = filt70_data_all.reset_index(drop=True)
# here we have 135 sequences
# save the file
filt70_data_all.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/filteredFiles/similarity70_FilesAA/simfilt70_data_all.txt'),
                         sep='\t', index=False)

print(len(data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
print(len(data_average_60.loc[data_average_60['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
# they both have 37 (complete linkage) and 39 (average linkage) sequences in common
print(len(filt80_data_all.loc[filt80_data_all['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
# just for the validation; all the sequences of 80% sim. filtering appear (as they should) also in the 70% sim. filtering

# save the sequences, that occur in complete, average and simfilt80
overlap_com_sim80 = data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]
overlap1 = overlap_com_sim80.loc[overlap_com_sim80['CDR3_AA'].isin(data_average_60.CDR3_AA)]
overlap1.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_filt_clustering_simfilt80.txt'),
                         sep='\t', index=False)
# 26 sequences occur in all

overlap_com_av = data_average_60.loc[data_average_60['CDR3_AA'].isin(data_complete_20.CDR3_AA)]
# 142 sequences occur in all

# save the sequences, that occur in complete, average and simfilt80
overlap_com_sim70 = data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]
overlap2 = overlap_com_sim70.loc[overlap_com_sim70['CDR3_AA'].isin(data_average_60.CDR3_AA)]
overlap2.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_filt_clustering_simfilt70.txt'),
                         sep='\t', index=False)
# here we have 36 sequences


# print the number of sequences that have different length compared to the target sequence; just for interst;
bools = []
for seq in filt70_data_all.CDR3_AA:
    if len(seq) == 17:
        bools.append('TRUE')
print(len(bools))
# 40 sequences have the same length like the target sequence;


####### Visualization in a Venn plot
import matplotlib_venn as venn
from matplotlib import pyplot as plt
# define the sets (in our case the sequences)
set_av60 = set(data_average_60['CDR3_AA'])
set_com20 = set(data_complete_20['CDR3_AA'])
set_sim80 = set(filt80_data_all['CDR3_AA'])
set_sim70 = set(filt70_data_all['CDR3_AA'])

# create a figure
fig = plt.figure(figsize=(10,6))
plt.subplot(121)
v1 = venn.venn3(subsets=[set_av60, set_com20, set_sim80],
                set_labels=('Clustering, av. linkage', 'Clustering, compl. linkage', 'Similarity 80%'))
for text in v1.set_labels:
    text.set_fontsize(7)
plt.subplot(122)
v2 = venn.venn3(subsets=[set_av60, set_com20, set_sim70], set_labels=('Clustering, av. linkage', 'Clustering, compl. linkage', 'Similarity 70%'))
for text in v2.set_labels:
    text.set_fontsize(7)
fig.suptitle("Venn diagram of the clustering and the similarity filtering", fontsize=16)
#fig.savefig('/media/lena/LENOVO/Dokumente/Masterarbeit/data/filteredFiles/similarity70_FilesAA/Venn_similarity_clustering_comparison.pdf')
fig.show()




################################ AFFINITY MATURATION CLUSTERING VS. SIMILARITY FILTERING
# Load the data of the cluster labels
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
labels_AP = pd.read_csv(str(file_path + 'Affinity_propagation/target_clust_data_AP.txt')
                                 , index_col=0, sep='\t', low_memory=True)
data_uniq_length = pd.read_csv(str(file_path + 'data_uniq_length_CDR3.txt'), index_col=0, sep='\t', low_memory=True)

# filter the original file for the sequences in the target cluster
data_AP = data_uniq_length.loc[data_uniq_length['CDR3_AA'].isin(labels_AP.CDR3_AA)]

# save it
data_AP.to_csv(str(file_path + 'Summary/affinity_propagation/tar_tot_data_AP.txt'),
                        sep = '\t', index=False)

# count the overlapping data between complete and average linkage;
print(len(data_AP))
# 549 sequences in the clustering found;

print(len(data_AP.loc[data_AP['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
print(len(data_AP.loc[data_AP['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]))
# they have very little overlapping sequences! only 15 (70%) and 11 (80%)

overlap_AP_sim70 = data_AP.loc[data_AP['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]
overlap_AP_sim80 = data_AP.loc[data_AP['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]

# save overlapping files
overlap_AP_sim70.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_AP_clustering_simfilt70.txt'),
                         sep='\t', index=False)
overlap_AP_sim80.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_AP_clustering_simfilt80.txt'),
                         sep='\t', index=False)

####### Visualization in a Venn plot
import matplotlib_venn as venn
from matplotlib import pyplot as plt
# define the sets (in our case the sequences)
set_av60 = set(data_average_60['CDR3_AA'])
set_com20 = set(data_complete_20['CDR3_AA'])
set_sim80 = set(filt80_data_all['CDR3_AA'])
set_sim70 = set(filt70_data_all['CDR3_AA'])
set_com_av = set(overlap_com_av['CDR3_AA'])
set_AP = set(data_AP['CDR3_AA'])

# create a figure
fig = plt.figure(figsize=(13,7))

# set up the venn plot
plt.subplot(121)
v1 = venn.venn3(subsets=[set_AP, set_sim80, set_sim70],
                set_labels=('Clustering, affinity propagation', 'Similarity 80%', 'Similarity 70%'))

for text in v1.set_labels:
    text.set_fontsize(7)

plt.subplot(122)
v2 = venn.venn3(subsets=[set_av60, set_com20, set_AP],
                set_labels=('Clustering, av. linkage', 'Clustering, compl. linkage', 'Clustering, affinity propagation'))

for text in v2.set_labels:
    text.set_fontsize(7)

fig.suptitle("Venn diagram of the clustering and the similarity filtering", fontsize=16)
fig.savefig('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Venn_similarity_clustering_AP_comparison.pdf')
fig.show()

# add venn plot with the clustering(complete & average), similarty filtering and the AP clustering
fig = plt.figure(figsize=(7,5))

# set up the venn plot
plt.plot()
v1 = venn.venn3(subsets=[set_AP, set_sim80, set_com_av],
                set_labels=('Clustering, affinity propagation', 'Similarity 80%', 'Clustering, av. & compl. linkage'))

for text in v1.set_labels:
    text.set_fontsize(7)

fig.suptitle('Venn diagram of the clustering and the similarity filtering', fontsize=16)
fig.savefig('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Venn_similarity_clustering_AP_comparison2.pdf')
fig.show()




