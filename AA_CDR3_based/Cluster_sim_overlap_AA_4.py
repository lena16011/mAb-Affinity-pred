'''
 to see how the similartiy filtering and the Clustering performed, and how much
 'overlap" we can see between the two methods, we want to compare the similarity
 filtering (80%, about 300 sequences) and the clustering with the length
 filtered files average linkage (cluster number: 60, 179 sequences) and complete linkage (cluster number: 20,
 199 sequences);


 EDIT: same is done with the affinity propagation clustering;
 Then we can visualize the overlap analysis in a Venn diagramm;
'''


import pandas as pd
import matplotlib_venn as venn
from matplotlib import pyplot as plt

################## SET INPUT PATHS
# set the input path to the folder that contains the clustering folders (complete_20_filt/ and
# average_60_filt
input_labels_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/'

# set the input path to the folder that contains the clustering labels of the affinity propagation clustering
input_AP = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Affinity_propagation/'

# set the input path of the data of the original unique CDR3 sequences
input_CDR3_uniq = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Filtered_files/data_uniq_length_CDR3.txt'

# set input path for the filtered CDR3 (80% similarity)
input_filt_CDR3_80 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Filtered_files/similarity80_FilesAA/simfilt80_data_all.txt'

# set input path for the filtered CDR3 (70% similarity)
input_filt_CDR3_70 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Filtered_files/similarity70_FilesAA/simfilt70_data_all.txt'




############# SET OUTPUT PATHS
# set boolean to save file of the sequences that appear in the target cluster (complete20, average60)
save_bool1 = False
output_dir1 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/'

# set boolean to save file of the overlapping sequences that appear in the target cluster
# (complete20, average60 and Affinity propagation) and similarity filtering
save_bool2 = False
output_dir2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/'

# set boolean to save venn plots
save_bool_plots = False
output_plots = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Plots/CDR3_Selection/'





#############################################
# HIERARCHICAL CLUSTERING VS. SIMILARITY FILTERING (with the length filtered sequences)
#############################################


# Load the data of the cluster labels
labels_complete_20 = pd.read_csv(str(input_labels_dir + 'complete_20_filt/complete_labels_20.txt')
                                 , index_col=0, sep='\t', low_memory=True)
labels_average_60 = pd.read_csv(str(input_labels_dir + 'average_60_filt/average_labels_60.txt')
                                , index_col=0, sep='\t', low_memory=True)

# Load the file with all unique CDR3 sequences
data_uniq_length = pd.read_csv(input_CDR3_uniq, index_col=0, sep='\t', low_memory=True)

# filter the original file for the sequences in the target cluster
data_complete_20 = data_uniq_length.loc[data_uniq_length['CDR3_AA'].isin(labels_complete_20.CDR3_AA)]
data_average_60 = data_uniq_length.loc[data_uniq_length['CDR3_AA'].isin(labels_average_60.CDR3_AA)]

# save it
if save_bool1 == True:
    data_complete_20.to_csv(str(output_dir1 + 'complete_20_filt/clust_data_complete_20.txt'),
                            sep = '\t', index=False)
    data_average_60.to_csv(str(output_dir1 + 'average_60_filt/clust_data_average_60.txt'),
                           sep = '\t', index=False)


# count the overlapping data between complete and average linkage;
print(len(data_complete_20))
print(len(data_average_60))
print(len(data_complete_20.loc[data_complete_20['CDR3_AA'].isin(data_average_60.CDR3_AA)]))
#142 sequences (of 199/179 are overlapping)



###### (1) Load in the data from the similarity filtering (80%)
filt80_data_all = pd.read_csv(input_filt_CDR3_80, sep='\t')

# print the number of sequences that are in the datasets left
print('# of unique CDR3 sequences found by:')
print('simfilt 80% only:', len(filt80_data_all))
print('simfilt 80% and complete20 clustering linkage:', len(data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]))
print('simfilt 80% and average60 clustering linkage:',len(data_average_60.loc[data_average_60['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]))
# they both share only 26 sequences in common



# print the number of sequences that have different length compared to the target sequence; just out of interest
bools = []
for seq in filt80_data_all.CDR3_AA:
    if len(seq) == 17:
        bools.append('TRUE')
print(len(bools))
# 26 sequences have the same length as the target sequence



############# (2) Load in the data from the similarity filtering (70%)
filt70_data_all = pd.read_csv(input_filt_CDR3_70, sep='\t')

# print the number of sequences that are in the datasets left
print('# of unique CDR3 sequences found by:')
print('simfilt 70% only:', len(filt70_data_all))
print('simfilt 70% and complete20 clustering linkage:', len(data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
print('simfilt 70% and average60 clustering linkage:',len(data_average_60.loc[data_average_60['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
# they both share only 37/39 sequences in common

# print the number of sequences that have different length compared to the target sequence; just out of interest
bools = []
for seq in filt70_data_all.CDR3_AA:
    if len(seq) == 17:
        bools.append('TRUE')
print(len(bools))

print(len(filt80_data_all.loc[filt80_data_all['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
# just for the validation; all the sequences of 80% sim. filtering appear (as they should) also in the 70% sim. filtering



#################### Save the output overlap files of the data

# save the sequences, that occur in complete, average and simfilt80
overlap_com_sim80 = data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]
overlap_com_av_sim80 = overlap_com_sim80.loc[overlap_com_sim80['CDR3_AA'].isin(data_average_60.CDR3_AA)]

if save_bool2 == True:
    overlap_com_av_sim80.to_csv(str(output_dir2 + 'overlap_filt_clustering_simfilt80.txt'),
                                sep='\t', index=False)
# 26 sequences occur in all

# save the sequences, that occur in complete, average
overlap_com_av = data_average_60.loc[data_average_60['CDR3_AA'].isin(data_complete_20.CDR3_AA)]

if save_bool2 == True:
    overlap_com_av.to_csv(str(output_dir2 + 'overlap_clustering_av_com.txt'),
                                sep='\t', index=False)
# 142 sequences occur in all


# save the sequences, that occur in complete, average and simfilt70
overlap_com_sim70 = data_complete_20.loc[data_complete_20['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]
overlap_com_av_sim70 = overlap_com_sim70.loc[overlap_com_sim70['CDR3_AA'].isin(data_average_60.CDR3_AA)]

if save_bool2 == True:
    overlap_com_av_sim70.to_csv(str(output_dir2 + 'overlap_clustering_simfilt70.txt') ,sep='\t', index=False)
# here we have 36 sequences







#######################################CHECK IF THE SEQUENCES HAVE SAME LENGTH OR NOT!!!!!
########################################### how does the isin function searches for????

# print the number of sequences that have different length compared to the target sequence; just for interst;
bools = []
for seq in filt70_data_all.CDR3_AA:
    if len(seq) == 17:
        bools.append('TRUE')
print(len(bools))
# 40 sequences have the same length like the target sequence;



################################### Visualization in a Venn plot

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


if save_bool_plots == True:
    fig.savefig(str(output_plots +'Venn_similarity_clustering_comparison.pdf'))

fig.show()





##################CHECK furhter here!!!!!!!!!!!!

################################ AFFINITY PROPAGATION CLUSTERING VS. SIMILARITY FILTERING
# Load the data of the cluster labels
labels_AP = pd.read_csv(str(input_AP + 'target_clust_data_AP.txt'), index_col=0, sep='\t',
                        low_memory=True)


# filter the original file for the sequences in the target cluster
data_AP = data_uniq_length.loc[data_uniq_length['CDR3_AA'].isin(labels_AP.CDR3_AA)]

# save it
if save_bool2 == True:
    data_AP.to_csv(str(output_dir2 + 'affinity_propagation/tar_clust_data_AP.txt'),
                        sep = '\t', index=False)

# count the sequences found in AP
print(len(data_AP))

####???#####
# 549 sequences in the clustering found;

print(len(data_AP.loc[data_AP['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]))
print(len(data_AP.loc[data_AP['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]))

####  ??????  #######
# they have very little overlapping sequences! only 15 (70%) and 11 (80%)

overlap_AP_sim70 = data_AP.loc[data_AP['CDR3_AA'].isin(filt70_data_all.CDR3_AA)]
overlap_AP_sim80 = data_AP.loc[data_AP['CDR3_AA'].isin(filt80_data_all.CDR3_AA)]

# save overlapping files
overlap_AP_sim70.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_AP_clustering_simfilt70.txt'),
                         sep='\t', index=False)
overlap_AP_sim80.to_csv(str('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Summary/overlap_AP_clustering_simfilt80.txt'),
                         sep='\t', index=False)










#################### Visualization in a Venn plot

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
# fig.savefig('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Venn_similarity_clustering_AP_comparison.pdf')
fig.show()



# add venn plot with the clustering(complete & average), similarty filtering and the AP clustering
fig = plt.figure(figsize=(7,5))

# set up the venn subplot
plt.subplot(121)
v1 = venn.venn3(subsets=[set_AP, set_sim80, set_com_av],
                set_labels=('Clustering, affinity propagation', 'Similarity 80%', 'Clustering, av. & compl. linkage'))

for text in v1.set_labels:
    text.set_fontsize(7)

# set up venn subplot
plt.subplot(122)
v2 = venn.venn2(subsets=[set_AP, set_sim80],
                set_labels=('Clustering, affinity propagation', 'Similarity 80%'))

for text in v2.set_labels:
    text.set_fontsize(7)

fig.suptitle('Venn diagram of the clustering and the similarity filtering', fontsize=16)
# fig.savefig('/media/lena/LENOVO/Dokumente/Masterarbeit/data/Venn_similarity_clustering_AP_comparison2.pdf')
fig.show()




