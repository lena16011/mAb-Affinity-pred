
# Now we have the different subsets of CDR3s that we want;
# in this script the according VDJs shall be analyzed in Network plots


import numpy as np
import pandas as pd

# specify the file path
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_comparison/'

# specify the target sequence
targetCDR3 = 'CTRDYYGSNYLAWFAYW'


###################################### OVERLAP AFFINITY PROPAGATION AND SIMILARITY 80%

# load in the data
data_AP_sim80 = pd.read_csv(str(file_path + 'overlap_AP_clustering_simfilt80.txt'), header=0, sep='\t', low_memory=True)

len(data_AP_sim80)



#### (1) Calculate a norm. Levenshtein distance matrix on VDJ-level
import stringdist

col=range(len(data_AP_sim80.VDJ_AA))

dist_matrix_norm_AP_sim80 = pd.DataFrame(columns=col)
dist_matrix_norm_AP_sim80.loc[0] = ['nan']*len(col)
data_ser = pd.DataFrame(columns=col)

# Calculate the distance matrix (w/o diagonal values same sequence vs. same sequence)
for i in range(len(data_AP_sim80.VDJ_AA)):
    data_ser = data_AP_sim80.VDJ_AA[:i].apply(stringdist.levenshtein_norm,
                                                 args=(data_AP_sim80.VDJ_AA[i],))
    dist_matrix_norm_AP_sim80.loc[i] = data_ser.T

# save dataframe
dist_matrix_norm_AP_sim80.to_csv(str(file_path + 'distance_matrices/dist_matrix_norm_AP_sim80.txt'), sep='\t')

# where in the matrix is the target sequence
target_entry = data_AP_sim80[data_AP_sim80['CDR3_AA']==targetCDR3]
print(target_entry)
#      ReadID Functionality  ...  Dataset Boost
# 5      34    productive  ...        A     3

# Calculate the max. similarity btw the sequences
min_sim = 1 - dist_matrix_norm_AP_sim80.max().max()
print(min_sim)

# calculate the mean of the distance values
mean_sim = 1 - dist_matrix_norm_AP_sim80.mean(1).mean()
print(mean_sim)
# the similarity of the VDJs is about 0.9517




##### (2) Calculate a abs. Levenshtein distance matrix on VDJ-level

dist_matrix_LD_AP_sim80 = pd.DataFrame(columns=col)

# Calculate the distance matrix
for i in range(len(data_AP_sim80.VDJ_AA)):
    data_ser = data_AP_sim80.VDJ_AA[:i].apply(stringdist.levenshtein,
                                                 args=(data_AP_sim80.VDJ_AA[i],))
    dist_matrix_LD_AP_sim80.loc[i] = data_ser.T


# print the max LD between the sequences
print(dist_matrix_LD_AP_sim80.max().max())
# the max LD that occurred btw. the sequences is 10


# mean of the LD
print(dist_matrix_LD_AP_sim80.mean().mean())
# mean of LD: 4.75


# compare the VDJs in length
pd.unique([len(x) for x in data_AP_sim80.VDJ_AA])
# all have the same length of 115 AA




########################### OVERLAP CLUSTERING AVER. LINK. AND COMPL. LINK.


# load in the data
data_av_com = pd.read_csv(str(file_path + 'overlap_clustering_av_com.txt'), header=0, sep='\t', low_memory=True)

len(data_av_com)




#### (1) Calculate a norm. Levenshtein distance matrix on VDJ-level

col=range(len(data_av_com.VDJ_AA))

dist_matrix_norm_av_com = pd.DataFrame(columns=col)
dist_matrix_norm_av_com.loc[0] = ['nan']*len(col)
data_ser = pd.DataFrame(columns=col)

# Calculate the distance matrix (w/o diagonal values same sequence vs. same sequence)
for i in range(len(data_av_com.VDJ_AA)):
    data_ser = data_av_com.VDJ_AA[:i].apply(stringdist.levenshtein_norm,
                                                 args=(data_av_com.VDJ_AA[i],))
    dist_matrix_norm_av_com.loc[i] = data_ser.T

# save dataframe
dist_matrix_norm_av_com.to_csv(str(file_path + 'distance_matrices/dist_matrix_norm_av_com.txt'), sep='\t')

# where in the matrix is the target sequence
target_entry = data_av_com[data_av_com['CDR3_AA']==targetCDR3]
print(target_entry)
#      ReadID Functionality  ...  Dataset Boost
# 84      34    productive  ...        A     3

# Calculate the max. similarity btw the sequences
min_sim = 1 - dist_matrix_norm_av_com.max().max()
print(min_sim)
# min similarty occurring in the distance/similarity matrix: 0.3621

# calculate the mean of the distance values
mean_sim = 1 - dist_matrix_norm_av_com.mean(1).mean()
print(mean_sim)
# the similarity of the VDJs is about 0.5973




##### (2) Calculate a abs. Levenshtein distance matrix on VDJ-level

dist_matrix_LD_av_com = pd.DataFrame(columns=col)

# Calculate the distance matrix
for i in range(len(data_av_com.VDJ_AA)):
    data_ser = data_av_com.VDJ_AA[:i].apply(stringdist.levenshtein,
                                                 args=(data_av_com.VDJ_AA[i],))
    dist_matrix_LD_av_com.loc[i] = data_ser.T


# print the max LD between the sequences
print(dist_matrix_LD_av_com.max().max())
# the max LD that occurred btw. the sequences is 74


# mean of the LD
print(dist_matrix_LD_av_com.mean().mean())
# mean of LD: 44.8

# compare the VDJs in length
pd.unique([len(x) for x in data_av_com.VDJ_AA])
# the sequences have lengths of 115, 114, 117, 113, 116 AAs




########################### SEQUENCES OF THE SIMILARITY FILTERING 80%

## scrip was repeated with the affinity propagation data

# specify input data and change output name at 'save dataframe' and run this part of the scrip
file_name = 'simfilt80_data_all.txt'

# load in the data
data_sim80 = pd.read_csv(str(file_path + file_name), header=0, sep='\t', low_memory=True)

len(data_sim80m)




#### (1) Calculate a norm. Levenshtein distance matrix on VDJ-level

col=range(len(data_av_com.VDJ_AA))

dist_matrix_norm_av_com = pd.DataFrame(columns=col)
dist_matrix_norm_av_com.loc[0] = ['nan']*len(col)
data_ser = pd.DataFrame(columns=col)

# Calculate the distance matrix (w/o diagonal values same sequence vs. same sequence)
for i in range(len(data_av_com.VDJ_AA)):
    data_ser = data_av_com.VDJ_AA[:i].apply(stringdist.levenshtein_norm,
                                                 args=(data_av_com.VDJ_AA[i],))
    dist_matrix_norm_av_com.loc[i] = data_ser.T

# save dataframe
dist_matrix_norm_av_com.to_csv(str(file_path + 'distance_matrices/dist_matrix_norm_AP.txt'), sep='\t')

# where in the matrix is the target sequence
target_entry = data_av_com[data_av_com['CDR3_AA']==targetCDR3]
print(target_entry)
#      ReadID Functionality  ...  Dataset Boost
# 9      34    productive  ...        A     3

# Calculate the max. similarity btw the sequences
min_sim = 1 - dist_matrix_norm_av_com.max().max()
print(min_sim)
# min similarty occurring in the distance/similarity matrix: 0.3621

# calculate the mean of the distance values
mean_sim = 1 - dist_matrix_norm_av_com.mean(1).mean()
print(mean_sim)
# the similarity of the VDJs is about 0.5973




##### (2) Calculate a abs. Levenshtein distance matrix on VDJ-level

dist_matrix_LD_av_com = pd.DataFrame(columns=col)

# Calculate the distance matrix
for i in range(len(data_av_com.VDJ_AA)):
    data_ser = data_av_com.VDJ_AA[:i].apply(stringdist.levenshtein,
                                                 args=(data_av_com.VDJ_AA[i],))
    dist_matrix_LD_av_com.loc[i] = data_ser.T


# print the max LD between the sequences
print(dist_matrix_LD_av_com.max().max())
# the max LD that occurred btw. the sequences is 74


# mean of the LD
print(dist_matrix_LD_av_com.mean().mean())
# mean of LD: 44.8

# compare the VDJs in length
pd.unique([len(x) for x in data_av_com.VDJ_AA])
# the sequences have lengths of 115, 114, 117, 113, 116 AAs




########################### SEQUENCES OF THE AFFINITY PROPAGATION

## scrip was repeated with the affinity propagation data

# specify input data and change output name at 'save dataframe' and run this part of the scrip
file_name = 'tar_clust.txt'

# load in the data
data_sim80 = pd.read_csv(str(file_path + file_name), header=0, sep='\t', low_memory=True)

len(data_sim80m)




#### (1) Calculate a norm. Levenshtein distance matrix on VDJ-level

col=range(len(data_av_com.VDJ_AA))

dist_matrix_norm_av_com = pd.DataFrame(columns=col)
dist_matrix_norm_av_com.loc[0] = ['nan']*len(col)
data_ser = pd.DataFrame(columns=col)

# Calculate the distance matrix (w/o diagonal values same sequence vs. same sequence)
for i in range(len(data_av_com.VDJ_AA)):
    data_ser = data_av_com.VDJ_AA[:i].apply(stringdist.levenshtein_norm,
                                                 args=(data_av_com.VDJ_AA[i],))
    dist_matrix_norm_av_com.loc[i] = data_ser.T

# save dataframe
dist_matrix_norm_av_com.to_csv(str(file_path + 'distance_matrices/dist_matrix_norm_AP.txt'), sep='\t')

# where in the matrix is the target sequence
target_entry = data_av_com[data_av_com['CDR3_AA']==targetCDR3]
print(target_entry)
#      ReadID Functionality  ...  Dataset Boost
# 9      34    productive  ...        A     3

# Calculate the max. similarity btw the sequences
min_sim = 1 - dist_matrix_norm_av_com.max().max()
print(min_sim)
# min similarty occurring in the distance/similarity matrix: 0.3621

# calculate the mean of the distance values
mean_sim = 1 - dist_matrix_norm_av_com.mean(1).mean()
print(mean_sim)
# the similarity of the VDJs is about 0.5973




##### (2) Calculate a abs. Levenshtein distance matrix on VDJ-level

dist_matrix_LD_av_com = pd.DataFrame(columns=col)

# Calculate the distance matrix
for i in range(len(data_av_com.VDJ_AA)):
    data_ser = data_av_com.VDJ_AA[:i].apply(stringdist.levenshtein,
                                                 args=(data_av_com.VDJ_AA[i],))
    dist_matrix_LD_av_com.loc[i] = data_ser.T


# print the max LD between the sequences
print(dist_matrix_LD_av_com.max().max())
# the max LD that occurred btw. the sequences is 74


# mean of the LD
print(dist_matrix_LD_av_com.mean().mean())
# mean of LD: 44.8

# compare the VDJs in length
pd.unique([len(x) for x in data_av_com.VDJ_AA])
# the sequences have lengths of 115, 114, 117, 113, 116 AAs


######### which and how many sequences have such a high LD value (in the ) ??????















