
'''

Script to select the VDJ sequences from the Annotated Tabels that occur with our selected CDR3s;
Further analysis of the sequence similarities and the lengths of the sequences

'''


import pandas as pd
import stringdist

######################### SELECTION OF VDJS FROM ANNOTATED TABLES FROM OVERLAP AP & SIMILARITY FILTERING 80%

# Load in the ess_HEL data sets (they were filtered only for unique VDJ nucleotide sequence)
dir_name = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/filtered_files/'

# create file names
list = ['A', 'B', 'C']
files = []
for i in range(3):
    files = files + [str('ess_HEL' + str(x) + 'BM' + list[i] + 'Annot_filtered.txt') for x in range(4)]

# create data names
data_names = []
for i in range(3):
    data_names = data_names + [str('data' + str(x) + list[i]) for x in range(4)]


# create dataframes with the datanames
for i in range(len(data_names)):
    locals()[data_names[i]] = pd.read_csv(str(dir_name + files[i]), sep='\t', header=None,
                                          names=['ReadID', 'Functionality', 'CDR3_AA', 'Table_CDR3', 'VDJ_AA', 'VDJ_NT'],
                                          skiprows=1)

# add cols for the dataset and boost
list_let = ['A', 'B', 'C']


for i in range(len(data_names)):
    for j in list_let:
        if data_names[i][5] == j:
        locals()[data_names[i]] = locals()[data_names[i]].assign(Dataset = [j]*len(locals()[data_names[i]]))


for i in range(len(data_names)):
    for j in range(4):
        if data_names[i][4] == str(j):
        locals()[data_names[i]] = locals()[data_names[i]].assign(Boost = [str(j)]*len(locals()[data_names[i]]))


# merge the A, B, C datasets
m_files = ['data0', 'data1', 'data2', 'data3']
count = 0
for i in range(len(m_files)):
    merge1 = locals()[data_names[count]]
    merge2 = locals()[data_names[count+1]]
    merge3 = locals()[data_names[count+2]]
    merge = merge1.append(merge2, ignore_index=True)
    merge = merge.append(merge3, ignore_index=True)
    locals()[m_files[i]] = merge
    count = count + 3
# merge the data of all mice
data_all = data0.append(data1, ignore_index=True)
data_all = data_all.append(data2, ignore_index=True)
data_all = data_all.append(data3, ignore_index=True)

# save the merged ess_HEL... files
data_all.to_csv(str(dir_name + 'ess_HEL_all_merged.txt'), sep='\t')


# load in the data of the selected CDR3 data to filter for
file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_selection/'
file_name = 'overlap_clustering_av_com.txt'
out_file1 = 'VDJ_data/tot_VDJs_from_Ann_Table_data_av_com.txt'
out_file2 = 'VDJ_data/uniq_VDJs_from_Ann_Table_data_av_com.txt'


data_AP_sim80 = pd.read_csv(str(file_path + file_name), header=0, sep='\t', low_memory=True)


data_filt = pd.DataFrame()
for CDR3 in data_AP_sim80.CDR3_AA:
    data_filt = data_filt.append(data_all[data_all.CDR3_AA == CDR3])

print(len(data_filt))
# now we have 331 sequences

print(len(data_filt.VDJ_AA.unique()))
# 193 unique VDJs

# save the file
data_filt.to_csv(str(file_path + out_file1), sep='\t', index=None)

# drop the duplicate sequences
data_filt_u = data_filt.drop_duplicates(['VDJ_AA'])

# save only the unique VDJs
data_filt_u.to_csv(str(file_path + out_file2), sep='\t', index=None)




######## (1) Calculate a norm. Levenshtein distance matrix on VDJ-level
col=range(len(data_filt_u.VDJ_AA))

dist_matrix_norm = pd.DataFrame(columns=col)
dist_matrix_norm.loc[0] = ['nan']*len(col)
data_ser = pd.DataFrame(columns=col)

# reset index, otherwise looping over it doesn't work
data_filt_u.reset_index(inplace=True, drop=True)

# Calculate the distance matrix (w/o diagonal values same sequence vs. same sequence)
for i in range(len(data_filt_u.VDJ_AA)):
    data_ser = data_filt_u.VDJ_AA[:i].apply(stringdist.levenshtein_norm, args=(data_filt_u.VDJ_AA[i],))
    dist_matrix_norm.loc[i] = data_ser.T

# Calculate the min. similarity btw the sequences
print('min. similarity:', 1 - dist_matrix_norm.max().max())
# min similarty occurring in the distance/similarity matrix: 0.3621

# calculate the mean of the distance values
print('mean similarity:', 1 - dist_matrix_norm.mean(1).mean())
# the similarity of the VDJs is about 0.5973




##### (2) Calculate a abs. Levenshtein distance matrix on VDJ-level

dist_matrix_LD = pd.DataFrame(columns=col)

# Calculate the distance matrix
for i in range(len(data_filt_u.VDJ_AA)):
    data_ser = data_filt_u.VDJ_AA[:i].apply(stringdist.levenshtein,
                                                 args=(data_filt_u.VDJ_AA[i],))
    dist_matrix_LD.loc[i] = data_ser.T


# print the max LD between the sequences
print('max. LD', dist_matrix_LD.max().max())
# the max LD that occurred btw. the sequences is 33

# mean of the LD
print('mean LD:', dist_matrix_LD.mean().mean())
# mean of LD: 7.78

# compare the VDJs in length
VDJ_len = pd.unique([len(x) for x in data_filt_u.VDJ_AA])
print(VDJ_len)
# the sequences have lengths of 105, 108, 114, 115 AAs

# how many sequences are in the data set with the certain length
for length in VDJ_len:
    print(length,  ': ', len([x for x in data_filt_u.VDJ_AA if len(x) == length]))
