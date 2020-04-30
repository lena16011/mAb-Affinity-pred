'''
Script to

1. load and organize the data sets for the subsequent calculation of the Levenshtein distances of
the CDR3 sequences to the binder/target CDR3 sequence. According to this distance the similarity of each
sequence to the binder CDR3 can be calculated (1-distance) and the CDR3 sequences were filtered to a certain
set threshold. The UNIQUE CDR3 sequences that were within this similarity threshold were saved.

2. calculate the distance matrices (Levenshtein, normed) and saved for
    - all unique CDR3 sequences
    - all sequences that have a CDR3 sequence of the same length as the target/binder sequence
'''


import pandas as pd
import numpy as np
import stringdist


##################### FUNCTIONS

def add_cols(data_name):
    for i in range(len(data_name)):
        if data_name[i][5] == "A":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Dataset=['A'] * len(globals()[data_name[i]]))
        elif data_name[i][5] == "B":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Dataset=['B'] * len(globals()[data_name[i]]))
        elif data_name[i][5] == "C":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Dataset=['C'] * len(globals()[data_name[i]]))

        if data_name[i][4] == "0":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Boost=['0'] * len(globals()[data_name[i]]))
        elif data_name[i][4] == "1":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Boost=['1'] * len(globals()[data_name[i]]))
        elif data_name[i][4] == "2":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Boost=['2'] * len(globals()[data_name[i]]))
        elif data_name[i][4] == "3":
            globals()[data_name[i]] = globals()[data_name[i]].assign(Boost=['3'] * len(globals()[data_name[i]]))


def create_var_lst(prefix):
    '''
    Creates a variable list for all the datasets with the given prefix[0-3][A-C]
    :param prefix
    :return:
    '''
    names = []
    for j in range(3):
        names = names + [str(prefix + str(i) + lst[j]) for i in range(4)]
    return names


def filter_dist(dict, similarity):
    '''
    Filter sequences accoding to a defined similarity;

    :param dict: input dictionary
    :param similarity:
    :return:
    '''
    return {k:v for (k,v) in dict.items() if 1-v > similarity}


def merge_datasets(file_lst):
    '''
    Function to merges our datasets to one dataset by appending the files in the order according
    the data names list;
    :param file_lst: list holding the ordered data names that are to be merged
    :return: merged dataset
    '''
    merge = pd.DataFrame()
    for file in file_lst:
        merge = merge.append(globals()[file], ignore_index=True)

    return merge


def calculate_distance_matrix(dataset):
    '''
    Calculate the distance matrix (norm Levenshtein) for a given dataset passed in the arguments
     :param dataset: as a dataseries with the sequences
     distances are stored only as a lower triangular matrix
    '''
    col = range(len(dataset))
    distance_matrix = pd.DataFrame(columns=col)
    data_ser = pd.DataFrame(columns=col)
    for i in range(len(dataset)):
        data_ser = dataset[:i + 1].apply(stringdist.levenshtein_norm,
                                                      args=(dataset[i],))
        distance_matrix.loc[i] = data_ser.T

    return distance_matrix


#####################

############# set the input directories
abs_path = 'D:/Dokumente/Masterarbeit/Lena/VDJ_Sequence_Selection/'
input_dir = abs_path+'data/Filtered_files/'
input_file_seq = abs_path+'data/Filtered_files/sequences.fasta'


#specify the similarity threshold to filter the sequences for
sim_threshold = 0.8



############# set the output directories

# set this boolean to true if the distance matrix should be calculated ONLY for CDR3 sequences
# that have the same length as the target CDR3 sequence
length_filtering = True

# set boolean to True if a file of the SIMILARITY-FILTERED unique sequences should be saved in a file
save_file_bool1 = True
output_dir_filtered = input_dir+'similarity'+str(int(sim_threshold*100))+'_FilesAA/'

# set boolean to True if a file of all NON-FILTERED, unique sequences should be saved in a file
save_file_bool2 = True
output_dir_data = input_dir

# set boolean to True if a file of the distance matrix of the NON-FILTERED sequences should be stored
save_file_bool3 = True
output_dir_matrix = input_dir


# set boolean to True if a file of all LENGTH-FILTERED, unique sequences should be saved in a file
save_file_bool4 = True
output_dir_data2 = input_dir

# set boolean to True if a file of the distance matrix of the LENGTH-FILTERED, unique CDR3 sequences
# should be stored
save_file_bool5 = True
output_dir_matrix2 = input_dir



##############################################################################
## Start of the script



######### LOAD INPUT

# prepare variable names
lst = ['A', 'B', 'C']
files = []
for j in range(3):
    files = files + [str(input_dir+"ess_HEL"+str(i)+"BM"+lst[j]+"Annot_filtered.txt") for i in range(4)]

# create data names
data_names = create_var_lst('data')

# Load all the data files
for i in range(len(data_names)):
    locals()[data_names[i]] = pd.read_csv(files[i], sep='\t', header=None,
                                          names=['ReadID', 'Functionality','CDR3_AA', 'Table_CDR3',
                                                 'VDJ_AA', 'VDJ_NT'], skiprows=1)

# add dataset and boost as columns
add_cols(data_names)


# load in the target sequence
with open(input_file_seq, 'r') as targetFile:
    lines= targetFile.readlines()
    target_CDR3_AA = lines[7].strip()




##### Calculate the Levenshtein distance to the target AA sequence ####
# of EACH DATASET separately;
# save the ReadID and the levenshtein distance as a dictionary for each dataset
lev_names = create_var_lst("lev_dist_AA_")

# Calculate the levenshtein distance of all sequences in all datasets to the target sequence and
# store it as a dict
for j in range(len(data_names)):
    locals()[lev_names[j]] = {}
    for i in range(len(locals()[data_names[j]])):
        lev = stringdist.levenshtein_norm(locals()[data_names[j]].CDR3_AA[i], target_CDR3_AA)
        locals()[lev_names[j]][locals()[data_names[j]].loc[i, 'ReadID']] = lev




########## (I) Similarity filtering

# filter for similarity and store new data sets
filt_lev_data_names = create_var_lst(str("filt_lev_dist_" + str(int(sim_threshold*100))))


for i in range(len(data_names)):
    locals()[filt_lev_data_names[i]] = filter_dist(locals()[lev_names[i]], sim_threshold)

# print number of sequences
for i in range(len(filt_lev_data_names)):
    print(str(filt_lev_data_names[i]), len(locals()[filt_lev_data_names[i]]))
# there is only in the dataset 3A, 2B, 2C sequences that have 80% similarity



# filter the original data for the found similar sequences and create new dataframes
filt_data_names = create_var_lst(str("filt_data_" + str(int(sim_threshold*100)) + "_AA"))

for j in range(len(data_names)):
    locals()[filt_data_names[j]] = pd.DataFrame(columns = data0A.columns)
    for i in range(len(locals()[data_names[j]][['ReadID']])):
        if locals()[data_names[j]].loc[i, 'ReadID'] in locals()[filt_lev_data_names[j]].keys():
            locals()[filt_data_names[j]].loc[i] = locals()[data_names[j]].loc[i]



# merge the filtered datasets to one
filt_data_all = merge_datasets(filt_data_names)

# sort the dataset
filt_data_uniqCDR3 = filt_data_all.sort_values(['Boost', 'Dataset'], ascending=[True, True])

# drop the duplicate CDR3 sequences
filt_data_uniqCDR3 = filt_data_uniqCDR3.drop_duplicates(['CDR3_AA'])

#reset the index of the unique entries
filt_data_uniqCDR3 = filt_data_uniqCDR3.reset_index(drop=True)

#save the filtered unique CDR3 sequences
if save_file_bool1 == True:
    filt_data_uniqCDR3.to_csv(str(output_dir_filtered + 'simfilt_' + str(int(sim_threshold*100)) + '_all.txt'),
                         sep = '\t', index = False)

# see target entry
# our target sequence (binder sequence) has the following entry:
target_entry = filt_data_uniqCDR3[filt_data_uniqCDR3['CDR3_AA']==target_CDR3_AA]
print(target_entry)
# 8      34    productive  ...        A     3




########## (II) Preparations for distance matrix

# merge the datasets before filtering
data_all = merge_datasets(data_names)


# sort the dataset
data_uniqCDR3 = data_all.sort_values(['Boost', 'Dataset'], ascending=[True, True])

# drop the duplicate sequences
data_uniqCDR3 = data_uniqCDR3.drop_duplicates(['CDR3_AA'])

#reset the index of the unique entries
data_uniqCDR3 = data_uniqCDR3.reset_index(drop=True)

# save this file for later
if save_file_bool2 == True:
    data_uniqCDR3.to_csv(str(output_dir_data + 'data_uniqCDR3.txt'), sep='\t')


############################## NOTE:
# here the dataframe Index is connected to the ReadID and thus to the information about which
# read comes from which mouse;
# the index, not the ReadID is used as "identifier" for the calculation of the distance matrices;
##############################


# our target sequence has the following entry:
target_entry = data_uniqCDR3[data_uniqCDR3['CDR3_AA']==target_CDR3_AA]
print(target_entry)
# 17358      34    productive  ...        A     3



###### (1) Calculate the distance matrix for ALL sequences

dist_matrix_CDR3 = calculate_distance_matrix(data_uniqCDR3.CDR3_AA)

# save the distance matrix in a txt file
if save_file_bool3 == True:
    dist_matrix_CDR3.to_csv(str(output_dir_matrix + 'uniqCDR3_dist_matrix.txt'),
                                      sep = '\t', float_format=np.float32, na_rep='0')

##### (2) Calculate the distance matrix for sequences of same CDR3 length
# filter the data set for CDR3 length

if length_filtering == True:
    data_uniq_length = pd.DataFrame(columns=data_uniqCDR3.columns)
    for i in range(len(data_uniqCDR3.CDR3_AA)):
        if len(data_uniqCDR3.CDR3_AA[i]) == len(target_CDR3_AA):
            uniq_length = data_uniqCDR3.iloc[i,]
            data_uniq_length = data_uniq_length.append(uniq_length)

    # sort and reset the index
    data_uniq_length.sort_values(['Boost', 'Dataset'], ascending=[True, True], inplace=True)
    data_uniq_length.reset_index(drop=True, inplace=True)

    # save this file for later
    if save_file_bool4 == True:
        data_uniq_length.to_csv(str(output_dir_data2 + 'data_uniq_length_uniqCDR3.txt'), sep='\t')

    # where in the matrix is the target sequence
    target_entry2 = data_uniq_length[data_uniq_length['CDR3_AA']==target_CDR3_AA]
    print(target_entry2)



    #Calculate a new distance matrix for the new sequences
    dist_matrix_uniq_length = calculate_distance_matrix(data_uniq_length.CDR3_AA)

    # save dataframe for clustering
    if save_file_bool5 == True:
        dist_matrix_uniq_length.to_csv(str(output_dir_matrix2 + 'uniq_length_dist_matrix.txt'), sep='\t')

