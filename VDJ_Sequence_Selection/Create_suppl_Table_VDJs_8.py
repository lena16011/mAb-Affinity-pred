'''
Create the supplementary table for the selected VDJs from the Affinity propagation clustering and similarity filtering.
The table should contain following columns:

VDJ_id  |   Count  --> dep. on in which file |   Frequency --> dep. on in which file  |   CDR1    |   CDR2    |   CDR3    |	SHM	|  [...] kinetic data (to be measured)
'''

import pandas as pd
import numpy as np
import os

# specify input path of the VDJ sequences
abs_path = 'D:/Dokumente/Masterarbeit/Lena/VDJ_Sequence_Selection'
path_VDJs = abs_path + '/data/VDJ_selection/original_data/uniq_VDJs_from_Ann_Table_data_AP_simfilt80.txt'

# specify the input path of the Annotated tables and the output file
path_annot_table = abs_path + '/data/Annotation_tables'

path_final_out = abs_path + '/data/VDJ_selection/VDJ_final_data'
if not os.path.exists(path_final_out):
    os.makedirs(path_final_out)



####### EXTRACTION OF READ COUNTS OF EACH SEQUENCE IN EACH FILE
# as this part runs pretty long we specify the file names only to the 2C and 3A datasets,
# because only in these datasets the VDJs occurr.

# create the file names of the annot. tables as a list
f_names = ['HEL_2_boost_BM_C_Annotated_Table.txt', 'HEL_3_boost_BM_A_Annotated_Table.txt']

# Load in the VDJ sequences
VDJ_sequences = pd.read_csv(path_VDJs, sep='\t')

# initialize 2d np.array with 331 rows (each VDJ) and 2 columns ()
VDJ_counts = np.zeros(shape=(len(VDJ_sequences), len(f_names)))

# read each file line per line
for j, file in enumerate(f_names):
    f_path = '{}/{}'.format(path_annot_table, file)

    #open file
    with open(f_path, 'r') as f_in:
        for line in f_in:
            cols = line.rstrip().split('\t')

            #iterate through the VDJs and count their occurrance in the line
            for i, VDJ in enumerate(VDJ_sequences.VDJ_AA):
                # check if the VDJs are in the column of the current line
                if VDJ == cols[26]:
                    VDJ_counts[i,j] += 1

# VDJ sequences only occurr in dataset 2C and 3A, as expected.

# extract the counts;
Count_2C = VDJ_counts[:, 0]
Count_3A = VDJ_counts[:, 1]

sum_count = np.sum(Count_2C) + np.sum(Count_3A)

Frequency_2C_selected = np.array(Count_2C/sum_count, dtype=np.float32)
Frequency_3A_selected = np.array(Count_3A/sum_count, dtype=np.float32)
# NOTE: divided by the sum of all selected sequences! not specific to the dataset


# set the number of reads occuring in the two datasets (w/o header; looked up from commandline)
all_reads_2C = 1803900 - 1
all_reads_3A = 2352915 - 1

# Frequency over all sequences in each dataset
Frequency_2C_overall = np.array(Count_2C/all_reads_2C, dtype=np.float32)
# Frequency_2C_overall_3A = np.array(Count_2C/all_reads_3A, dtype=np.float32)
Frequency_3A_overall = np.array(Count_3A/all_reads_3A, dtype=np.float32)
# Frequency_3A_overall_3A = np.array(Count_3A/all_reads_3A, dtype=np.float32)


# put the info together in a dataframe and save it as a file
# specify columns for the dataframe
cols = ['ReadID', 'Boost', 'Dataset', 'VDJ_AA','Count_2C', 'Frequency_2C_selected','Frequency_2C_overall',
         'Count_3A', 'Frequency_3A_selected',   'Frequency_3A_overall',
        'SHM_tot_2C', 'SHM_NS_2C', 'SHM_tot_3A', 'SHM_NS_3A', 'VDJ_NT']
data = pd.DataFrame(columns=cols)

# fill in the respective information
for col in cols:
    if col in VDJ_sequences.columns:
        data.loc[:,str(col)] = VDJ_sequences.loc[:,str(col)]

# make an add. list for the counts and frequencies
lst = ['Count_2C', 'Frequency_2C_selected','Frequency_2C_overall', 'Count_3A',
       'Frequency_3A_selected', 'Frequency_3A_overall',]

for col in lst:
    data.loc[:,str(col)] = locals()[col]


# ############################### save intermediate data
# path_intermed_file = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_selection/VDJ_final_data/INTERMEDIATE_DATA.txt'
# data.to_csv(path_intermed_file, sep='\t', index=False)
#
# # Load the saved intermediate data
# data_intermed = pd.read_csv(path_intermed_file, sep='\t', index_col=0)
# #############################################



# how many sequences actually occur in both data sets
[print(x) for x in range(len(Count_2C)) if (data.Count_2C[x] > 0 and data.Count_3A[x] > 0)]
# only 1


# # make a copy of the data and reindex
# data = data_intermed.copy()
# data.reset_index(drop=True, inplace=True)

# add SHM column in the data from the Annot. Table
# the respective columns in the Annot. Table are: 10, 11 (in terminal) so 9, 10 in python
#  read each file line per line
for j, file in enumerate(f_names):
    f_path = '{}/{}'.format(path_annot_table, file)

    #open file
    with open(f_path, 'r') as f_in:
        for line in f_in:
            cols = line.rstrip().split('\t')

            #iterate through the VDJs and extract SHMs
            for i, VDJ in enumerate(VDJ_sequences.VDJ_AA):
                # check if the VDJs are in the column of the current line
                if VDJ == cols[26]:
                    # set j-1 to not account for the header line
                    if file == 'HEL_2_boost_BM_C_Annotated_Table.txt':
                        data.loc[i,'SHM_tot_2C'] = cols[9]
                        data.loc[i,'SHM_NS_2C'] = cols[10]
                    if file == 'HEL_3_boost_BM_A_Annotated_Table.txt':
                        data.loc[i,'SHM_tot_3A'] = cols[9]
                        data.loc[i,'SHM_NS_3A'] = cols[10]

# save the final file
data.to_csv(path_final_out+'/FINAL_VDJ_Suppl_Table.txt', sep='\t', index=False)

