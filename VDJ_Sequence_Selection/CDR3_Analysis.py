import pandas as pd
import stringdist
import numpy as np

'''
Script for calculating some statistics from the final selection of the CDR3 sequences (26).
'''

# calculate a distance matrix
def calculate_norm_dist_matrix(seq_lst):
    '''
    function to calculate the distance matrix for further k-Medoids clustering
    seq_lst is a list of the input sequences for pairwise distance calculation
    :return: is a pandas dataframe of all distances of the n input sequences to each other;
             the lower triangular of the matrix is filled;
    '''
    # initialize numpy 2d array for the matrix
    seqs = pd.Series(np.array(seq_lst))
    D = pd.DataFrame(columns=range(len(seqs)))
    row = pd.DataFrame(columns=range(len(seqs)))
    for i, seq in enumerate(seqs):
        # Calculate the distances as rows of the distance matrix
        row = seqs.iloc[:].apply(stringdist.levenshtein_norm, args=(seqs.loc[i],))
        D.loc[i] = row.T
    return D.values

# calculate a distance matrix
def calculate_LD_dist_matrix(seq_lst):
    '''
    function to calculate the distance matrix for further k-Medoids clustering
    seq_lst is a list of the input sequences for pairwise distance calculation
    :return: is a pandas dataframe of all distances of the n input sequences to each other;
             the lower triangular of the matrix is filled;
    '''
    # initialize numpy 2d array for the matrix
    seqs = pd.Series(np.array(seq_lst))
    D = pd.DataFrame(columns=range(len(seqs)))
    row = pd.DataFrame(columns=range(len(seqs)))
    for i, seq in enumerate(seqs):
        # Calculate the distances as rows of the distance matrix
        row = seqs.iloc[:].apply(stringdist.levenshtein, args=(seqs.loc[i],))
        D.loc[i] = row.T
    return D.values

# prepare the sequence list for the network plots; create 3-tuples; just from another script
# re-used to calculate some statistics
def ebunch_LD(seqs_lst, LD = None):
    '''
        Function to calculate the distance matrix (Levenshtein distance) and then store the
        sequence pairs as 3-tuples with the connected sequences and their corresponding
        Levenshtein distance;

        - seq_lst is a list of the input sequences for pairwise distance calculation
        - LD is the Levenshtein distance of which want the function to return the ebunches
        :return ebunch : is a list of 3-tuples;
    '''

    # initialize numpy 2d array for the matrix
    seqs = pd.Series(np.array(seqs_lst))
    D = pd.DataFrame(columns=range(len(seqs)))
    row = pd.DataFrame(columns=range(len(seqs)))
    for i, seq in enumerate(seqs):
        # Calculate the distances as rows of the distance matrix
        row = seqs.iloc[:].apply(stringdist.levenshtein, args=(seqs.loc[i],))
        D.loc[i] = row.T

    # now we can store the 3-tuples with the distances as weights (ebunch for the network graph)
    data_tup = []
    for i in range(len(seqs_lst)):
        for j in range(i + 1, len(seqs_lst)):
            if LD !=None:
                    if D.iloc[j, i] == LD:
                        val = (i, j, D.iloc[j, i])
                        data_tup.append(val)
            elif LD == None:
                val = (i, j, D.iloc[j, i])
                data_tup.append(val)
    return data_tup

def ebunch_norm(seqs_lst, LD = None):
    '''
        Function to calculate the distance matrix (Levenshtein distance) and then store the
        sequence pairs as 3-tuples with the connected sequences and their corresponding
        Levenshtein distance;

        - seq_lst is a list of the input sequences for pairwise distance calculation
        - LD is the Levenshtein distance of which want the function to return the ebunches
        :return ebunch : is a list of 3-tuples;
    '''

    # initialize numpy 2d array for the matrix
    seqs = pd.Series(np.array(seqs_lst))
    D = pd.DataFrame(columns=range(len(seqs)))
    row = pd.DataFrame(columns=range(len(seqs)))
    for i, seq in enumerate(seqs):
        # Calculate the distances as rows of the distance matrix
        row = seqs.iloc[:].apply(stringdist.levenshtein_norm, args=(seqs.loc[i],))
        D.loc[i] = row.T

    # now we can store the 3-tuples with the distances as weights (ebunch for the network graph)
    data_tup = []
    for i in range(len(seqs_lst)):
        for j in range(i + 1, len(seqs_lst)):
            if LD !=None:
                    if D.iloc[j, i] == LD:
                        val = (i, j, D.iloc[j, i])
                        data_tup.append(val)
            elif LD == None:
                val = (i, j, D.iloc[j, i])
                data_tup.append(val)
    return data_tup



file_in = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_selection/original_data/overlap_AP_clustering_simfilt80.txt'
data_cdr3 = pd.read_csv(file_in, sep='\t')

# get the CDR3s as list
seq_lst = list(data_cdr3.CDR3_AA)

# calculate norm distance matrix
dist_norm = calculate_norm_dist_matrix(seq_lst)

# calculate LD distance matrix
dist_LD = calculate_LD_dist_matrix(seq_lst)


# create ebunches to calculate stats
eb = ebunch(seq_lst)
mean_LD = np.mean([eb[x][2] for x in range(len(eb))])
max_LD = max([eb[x][2] for x in range(len(eb))])

# similarity
eb_norm = ebunch_norm(seq_lst)
mean_sim = 1 - np.mean([eb_norm[x][2] for x in range(len(eb_norm))])
min_sim = min([1-eb_norm[x][2] for x in range(len(eb_norm))])
max_sim = max([1-eb_norm[x][2] for x in range(len(eb_norm))])


# print the stats
print("# selected CDR3s",'\t',len(dist_LD))
print("length of CDR3s",'\t',np.unique([len(x) for x in seq_lst]))

print("range of selected LDs",'\t',np.unique(dist_LD))
print("mean LD",'\t',mean_LD)
print("max LD",'\t',max_LD)

print("mean Similarity",'\t',mean_sim)
print("min Similarity",'\t',min_sim)
print("max Similarity",'\t',max_sim)

# print a sequence list
[print(x) for x in seq_lst]



### repeat with the final VDJs


file_in = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_selection/VDJ_final_data/final_50_selection/final_50_selection_data_115AA.txt'
data = pd.read_csv(file_in, sep='\t')

# get the CDR3s as list
seq_lst = list(data.VDJ_AA)

# calculate norm distance matrix
dist_norm = calculate_norm_dist_matrix(seq_lst)

# calculate LD distance matrix
dist_LD = calculate_LD_dist_matrix(seq_lst)


# create ebunches to calculate stats
eb = ebunch_LD(seq_lst)
mean_LD = np.mean([eb[x][2] for x in range(len(eb))])
max_LD = max([eb[x][2] for x in range(len(eb))])

# similarity
eb_norm = ebunch_norm(seq_lst)
mean_sim = 1 - np.mean([eb_norm[x][2] for x in range(len(eb_norm))])
min_sim = min([1-eb_norm[x][2] for x in range(len(eb_norm))])
max_sim = max([1-eb_norm[x][2] for x in range(len(eb_norm))])


# print the stats
print("# selected VDJss",'\t',len(dist_LD))
print("length of VDJs",'\t',np.unique([len(x) for x in seq_lst]))

print("range of selected LDs",'\t',np.unique(dist_LD))
print("mean LD",'\t',mean_LD)
print("max LD",'\t',max_LD)

print("mean Similarity",'\t',mean_sim)
print("min Similarity",'\t',min_sim)
print("max Similarity",'\t',max_sim)

# print a sequence list
[print(x) for x in seq_lst]


