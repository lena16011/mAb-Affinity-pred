
import pandas as pd
import numpy as np
import stringdist

# calculate a distance matrix
def calculate_norm_dist_matrix(seq_lst, verbose=False):
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
        if verbose == True and i % 10 == 0:
            print(i+" seqs processed")

    return D.values

# calculate a distance matrix
def calculate_LD_dist_matrix(seq_lst, verbose=False):
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
        if verbose == True and i % 10 == 0:
            print(i+" seqs processed")

    return D.values

# prepare the sequence list for the network plots; create 3-tuples; just from another script
# re-used to calculate some statistics
def ebunch_LD(seqs_lst, LD = None):
    '''
        Function to calculate the distance matrix (Levenshtein distance) and then store the
        sequence pairs as 3-tuples with the connected sequences and their corresponding
        Levenshtein distance;

        - seq_lst is a list of the input sequences for pairwise distance calculation
        - LD (Levenshtein distance); if not none, function returns only the ebunches of the defined LD
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
        - LD (Levenshtein distance) can be set, if not none, function returns only the ebunches of the defined LD
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
                        val = (i, j, 1-D.iloc[j, i])
                        data_tup.append(val)
            elif LD == None:
                val = (i, j, 1-D.iloc[j, i])
                data_tup.append(val)
    return data_tup
