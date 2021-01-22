'''
Script to load the amino acid sequences and to one-hot-encode sequences and to create file;
'''

import os
import numpy as np
import pandas as pd

### input ###
input_file = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_selection/VDJ_final_data/final_50_selection/Final_49_AA_from_geneious.csv'
### functions ###

def one_hot_encode(seq):
    ''' convert an amino acid sequence into a one hot encoded matrix as nd array
    Parameters:
        seq: amino acid sequence as string of amino acids (length = 115)
    Return:
        X (Lx20 nd array): of one-hot encoded sequence; rows represent positions in the sequences
        and columns each possible amino acid;
    '''

    # create an amino acid directory
    encode_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,\
               'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,\
               'V':17,'W':18,'Y':19}

    l = len(seq)

    # initialize empty nd array
    X = np.zeros((l, 20))

    # iterate through sequence and replace 0 by 1 at specific position in the nd array
    for i, aa in enumerate(seq):
        X[i, encode_dict[aa]] = 1

    return X

def one_hot_decode(X):
    ''' convert a one hot encoded matrix as nd array into an amino acid sequence
    Parameters:
        X (Lx20 nd array): of one-hot encoded sequence; rows represent positions in the sequences
        and columns each possible amino acid;
    Return:
        seq: amino acid sequence as string of amino acids (length = 115)
    '''
    decode_dict = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', \
                   9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', \
                   17: 'V', 18: 'W', 19: 'Y'}
    l = X.shape[0]
    # initialize empty nd array
    s = []

    # iterate through each sequence position and
    for i in range(l):
        s.append(decode_dict[X[i].argmax()])

    # convert list of characters to string
    seq = "".join(s)
    return seq

# Load sequence data
df_seq = pd.read_csv(input_file, usecols=['SampleID', 'Sequence'])

# iterate through df of sequences and safe each one-hot encoded matrix as a file

