import numpy as np
import pandas as pd


### substitution matrix per default loaded from this location; otherwise change in CDRdist() function
# matrix_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/SW_aligner/ProteinNucleotideSequenceAlignment-master/matrices/blosum45_edited.csv'
# dict_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/SW_aligner/ProteinNucleotideSequenceAlignment-master/matrices/AA_dict.csv'



def blosum_distance(seq1, seq2, sub_matrix, AA_dict):
    '''
    Function to calculate a distance value based on BLOSUM45 matrix
    :param seq1, seq2: AA sequences to compare

    return value of distance based on BLOSUM45 substitution matrix

    '''

    # convert aa sequence according to AA_dict to numbers (to access sub_matrix easier)
    seq1_conv = []
    for aa in seq1:
        seq1_conv.append(AA_dict[str(aa)])

    seq2_conv = []
    for aa in seq2:
        seq2_conv.append(AA_dict[str(aa)])

    # initialize scoring value
    val = 0

    # iterate through sequence and get values of subst. matrix to add up to scoring value
    for aa1, aa2 in zip(seq1_conv, seq2_conv):
        val_add = sub_matrix.iloc[int(aa1),int(aa2)]
        val += val_add
    return val


def CDRdist45(seq1, seq2):
    '''
    Function to calculate the CDRdist between 2 input sequences; based on BLOSUM45 substituion matrix;
    !!! Substitution matrix and AA_dict are loaded locally per default !!!

    :param seq1:
    :param seq2:
    :param sub_matrix: substitution matrix (BLOSUM)
    :param AA_dict: dictionary according to BLOSUM dataframe
    :return: CDRdist : distance value between 2 sequences
    '''
    # load substitution matrix and AA_dict
    matrix_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/substitution_matrices/blosum45_edited.csv'
    dict_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/substitution_matrices/AA_dict.csv'

    # read AA dict from csv file
    AA_dict = {}
    with open(dict_infile, mode='r') as f_in:
        for line in f_in:
            parts = line.rstrip().split("\t")
            AA_dict[parts[0]] = parts[1]

    # read BLOSUM45 substitution matrix
    sub_matrix = pd.read_csv(matrix_infile, delimiter=',')

    # calculate the blosum distances
    dist12 = blosum_distance(seq1, seq2, sub_matrix, AA_dict)
    dist11 = blosum_distance(seq1, seq1, sub_matrix, AA_dict)
    dist22 = blosum_distance(seq1, seq1, sub_matrix, AA_dict)

    # calculate the CDRdist
    cdr_dist = 1 - (np.sqrt(dist12 ** 2 / dist11 * dist22))

    return cdr_dist


def CDRdist62(seq1, seq2):
    '''
    Function to calculate the CDRdist between 2 input sequences. based on BLOSUM62 substituion matrix;
    !!! Substitution matrix and AA_dict are loaded locally per default !!!

    :param seq1:
    :param seq2:
    :param sub_matrix: substitution matrix (BLOSUM)
    :param AA_dict: dictionary according to BLOSUM dataframe
    :return: CDRdist : distance value between 2 sequences
    '''
    # load substitution matrix and AA_dict
    matrix_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/substitution_matrices/BLOSUM62_edited.csv'
    dict_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/substitution_matrices/AA_dict.csv'

    # read AA dict from csv file
    AA_dict = {}
    with open(dict_infile, mode='r') as f_in:
        for line in f_in:
            parts = line.rstrip().split("\t")
            AA_dict[parts[0]] = parts[1]

    # read BLOSUM45 substitution matrix
    sub_matrix = pd.read_csv(matrix_infile, delimiter=',')

    # calculate the blosum distances
    dist12 = blosum_distance(seq1, seq2, sub_matrix, AA_dict)
    dist11 = blosum_distance(seq1, seq1, sub_matrix, AA_dict)
    dist22 = blosum_distance(seq1, seq1, sub_matrix, AA_dict)

    # calculate the CDRdist
    cdr_dist = 1 - (np.sqrt(dist12**2/dist11 * dist22))

    return cdr_dist


def CDRdistPAM40(seq1, seq2):
    '''
    Function to calculate the CDRdist between 2 input sequences. based on BLOSUM62 substituion matrix;
    !!! Substitution matrix and AA_dict are loaded locally per default !!!

    :param seq1:
    :param seq2:
    :param sub_matrix: substitution matrix (BLOSUM)
    :param AA_dict: dictionary according to BLOSUM dataframe
    :return: CDRdist : distance value between 2 sequences
    '''
    # load substitution matrix and AA_dict
    matrix_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/substitution_matrices/PAM40_edited.csv'
    dict_infile = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/substitution_matrices/AA_dict.csv'

    # read AA dict from csv file
    AA_dict = {}
    with open(dict_infile, mode='r') as f_in:
        for line in f_in:
            parts = line.rstrip().split("\t")
            AA_dict[parts[0]] = parts[1]

    # read BLOSUM45 substitution matrix
    sub_matrix = pd.read_csv(matrix_infile)

    # calculate the blosum distances
    dist12 = blosum_distance(seq1, seq2, sub_matrix, AA_dict)
    dist11 = blosum_distance(seq1, seq1, sub_matrix, AA_dict)
    dist22 = blosum_distance(seq1, seq1, sub_matrix, AA_dict)

    # calculate the CDRdist
    cdr_dist = 1 - (np.sqrt(dist12 ** 2 / dist11 * dist22))

    return cdr_dist


