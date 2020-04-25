'''
Script to convert predicted mus back to KD values ===>> NEVER REALLY USED AND UNFINISHED SCRIPT
'''



import pandas as pd
import numpy as np

def unnormalize(n_data, un_data):
    '''
    Function to convert standardized values predicted in Gaussian process back to KD values.
    :param n_data: standardized data to be backconverted
    :param un_data: unnormalized data that was used to standardized
    :return:
    '''
    log_un_data = np.log(un_data)
    log_data = n_data*np.std(log_un_data) + np.mean(log_un_data)
    return np.exp(log_data)



###### SET INPUT DIRECTORIES ######
abs_path = 'D:/Dokumente/Masterarbeit/Lena/GP_implementation'

input_dir1 = abs_path + '/data/input/'
input_dir2 = abs_path + '/data/gen_seqs_muvar/'


input_f_seq1 = input_dir1 + 'CV_matern_data.csv'
input_f_seq2 = input_dir2 + 'pred_seqs_data.csv'


# Load data
pred_data = pd.read_csv(input_f_seq1, index_col=0)
newseq_data = pd.read_csv(input_f_seq2)


norm_KD_cor = newseq_data['mus'].values

predicted = pred_data['KD_pred'].values
measured = pred_data['KD'].values









