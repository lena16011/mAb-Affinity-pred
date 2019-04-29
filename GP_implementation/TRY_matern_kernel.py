import os
import numpy as np
import pandas as pd
import stringdist as sd
from sklearn.model_selection import train_test_split
from scipy import optimize, linalg
from GP_implementation import one_hot_encoding as ohc
from sklearn.metrics.pairwise import euclidean_distances
from GP_implementation import GP_fcts as GP





def matern_5_2_kernel(X, X_, hypers=1.0):
    """ Calculate the Matern kernel between X and X_.
    Parameters:
        X (np.ndarray):
        X_ (np.ndarray)
        hypers (iterable): default is ell=1.0.
    Returns:
        K (np.ndarray)
    """
    # D = distance.cdist(X, X_)
    D = euclidean_distances(X, X_)
    D_L = D / hypers[0]

    first = (1.0 + np.sqrt(5.0) * D_L) + 5.0 * D_L ** 2 / 3.0
    second = np.exp(-np.sqrt(5.0) * D_L)

    K = first * second

    return K

def one_hot_encode_matern(seq_input):
    ''' convert an amino acid sequence into a one hot encoded matrix as nd array;
    Note: for Matern kernel
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

    L = len(seq_input[0])
    n = len(seq_input)

    # initialize empty nd array
    X = np.zeros((n, len(encode_dict)*L))

    # iterate through sequence and replace 0 by 1 at specific position in the nd array
    for i, seq in enumerate(seq_input):
        for j, aa in enumerate(seq):
            aa_indx = encode_dict[aa]
            X[i][20 * j + aa_indx] = 1

    return X

def predict_GP_mat(X_train, y_train, X_test, prams):
    """ Gaussian process regression predictions.
    Parameters:
        X_train (np.ndarray): n x d training inputs as one-hot encoded
        y_train (np.ndarray): n training observations
        X_test (np.ndarray): m x d points to predict
    Returns:
        mu (np.ndarray): m predicted means
        var (np.ndarray): m predictive variances
    """
    # Evaluate kernel on training data
    K = matern_5_2_kernel(X_train, X_train, prams[1:])
    # To invert K_y we use the Cholesky decomposition (L)
    L = np.linalg.cholesky(K + np.eye(np.shape(X_train)[0]) * prams[0] ** 2)
    # solve for z=L^-1y
    z = linalg.solve_triangular(L, y_train, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star = matern_5_2_kernel(X_train, X_test, prams[1:])
    mu = np.matmul(K_star.T, alpha)
    # Compute the variance at the test points
    z = linalg.solve_triangular(L, K_star, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star_star = matern_5_2_kernel(X_test, X_test, prams[1:])
    v = np.diag(K_star_star) - np.dot(K_star.T, alpha)
    v = np.diag(v)
    return mu, v

def neg_log_marg_likelihood_mat(log_prams, X, y):
    """ Calculate the negative log marginal likelihood loss.
    We pass the log hypers here because it makes the optimization
    more stable.
    Parameters:
        log_hypers (np.ndarray): natural log of the hyper-parameters
        X (np.ndarray)
        y (np.ndarray)
    Returns:
        (float) The negative log marginal likelihood.
    """

    non_log_prams = np.exp(log_prams)
    # print(non_log_prams)

    # Evaluate kernel on training data
    K = matern_5_2_kernel(X, X, non_log_prams[1:])

    # To invert K we use the Cholesky decomposition (L), because symmetric and positive  definite
    n = len(y)
    L = np.linalg.cholesky(K + np.eye(np.shape(X)[0]) * non_log_prams[0])
    z = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)  # dont know about this

    log_p_y_X = 0.5 * np.matmul(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)


    return log_p_y_X














#### LOAD INPUT ####
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'
input_f_seq = input_dir + 'Final_49_AA_from_geneious.csv'
input_f_KD = input_dir + 'HC_KDvals.csv'

# Load sequence data
df_seq = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence'])

# Load KD values and add them together to a new dataframe (remove samples that w/o measured KD value)
KDs = pd.read_csv(input_f_KD, usecols=['SampleID', 'KD'], sep = ';')
data = pd.merge(df_seq,KDs, on='SampleID')

#### DATA PROCESSING ####

# normalize data
data['KD'] = GP.normalize_test_train_set(data['KD'])

# split into train and test data
X_train, X_test, y_train, y_test = GP.split_data(data, 5, r_state=123)

# one-hot encode sequences
X_train, X_test =

mu, vars = predict_GP_mat(X_train, y_train, X_test, [0.5,1])





