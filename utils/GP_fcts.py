import numpy as np
import pandas as pd
import stringdist as sd
from sklearn.model_selection import train_test_split, KFold
from scipy import optimize, linalg
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from utils import CDRdist_fcts as cd
from sklearn.metrics.pairwise import euclidean_distances


####### DATA PROCESSING ########

def split_data(df, n, r_state):
    '''
    Convert a pd.DataFrame of the sequences and KD values into test (X_test, y_true)
    and traininig data (X_train, y_train) by randomly sampling n samples.

    :param df: (pd.dataframe) with the sequence and and KD information of the whole dataset
    :param n: (int) #samples that are put into the test set
    :param r_state (int) seed used by random number generator
    :return: (np.array) X_train, y_train, X_test, y_true
    '''
    X = df.Sequence.values
    y = df.KD.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n, random_state=r_state)
    return X_train, X_test, y_train, y_test

def normalize_test_train_set(*args):
    '''
     normalize one or two datasets (train and test set)

     :param: y1, y2 (np.array) having assigned the unnormalized KD values
     :return: (np.array) one or two normalized arrays of KD values
    '''

    if len([*args]) == 1:
        log_data = np.log([*args][0])
        norm = (log_data - np.mean(log_data)) / np.std(log_data)
        return norm

    else:
        data_ls_n = []
        for kds in args:
            log_data = np.log(kds)
            norm = (log_data - np.mean(log_data)) / np.std(log_data)
            data_ls_n.append(norm)
        return data_ls_n[0], data_ls_n[1]

def unnormalize(n_data, un_data):
    unnorm = np.array
    unlog_data = np.log(un_data)
    log_data = n_data*np.std(unlog_data) + np.mean(unlog_data)
    return log_data

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





######## KERNEL IMPLEMENTATION ###########

def LD_kernel(in_seq1, in_seq2 = False):
    '''
    Calculate the similarity (1-Levenshtein distance) for a given sequence dataset; if sequence = a dataframe,
    the function calculates similarity of the sequence to all sequences.
     :param
     seq1: (pd.DataSeries) nxn dataframe of sequences (X_train)
     seq2: (str) sequence that we want to calculate the similarity to (X_test)

     :return:
     (pd.DataFrame) distances btw. all sequences n x n matrix
     (pd.DataFrame) distances btw. in_seq1 and in_seq2 sequences n x m matrix
     (pd.Series) data series of the similarities to the the target sequence 1 x n
    '''
    # distance from sequences within
    if isinstance(in_seq2, (np.bool)):
        seq1 = pd.Series(in_seq1)
        n = range(len(seq1))
        k=pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        for i in range(len(seq1)):
            data_ser = 1-seq1[0:].apply(sd.levenshtein_norm, args=(seq1[i],))
            k.loc[i] = data_ser

        ker = k

    # if seq2 is only one sequence
    elif type(in_seq2) == str:
        seq1 = pd.Series(in_seq1)
        ker=pd.Series()
        for i in range(len(seq1)):
            ker = 1-seq1[0:].apply(sd.levenshtein_norm, args=(in_seq2,))
    else:
        seq1 = pd.Series(in_seq1)
        seq2 = pd.Series(in_seq2)
        n = range(len(seq1))
        k = pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        for i in range(len(seq2)):
            data_ser = 1 - seq1[0:].apply(sd.levenshtein_norm, args=(seq2[i],))
            k.loc[i] = data_ser

        ker = k.transpose()

    return ker

def matern_5_2_kernel(X, X_, hypers):
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

def CDRdist_kernel45(in_seq1, in_seq2 = False):
    '''
    Calculate the similarity based on CDRdist for a given sequence dataset; if sequence = a dataframe,
    the function calculates similarity of the sequence to all sequences.
     :param
     seq1: (pd.DataSeries) nxn dataframe of sequences (X_train)
     seq2: (str) sequence that we want to calculate the similarity to (X_test)

     :return:
     (pd.DataFrame) distances btw. all sequences n x n matrix
     (pd.DataFrame) distances btw. in_seq1 and in_seq2 sequences n x m matrix
     (pd.Series) data series of the similarities to the the target sequence 1 x n

    '''

    if isinstance(in_seq2, (np.bool)):
        seq1 = pd.Series(in_seq1)
        n = range(len(seq1))
        ker=pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        k = pd.DataFrame(columns=n)
        for i in range(len(seq1)):
            data_ser = 1-seq1[0:].apply(cd.CDRdist45, args=(seq1[i],))
            k.loc[i] = data_ser

        ker = k


        # if seq2 is only one sequence
    elif type(in_seq2) == str:
        seq1 = pd.Series(in_seq1)
        ker = pd.Series()
        for i in range(len(seq1)):
            ker = 1 - seq1[0:].apply(cd.CDRdist45, args=(in_seq2,))
    else:
        seq1 = pd.Series(in_seq1)
        seq2 = pd.Series(in_seq2)
        n = range(len(seq1))
        k = pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        for i in range(len(seq2)):
            data_ser = 1 - seq1[0:].apply(cd.CDRdist45, args=(seq2[i],))
            k.loc[i] = data_ser

        ker = k.transpose()

    return ker

def CDRdist_kernel62(in_seq1, in_seq2 = False):
    '''
    Calculate the similarity based on CDRdist for a given sequence dataset; if sequence = a dataframe,
    the function calculates similarity of the sequence to all sequences.
     :param
     seq1: (pd.DataSeries) nxn dataframe of sequences (X_train)
     seq2: (str) sequence that we want to calculate the similarity to (X_test)

     :return:
     (pd.DataFrame) distances btw. all sequences n x n matrix
     (pd.DataFrame) distances btw. in_seq1 and in_seq2 sequences n x m matrix
     (pd.Series) data series of the similarities to the the target sequence 1 x n

    '''

    if isinstance(in_seq2, (np.bool)):
        seq1 = pd.Series(in_seq1)
        n = range(len(seq1))
        ker=pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        k = pd.DataFrame(columns=n)
        for i in range(len(seq1)):
            data_ser = 1-seq1[0:].apply(cd.CDRdist62, args=(seq1[i],))
            k.loc[i] = data_ser

        ker = k

        # if seq2 is only one sequence
    elif type(in_seq2) == str:
        seq1 = pd.Series(in_seq1)
        ker = pd.Series()
        for i in range(len(seq1)):
            ker = 1 - seq1[0:].apply(cd.CDRdist62, args=(in_seq2,))

    else:
        seq1 = pd.Series(in_seq1)
        seq2 = pd.Series(in_seq2)
        n = range(len(seq1))
        k = pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        for i in range(len(seq2)):
            data_ser = 1 - seq1[0:].apply(cd.CDRdist62, args=(seq2[i],))
            k.loc[i] = data_ser

        ker = k.transpose()

    return ker

def CDRdist_kernelPAM40(in_seq1, in_seq2 = False):
    '''
    Calculate the similarity based on CDRdist for a given sequence dataset; if sequence = a dataframe,
    the function calculates similarity of the sequence to all sequences.
     :param
     seq1: (pd.DataSeries) nxn dataframe of sequences (X_train)
     seq2: (str) sequence that we want to calculate the similarity to (X_test)

     :return:
     (pd.DataFrame) distances btw. all sequences n x n matrix
     (pd.DataFrame) distances btw. in_seq1 and in_seq2 sequences n x m matrix
     (pd.Series) data series of the similarities to the the target sequence 1 x n

    '''

    if isinstance(in_seq2, (np.bool)):
        seq1 = pd.Series(in_seq1)
        n = range(len(seq1))
        ker=pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        k = pd.DataFrame(columns=n)
        for i in range(len(seq1)):
            data_ser = 1-seq1[0:].apply(cd.CDRdistPAM40, args=(seq1[i],))
            k.loc[i] = data_ser

        ker = k


        # if seq2 is only one sequence
    elif type(in_seq2) == str:
        seq1 = pd.Series(in_seq1)
        ker = pd.Series()
        for i in range(len(seq1)):
            ker = 1 - seq1[0:].apply(cd.CDRdistPAM40, args=(in_seq2,))
    else:
        seq1 = pd.Series(in_seq1)
        seq2 = pd.Series(in_seq2)
        n = range(len(seq1))
        k = pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        for i in range(len(seq2)):
            data_ser = 1 - seq1[0:].apply(cd.CDRdistPAM40, args=(seq2[i],))
            k.loc[i] = data_ser

        ker = k.transpose()

    return ker




######## GAUSSIAN PROCESS LD #######

def predict_GP(X_train, y_train , X_test, param):
    """ Gaussian process regression predictions.
    Parameters:
        X_train (np.ndarray): n x d training inputs
        y_train (np.ndarray): n training observations
        X_test (np.ndarray): m x d points to predict
        param (float): noise hyperparameter (determined by maximizing the log marginal likelihood
    Returns:
        mu (np.ndarray): m predicted means
        var (np.ndarray): m predictive variances
    """

    # Evaluate kernel on training data
    K = LD_kernel(X_train)

    # To invert K_y we use the Cholesky decomposition (L)
    L = np.linalg.cholesky(K + np.eye(np.shape(X_train)[0]) * param ** 2)

    # solve for z=L^-1y
    z = linalg.solve_triangular(L, y_train, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star = LD_kernel(X_train, X_test)
    mu = np.matmul(K_star.T, alpha)

    # Compute the variance at the test points
    z = linalg.solve_triangular(L, K_star, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star_star = LD_kernel(X_test)
    v = np.diag(K_star_star) - np.dot(K_star.T, alpha)
    v = np.diag(v)

    return mu, v

def neg_log_marg_likelihood(log_pram, X, y):
    """ Calculate the negative log marginal likelihood loss. This
    will be minimized (= maximizing the log marginal likelihood)
    to determine the hyperparameter;
    We pass the log hypers here because it makes the optimization
    more stable.
    Parameters:
        log_hypers (np.ndarray): natural log of the hyper-parameters
        X (np.ndarray)
        y (np.ndarray)
    Returns:
        (float) The negative log marginal likelihood.
    """

    non_log_pram = np.exp(log_pram)
    # print(non_log_prams)

    # Evaluate kernel on training data
    K = LD_kernel(X)

    # To invert K we use the Cholesky decomposition (L), because symmetric and positive  definite
    n = len(y)
    L = np.linalg.cholesky(K + np.eye(np.shape(X)[0])*non_log_pram)
    z = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)  # dont know about this

    log_p_y_X = 0.5 * np.matmul(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

    return log_p_y_X

def get_params(X_train, y_train, init_param = [0.1]):
    '''
    Optimize the neg. log likelihood to get the optimal hyperparameter
    :param X_train:
    :param y_train:
    :param init_param: initial parameter to optimize
    :return: opt_param
    '''

    initial = init_param
    # take the log of the initial guess for optimiziation
    initial_guess_log = np.log(initial)

    # optimize to fit model by minimizing the neg. log likelihood
    result = optimize.minimize(neg_log_marg_likelihood, initial_guess_log,
                               args=(X_train, y_train), method='L-BFGS-B')

    # next set of hyper prams
    opt_param = np.exp(result.x[0]) ** 2

    return opt_param

def cv_param_tuning(X, y, k, init_param = [0.1]):
    """
    Cross validation framework to find the optimal noise hyperparameter sigma^2 by
    minimizing the neg log likelihood.

    :param
        X (ndarray) of the sequences: should already be X_train data split (from
            outer CV for the model accuracy)
        y (ndarray) of normalized, measured values (KD); already split y_train data
            (from outer CV for the model accuracy)
        k (int) number of k for k-folds; number of splits

    :return

    """
    kf = KFold(n_splits=k) # Define the n_split = number of folds

    # initialize lists to store the
    mu_s = []
    var_s = []
    y_s = []
    params_test = []
    cycle_num = 1

    # loop for the CV
    for train_index, test_index in kf.split(X):

        #split in train and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        initial_guess = init_param

        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)

        # optimize to fit model by minimizing the neg. log likelihood
        result = optimize.minimize(neg_log_marg_likelihood, initial_guess_log,
                                         args=(X_train,y_train), method='L-BFGS-B')

        # next set of hyper prams
        prams_me = np.exp(result.x[0])**2

        print(cycle_num)
        cycle_num += 1

        # next used trained GP model to predict on test data
        mu, var = predict_GP(X_train, y_train, X_test, prams_me)
        mu_s.append(mu)
        var_s.append(var)
        y_s.append(y_test)
        params_test.append(prams_me)


    y_s_all = [datapt for iteration in y_s for datapt in iteration]
    mu_s_all = [datapt for iteration in mu_s for datapt in iteration]
    var_s_all = [datapt for iteration in var_s for datapt in iteration]


    return mu_s_all, var_s_all, y_s_all, params_test




# Matern kernel

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

def get_params_mat(X_train, y_train, init_param = [0.1, 10]):
    '''
    Optimize the neg. log likelihood to get the optimal hyperparameter
    :param X_train:
    :param y_train:
    :param init_param: list of 2 values as initial hyper parameters; default: [0.1, 10]
    :return: opt_param
    '''

    initial = init_param
    # take the log of the initial guess for optimiziation
    initial_guess_log = np.log(initial)

    # optimize to fit model by minimizing the neg. log likelihood
    result = optimize.minimize(neg_log_marg_likelihood_mat, initial_guess_log,
                               args=(X_train, y_train), method='L-BFGS-B')

    # next set of hyper prams
    opt_param = [np.exp(result.x[0]) ** 2, np.exp(result.x[1])]

    return opt_param

def cv_param_tuning_mat(X, y, k, init_param = [0.1, 10]):
    """
    Cross validation framework to find the optimal noise hyperparameter sigma^2 and the kernel
    parameter by minimizing the neg log likelihood.

    :param
        X (ndarray) of the sequences: should already be X_train data split (from
            outer CV for the model accuracy)
        y (ndarray) of normalized, measured values (KD); already split y_train data
            (from outer CV for the model accuracy)
        k (int) number of k for k-folds; number of splits

    :return

    """
    kf = KFold(n_splits=k) # Define the n_split = number of folds

    # initialize lists to store the
    mu_s = []
    var_s = []
    y_s = []
    params_test = []
    cycle_num = 1

    # loop for the CV
    for train_index, test_index in kf.split(X):

        #split in train and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        initial_guess = init_param

        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)

        # optimize to fit model by minimizing the neg. log likelihood
        result = optimize.minimize(neg_log_marg_likelihood_mat, initial_guess_log,
                                         args=(X_train,y_train), method='L-BFGS-B')

        # next set of hyper prams
        prams_me = [np.exp(result.x[0])**2, np.exp(result.x[1])]

        print(cycle_num)
        cycle_num += 1

        # next used trained GP model to predict on test data
        mu, var = predict_GP_mat(X_train, y_train, X_test, prams_me)
        mu_s.append(mu)
        var_s.append(var)
        y_s.append(y_test)
        params_test.append(prams_me)


    y_s_all = [datapt for iteration in y_s for datapt in iteration]
    mu_s_all = [datapt for iteration in mu_s for datapt in iteration]
    var_s_all = [datapt for iteration in var_s for datapt in iteration]


    return mu_s_all, var_s_all, y_s_all, params_test




# CDRdist kernel
# BLOSUM45

def predict_GP_CDRd45(X_train, y_train, X_test, param):
    """ Gaussian process regression prediction with CDRdist kernel.
    Parameters:
        X_train (np.ndarray): n x d training inputs as one-hot encoded
        y_train (np.ndarray): n training observations
        X_test (np.ndarray): m x d points to predict
    Returns:
        mu (np.ndarray): m predicted means
        var (np.ndarray): m predictive variances
    """
    # Evaluate kernel on training data
    K = CDRdist_kernel45(X_train)

    # To invert K_y we use the Cholesky decomposition (L)
    L = np.linalg.cholesky(K + np.eye(np.shape(X_train)[0]) * param ** 2)

    # solve for z=L^-1y
    z = linalg.solve_triangular(L, y_train, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star = CDRdist_kernel45(X_train, X_test)
    mu = np.matmul(K_star.T, alpha)

    # Compute the variance at the test points
    z = linalg.solve_triangular(L, K_star, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star_star = CDRdist_kernel45(X_test)
    v = np.diag(K_star_star) - np.dot(K_star.T, alpha)
    v = np.diag(v)

    return mu, v

def neg_log_marg_likelihood_CDRd45(log_pram, X, y):
    """ Calculate the negative log marginal likelihood loss. This
    will be minimized (= maximizing the log marginal likelihood)
    to determine the hyperparameter;
    We pass the log hypers here because it makes the optimization
    more stable.
    Parameters:
        log_hypers (np.ndarray): natural log of the hyper-parameters
        X (np.ndarray)
        y (np.ndarray)
    Returns:
        (float) The negative log marginal likelihood.
    """

    non_log_pram = np.exp(log_pram)
    # print(non_log_prams)

    # Evaluate kernel on training data
    K = CDRdist_kernel45(X)

    # To invert K we use the Cholesky decomposition (L), because symmetric and positive  definite
    n = len(y)
    L = np.linalg.cholesky(K + np.eye(np.shape(X)[0])*non_log_pram)
    z = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)  # dont know about this

    log_p_y_X = 0.5 * np.matmul(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

    return log_p_y_X

def get_params_CDRd45(X_train, y_train):
    '''
    Optimize the neg. log likelihood to get the optimal hyperparameter
    :param X_train:
    :param y_train:
    :return: opt_param
    '''

    initial = 2
    # take the log of the initial guess for optimiziation
    initial_guess_log = np.log(initial)

    # optimize to fit model by minimizing the neg. log likelihood
    result = optimize.minimize(neg_log_marg_likelihood_CDRd45, initial_guess_log,
                               args=(X_train, y_train), method='L-BFGS-B')

    # next set of hyper prams
    opt_param = np.exp(result.x[0]) ** 2

    return opt_param

def cv_param_tuning_CDRd45(X, y, k):
    """
    Cross validation framework to find the optimal noise hyperparameter sigma^2 by
    minimizing the neg log likelihood.

    :param
        X (ndarray) of the sequences: should already be X_train data split (from
            outer CV for the model accuracy)
        y (ndarray) of normalized, measured values (KD); already split y_train data
            (from outer CV for the model accuracy)
        k (int) number of k for k-folds; number of splits

    :return

    """
    kf = KFold(n_splits=k) # Define the n_split = number of folds

    # initialize lists to store the
    mu_s = []
    var_s = []
    y_s = []
    params_test = []
    cycle_num = 1
    # loop for the CV
    for train_index, test_index in kf.split(X):

        #split in train and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        initial_guess = 2

        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)

        # optimize to fit model by minimizing the neg. log likelihood
        result = optimize.minimize(neg_log_marg_likelihood_CDRd45, initial_guess_log,
                                         args=(X_train,y_train), method='L-BFGS-B')

        # next set of hyper prams
        prams_me = np.exp(result.x[0])**2


        # next used trained GP model to predict on test data
        mu, var = predict_GP_CDRd45(X_train, y_train, X_test, prams_me)
        mu_s.append(mu)
        var_s.append(var)
        y_s.append(y_test)
        params_test.append(prams_me)

        # print kv cycle
        print(cycle_num)
        cycle_num += 1

    y_s_all = [datapt for iteration in y_s for datapt in iteration]
    mu_s_all = [datapt for iteration in mu_s for datapt in iteration]
    var_s_all = [datapt for iteration in var_s for datapt in iteration]


    return mu_s_all, var_s_all, y_s_all, params_test



# BLOSUM62

def predict_GP_CDRd62(X_train, y_train, X_test, param):
    """ Gaussian process regression prediction with CDRdist kernel.
    Parameters:
        X_train (np.ndarray): n x d training inputs as one-hot encoded
        y_train (np.ndarray): n training observations
        X_test (np.ndarray): m x d points to predict
    Returns:
        mu (np.ndarray): m predicted means
        var (np.ndarray): m predictive variances
    """
    # Evaluate kernel on training data
    K = CDRdist_kernel62(X_train)

    # To invert K_y we use the Cholesky decomposition (L)
    L = np.linalg.cholesky(K + np.eye(np.shape(X_train)[0]) * param ** 2)

    # solve for z=L^-1y
    z = linalg.solve_triangular(L, y_train, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star = CDRdist_kernel62(X_train, X_test)
    mu = np.matmul(K_star.T, alpha)

    # Compute the variance at the test points
    z = linalg.solve_triangular(L, K_star, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star_star = CDRdist_kernel62(X_test)
    v = np.diag(K_star_star) - np.dot(K_star.T, alpha)
    v = np.diag(v)

    return mu, v

def neg_log_marg_likelihood_CDRd62(log_pram, X, y):
    """ Calculate the negative log marginal likelihood loss. This
    will be minimized (= maximizing the log marginal likelihood)
    to determine the hyperparameter;
    We pass the log hypers here because it makes the optimization
    more stable.
    Parameters:
        log_hypers (np.ndarray): natural log of the hyper-parameters
        X (np.ndarray)
        y (np.ndarray)
    Returns:
        (float) The negative log marginal likelihood.
    """

    non_log_pram = np.exp(log_pram)
    # print(non_log_prams)

    # Evaluate kernel on training data
    K = CDRdist_kernel62(X)

    # To invert K we use the Cholesky decomposition (L), because symmetric and positive  definite
    n = len(y)
    L = np.linalg.cholesky(K + np.eye(np.shape(X)[0])*non_log_pram)
    z = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)  # dont know about this

    log_p_y_X = 0.5 * np.matmul(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

    return log_p_y_X

def get_params_CDRd62(X_train, y_train):
    '''
    Optimize the neg. log likelihood to get the optimal hyperparameter
    :param X_train:
    :param y_train:
    :return: opt_param
    '''

    initial = 2
    # take the log of the initial guess for optimiziation
    initial_guess_log = np.log(initial)

    # optimize to fit model by minimizing the neg. log likelihood
    result = optimize.minimize(neg_log_marg_likelihood_CDRd62, initial_guess_log,
                               args=(X_train, y_train), method='L-BFGS-B')

    # next set of hyper prams
    opt_param = np.exp(result.x[0]) ** 2

    return opt_param

def cv_param_tuning_CDRd62(X, y, k):
    """
    Cross validation framework to find the optimal noise hyperparameter sigma^2 by
    minimizing the neg log likelihood.

    :param
        X (ndarray) of the sequences: should already be X_train data split (from
            outer CV for the model accuracy)
        y (ndarray) of normalized, measured values (KD); already split y_train data
            (from outer CV for the model accuracy)
        k (int) number of k for k-folds; number of splits

    :return

    """
    kf = KFold(n_splits=k) # Define the n_split = number of folds

    # initialize lists to store the
    mu_s = []
    var_s = []
    y_s = []
    params_test = []
    cycle_num = 1
    # loop for the CV
    for train_index, test_index in kf.split(X):

        #split in train and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        initial_guess = 2

        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)

        # optimize to fit model by minimizing the neg. log likelihood
        result = optimize.minimize(neg_log_marg_likelihood_CDRd62, initial_guess_log,
                                         args=(X_train,y_train), method='L-BFGS-B')

        # next set of hyper prams
        prams_me = np.exp(result.x[0])**2


        # next used trained GP model to predict on test data
        mu, var = predict_GP_CDRd62(X_train, y_train, X_test, prams_me)
        mu_s.append(mu)
        var_s.append(var)
        y_s.append(y_test)
        params_test.append(prams_me)

        # print kv cycle
        print(cycle_num)
        cycle_num += 1

    y_s_all = [datapt for iteration in y_s for datapt in iteration]
    mu_s_all = [datapt for iteration in mu_s for datapt in iteration]
    var_s_all = [datapt for iteration in var_s for datapt in iteration]


    return mu_s_all, var_s_all, y_s_all, params_test



# PAM40

def predict_GP_CDRdPAM40(X_train, y_train, X_test, param):
    """ Gaussian process regression prediction with CDRdist kernel.
    Parameters:
        X_train (np.ndarray): n x d training inputs as one-hot encoded
        y_train (np.ndarray): n training observations
        X_test (np.ndarray): m x d points to predict
    Returns:
        mu (np.ndarray): m predicted means
        var (np.ndarray): m predictive variances
    """
    # Evaluate kernel on training data
    K = CDRdist_kernelPAM40(X_train)

    # To invert K_y we use the Cholesky decomposition (L)
    L = np.linalg.cholesky(K + np.eye(np.shape(X_train)[0]) * param ** 2)

    # solve for z=L^-1y
    z = linalg.solve_triangular(L, y_train, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star = CDRdist_kernelPAM40(X_train, X_test)
    mu = np.matmul(K_star.T, alpha)

    # Compute the variance at the test points
    z = linalg.solve_triangular(L, K_star, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)
    K_star_star = CDRdist_kernelPAM40(X_test)
    v = np.diag(K_star_star) - np.dot(K_star.T, alpha)
    v = np.diag(v)

    return mu, v

def neg_log_marg_likelihood_CDRdPAM40(log_pram, X, y):
    """ Calculate the negative log marginal likelihood loss. This
    will be minimized (= maximizing the log marginal likelihood)
    to determine the hyperparameter;
    We pass the log hypers here because it makes the optimization
    more stable.
    Parameters:
        log_hypers (np.ndarray): natural log of the hyper-parameters
        X (np.ndarray)
        y (np.ndarray)
    Returns:
        (float) The negative log marginal likelihood.
    """

    non_log_pram = np.exp(log_pram)
    # print(non_log_prams)

    # Evaluate kernel on training data
    K = CDRdist_kernelPAM40(X)

    # To invert K we use the Cholesky decomposition (L), because symmetric and positive  definite
    n = len(y)
    L = np.linalg.cholesky(K + np.eye(np.shape(X)[0])*non_log_pram)
    z = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, z, lower=False)  # dont know about this

    log_p_y_X = 0.5 * np.matmul(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

    return log_p_y_X

def get_params_CDRdPAM40(X_train, y_train):
    '''
    Optimize the neg. log likelihood to get the optimal hyperparameter
    :param X_train:
    :param y_train:
    :return: opt_param
    '''

    initial = 2
    # take the log of the initial guess for optimiziation
    initial_guess_log = np.log(initial)

    # optimize to fit model by minimizing the neg. log likelihood
    result = optimize.minimize(neg_log_marg_likelihood_CDRdPAM40, initial_guess_log,
                               args=(X_train, y_train), method='L-BFGS-B')

    # next set of hyper prams
    opt_param = np.exp(result.x[0]) ** 2

    return opt_param

def cv_param_tuning_CDRdPAM40(X, y, k):
    """
    Cross validation framework to find the optimal noise hyperparameter sigma^2 by
    minimizing the neg log likelihood.

    :param
        X (ndarray) of the sequences: should already be X_train data split (from
            outer CV for the model accuracy)
        y (ndarray) of normalized, measured values (KD); already split y_train data
            (from outer CV for the model accuracy)
        k (int) number of k for k-folds; number of splits

    :return

    """
    kf = KFold(n_splits=k) # Define the n_split = number of folds

    # initialize lists to store the
    mu_s = []
    var_s = []
    y_s = []
    params_test = []
    cycle_num = 1
    # loop for the CV
    for train_index, test_index in kf.split(X):

        #split in train and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        initial_guess = 2

        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)

        # optimize to fit model by minimizing the neg. log likelihood
        result = optimize.minimize(neg_log_marg_likelihood_CDRdPAM40, initial_guess_log,
                                         args=(X_train,y_train), method='L-BFGS-B')

        # next set of hyper prams
        prams_me = np.exp(result.x[0])**2


        # next used trained GP model to predict on test data
        mu, var = predict_GP_CDRdPAM40(X_train, y_train, X_test, prams_me)
        mu_s.append(mu)
        var_s.append(var)
        y_s.append(y_test)
        params_test.append(prams_me)

        # print kv cycle
        print(cycle_num)
        cycle_num += 1

    y_s_all = [datapt for iteration in y_s for datapt in iteration]
    mu_s_all = [datapt for iteration in mu_s for datapt in iteration]
    var_s_all = [datapt for iteration in var_s for datapt in iteration]


    return mu_s_all, var_s_all, y_s_all, params_test




### MODEL EVALUATION ###

def calc_print_scores(measured, predicted, k=None):
    """
    Calculates and prints the R2, correlation coefficient and
    mean squared error (MSE) for measured and predicted values;
    :param predicted: (list) predicted values
    :param measured:  (list) true values
    :return: r2, r, mse (float) coefficient of determination,
            correlation coefficient and MSE
    """

    # calculate scores
    r2 = r2_score(measured, predicted)
    mse = mean_squared_error(measured, predicted)
    r = np.corrcoef(measured, predicted)[1][0]

    print(k, '-fold cross validation of GP regression model')
    print('R2 = %0.2f' % r2)
    print('Correlation coef = %0.2f' % r)
    print('MSE = %0.2f' % mse)

    return r2, r, mse







### VISUALIZATION OF CORRELATION

def correlation_plot(measured, predicted, cor_line = True, x_lim = 2.5, y_lim = 2.5,
                     save_fig = False, out_file=None):
    '''
    Draws and returns a simple correlation plot
    :param measured: (list) true values
    :param predicted: (list) predicted values
    :param cor_line: (bool) True if the line's slope should be 1, otherwise
    slope is inferred from the data
    :return: plt object
    '''
    plt.figure('My GP test set evaluation', figsize=(2, 2))
    plt.title('KD values')
    plt.plot(measured, predicted, 'o', color='k', ms=3)
    # set y and x axis
    y_lim = [-y_lim, y_lim]
    x_lim = [-x_lim, x_lim]
    plt.ylim(y_lim)
    plt.xlim(x_lim)

    if cor_line == False:
        # fit a linear function to the data to draw a line indicating correlation
        par = np.polyfit(measured, predicted, 1, full=True)
        slope = par[0][0]
        intercept = par[0][1]
        plt.plot(x_lim, [(y_lim[0]*slope+intercept), (y_lim[1]*slope+intercept)], '-', color='k')

    else:

        plt.plot(x_lim, y_lim, '-', color='k')

    # plt.savefig(path_outputs + str(property_)+'_matern_kernel_CV.pdf', bbox_inches='tight', transparent=True)

    if save_fig == True:
        plt.savefig(fname=out_file)

    plt.show()


def corr_var_plot(measured, predicted, vars=False, x_std=1, legend = False, method = None,
                  R2=None, corr_coef=None, MSE=None, save_fig = False, out_file=None):
    '''
    Correlation plot with filled areas of standard deviation
    (2nd order polynomial is fitted to the stdevs)

    :param: measured (ndarray) of the measured values
    :param: predicted (ndarray) of the predicted values
    :param: vars (ndarray) variance (calculated of the GP function
    :param: x_std (int) defining, whether 1 or 2 x standard deviation
            should be plotted
    :return: plot
    '''

    # Data prep
    y_pred = np.asarray(predicted)
    x = np.asarray(measured)

    # correlation line for the plot
    par = np.polyfit(x, y_pred, 1, full=True)
    slope = par[0][0]
    intercept = par[0][1]
    # set y and x axis
    x_lim = [-2.5, 2.5]
    y_lim = [-2.5, 2.5]

    ###### setup filled areas as standard deviation
    #if vars != False:
    std = x_std*np.sqrt(vars)
    # y_values of the correlation line to add to the stds
    y_corline = np.asarray([i * slope + intercept for i in x])

    std_pos = np.add(y_corline, np.abs(std))
    std_neg = np.subtract(y_corline, np.abs(std))

    # fit line to the stds and get y values of fit
    l_pos = np.polyfit(x, std_pos, 2)
    l_neg = np.polyfit(x, std_neg, 2)

    # get x and y values
    x_var = np.append(x, x_lim[1])
    x_var = np.insert(x_var, 0, x_lim[0])
    l_pos_y = x_var ** 2 * l_pos[0] + x_var * l_pos[1] + l_pos[2]
    l_neg_y = x_var ** 2 * l_neg[0] + x_var * l_neg[1] + l_neg[2]

    # combine values in dataframe to sort according to x
    var_df = pd.DataFrame()
    var_df['x_var'] = np.asarray(x_var)
    var_df['std_positive'] = np.asarray(l_pos_y)
    var_df['std_negative'] = np.asarray(l_neg_y)
    var_df.sort_values('x_var', inplace=True)


    ###### Start plot
    # plot
    plt.figure('GP', figsize=(5, 5))
    # set title and axis labels
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.title(str('Correlation of measured and predicted KD values' + method))
    plt.xlabel('measured')
    plt.ylabel('predicted')

    # add data points
    plt.scatter(x, y_pred, color='k')

    # add a diagonal and correlation line
    plt.plot(x_lim, [x_lim[0] * slope + intercept, x_lim[1] * slope + intercept], '-', color='k')
    # plt.plot(x_lim, y_lim, linestyle = '--',color='k')

    if vars is False:
        alpha = 0
    else:
        alpha = 1

    plt.scatter(x, std_pos, color='b', alpha=alpha, s=4)
    plt.scatter(x, std_neg, color='b', alpha=alpha, s=4)
    plt.fill_between(var_df['x_var'], var_df['std_positive'], var_df['std_negative'],
                     interpolate=True, alpha=0.3, color='orange')

    # define legend - note: can only plot 4 values (defined by the # of values plotted)
    if legend == True:
        l0 = "slope = {:.4f}".format(slope)
        l1 = "R2 = {:.4f}".format(R2)
        l2 = "Corr. coeff. = {:.4f}".format(corr_coef)
        l3 = "MSE = {:.4f}".format(MSE)

        leg = plt.legend(labels = [l0, l1, l2,l3], handlelength=0, handletextpad=0,
                     loc = 4)
        for item in leg.legendHandles:
            item.set_visible(False)

    if save_fig == True:
        plt.savefig(fname=out_file)

    plt.show()
    plt.close()


def corr_var_plot_highlighted(measured_train, predicted_train, var_train,
                            measured_test, predicted_test, var_test, legend=False,
                            R2=None, cor_coef=None, MSE=None, save_fig = False, out_file=None):

    # plot with highligted data points
    # set values
    x_test = measured_test
    x_train = measured_train

    y_pred_test = predicted_test
    y_pred_train = predicted_train

    std_test = 2*np.sqrt(var_test)
    std_train = 2*np.sqrt(var_train)

    x = np.concatenate((x_train, x_test))
    y_pred = np.concatenate((y_pred_train, y_pred_test))
    std = np.concatenate((std_train, std_test))


    # correlation line for all values
    par = np.polyfit(x, y_pred, 1, full=True)
    slope = par[0][0]
    intercept = par[0][1]

    # get coordinates of the standard devs
    y_corline = np.asarray([i*slope + intercept for i in x])
    std_pos = np.add(y_corline, np.abs(std))
    std_neg = np.subtract(y_corline, np.abs(std))

    # fit line to the stds and get y values of fit in a set range
    l_pos = np.polyfit(x, std_pos, 2)
    l_neg = np.polyfit(x, std_neg, 2)
    # set y and x axis
    x_lim = [-2.5, 2.5]
    y_lim = [-2.5, 2.5]
    # get x and y values
    x_area = np.append(x, x_lim[1])
    x_area = np.insert(x_area, 0, x_lim[0])
    l_pos_y = x_area ** 2 * l_pos[0] + x_area * l_pos[1] + l_pos[2]
    l_neg_y = x_area ** 2 * l_neg[0] + x_area * l_neg[1] + l_neg[2]

    # combine values in dataframe to sort according to x
    var_df = pd.DataFrame()
    var_df['x_area'] = x_area
    var_df['std_positive'] = l_pos_y
    var_df['std_negative'] = l_neg_y
    var_df.sort_values('x_area', inplace=True)

    plt.figure('GP testset evaluation', figsize=(5, 5))

    plt.title('Correlation of measured and predicted KD values')
    plt.xlabel('measured')
    plt.ylabel('predicted')
    # set y and x axis
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.fill_between(var_df['x_area'], var_df['std_positive'], var_df['std_negative'],
                     alpha = 0.3 , interpolate=True, color = 'orange')

    # set color list for the data points
    cols = list(np.concatenate((np.repeat('b',len(x_train)),np.repeat('r',len(x_test)))))

    plt.scatter(x_train, y_pred_train, color='k')
    plt.scatter(x_test, y_pred_test, color='r')
    # plt.errorbar(x, y_pred, fmt='ko', yerr=std, alpha = 0.5)
    plt.scatter(x, std_pos, color=cols, s=4, marker = ".")
    plt.scatter(x, std_neg, color=cols, s=4, marker = ".")
    plt.plot(x_lim, [x_lim[0] * slope + intercept, x_lim[1] * slope + intercept], '-', color='k')

    # define legend
    if legend == True:
        l0 = "slope = {:.4f}".format(slope)
        l1 = "R2 = {:.4f}".format(R2)
        l2 = "Corr. coeff. = {:.4f}".format(cor_coef)
        l3 = "MSE = {:.4f}".format(MSE)

        leg = plt.legend(labels = [l0, l1, l2,l3], handlelength=0, handletextpad=0,
                     loc = 4)
        for item in leg.legendHandles:
            item.set_visible(False)

    if save_fig == True:
        plt.savefig(fname=out_file)

    plt.show()






# ############################################## TEST CDRdist
# #
# #### LOAD INPUT ####
# input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'
# input_f_seq = input_dir + 'Final_49_AA_from_geneious.csv'
# input_f_KD = input_dir + 'HC_KDvals.csv'
#
# # output file directory
# dir_out = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Plots/GP_model/LD_Correlation/'
#
# # Load sequence data
# df_seq = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence'])
#
# # Load KD values and add them together to a new dataframe (remove samples that w/o measured KD value)
# KDs = pd.read_csv(input_f_KD, usecols=['SampleID', 'KD'])
# data = pd.merge(df_seq,KDs, on='SampleID')
#
# KDs_unnormalized = data
#
# #### DATA PROCESSING ####
#
# # normalize data
# data['KD'] = normalize_test_train_set(data['KD'])
#
# # split into train and test data
# X_train, X_test, y_train, y_test = split_data(data, 5, r_state=123)
#
#
#
#
# seqs = data['Sequence'].values
# KDs = data['KD'].values
#
#
# # test inner cv loop for hyperparameter tuning
# k = 35
# mus, vars, y_true, prams_test = cv_param_tuning(seqs, KDs, k, init_param=100)
#
# #
#
#
# # calculate and print scores
# r2, cor_coef, MSE= calc_print_scores(y_true, mus, k)
#
# # draw simple correlation plot
# correlation_plot(y_true, mus, cor_line=False, save_fig=False)
#                     # , out_file =str(dir_outLD+'LD_corr_CV_simple.png'))
#
#
# # draw correlation plot with standard deviation
# corr_var_plot(y_true, mus, vars, x_std=2, legend=True, R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=False, method='outercycle')
#                  # , out_file = str(dir_outLD+'LD_corr_variance_CV.png'))


### Test sensitivity of hyperparameter; same results for init_param: 0.5,1.2.10,100 for LD













