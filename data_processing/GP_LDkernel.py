
import numpy as np
import pandas as pd
import stringdist as sd
from sklearn.model_selection import train_test_split, KFold
from scipy import optimize, linalg
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

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
    if isinstance(in_seq2, (np.bool)):
        seq1 = pd.Series(in_seq1)
        n = range(len(seq1))
        ker=pd.DataFrame(columns=n)
        data_ser = pd.DataFrame(columns=n)
        for i in range(len(seq1)):
            data_ser = 1-seq1[0:].apply(sd.levenshtein_norm, args=(seq1[i],))
            ker.loc[i] = data_ser
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


def split_data(df, n):
    '''
    Convert a pd.DataFrame of the sequences and KD values into test (X_test, y_true)
    and traininig data (X_train, y_train) by randomly sampling n samples.

    :param df: (pd.dataframe) with the sequence and and KD information of the whole dataset
    :param n: (int) #samples that are put into the test set
    :return: (np.array) X_train, y_train, X_test, y_true
    '''
    X = df.Sequence.values
    y = df.KD.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n)
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


def predict_GP(X_train, y_train, X_test, param):
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

def cv_param_tuning(X, y, k):
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

    # loop for the CV
    for train_index, test_index in kf.split(X):

        #split in train and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        initial_guess = 2

        # take the log of the initial guess for optimiziation
        initial_guess_log = np.log(initial_guess)

        # optimize to fit model by minimizing the neg. log likelihood
        result = optimize.minimize(neg_log_marg_likelihood, initial_guess_log,
                                         args=(X_train,y_train), method='L-BFGS-B')

        # next set of hyper prams
        prams_me = np.exp(result.x[0])**2


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

def correlation_plot(measured, predicted):
    '''
    Draws and returns a simple correlation plot
    :param measured: (list) true values
    :param predicted: (list) predicted values
    :return: plt object
    '''
    plt.figure('My GP test set evaluation', figsize=(2, 2))
    plt.title('KD values')
    plt.plot(measured, predicted, 'o', color='k', ms=3)
    # set y and x axis
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])

    # fit a linear function to the data to draw a line indicating correlation
    par = np.polyfit(measured, predicted, 1, full=True)
    slope = par[0][0]
    intercept = par[0][1]

    plt.plot([-2, 2], [1 * -2 + intercept, slope * 2 + intercept], '-', color='k')
    # plt.savefig(path_outputs + str(property_)+'_matern_kernel_CV.pdf', bbox_inches='tight', transparent=True)
    plt.show()


def get_params(X_train, y_train):
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
    result = optimize.minimize(neg_log_marg_likelihood, initial_guess_log,
                               args=(X_train, y_train), method='L-BFGS-B')

    # next set of hyper prams
    opt_param = np.exp(result.x[0]) ** 2

    return opt_param





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

# split into train and test data
X_train, X_test, y_train, y_test = split_data(data, 5)

# normalize data
y_train, y_test = normalize_test_train_set(y_train, y_test)


#### CROSS VALIDATION ####

# test inner cv loop for hyperparameter tuning
k = 30
mus, vars, y_true, prams_test = cv_param_tuning(X_train, y_train, k)
# calculate and print scores
calc_print_scores(y_true, mus, k)
# draw correlation plot
correlation_plot(y_true, mus)



#### test on whole data set
opt_param = get_params(X_train, y_train)

# predict the test set
mu_test, var_test = predict_GP(X_train, y_train, X_test, opt_param)
# predict training set
mu_train, var_test = predict_GP(X_train, y_train, X_train, opt_param)

predicted = np.concatenate((mu_train, mu_test))
measured = np.concatenate((y_train, y_test))


calc_print_scores(predicted, measured)
# draw correlation plot
correlation_plot(predicted, measured)

print("Test set: ")
for m, p in zip(y_test, mu_test):
    print("measured value: {0:.2f} and predicted value: {1:.2f}".format(m, p))


print("Training set: ")
for m, p in zip(y_train, mu_train):
    print("measured value: {0:.2f} and predicted value: {1:.2f}".format(m, p))







































###### ################calc correlation acc. gihub code of Bedbrook
# why manual calculation ???
# fit a linear function to the data points (least squares)
par = np.polyfit(measured, predicted, 1, full=True)
slope=par[0][0]
intercept=par[0][1]

variance = np.var(predicted)
residuals = np.var([(slope*xx + intercept - yy) for xx, yy in zip(measured, predicted)])
Rsqr = np.round(1-residuals/variance, decimals=2)
######################################################



