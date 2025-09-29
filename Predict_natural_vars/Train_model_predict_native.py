"""
Script to train ML models - GP

Setup pipeline to:

1.a Train models on training set
1.b predict novels set
2.a Train models on training+novel set
2.b Predict the 193 VDJ variants from the CDR3 based selection


Lena Erlach
17.11.2023
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import utils.GP_fcts as GP
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import Levenshtein
# Calculate levenshtein distance (normalized) matrix
def calc_norm_levens_dist(seqs: list, verbose=1):
    sim_matrix = np.ndarray((len(seqs), len(seqs)))
    for j in range(len(seqs)):

        if verbose > 0:
            if (j % 100 == 0):  print(j)

        LD_arr = []
        for i in range(len(seqs)):
            LD_arr.append(Levenshtein.distance(seqs[j], seqs[i]))

        # store distances in matrix
        sim_matrix[j, :] = LD_arr

    # return distance matrix
    # dist_matrix = 1 - sim_matrix
    return sim_matrix

def min_ED_seqs(seq_ref, seq):
    ''' Function that calculates the min distance to the seq_ref'''
    l_t = len(seq_ref)
    list_seqs = np.concatenate((seq_ref, seq))

    dist_matrix = calc_norm_levens_dist(list_seqs, verbose=1)
    min_dist = np.min(dist_matrix[l_t:, :l_t], axis=1)

    return min_dist

#########################################################################################################
###### PREDICTION CLASS ######
#########################################################################################################



class GP_train_predict():
    '''
    Class for training ML models and predicting affinity for sequences in native immune repertoire dataset.


    '''


    def __init__(self, reg = GaussianProcessRegressor(random_state=123, n_restarts_optimizer=10),
                 model_name: str = "GaussianProcess_RBF",
                 metrics: dict = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}):
        """
        Initializes the instance based on regression model, data, model name and metrics and param_grid.
        """
        # self.X_OH = X_OH
        # self.y = y
        self.reg = reg
        self.model_name = model_name
        self.metrics = metrics
        self.param_grid = None
        self.y_pred = None
        self.vars = None
        self.scaler = StandardScaler()
        self.best_model = None # best model from non-nested CV
        self.best_score = None # score from best model from non-nested CV



    def train_model(self, X_train, y_train, k=5,
                    param_grid = {'regressor__kernel': [RBF(l) for l in np.logspace(-1, 1, 3)], 'regressor__alpha': [1e-10, 1e-3, 0.1]},
                    scoring=None):
        '''
        Train regression model on the training dataset
        '''

        # Define a kernel with parameters
        # kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

        # Create Gaussian Process model
        gp = GaussianProcessRegressor(kernel=RBF(), random_state=123, n_restarts_optimizer=10)
        # gp = GaussianProcessRegressor(random_state=123, n_restarts_optimizer=5)
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('regressor', self.reg)
        ])




        # param_grid = {
        #            'reg__kernel': [RBF(l) for l in np.logspace(-1, 1, 3)].append([RBF(l) + ]),
        #            'reg__alpha': [1e-10, 1e-3, 0.1]}

        # # Define parameter grid
        # param_grid = {
        #     'reg__alpha': [1e-10, 1e-2, 1, 1e2],
        #     'reg__kernel__k1__length_scale': [l for l in np.logspace(-1, 1, 3)],    # Correct parameter name for RBF kernel
        #     'reg__kernel__k2__noise_level': [1e-10, 1e-2, 1, 10]
        # }


        # Grid search with cross-validation
        if scoring is None:
            print('metrics')
            scoring = self.metrics

        gp_grid = GridSearchCV(pipeline, param_grid, cv=k, scoring = scoring, refit='r2')

        # Fit the model
        gp_grid.fit(X_train, y_train)

        self.best_model = gp_grid.best_estimator_
        self.best_score = gp_grid.best_score_


        return gp_grid





###### SET MODEL AND PARAMETERS #####
random.seed(123)
randomized = False # set true to run the models with randomized labels for evaluation

k_inner = 5
k_outer = 5
verbose = 1
n_jobs = -1



s_fig = True








#########################################################################################################
###### SET INPUT/OUTPUT DIRECTORIES ######
#########################################################################################################

ROOT_DIR = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/'
input_train_seq = os.path.join(ROOT_DIR, 'GP_implementation/data/input/input_HCs.csv')
input_test_seq = os.path.join(ROOT_DIR, 'GP_implementation/data/final_validation/novel_variants_AA_KDs.csv')
input_VDJ_seq = os.path.join(ROOT_DIR, 'Predict_natural_vars/data/TEMP_mixcr_VDJ_HEL_clonotyped_80CDR3sim.csv')
# the raw data misses the first part of the mixcr file that's for matching the ones from the mixcr
VDJ_raw = os.path.join(ROOT_DIR,
                       'VDJ_Sequence_Selection/data/VDJ_selection/original_data/uniq_VDJs_from_Ann_Table_data_AP_simfilt80.txt')

# output
dir_out = os.path.join(ROOT_DIR, 'Analysis_for_publication_R/CorrPlot_designed_vars')


#########################################################################################################
############ LOAD DATA #############
#########################################################################################################

data_train = pd.read_csv(input_train_seq, usecols=['SampleID', 'Sequence', 'KD'])
data_test_all = pd.read_csv(input_test_seq)

# drop the variant with bad performance
data_test_expressed = data_test_all.iloc[:8,].drop(4).reset_index(drop=True)
data_test = data_test_all.drop(4).reset_index(drop=True)


# add value for expression
data_test["expressed"] = False
data_test.loc[~data_test['KD_nM'].isna(), 'expressed'] = True

data_test["min_ED_to_train"] = min_ED_seqs(data_train['Sequence'].values, data_test['VDJ_AA'].values).astype(int)







#########################################################################################################
############  SETUP DATASET 1 ############
#########################################################################################################

# train set of original training data and newly designed test set
X_train = data_train['Sequence'].values
y_train = data_train['KD'].values

# one-hot encoding
X_train_OH = GP.one_hot_encode_matern(X_train)
y_train = np.log(y_train + 1)



# test set
X_test = data_test['VDJ_AA'].values
y_test = data_test['KD_nM'].values

# one-hot encoding
X_test_OH = GP.one_hot_encode_matern(X_test)
y_test = np.log(y_test + 1)

#########################################################################################################
############  SETUP DATASET 2 ############
# #########################################################################################################
# # train set of original training data and newly designed test set
# X_train = np.concatenate((data_train['Sequence'].values, data_test_expressed['VDJ_AA'].values))
# y_train = np.concatenate((data_train['KD'].values, data_test_expressed['KD_nM'].values)).reshape(-1, 1)
#
# # one-hot encoding
# X_train_OH = GP.one_hot_encode_matern(X_train)
# y_train = np.log(y_train + 1)


# #########################################################################################################
# ############ LOAD NATIVE SEQS #############
# #########################################################################################################
#
# data_natvars_total = pd.read_csv(input_VDJ_seq, sep=',')
# data_natvars = pd.read_csv(VDJ_raw, sep='\t')['VDJ_AA']
#
# # prepare the sequences
# native_sel_seqs = prep_native_seqs(data_natvars, data_natvars_total)
# # remove original, selected variants from test set
# duplicated_id = [i for i, s in enumerate(native_sel_seqs) if s in X_train]
# native_sel_seqs_u = np.delete(native_sel_seqs, duplicated_id)
#
# # natural variants
# X_natvar = native_sel_seqs_u
# X_natvar_OH = GP.one_hot_encode_matern(X_natvar)








#########################################################################################################
##################           MODEL SETUP           ##################
#########################################################################################################

def main():
    gp = GP_train_predict()

    gp_trained = gp.train_model(X_train_OH, y_train, k=5,
                        param_grid = {'regressor__kernel': [RBF(l) for l in np.logspace(-1, 1, 3)], 'regressor__alpha': [1e-10, 1e-3, 0.1]},
                        scoring=None)

    y_pred, y_var = gp.best_model.predict(X_test_OH, return_std = True)

    data_test['y_pred'] = np.expm1(y_pred)
    data_test['y_var'] = y_var




    #########################################################################################################
    #################           CORRELATION PLOTS           ##################
    #########################################################################################################



    ##### Correlation plot - kD_nM
    plt.scatter(data_test['y_pred'][data_test['expressed'] == False], [0] * len(data_test['y_pred'][data_test['expressed'] == False]),
                color='k', label='not expressed')
    plt.scatter(data_test['y_pred'][data_test['expressed'] == True], data_test['KD_nM'][data_test['expressed'] == True],
                color= '#1874cd', label='expressed')


    xs = data_test['y_pred'][data_test['expressed'] == True].values
    ys = data_test['KD_nM'][data_test['expressed'] == True].values
    txts = data_test['min_ED_to_train'][data_test['expressed'] == True].values


    plt.ylabel("measured")
    plt.xlabel("predicted")
    plt.title('predicted and measured kD_nM')
    plt.legend()
    plt.savefig(os.path.join(dir_out, 'Correlation_plot_novelvars_with_noexpr.pdf'))
    # Loop for annotation of all points
    for i in range(len(xs)):
        plt.annotate(txts[i], (xs[i], ys[i] + 0.1))

    plt.savefig(os.path.join(dir_out, 'Correlation_plot_novelvars_with_noexpr_withEDs.pdf'))
    plt.show()






    ##### Correlation plot - log(kD_nM)
    plt.scatter(y_pred[data_test['expressed'] == False], [0] * len(y_test[data_test['expressed'] == False]),
                color='k', label='not expressed')
    plt.scatter(y_pred[data_test['expressed'] == True], y_test[data_test['expressed'] == True],
                color='#1874cd', label='expressed')
    plt.ylabel("measured")
    plt.xlabel("predicted")
    plt.title('predicted and measured log(kD_nM + 1)')
    plt.legend()
    plt.savefig(os.path.join(dir_out, 'Correlation_plot_novelvars_with_noexpr_logKDs.pdf'))

    xs = y_pred[data_test['expressed'] == True]
    ys = y_test[data_test['expressed'] == True]
    txts = data_test['min_ED_to_train'][data_test['expressed'] == True].values

    # Loop for annotation of all points
    for i in range(len(xs)):
        plt.annotate(txts[i], (xs[i], ys[i] + 0.05))
    plt.savefig(os.path.join(dir_out, 'Correlation_plot_novelvars_with_noexpr_logKDs_withEDs.pdf'))
    plt.show()






    #########################################################################################################
    #################           HISTOGRAM OF KDS           ##################
    ########################################################################################################

    y_pred_ex = data_test['y_pred'][data_test['expressed'] == True]
    y_pred_nex = data_test['y_pred'][data_test['expressed'] == False]
    plt.hist([y_pred_ex, y_pred_nex], color=['#000000', '#A9A9A9'], label=['expressed', 'not expressed'])
    plt.ylabel('count')
    plt.xlabel('predicted kD_nM values')
    plt.title('Predicted kDs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()





