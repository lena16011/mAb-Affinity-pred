'''
Script to visualize the predictions of the novel, designed sequences with correlation plots.

With different models:
- GP Matern
- GP RBF
- Kernel Ridge

'''


import pandas as pd
import numpy as np
from utils import GP_fcts as GP
from GP_implementation.Regression_Evaluation_framework import Regression_evaluation_paramTuning as REV
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, OneHotEncoder



###### SET INPUT DIRECTORIES ######
in_dir2 = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/input'
in_f2 = os.path.join(in_dir2, 'input_HCs.csv')
path_toKD = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/final_validation/novel_variants_AA_KDs.csv"



###### LOAD DATA ######
data = pd.read_csv(in_f2, usecols=['SampleID', 'Sequence', 'KD'])
data.columns = ['SampleID', 'VDJ_AA', 'KD_nM'] # rename columns
novel_data = pd.read_csv(path_toKD, delimiter = ',')

# check for duplicates of original sequences
print("Sequences identical to original {} variants: {}".format(len(data.VDJ_AA), any(novel_data.VDJ_AA.isin(data.VDJ_AA)) == True))
print()
"No duplicates found (true), if {}".format(len(novel_data.VDJ_AA) == len(np.unique(novel_data.VDJ_AA)))


# model names for saving files etc.
model_names = ["GaussianProcess_RBF", "GaussianProcess_Matern", "KernelRidge"] #, "RandomForestRegression", "OrdinalLinearRegression"]
# list of parameters to test per model (take care of order!)
param_list = [{'kernel': [None, RBF()],
               'alpha': [1e-10, 0.1]},
              {'kernel': [None, Matern()],
               'alpha': [1e-10, 0.1]},
              {'alpha': [0.1, 1.0, 10.0],
               'kernel': ['linear', 'rbf', 'polynomial'],
               'degree': [2, 3, 4],
               'gamma': [0.1, 1.0, 10.0]}#,
              # {'n_estimators': [10, 100, 200],
              #  'max_depth': [2, 5, 10]},
              # {'fit_intercept': [True, False]}
              ]
# set True for GPs, False for other models; (again order!)
vars_list = [True, True, False]#, False, False]

model_list = [GaussianProcessRegressor(random_state=1),
              GaussianProcessRegressor(random_state=1),
              KernelRidge()# ,
              # RandomForestRegressor(random_state=1),
              # LinearRegression()
              ]
# define the metrics to be used; so far it only works with 2 max, and the first one will be the one the model will
# be optimized for
metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}

# create an amino acid directory
# encode_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,\
#            'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,\
#            'V':17,'W':18,'Y':19}
# create an amino acid directory
encode_dict = ['A','C','D', 'E','F','G','H','I','K',\
           'L','M','N','P','Q','R','S','T',\
           'V','W','Y']



#### DATA PROCESSING ####
# normalize data
scaler = StandardScaler().fit(np.array(data['KD_nM']).reshape(len(data['KD_nM']),1))
data['KD_norm'] = scaler.transform(np.array(data['KD_nM']).reshape(len(data['KD_nM']),1))
novel_data['KD_norm'] = scaler.transform(np.array(novel_data['KD_nM']).reshape(len(novel_data['KD_nM']),1))

X = data['VDJ_aa'].values
y = data['KD_norm'].values
# # random labels
# if randomized is True:
#     y = data['KD_norm'].sample(frac=1).reset_index(drop = True).values

# one-hot encoding
OH_encoder = OneHotEncoder(categories=encode_dict).fit(X)





















class regression_model_predict_novel_sequences:
    """ Train regression models with with a training set and predict on test set;

        This class can be defned with various models, which will be:
         - trained on a specified train set (GP project: initially selected and characterized sequences
         - kD values of newly designed sequences will be predicted
         - visualizations of the predicted and measured KD values (correlation plot

        Attributes:
            X_OH: numpy.ndarray of shape (n_samples, p_length * 20)
                protein sequence data, one-hot encoded;
            y: numpy.ndarray of shape (n_samples, 1)
                log-standardized kD value to be predicted from the protein sequences;
            reg: instance of a sklearn regressor
                sklearn estimator for regression; e.g. reg = sklearn.kernel_ridge.KernelRidge() with or without parameters
            model_name: str
                name of the model type as a prefix for saving files/plots;
            metrics: dict
                metric names (str) as keys and strings as values of the corresponding sklearn metrics for fitting the models;
                currently up to 2 metrics are supported in this class; the first one in the dictionary will be used as the main
                metric for refitting the model; the second will be only reported; optimized for MSE and R2 value;
                for default set to: {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
            param_grid: dict
                parameters names (str) as keys and lists of parameter settings for parameter tuning of the model
            k_i: int, default 3;
                number of k-folds for the inner nested CV (parameter tuning)
            corr_plot_output: boolean
                if True, a plot for the k evaluation will be generated
            y_pred: numpy.ndarray of shape (n_test_samples, 1)
                predicitons  of the test set (in case of GPs: mu)
            vars: numpy.ndarray of shape (n_test_samples, 1)
                variance of predicitons  of the test set (in case of GPs only)

        """

    def __init__(self, X_OH: np.ndarray, y: np.ndarray, reg, model_name: str, metrics: dict):
        """
        Initializes the instance based on regression model, data, model name and metrics.
        """
        self.X_OH = X_OH
        self.y = y
        self.reg = reg
        self.model_name = model_name
        self.metrics = metrics
        self.param_grid = None
        self.k_i = 3
        self.corr_plot_output = None
        self.y_pred = None
        self.vars = None


    def k_CV_and_plot(self, param_grid: dict, k: int, plot: bool = True, save_fig: bool=False, w_vars = False, save_path=None):

        kf = KFold(n_splits=k, shuffle=True, random_state=1)  # Define the n_split = number of folds

        # Define the grid search object - non-nested
        grid_search = GridSearchCV(estimator=self.reg,
                                   param_grid=param_grid,
                                   scoring=self.metrics,
                                   cv=kf,
                                   refit=list(self.metrics.keys())[0])

        # Non_nested parameter search and scoring
        grid_search.fit(self.X_OH, self.y)
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_
        print('Best model: ', best_model)
        print('Best score: ', best_score)

        # Obtain the predicted values using cross-validation
        self.y_pred = cross_val_predict(best_model, self.X_OH, self.y, cv=kf, verbose=1)
        if w_vars == True:
            # compute std separately
            self.vars = np.empty_like(self.y_pred)
            for i, (train_index, test_index) in enumerate(kf.split(self.X_OH)):
                best_model.fit(self.X_OH[train_index], self.y[train_index])
                _, self.vars[test_index] = best_model.predict(self.X_OH[test_index], return_std=True)

        elif w_vars == False:
            self.vars = False


        r2, cor_coef, MSE = GP.calc_print_scores(self.y, self.y_pred, k)

        # plot the results
        if plot == True:
            GP.corr_var_plot(self.y, self.y_pred, vars=self.vars, x_std=2, legend=True, method="\n" + self.model_name + " regression",
                             R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig=save_fig, out_file=save_path)

        # summarize in df
        model_score_df = pd.DataFrame(data = {'Model': [self.model_name], 'R2': [r2], 'Corr_coef': [cor_coef], 'MSE': [MSE], 'params': [str(best_model)]}, index=[0])

        return model_score_df