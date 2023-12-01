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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Suppress the warning from sklearn.metrics module
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)#, module='sklearn.metrics')


def run():
    ###### SET PARAMETERS ######
    random.seed(123)
    log_transform = True  # test if the models predict differently


    ###### SET INPUT DIRECTORIES ######
    in_dir2 = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/input'
    in_f2 = os.path.join(in_dir2, 'input_HCs.csv')
    path_toKD = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/final_validation/novel_variants_AA_KDs.csv"

    # output plot for the plots
    dir_output_plots = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/Plots/final_validation/log_transformed_data'
    dir_out_eval = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/final_validation/log_transformed_data'
    if not os.path.exists(dir_out_eval):
        os.makedirs(dir_out_eval)

    ###### LOAD DATA ######
    data = pd.read_csv(in_f2, usecols=['SampleID', 'Sequence', 'KD'])
    data.columns = ['SampleID', 'VDJ_AA', 'KD_nM'] # rename columns
    novel_data = pd.read_csv(path_toKD, delimiter = ',')

    # check for duplicates of original sequences
    print("Sequences identical to original {} variants: {}".format(len(data.VDJ_AA), any(novel_data.VDJ_AA.isin(data.VDJ_AA)) == True))
    print()
    "No duplicates found (true), if {}".format(len(novel_data.VDJ_AA) == len(np.unique(novel_data.VDJ_AA)))


    #### DATA PROCESSING ####
    X_seq_train = data['VDJ_AA'].values
    X_seq_test = novel_data['VDJ_AA'].values[~np.isnan(novel_data['KD_nM'])]

    y_train = data['KD_nM'].values.reshape(-1,1)
    y_test = novel_data['KD_nM'].values[~np.isnan(novel_data['KD_nM'])].reshape(-1,1)
    # log transform
    if log_transform is True:
        y_train = np.log(y_train + 1)
        y_test = np.log(y_test + 1)

    # one-hot encoding
    X_train = GP.one_hot_encode_matern(X_seq_train)
    X_test = GP.one_hot_encode_matern(X_seq_test)



    #### SET MODEL PARAMETERS ####
    # test the framework
    # model names for saving files etc.
    model_names = ["GaussianProcess_RBF"#, "GaussianProcess_Matern",
                   # "KernelRidge",
                   # "RandomForestRegression",
                   # "OrdinalLinearRegression"
    ]
    # list of parameters to test per model (take care of order!)
    param_list = [{'regressor__kernel': [None, RBF()],
                   'regressor__alpha': [1e-10, 0.1]},
                  {'regressor__kernel': [None, Matern()],
                   'regressor__alpha': [1e-10, 0.1]},
                  {'regressor__alpha': [0.1, 1.0, 10.0],
                   'regressor__kernel': ['linear', 'rbf', 'polynomial'],
                   'regressor__degree': [2, 3, 4],
                   'regressor__gamma': [0.1, 1.0, 10.0]},
                  {'regressor__n_estimators': [10, 100, 200],
                   'regressor__max_depth': [2, 5, 10]},
                  {'regressor__fit_intercept': [True, False]}
                  ]
    # set True for GPs, False for other models; (again order!)
    vars_list = [True, True,
                 False, False, False]


    model_list = [GaussianProcessRegressor(random_state=1),
                  GaussianProcessRegressor(random_state=1),
                  KernelRidge(),
                  RandomForestRegressor(random_state=1),
                  LinearRegression()
                  ]
    # define the metrics to be used; so far it only works with 2 max, and the first one will be the one the model will
    # be optimized for
    metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
    lim=[-0.5,2.5] # x & y limits for the plots

    test_score_df = pd.DataFrame(columns=['Model', 'R2', 'Corr_coef', 'MSE'])
    prediction_df = pd.DataFrame(columns=['IDs', 'VDJ_AA', 'y', 'y_pred', 'train_label', 'y_var', 'Model'])

    s_figs = False


    #### LOOP THROUGH ALL THE MODELS ####
    for i, model_name in enumerate(model_names):

        param_grid = param_list[i]
        reg = model_list[i]
        w_vars = vars_list[i]
        model_name = model_names[i]
        print()
        print('--- Start model evaluation: ' + model_name)

        #### SET OUTPUT DIRECTORIES (for plots to save) ####
        dir_out = os.path.join(dir_output_plots, model_name)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)


        # define class model
        print('\n--- Optimize parameters')

        kernel = REV.regression_model_evaluation(X_train, y_train.reshape(-1,), reg, model_name, metrics)

        kernel.k_CV_and_plot(param_grid, k=len(X_train), plot = True, save_fig=s_figs, x_lim=lim, x_std=2,
                                                      y_lim=lim, w_vars = w_vars,
                                                      save_path=os.path.join(dir_out, model_name+'_corr_plot.pdf'))


        print('\n--- Predict designed sequences as test set\n')
        # get the predictions for the
        if w_vars is True:
            y_test_pred, y_test_vars = kernel.best_model.predict(X_test, return_std = True)
            y_test_vars = y_test_vars**2
        elif w_vars is False:
            y_test_pred = kernel.best_model.predict(X_test)

        # make a dataframe with ids, y, y_pred, category (train, test (hi, mid, lo kd
        pred_df = pd.DataFrame({'IDs': np.append(data.SampleID, novel_data.IDs[~np.isnan(novel_data['KD_nM'])]),
                                'VDJ_AA': np.append(X_seq_train, X_seq_test), 'y': np.append(kernel.y , y_test),
                                'y_pred': np.append(kernel.y_pred, y_test_pred),
                                'train_label': np.append(np.asarray(['train'] * len(kernel.y)).reshape(-1,), novel_data.label[~np.isnan(novel_data['KD_nM'])])
                                })
        if w_vars is True:
            pred_df['y_var'] = np.append(kernel.vars , y_test_vars)

        pred_df["Model"] = model_name
        prediction_df = prediction_df.append(pred_df)

        # print scores
        r2, cor_coef, MSE = GP.calc_print_scores(y_test.reshape(-1,), y_test_pred.reshape(-1,), k=len(X_train))

        # combine to dataframe
        test_score_df.loc[i, 'R2'] = r2
        test_score_df.loc[i, 'Corr_coef'] = cor_coef
        test_score_df.loc[i, 'MSE'] = MSE
        test_score_df.loc[i, 'Model'] = str(kernel.best_model[1])



        #### MAKE CORRELATION PLOT WITH THE TEST SET HIGHLIGHTED - CONFIDENCE INTERVAL IS FIT ON THE TEST SET TOO! ####
        # CONFIDENCE INTERVAL IS FIT ON THE TEST SET TOO! Can be adjusted for non probablistic models
        if w_vars is True:
            # adjust correlation plot for highlighting the novel sequences
            GP.corr_var_plot_highlighted(measured_train = kernel.y, predicted_train = kernel.y_pred, vars_train = kernel.vars, x_std=2,
                                         measured_test = y_test, predicted_test = y_test_pred,x_lim = lim, y_lim = lim,
                                         vars_test = y_test_vars, legend=True, R2=r2, cor_coef=cor_coef, MSE=MSE, save_fig = s_figs,
                                         out_file=os.path.join(dir_out, 'Corr_plot_with_testset_highlighted.pdf'))



        #### MAKE CORRELATION PLOT WITH THE TEST SET HIGHLIGHTED BY CATEGORY ####
        label_n = ['hiKD_rat', 'loKD_rat']

        # setup dictionarys for plotting and highlighting the categories
        y = pred_df.y.values[pred_df.train_label == 'train']
        y_pred = pred_df.y_pred.values[pred_df.train_label == 'train']

        y_test_set_dict = {'hiKD_rat': pred_df.y.values[pred_df.train_label == 'hiKD_rat'], 'loKD_rat': pred_df.y.values[pred_df.train_label == 'loKD_rat']}
        y_pred_test_set_dict = {'hiKD_rat': pred_df.y_pred.values[pred_df.train_label == 'hiKD_rat'], 'loKD_rat': pred_df.y_pred.values[pred_df.train_label == 'loKD_rat']}

        # define stds, if model predicts so
        if w_vars is True:
            y_var = pred_df.y_var.values[pred_df.train_label == 'train']
            y_var_test_set_dict = {'hiKD_rat': pred_df.y_var.values[pred_df.train_label == 'hiKD_rat'],
                                   'loKD_rat': pred_df.y_var.values[pred_df.train_label == 'loKD_rat']}
        else:
            y_var_test_set_dict = False
            y_var = False

        # Correlation plot
        GP.corr_var_plot_highlighted_extended(y, y_pred, y_var, label_n, y_test_set_dict, y_pred_test_set_dict,
                                               y_var_test_set_dict, x_std=2, colors = ['#e01212', '#1f1fab', '#0f6e02', '#f2c40a'],
                                               x_lim=lim, y_lim=lim, errbar=True, std_scatter = False,
                                               std_scatter_test = False, save_fig=s_figs, out_file=os.path.join(dir_out, 'Corr_plot_with_testset_highl_errbar_only_rat.pdf'))


        #### MAKE CORRELATION PLOT WITH THE TEST SET HIGHLIGHTED BY CATEGORY - with the random designed variants ####
        label_n = ['hiKD_rat', 'loKD_rat', 'germ_line_HC50', 'loLD_pos', 'loLD_mid']
        y_test_set_dict = {'hiKD_rat': pred_df.y.values[pred_df.train_label == 'hiKD_rat'],
                           'loKD_rat': pred_df.y.values[pred_df.train_label == 'loKD_rat'],
                            'germ_line_HC50': pred_df.y.values[pred_df.train_label == 'germ_line_HC50'],
                            'loLD_pos': pred_df.y.values[pred_df.train_label == 'loLD_pos'],
                            'loLD_mid': pred_df.y.values[pred_df.train_label == 'loLD_mid'],
                           }
        y_pred_test_set_dict = {'hiKD_rat': pred_df.y_pred.values[pred_df.train_label == 'hiKD_rat'],
                                'loKD_rat': pred_df.y_pred.values[pred_df.train_label == 'loKD_rat'],
                                'germ_line_HC50': pred_df.y_pred.values[pred_df.train_label == 'germ_line_HC50'],
                                'loLD_pos': pred_df.y_pred.values[pred_df.train_label == 'loLD_pos'],
                                'loLD_mid': pred_df.y_pred.values[pred_df.train_label == 'loLD_mid']}
        # define stds, if model predicts so
        if w_vars is True:
            y_var_test_set_dict = {'hiKD_rat': pred_df.y_var.values[pred_df.train_label == 'hiKD_rat'],
                                   'loKD_rat': pred_df.y_var.values[pred_df.train_label == 'loKD_rat'],
                                   'germ_line_HC50': pred_df.y_var.values[pred_df.train_label == 'germ_line_HC50'],
                                   'loLD_pos': pred_df.y_var.values[pred_df.train_label == 'loLD_pos'],
                                   'loLD_mid': pred_df.y_var.values[pred_df.train_label == 'loLD_mid']}
        else:
            y_var_test_set_dict = False


        GP.corr_var_plot_highlighted_extended(y, y_pred, y_var, label_n, y_test_set_dict, y_pred_test_set_dict,
                                               y_var_test_set_dict, x_std=1, colors = ['#e01212', '#1f1fab', '#0f6e02', '#f2c40a', '#6f07b0'],
                                               x_lim=lim, y_lim=lim, errbar = True, std_scatter = False,
                                               std_scatter_test = False, save_fig=s_figs, out_file=os.path.join(dir_out, 'Corr_plot_with_testset_highl_errbar.pdf'))

        print("--------------------------------------------------------")
        print("DONE with " + model_name)
        print("---------------------++++++++++++-----------------------")



    # --- SAVE THE SUMMARIZED DATAFRAMES - CV MODEL SCORES AND TEST SET MODEL SCORES
    test_score_df.to_csv(os.path.join(dir_out_eval, 'Test_scores_designed_vars_tuned_model.csv'))
    prediction_df.to_csv(os.path.join(dir_out_eval, 'Predictions_allvariants_tuned_model.csv'))


def main():
    run()




if __name__ == '__main__':
    main()



