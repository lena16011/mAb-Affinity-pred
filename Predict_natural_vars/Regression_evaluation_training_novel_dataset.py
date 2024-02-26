"""
Script to implement ML models (GP, KRR,  regression with sklearn

Setup pipeline to:

1. Train models on training set
2. Predict the 193 VDJ variants from the CDR3 based selection
3. optional?? predict the variants from the clonal lineages (check out the clonotyped mixcr file!!)

Lena Erlach
17.11.2023
"""


import pandas as pd
import numpy as np
import os
import utils.GP_fcts as GP
import warnings, random
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor







#########################################################################################################
######  SETUP  #####
#########################################################################################################

def run():
    ##########################################################################################################
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # , module='sklearn.metrics')

    # ###### SET MODEL AND PARAMETERS #####
    random.seed(123)
    randomized = False # set true to run the models with randomized labels for evaluation

    k_inner = 5
    k_outer = 5
    verbose = 1
    n_jobs = -1





    # model names for saving files etc.
    model_names = ["GaussianProcess_RBF",
                   "GaussianProcess_RBF_WhiteKernel",
                   "GaussianProcess_Matern",
                   "KernelRidge", "RandomForestRegression", "OrdinalLinearRegression"
                   ]
    # list of parameters to test per model (take care of order!)
    param_list = [
                  {'regressor__kernel': [RBF(l) for l in np.logspace(-1, 1, 3)],
                   'regressor__alpha': [1e-10, 1e-3, 0.1]},

                {'regressor__kernel': [RBF()+WhiteKernel()],
                'regressor__kernel__k1__length_scale': [l for l in np.logspace(-1, 1, 3)],    # Correct parameter name for RBF kernel
                'regressor__kernel__k2__noise_level': [1e-10, 1e-2, 1, 10],
                'regressor__alpha': [1e-10, 1e-3, 0.1]},

                  {'regressor__kernel': [Matern(l) for l in np.logspace(-1, 1, 3)],
                   'regressor__alpha': [1e-10, 1e-3, 0.1]},
                  {'regressor__alpha': [1e-3, 0.1, 1.0, 10.0],
                   'regressor__kernel': ['linear', 'rbf', 'polynomial'],
                   'regressor__degree': [2, 3],
                   'regressor__gamma': [0.1, 1.0, 10.0]},
                  {'regressor__n_estimators': [10, 100, 200],
                   'regressor__max_depth': [2, 5, 10]},
                  {'regressor__fit_intercept': [True, False]}
                  ]
    # set True for GPs, False for other models; (again order!)
    vars_list = [True,
                 True,
                 True,
                 False,
                 False, False
                ]

    model_list = [
                  GaussianProcessRegressor(random_state=1),
                  GaussianProcessRegressor(random_state=1),
                  GaussianProcessRegressor(random_state=1),
                  KernelRidge(),
                  RandomForestRegressor(random_state=1),
                  LinearRegression()
                  ]
    # define the metrics to be used; so far it only works with 2 max, and the first one will be the one the model will
    # be optimized for
    metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}

    s_fig = True

    #########################################################################################################
    ###### SET INPUT/OUTPUT DIRECTORIES ######
    #########################################################################################################

    ROOT_DIR = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/'
    input_train_seq = os.path.join(ROOT_DIR, 'GP_implementation/data/input/input_HCs.csv')
    input_test_seq = os.path.join(ROOT_DIR, 'GP_implementation/data/final_validation/novel_variants_AA_KDs.csv')
    input_VDJ_seq = os.path.join(ROOT_DIR, 'Predict_natural_vars/data/TEMP_mixcr_VDJ_HEL_clonotyped_80CDR3sim.csv')
    # the raw data misses the first part of the mixcr file that's for matching the ones from the mixcr
    VDJ_raw = os.path.join(ROOT_DIR, 'VDJ_Sequence_Selection/data/VDJ_selection/original_data/uniq_VDJs_from_Ann_Table_data_AP_simfilt80.txt')



    # output
    dir_out_model = os.path.join(ROOT_DIR, 'Predict_natural_vars/data/model_evaluation/log_transformed_input_outlier_rm/Plots')
    dir_out_eval = os.path.join(ROOT_DIR, 'Predict_natural_vars/data/model_evaluation/log_transformed_input_outlier_rm')


    #########################################################################################################
    ############ LOAD DATA #############
    #########################################################################################################

    data_train = pd.read_csv(input_train_seq, usecols=['SampleID', 'Sequence', 'KD'])
    data_test_all = pd.read_csv(input_test_seq)


    # drop the variant with bad performance
    data_test = data_test_all.iloc[:8, :].drop(4).reset_index(drop = True)



    #########################################################################################################
    ############  SETUP DATASET  ############
    #########################################################################################################

    # train set of original training data and newly designed test set
    X_train = np.concatenate((data_train['Sequence'].values, data_test['VDJ_AA'].values))
    y_train = np.concatenate((data_train['KD'].values, data_test['KD_nM'].values)).reshape(-1,1)

    # one-hot encoding
    X_train_OH = GP.one_hot_encode_matern(X_train)
    y_train = np.log(y_train + 1)



    #########################################################################################################
    ############ LOAD NATIVE SEQS #############
    #########################################################################################################

    data_natvars_total = pd.read_csv(input_VDJ_seq, sep=',')
    data_natvars = pd.read_csv(VDJ_raw, sep='\t')['VDJ_AA']

    # prepare the sequences
    native_sel_seqs = prep_native_seqs(data_natvars, data_natvars_total)
    # remove original, selected variants from test set
    duplicated_id = [i for i, s in enumerate(native_sel_seqs) if s in X_train]
    native_sel_seqs_u = np.delete(native_sel_seqs, duplicated_id)


    # natural variants
    X_natvar = native_sel_seqs_u
    X_natvar_OH = GP.one_hot_encode_matern(X_natvar)




    # Dataframes to be saved
    Nested_Scores_df= pd.DataFrame(columns=['model_name', 'MSE_nonnested', 'R2_nonnested', 'best_params_nonnested',
                                          'MSE_nested','R2_nested', 'k_outer', 'k_inner', 'mostselected_params_nested'])
    LOO_Scores_df = pd.DataFrame(columns=['Model', 'R2', 'Corr_coef', 'MSE', 'params'])



    # list of ks to evaluate
    k_l = list(range(2, len(X_train_OH), 3))
    k_l.append(len(X_train_OH))


    #########################################################################################################
    ###### EVALUATION OF Ks; NESTED CV #####
    #########################################################################################################

    for i, model_name in enumerate(model_names):

        print('\nStart model evaluation: '+ model_name)
        param_grid = param_list[i]
        reg = model_list[i]
        w_vars = vars_list[i]

        ## SET OUTPUT DIRECTORIES (for plots to save)
        dir_out = os.path.join(dir_out_model, model_name)

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        if not os.path.exists(dir_out_eval):
            os.makedirs(dir_out_eval)

        # define class model
        kernel = REV.regression_model_evaluation(X_train_OH, y_train, reg, model_name, metrics)

        ### EVALUATE Ks FOR CV
        print('Evaluate k')
        kernel.evaluate_k(param_grid, k_in=k_inner, k_l=k_l, plot = True, save_fig= s_fig, n_jobs= n_jobs,
                          save_path=os.path.join(dir_out, model_name+'_k_sensitivity_nestedCV_test_ki'+ str(k_inner) + '.pdf'), verbose=0)

        ### MODEL EVALUATION WITH NESTED CV (set k values)
        print('\nPerform nested CV')
        non_nested_cv_df, nested_cv_df, scores_df = kernel.nested_param_tuning_eval(param_grid, k_o=k_outer, k_i=k_inner,
                                                                                    n_jobs = n_jobs, verbose=verbose)
        # save scores dataframe
        Nested_Scores_df = Nested_Scores_df.append(scores_df)

        ###### HYPERPARAMETER TUNING AND LOO-CV WITH BEST PERFORMING PARAMETERS #####
        ### LOO CV
        print('\nPerform LOO-CV and make correlation plot')
        model_score_df = kernel.k_CV_and_plot(param_grid, k=len(X_train_OH), plot = True, save_fig=s_fig, x_lim=[-0.5,2.5],
                                              y_lim=[-0.5,2.5], w_vars = w_vars,
                                              save_path=os.path.join(dir_out, model_name+'_corr_plot.pdf'))

        LOO_Scores_df = LOO_Scores_df.append(model_score_df)
        ##########################################################################################################
        print("--------------------------------------------------------")
        print("DONE with " + model_name)
        print("---------------------++++++++++++-----------------------")

    #
    # save dataframes
    Nested_Scores_df.to_csv(os.path.join(dir_out_eval, 'Nested_CV_scores_ki'+str(k_inner)+'_ko'+str(k_outer)+'.csv'))
    LOO_Scores_df.to_csv(os.path.join(dir_out_eval, 'Param_tuned_LOOCV_scores.csv'))





    #########################################################################################################
    ###### MAKE CORRELATION PLOT WITH THE TEST SET HIGHLIGHTED BY CATEGORY #####
    #########################################################################################################
    s_fig = True
    for i, model_name in enumerate(model_names):

        print('\nStart model evaluation: '+ model_name)
        param_grid = param_list[i]
        reg = model_list[i]
        w_vars = vars_list[i]

        ## SET OUTPUT DIRECTORIES (for plots to save)
        dir_out = os.path.join(dir_out_model, model_name)

        # define class model
        kernel = REV.regression_model_evaluation(X_train_OH, y_train, reg, model_name, metrics)

        print('\nPerform LOO-CV and make correlation plot')
        model_score_df = kernel.k_CV_and_plot(param_grid, k=len(X_train_OH), plot = True, save_fig=s_fig, x_lim=[-0.5,2.5],
                                                  y_lim=[-0.5,2.5], w_vars = w_vars,
                                                  save_path=os.path.join(dir_out, model_name+'_corr_plot.pdf'))


        label_n= ['hiKD_rat', 'loKD_rat']
        lim = [-1, 2.5]


        pred_df = pd.DataFrame(columns=['ID', 'y', 'y_pred', 'y_var', 'train_label'])
        pred_df['y'] = y_train.reshape(-1,)
        pred_df['y_pred'] = kernel.y_pred
        pred_df['y_var'] = kernel.vars


        pred_df.loc[:len(data_train.SampleID), 'ID'] = data_train['SampleID']
        pred_df.loc[len(data_train.SampleID):, 'ID'] = data_test['IDs'].values

        pred_df.loc[:35, 'train_label'] = 'train'
        pred_df.loc[35:, 'train_label'] = data_test.label.values

        # setup dictionarys for plotting and highlighting the categories
        y = pred_df.y.values[pred_df.train_label == 'train']
        y_pred = pred_df.y_pred.values[pred_df.train_label == 'train']
        # setup dictionaries for plotting and highlighting the categories
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
                                               std_scatter_test = False, save_fig=s_fig, out_file=os.path.join(dir_out, 'Corr_plot_with_testset_highl_errbar_only_rat.pdf'))


    print('DONE')






def main():
    run()




if __name__ == '__main__':
    main()













#########################################################################################################
######      GENERATE PREDICTIONS ON THE NATIVE SEQUENCE VARIANTS          #####
#########################################################################################################







