'''
Script to visualize the relation between variance and kD values.

With different models:
- GP Matern
- GP RBF
- Kernel Ridge
Author: Lena Erlach
Date: 11.07.2023
'''


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

# set, if log the data
log = False # looks better/more informative
fig_size = (5, 4.5)
save_fig_b = True

# set paths
in_path = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/final_validation/log_transformed_data/Predictions_allvariants_tuned_model.csv"
save_fig = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Analysis_for_publication_R/Variance_analysis/"
save_names = ['kD_pred_variance_scatter_rat_designed_vars.pdf', 'kD_pred_variance_scatter_all_designed_vars.pdf']
save_names1 = ['MSE_variance_scatter_rat_designed_vars.pdf', 'MSE_variance_scatter_all_designed_vars.pdf']
model_names = ['GaussianProcess_RBF', 'GaussianProcess_Matern']

# instantiate corr coef dataframe
corr_coeff_df = pd.DataFrame(columns=['pearson_corr_coef', 'p_val_pearson', 'spearman_corr_coef', 'p_val_spearman', 'Model',
                                   'with_random_design'])

# Load dataframe
prediction_df = pd.read_csv(in_path, index_col=0)
if log is True:
    prediction_df['y'] = np.log(prediction_df['y'])
    prediction_df['y_var'] = np.log(prediction_df['y_var'])

# loop throught the models
for i, model_name in enumerate(model_names):
    # i = 0
    # model_name = model_names[i]
    # filter for model
    p_df1 = prediction_df.loc[prediction_df['Model'] == model_name,]
    p_df1.reset_index(inplace=True, drop=True)

    # Calculate correlation coefficients for kD and Variances
    pears_corr, pears_p = scipy.stats.pearsonr(p_df1.y, p_df1.y_var)  # Pearson's r
    spear_corr, spearm_p = scipy.stats.spearmanr(p_df1.y, p_df1.y_var) # Spearman's rho

    corr_coeff_df = corr_coeff_df.append({'pearson_corr_coef': pears_corr, 'p_val_pearson': pears_p, 'spearman_corr_coef': spear_corr,
                                   'p_val_spearman': spearm_p, 'Model': model_name, 'with_random_design': True}, ignore_index=True)

    # setup scatter plot without the random designed vars
    sns.set(
        style="ticks",
        rc={"figure.figsize": fig_size,  # width = 6, height = 9
            })  # Axes colour
    b = sns.scatterplot(data=p_df1, x='y', y='y_var', hue='train_label')
    # y axis
    b.set_ylabel("predicted variance", fontsize=14)  # remove y label
    # y axis
    b.set_xlabel("measured kD value", fontsize=14)  # Set the x axis label and font size
    b.set(title='Predicted variance ' + model_name)
    b.set_ylim(-0.01, .55)
    plt.tight_layout()
    # sns.despine() # removes the top and right border line of the plot
    if save_fig_b is True:
        plt.savefig(os.path.join(save_fig, model_name + "_" + save_names[1]))

    plt.show()


    # FILTER OUT RANDOMLY DESIGNED
    # filter the data for the rationally designed only
    p_df = p_df1.loc[(p_df1['train_label'] != 'germ_line_HC50') & (p_df1['train_label'] != 'loLD_pos') &
                    (p_df1['train_label'] != 'loLD_neg') & (p_df1['train_label'] != 'loLD_mid')]
    p_df.reset_index(drop=True, inplace=True)

    # Calculate correlation coefficients for kD and Variances
    pears_corr, pears_p = scipy.stats.pearsonr(p_df.y, p_df.y_var)  # Pearson's r
    spear_corr, spearm_p = scipy.stats.spearmanr(p_df.y, p_df.y_var)  # Spearman's rho

    corr_coeff_df = corr_coeff_df.append(
        {'pearson_corr_coef': pears_corr, 'p_val_pearson': pears_p, 'spearman_corr_coef': spear_corr,
         'p_val_spearman': spearm_p, 'Model': model_name, 'with_random_design': False}, ignore_index=True)

    # setup scatter plot without the random designed vars
    b = sns.scatterplot(data=p_df, x='y', y='y_var', hue='train_label')
    # y axis
    b.set_ylabel("predicted variance", fontsize=14) # remove y label
    # y axis
    b.set_xlabel("measured kD value", fontsize=14) # Set the x axis label and font size
    b.set(title='Predicted variance - '+ model_name)
    b.set_ylim(-0.01, .22)
    plt.tight_layout()
    #sns.despine() # removes the top and right border line of the plot
    if save_fig_b is True:
        plt.savefig(os.path.join(save_fig, model_name + "_" + save_names[0]))
    plt.show()

corr_coeff_df.to_csv(os.path.join(save_fig, "Correlation_coefficient_kD_pred_vars.csv"))






############################################################

############ Visualize the variance vs MSE

############################################################

for i, model_name in enumerate(model_names):
    # i = 0
    # model_name = model_names[i]

    # Load dataframe
    prediction_df = pd.read_csv(in_path, index_col=0)


    # filter for model
    p_df1 = prediction_df.loc[prediction_df['Model'] == model_name,]
    p_df1.reset_index(inplace=True, drop=True)


    ##### Run for all the sequences ######
    ms_error = np.square(p_df1['y'] - p_df1['y_pred'])
    variances = p_df1['y_var']

    # setup scatter plot without the random designed vars
    sns.set(
        style="ticks",
        rc={"figure.figsize": fig_size,  # width = 6, height = 9
            })  # Axes colour
    b = sns.scatterplot(x=ms_error, y=variances, hue=p_df1['train_label'])
    # y axis
    b.set_ylabel("predicted variance", fontsize=14)  # remove y label
    # y axis
    b.set_xlabel("MSE", fontsize=14)  # Set the x axis label and font size
    b.set(title='Predicted variance vs. MSE ' + model_name)
    # b.set_ylim(-0.01, .55)
    plt.tight_layout()
    if save_fig_b is True:
        print('save1')
        plt.savefig(os.path.join(save_fig, model_name + "_" + save_names1[1]))
    plt.show()



    ##### Run for only rationally designed sequences ######
    # filter the data for the rationally designed only
    p_df = p_df1.loc[(p_df1['train_label'] != 'germ_line_HC50') & (p_df1['train_label'] != 'loLD_pos') &
                    (p_df1['train_label'] != 'loLD_neg') & (p_df1['train_label'] != 'loLD_mid')]
    p_df.reset_index(drop=True, inplace=True)


    ms_error = np.sqrt(np.square(p_df['y'] - p_df['y_pred']))
    variances = p_df['y_var']


    b = sns.scatterplot(x=ms_error, y=variances, hue=p_df['train_label'])
    # y axis
    b.set_ylabel("predicted variance vs. MSE ", fontsize=14)  # remove y label
    # y axis
    b.set_xlabel("MSE", fontsize=14)  # Set the x axis label and font size
    b.set(title='Predicted variance ' + model_name)
    # b.set_ylim(-0.01, .55)
    plt.tight_layout()
    if save_fig_b is True:
        print('save2')
        plt.savefig(os.path.join(save_fig, model_name + "_" + save_names1[0]))

    plt.show()






