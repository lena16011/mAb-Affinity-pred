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

# set, if log the data
log = False # looks better/more informative


# set paths
in_path = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/final_validation/log_transformed_data/Predictions_allvariants_tuned_model.csv"
save_fig = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Analysis_for_publication_R/Variance_analysis/"
save_names = ['kD_pred_variance_scatter_rat_designed_vars.pdf', 'kD_pred_variance_scatter_all_designed_vars.pdf']
model_names = ['GaussianProcess_RBF', 'GaussianProcess_Matern']

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


    # setup scatter plot without the random designed vars
    b = sns.scatterplot(data=p_df1, x='y', y='y_var', hue='train_label')
    # y axis
    b.set_ylabel("predicted variance", fontsize=14)  # remove y label
    # y axis
    b.set_xlabel("measured kD value", fontsize=14)  # Set the x axis label and font size
    b.set(title='Predicted variance - ' + model_name)
    b.set_ylim(0, .55)
    plt.tight_layout()
    # sns.despine() # removes the top and right border line of the plot
    plt.savefig(os.path.join(save_fig, model_name + "_" + save_names[1]))
    plt.show()


    # filter the data for the ratinoally designed only
    p_df = p_df1.loc[(p_df1['train_label'] != 'germ_line_HC50') & (p_df1['train_label'] != 'loLD_pos') &
                    (p_df1['train_label'] != 'loLD_neg') & (p_df1['train_label'] != 'loLD_mid')]
    p_df.reset_index(drop=True, inplace=True)

    # setup scatter plot without the random designed vars
    b = sns.scatterplot(data=p_df, x='y', y='y_var', hue='train_label')
    # y axis
    b.set_ylabel("predicted variance", fontsize=14) # remove y label
    # y axis
    b.set_xlabel("measured kD value", fontsize=14) # Set the x axis label and font size
    b.set(title='Predicted variance - '+ model_name)
    b.set_ylim(0, .2)
    plt.tight_layout()
    #sns.despine() # removes the top and right border line of the plot
    plt.savefig(os.path.join(save_fig, model_name + "_" + save_names[0]))
    plt.show()




