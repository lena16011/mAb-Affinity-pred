'''
Script to visualize the metrics for the

 - Barplot of model metrics

'''


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

###### BOX PLOT - FUNCTION ######

###### SET INPUT DIRECTORIES & LOAD DATA ######
# Set paths to originally selected and novel sequences
in_f1 = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/log_transformed_input/Param_tuned_LOOCV_scores.csv'
in_f2 = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/log_transformed_input/Nested_CV_scores_ki5_ko5.csv"

# set path to output folder
save_path = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Analysis_for_publication_R/Model_comparison_plots"

# load metrics
data = pd.read_csv(in_f1, index_col=0)
data.reset_index(inplace=True, drop=True)
# rename model entries
data.loc[0, 'Model'] = 'GP RBF'
data.loc[1, 'Model'] = 'GP Matern'
data.loc[2, 'Model'] = 'Kernel Ridge'
data.loc[3, 'Model'] = 'Random Forest'
# exclude Linear model
data.drop(index=4, inplace=True)
# data.loc[4, 'Model'] = 'Linear'
data.rename(columns={"R2": "R^2 LOO-CV", "MSE": "MSE LOO-CV","Corr_coef": "Correlation coefficient"}, inplace=True)
d_m1 = pd.melt(data.loc[:,['Model', 'MSE LOO-CV', 'Correlation coefficient', 'R^2 LOO-CV']], id_vars=['Model'], var_name='Metric')


# add data from nested cross validation
data_nested = pd.read_csv(in_f2, index_col=0)
data_nested.reset_index(inplace=True, drop=True)
# rename model entries
data_nested.loc[0, 'Model'] = 'GP RBF'
data_nested.loc[1, 'Model'] = 'GP Matern'
data_nested.loc[2, 'Model'] = 'Kernel Ridge'
data_nested.loc[3, 'Model'] = 'Random Forest'
# exclude Linear model
data_nested.drop(index=4, inplace=True)
data_nested.rename(columns={"R2_nested": "R^2 nested", "MSE_nested": "MSE nested"}, inplace=True)
d_m2 = pd.melt(data_nested.loc[:,['Model', 'MSE nested', 'R^2 nested']], id_vars=['Model'], var_name='Metric')

# append to dataframe and set order
d_m = pd.concat([d_m2[d_m2.Metric == "MSE nested"], d_m1[d_m1.Metric == "MSE LOO-CV"],
                 d_m2[d_m2.Metric == "R^2 nested"], d_m1[d_m1.Metric == "R^2 LOO-CV"],
                 d_m1[d_m1.Metric == "Correlation coefficient"]])



c_pal = 'Paired'
fig_size = (6, 8)
x='Model'
y='value'
hue='Metric'
save_fig = os.path.join(save_path, 'R2_CorrCoef_Barplot_updated11092023.pdf')


# setup a box plot with the original sequences
sns.set(
    style="ticks",  # The 'ticks' style
    rc={"figure.figsize": fig_size,  # width = 6, height = 9
        })  # Axes colour
b = sns.barplot(data=d_m, x=x, y=y, hue=hue,
                palette=c_pal)  # Stop showing the fliers)
# y axis
b.set(ylabel=None) # remove y label
b.set_ylim(0, 1)
# # y axis
b.set_xlabel("Regression model", fontsize=14) # Set the x axis label and font size
b.set_xticklabels(b.get_xticklabels(), rotation=45, horizontalalignment='right')

#plt.title("LOO-CV model evaluation")
plt.legend(bbox_to_anchor=(0.5, 1.12), ncol=3, loc='upper center', borderaxespad=0)
plt.tight_layout()
plt.savefig(save_fig)
plt.show()


