

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

###### BOX PLOT - FUNCTION ######

###### SET INPUT DIRECTORIES & LOAD DATA ######
# Set paths to originally selected and novel sequences
in_f1 = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/final_validation/log_transformed_data/Test_scores_designed_vars_tuned_model.csv'


# set path to output folder
save_path = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Analysis_for_publication_R/Model_comparison_plots"

# load selected sequences
data = pd.read_csv(in_f1, index_col=0)
data.drop(index=4, inplace=True)
data["Model_params"] = data["Model"]
data.loc[0, 'Model'] = 'GP RBF'
data.loc[1, 'Model'] = 'GP Matern'
data.loc[2, 'Model'] = 'Kernel Ridge'
data.loc[3, 'Model'] = 'Random Forest'
data.rename(columns={"Corr_coef": "Correlation coefficient"}, inplace=True)

d_m = pd.melt(data.loc[:,['Model', 'MSE', 'Correlation coefficient']], id_vars=['Model'], var_name='Metric')


c_pal = 'Paired'
fig_size = (5, 7)
x='Model'
y='value'
hue='Metric'
save_fig = os.path.join(save_path, 'R2_CorrCoef_Barplot_LOOCV_novVars.pdf')


# setup a box plot with the original sequences
sns.set(
    style="ticks",  # The 'ticks' style
    rc={"figure.figsize": fig_size,  # width = 6, height = 9
        })  # Axes colour
b = sns.barplot(data=d_m, x=x, y=y, hue=hue,
                palette=c_pal)  # Stop showing the fliers)
# y axis
b.set(ylabel=None) # remove y label
b.set_ylim(0, 1.0)
# # y axis
b.set_xlabel("Regression model", fontsize=14) # Set the x axis label and font size
b.set_xticklabels(b.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.tight_layout()
plt.savefig(save_fig)
plt.show()

