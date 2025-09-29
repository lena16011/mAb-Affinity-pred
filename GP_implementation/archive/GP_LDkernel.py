'''
Script does the LOO-CV for the measured VDJ variants with a kernel based on the **LD** between the sequences.

- for each cycle, the model is trained on 34 of the variants and predicts the affinity of the remaining one
- all of the predicted and measured affinities are compared and R^2, pearson correlation coefficient, MSE are
calculated.
- correlation plots are drawn and saved
'''

import pandas as pd
import os
from utils import GP_fcts as GP
import matplotlib.pyplot as plt



###### SET INPUT DIRECTORIES ######
abs_path = 'D:/Dokumente/Masterarbeit/Lena/GP_implementation'
input_dir = abs_path + '/data/input/'

input_f_seq = input_dir + 'input_HCs.csv'


## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = abs_path + '/data/Plots/GP_model/CV_correlation/LD_kernel/'

# If the output directories do not exist, then create it
if not os.path.exists(dir_out):
    os.makedirs(dir_out)



###### LOAD DATA #######
data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])


#### DATA PROCESSING ####
# normalize data
data['KD_norm'] = GP.normalize_test_train_set(data['KD'])

X_train = data['Sequence'].values
y_train = data['KD_norm'].values



#### CROSS VALIDATION ####

# test model in Leave one out-cv loop for hyperparameter tuning in cv_param_tuning function
k = 35
mus, vars, y_true, prams_test = GP.cv_param_tuning(X_train, y_train, k)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig = True, out_file = dir_out + 'LD_corr_plot_simple.png')


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nLD kernel',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig = True, out_file= dir_out + 'LD_corr_plot.png')


### Plot the distribution of predicted values  ===> doesn't make sense, because GPs assume
# Gaussian distribution of the data!!
# fig, axs = plt.subplots(1, 2, tight_layout=True)
#
# axs[0].hist(mus, bins = 30)
# axs[1].hist(y_train, bins = 30)
# axs[0].title.set_text('Distribution of predictions')
# axs[1].title.set_text('Distribution of true KDs')
# plt.savefig(fname=dir_out + 'LD_predDistribution.png', format='png')
# plt.show()





