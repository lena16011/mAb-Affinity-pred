'''
Script does the LOO-CV for the measured VDJ variants with a kernel based on the **CDRdist measure**
(which is based on BLOSUM and PAM substitution matrices).

- for each cycle, the model is trained on 34 of the variants and predicts the affinity of the remaining one
- all of the predicted and measured affinities are compared and R^2, pearson correlation coefficient, MSE are
calculated.
- correlation plots are drawn and saved
'''


import pandas as pd
import os
from utils import GP_fcts as GP
import numpy as np
import matplotlib.pyplot as plt



###### SET INPUT DIRECTORIES ######
abs_path = 'D:/Dokumente/Masterarbeit/Lena/GP_implementation'
input_dir = abs_path + '/data/input/'

input_f_seq = input_dir + 'input_HCs.csv'


## SET OUTPUT DIRECTORIES (for plots to save)

dir_out = abs_path + '/data/Plots/GP_model/CV_correlation/LD_kernel/'

dir_out1 = abs_path + '/data/Plots/GP_model/CV_correlation/CDRdistB45/'
dir_out2 = abs_path + '/data/Plots/GP_model/CV_correlation/CDRdistB62/'
dir_out3 = abs_path + '/data/Plots/GP_model/CV_correlation/CDRdistPAM140/'

# If the output directories do not exist, then create it
if not os.path.exists(dir_out1):
    os.makedirs(dir_out1)
if not os.path.exists(dir_out2):
    os.makedirs(dir_out2)
if not os.path.exists(dir_out3):
    os.makedirs(dir_out3)


###### LOAD DATA #######
data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])


#### DATA PROCESSING ####
# normalize data
data['KD_norm'] = GP.normalize_test_train_set(data['KD'])

X_train = data['Sequence'].values
y_train = data['KD_norm'].values



#### CROSS VALIDATION ####
### BLOSUM$45

# test inner cv loop for hyperparameter tuning in cv_param_tuning function
k = 35
mus1, vars1, y_true1, prams_test1 = GP.cv_param_tuning_CDRd45(X_train, y_train, k)

# calculate and print scores
r21, cor_coef1, MSE1 = GP.calc_print_scores(y_true1, mus1, k)

# draw simple correlation plot
GP.correlation_plot(y_true1, mus1, cor_line=False, save_fig = True, out_file = dir_out1 + 'B45_corr_plot_simple.png')



##########################################################
#
# weird mathematical issue...
#
##########################################################





#################  RERAN with B62


out_file = dir_out3 + 'PAM140_corr_plot_laxis.png'
# set the axis
ax_limit = 10

corr_coef = cor_coef3
R2 = r23
legend = True
save_fig = True

method = 'PAM140'
measured = y_true3
predicted = mus3
vars=vars3
MSE = MSE3

x_std = 2


# Correlation plot with filled areas as standard deviation
std = x_std * np.sqrt(vars)
y_pred = np.asarray(predicted)
x = np.asarray(measured)

# correlation line for the plot
par = np.polyfit(x, y_pred, 1, full=True)
slope = par[0][0]
intercept = par[0][1]

# y_values of the correlation line to add to the stds
y_corline = np.asarray([i * slope + intercept for i in x])
std_pos = np.add(y_corline, np.abs(std))
std_neg = np.subtract(y_corline, np.abs(std))

# fit line to the stds and get y values of fit
l_pos = np.polyfit(x, std_pos, 2)
l_neg = np.polyfit(x, std_neg, 2)

# set y and x axis

x_lim = [-ax_limit, ax_limit]
y_lim = [-ax_limit, ax_limit]
# get x and y values
x_var = np.append(x, x_lim[1])
x_var = np.insert(x_var, 0, x_lim[0])
l_pos_y = x_var ** 2 * l_pos[0] + x_var * l_pos[1] + l_pos[2]
l_neg_y = x_var ** 2 * l_neg[0] + x_var * l_neg[1] + l_neg[2]

# combine values in dataframe to sort according to x
var_df = pd.DataFrame()
var_df['x_var'] = np.asarray(x_var)
var_df['std_positive'] = np.asarray(l_pos_y)
var_df['std_negative'] = np.asarray(l_neg_y)
var_df.sort_values('x_var', inplace=True)

# plot
plt.figure('GP', figsize=(5, 5))
# set title and axis labels
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.title(str('Correlation of measured and predicted KD values ' + method))
plt.xlabel('measured')
plt.ylabel('predicted')

# add data points
plt.scatter(x, y_pred, color='k')

# add a diagonal and correlation line
plt.plot(x_lim, [x_lim[0] * slope + intercept, x_lim[1] * slope + intercept], '-', color='k')
# plt.plot(x_lim, y_lim, linestyle = '--',color='k')

# plt.errorbar(x, y_pred, fmt='ko', yerr=std, alpha = 0.5)
plt.scatter(x, std_pos, color='b', s=4)
plt.scatter(x, std_neg, color='b', s=4)
plt.fill_between(var_df['x_var'], var_df['std_positive'], var_df['std_negative'],
                 interpolate=True, alpha=0.3, color='orange')
# define legend
if legend == True:
    l0 = "slope = {:.4f}".format(slope)
    l1 = "R2 = {:.4f}".format(R2)
    l2 = "Corr. coeff. = {:.4f}".format(corr_coef)
    l3 = "MSE = {:.4f}".format(MSE)

    leg = plt.legend(labels=[l0, l1, l2, l3], handlelength=0, handletextpad=0,
                     loc=4)
    for item in leg.legendHandles:
        item.set_visible(False)

if save_fig == True:
    plt.savefig(fname=out_file, format='png')

plt.show()



##### doesn't work

# draw correlation plot with standard deviation
# GP.corr_var_plot(y_true1, mus1, vars1, x_std=2, legend=True, method = '\nBLOSUM45',
#               R2=r21, corr_coef=cor_coef1, MSE = MSE1, save_fig = True, out_file = dir_out1 + 'B45_corr_plot.png')

##### doesn't work







### BLOSUM62

# test inner cv loop for hyperparameter tuning in cv_param_tuning function
k = 35
mus2, vars2, y_true2, prams_test2 = GP.cv_param_tuning_CDRd62(X_train, y_train, k)

# calculate and print scores
r22, cor_coef2, MSE2 = GP.calc_print_scores(y_true2, mus2, k)

# draw simple correlation plot
GP.correlation_plot(y_true2, mus2, cor_line=False, save_fig = False, out_file=dir_out2+'B62_corr_plot_simple.png')


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus2, vars2, x_std=2, legend=True, method = '\nBLOSUM62',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig = True, out_file=dir_out2+'BLOSUM62_new.png')



### PAM140%

# test inner cv loop for hyperparameter tuning in cv_param_tuning function
k = 35
mus3, vars3, y_true3, prams_test3 = GP.cv_param_tuning_CDRdPAM40(X_train, y_train, k)

# calculate and print scores
r23, cor_coef3, MSE3 = GP.calc_print_scores(y_true3, mus3, k)

# draw simple correlation plot
GP.correlation_plot(y_true3, mus3, cor_line=False, save_fig = True, out_file = dir_out3+'PAM40_corr_plot_simple.png')


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true3, mus3, vars3, x_std=2, legend=True, method = '\nPAM40',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig = True, out_file = dir_out3+'PAM40_corr_plot_simple.png')

