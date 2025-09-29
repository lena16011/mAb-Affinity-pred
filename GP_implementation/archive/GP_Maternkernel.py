'''
Script does the LOO-CV for the measured VDJ variants with a Matern kernel based on the one-hot encoded sequences.

- for each cycle, the model is trained on 34 of the variants and predicts the affinity of the remaining one
- all of the predicted and measured affinities are compared and R^2, pearson correlation coefficient, MSE are
calculated.
- correlation plots are drawn and saved
'''


import pandas as pd
import os
from utils import GP_fcts as GP
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

###### SET INPUT DIRECTORIES ######
abs_path = 'D:/Dokumente/Masterarbeit/Lena/GP_implementation'
input_dir = abs_path + '/data/input/'

input_f_seq = input_dir + 'input_HCs.csv'


## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = abs_path + '/data/Plots/GP_model/CV_correlation/Matern_kernel/'

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

# one-hot encoding
X_train_OH = GP.one_hot_encode_matern(X_train)


#### CROSS VALIDATION ####

k = 35
mus, vars, y_true, prams_test = GP.cv_param_tuning_mat(X_train_OH, y_train, k)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig = False, out_file=dir_out + 'Matern_corr_plot_simple.png')

def corr_var_plot(measured, predicted, vars, x_std=1, legend = False, method = None,
                  R2=None, corr_coef=None, MSE=None, save_fig = False, out_file=None):
    '''
    Correlation plot with filled areas of standard deviation
    (2nd order polynomial is fitted to the stdevs)

    :param: measured (ndarray) of the measured values
    :param: predicted (ndarray) of the predicted values
    :param: vars (ndarray) variance (calculated of the GP function
    :param: x_std (int) defining, whether 1 or 2 x standard deviation
            should be plotted
    :return: plot
    '''

    # Correlation plot with filled areas as standard deviation
    std = x_std*np.sqrt(vars)
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
    x_lim = [-2.5, 2.5]
    y_lim = [-2.5, 2.5]
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
    plt.title(str('Correlation of measured and predicted KD values' + method))
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

        leg = plt.legend(labels = [l0, l1, l2,l3], handlelength=0, handletextpad=0,
                     loc = 4)
        for item in leg.legendHandles:
            item.set_visible(False)

    if save_fig == True:
        plt.savefig(fname=out_file)#, format='svg')

    plt.show()
# draw correlation plot with standard deviation
corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nMatern kernel',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig = False, out_file=dir_out + 'Matern_corr_plot.png')


### Plot the distribution of predicted values  ===> doesn't make sense, because GPs assume
# Gaussian distribution of the data!!
# fig, axs = plt.subplots(1, 2, tight_layout=True)
#
# axs[0].hist(mus, bins = 30)
# axs[1].hist(y_train, bins = 30)
# axs[0].title.set_text('Distribution of predictions')
# axs[1].title.set_text('Distribution of true KDs')
# plt.savefig(fname=dir_out + 'Matern_predDistribution.png', format='png')
# plt.show()



# create plot with the variants for the final validation

def corr_var_plot_highlighted(measured_train, predicted_train, var_train,
                            measured_test, predicted_test, var_test, legend=False,
                            R2=None, cor_coef=None, MSE=None, save_fig = False, out_file=None):

    # plot with highligted data points
    # set values
    x_test = measured_test
    x_train = measured_train

    y_pred_test = predicted_test
    y_pred_train = predicted_train

    std_test = 2*np.sqrt(var_test)
    std_train = 2*np.sqrt(var_train)

    x = np.concatenate((x_train, x_test))
    y_pred = np.concatenate((y_pred_train, y_pred_test))
    std = np.concatenate((std_train, std_test))


    # correlation line for all values
    par = np.polyfit(x_train, y_pred_train, 1, full=True)
    slope = par[0][0]
    intercept = par[0][1]

    # get coordinates of the standard devs
    y_corline = np.asarray([i*slope + intercept for i in x])
    std_pos = np.add(y_corline, np.abs(std))
    std_neg = np.subtract(y_corline, np.abs(std))

    # fit line to the stds and get y values of fit in a set range
    l_pos = np.polyfit(x, std_pos, 2)
    l_neg = np.polyfit(x, std_neg, 2)
    # set y and x axis
    x_lim = [-2.5, 2.5]
    y_lim = [-2.5, 2.5]
    # get x and y values
    x_area = np.append(x, x_lim[1])
    x_area = np.insert(x_area, 0, x_lim[0])
    l_pos_y = x_area ** 2 * l_pos[0] + x_area * l_pos[1] + l_pos[2]
    l_neg_y = x_area ** 2 * l_neg[0] + x_area * l_neg[1] + l_neg[2]

    # combine values in dataframe to sort according to x
    var_df = pd.DataFrame()
    var_df['x_area'] = x_area
    var_df['std_positive'] = l_pos_y
    var_df['std_negative'] = l_neg_y
    var_df.sort_values('x_area', inplace=True)

    # set up figure
    plt.figure('GP testset evaluation', figsize=(5, 5))

    plt.title('Correlation of measured and predicted KD values')
    plt.xlabel('measured')
    plt.ylabel('predicted')
    # set y and x axis
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.fill_between(var_df['x_area'], var_df['std_positive'], var_df['std_negative'],
                     alpha = 0.3 , interpolate=True, color = 'orange')

    # set color list for the std dev points
    cols = list(np.concatenate((np.repeat('b',len(x_train)),np.repeat('r',len(x_test)))))
    # make list for the novel variants (hide germline sequence)
    cols2 = list(np.concatenate((np.repeat('b', 3), np.repeat('r', 4), np.repeat('w', 1), np.repeat('y', 1), np.repeat('g', 2))))

    plt.scatter(x_train, y_pred_train, color='k')
    plt.scatter(x_test, y_pred_test, color='r')
    # plt.errorbar(x, y_pred, fmt='ko', yerr=std, alpha = 0.5)
    plt.scatter(x, std_pos, color=cols, s=4, marker = ".")
    plt.scatter(x, std_neg, color=cols, s=4, marker = ".")
    plt.plot(x_lim, [x_lim[0] * slope + intercept, x_lim[1] * slope + intercept], '-', color='k')

    # define legend
    if legend == True:
        l0 = "slope = {:.4f}".format(slope)
        l1 = "R2 = {:.4f}".format(R2)
        l2 = "Corr. coeff. = {:.4f}".format(cor_coef)
        l3 = "MSE = {:.4f}".format(MSE)

        leg = plt.legend(labels = [l0, l1, l2,l3], handlelength=0, handletextpad=0,
                     loc = 4)
        for item in leg.legendHandles:
            item.set_visible(False)

    if save_fig == True:
        plt.savefig(fname=out_file, format='png')

    plt.show()






# Load novel variants data
data_novvars = pd.read_csv(abs_path+"/data/final_validation/novel_variants_KD_vs_preds.csv")
measured_novvars = GP.normalize_test_train_set(data_novvars['KD value [nM]']).values
predicted_novvars = data_novvars['pred_normalized'].values
variances_novvars = data_novvars['variances'].values

corr_var_plot_highlighted(y_true, mus, vars,
                            measured_novvars, predicted_novvars, variances_novvars, legend=True,
                            R2=r2, cor_coef=cor_coef, MSE=MSE, save_fig = True, out_file=dir_out + 'Matern_corr_plot_newvarshighlighted.png')





# seaborn cluster map with dendogram
import seaborn as sns; sns.set(color_codes=True)

seqs = data_novvars['Sequences']
kd_vals = data_novvars['KD value [nM]']
seq_labels = ["{}, KD: {:.3f} nM".format(y, kd_vals[i]) for i, y in enumerate([" ".join(["tHC", str(x+1)]) for x in range(len(seqs))])]
out_dir = dir_out + 'Sequence_heatmap_novel_vars.png'


def sequence_heatmap(seqs, seq_labels, col_map="mako", save_fig=False, out_dir=None):
    '''
    Function to create a heatmap like plot for sequence comparison with included dendogram acc. to seaborn.clustermap().
    :param seqs: data series or array with the sequences of to plot
    :param seq_labels: labels of each sequence
    :param col_map: matplotlib colormap
    :param save_fig: bool, if figure should be saved
    :param out_dir: destination and file name, if save_fig == True
    :return:
    '''
    # convert sequences to 2-D array of numbers
    ls = []
    for seq in seqs:
        l = [ord(x) - 64 for x in seq]
        ls.append(np.asarray(l))
    ls = np.asarray(ls)

    ax = sns.clustermap(ls, figsize=(30, 5),
                        cbar_pos=None,
                        col_cluster=False,
                        cmap=col_map,
                        yticklabels=seq_labels,
                        xticklabels=2,
                        dendrogram_ratio=(.05, .0),
                        linewidths=.7, linecolor='black')

    # make dendogram lines thicker
    plt.yticks(rotation=0)
    for a in ax.ax_row_dendrogram.collections:
        a.set_linewidth(3)

    plt.tick_params(labelsize=15, labelbottom = False, bottom=False, top = True, labeltop=True)

    plt.show()

    if save_fig == True:
        plt.savefig(out_dir)




sequence_heatmap(seqs, seq_labels, col_map="mako", save_fig=False, out_dir=None)



