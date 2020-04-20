import pandas as pd
import os
from GP_implementation import GP_fcts as GP
import matplotlib.pyplot as plt

###### SET INPUT DIRECTORIES ######
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'

input_f_seq = input_dir + 'input_HCs.csv'



## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Plots/GP_model/CV_correlation/Maternlatest0420/'


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
GP.correlation_plot(y_true, mus, cor_line=False, save_fig = True, out_file=dir_out + 'Matern_corr_plot_simple.png')


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nMatern kernel',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig = True, out_file=dir_out + 'Matern_corr_plot.png')




### Plot the distribution of predicted values
fig, axs = plt.subplots(1, 2, tight_layout=True)

axs[0].hist(mus, bins = 30)
axs[1].hist(y_train, bins = 30)
axs[0].title.set_text('Distribution of predictions')
axs[1].title.set_text('Distribution of true KDs')
plt.savefig(fname=dir_out + 'Matern_predDistribution.png', format='png')
plt.show()