
import pandas as pd
import os
from GP_implementation import GP_fcts as GP


###### SET INPUT DIRECTORIES ######
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'

input_f_seq = input_dir + 'input_HCs.csv'



## SET OUTPUT DIRECTORIES (for plots to save)
dir_outLD = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Plots/GP_model/LD_kernel/'

# If the output directories do not exist, then create it
if not os.path.exists(dir_outLD):
    os.makedirs(dir_outLD)





###### LOAD DATA #######
data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])


#### DATA PROCESSING ####
# normalize data
data['KD_norm'] = GP.normalize_test_train_set(data['KD'])

X_train = data['Sequence'].values
y_train = data['KD_norm'].values



#### CROSS VALIDATION ####

# test inner cv loop for hyperparameter tuning in cv_param_tuning function
k = 35
mus, vars, y_true, prams_test = GP.cv_param_tuning(X_train, y_train, k, init_param=1)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig = False, out_file=None)


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nLD kernel',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig = False, out_file=None)
















