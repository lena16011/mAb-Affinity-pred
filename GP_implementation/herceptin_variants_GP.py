
import os
import numpy as np
import pandas as pd
import stringdist as sd
from sklearn.model_selection import train_test_split, KFold
from scipy import optimize, linalg
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from GP_implementation import GP_fcts as GP



## SET INPUT DIRECTORIES ####
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/herceptin/'
input_f = input_dir + 'herceptin.csv'


## SET OUTPUT DIRECTORIES (for plots to save)
dir_outSubMa = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/herceptin/Plots/substitution_matrices/'
dir_outLD = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/herceptin/Plots/LD/'
dir_outMa = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/herceptin/Plots/Matern/'

dirs = [dir_outSubMa, dir_outLD, dir_outMa]

# If the output directories do not exist, then create it
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)



#### LOAD INPUT ####
# Load sequence data
data = pd.read_csv(input_f, usecols=['Variant', 'Sequence', 'KD'])

# Load KD values and add them together to a new dataframe (remove samples that w/o measured KD value)

KDs_unnormalized = data['KD']



#### DATA PROCESSING ####

# normalize data
data['KD'] = GP.normalize_test_train_set(data['KD'])

# split into train and test data
# X_train, X_test, y_train, y_test = GP.split_data(data, 5, r_state=123)

# specify X_train, y_train
X_train = data['Sequence'].values
y_train = data['KD'].values




######## LD KERNEL #########
#### CROSS VALIDATION

# test inner cv loop for hyperparameter tuning
k = 30
mus, vars, y_true, prams_test = GP.cv_param_tuning(X_train, y_train, k, init_param=1)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig=False)
                    # , out_file =str(dir_outLD+'LD_corr_CV_simple.png'))


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nLD kernel ',
                 R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=False) #, out_file=str(dir_outLD+'LD_corr_variance_CV.png'))



################################### NECESSARY ??? #########################################
#### TEST TEST SET ####

# get optimal noise parameter
opt_param = GP.get_params(X_train, y_train)

# predict the test set and training set
mu_test, var_test = GP.predict_GP(X_train, y_train, X_test, opt_param)
mu_train, var_train = GP.predict_GP(X_train, y_train, X_train, opt_param)

# merge data set
predicted = np.concatenate((mu_train, mu_test))
measured = np.concatenate((y_train, y_test))
vars = np.concatenate((var_train, var_test))

# calculate scoring values
r2, cor_coef, MSE = GP.calc_print_scores(measured, predicted)

# draw simple correlation plot
GP.correlation_plot(measured, predicted, cor_line=False)

print("Test set: ")
for m, p in zip(y_test, mu_test):
    print("measured value: {0:.2f} and predicted value: {1:.2f}".format(m, p))


print("Training set: ")
for m, p in zip(y_train, mu_train):
    print("measured value: {0:.2f} and predicted value: {1:.2f}".format(m, p))

# correlation plot of training data
GP.corr_var_plot(y_train, mu_train, var_train, x_std=2, legend=True,
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=True,
                 out_file=str(dir_out+'LD_corr_variance_trainingset.png'))

# correlation plot of test data
GP.corr_var_plot(y_test, mu_test, var_test, x_std=2, legend=True,
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=True,
                 out_file=str(dir_out+'LD_corr_variance_testset.png'))

# correlation plot of both data sets
GP.corr_var_plot_highlighted(y_train, mu_train, var_train,  y_test, mu_test, var_test,
                        legend=True, R2=r2,cor_coef=cor_coef, MSE=MSE, save_fig=True,
                 out_file=str(dir_out+'LD_corr_variance_highlighted.png'))

##############################################################################################################







####### MATERN KERNEL #########


# one-hot encode sequences
X_trainOH = GP.one_hot_encode_matern(X_train)
# X_testOH = GP.one_hot_encode_matern(X_test)


#### CROSS VALIDATION
# test inner cv loop for hyperparameter tuning
k = 30
mus, vars, y_true, prams = GP.cv_param_tuning_mat(X_trainOH, y_train, k, init_param=(1,50))

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig=False) #, out_file =str(dir_outMa+'Ma_corr_CV_simple.png'))

# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nCMatern kernel ',
              R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=False) #, out_file=str(dir_outMa+'Ma_corr_variance_CV.png'))

### hyperparameter just changes minor






#BLOSUM45

## CROSS VALIDATION ####

# test inner cv loop for hyperparameter tuning
k = 25
mus, vars, y_true, prams_test = GP.cv_param_tuning_CDRd45(X_train, y_train, k)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig=True, out_file =str(dir_outSubMa+'B45_corr_CV_simple.png'))


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nCDRdist BLOSUM45 ',
                 R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=True, out_file=str(dir_outSubMa+'B45_corr_variance_CV.png'))



#BLOSUM62

## CROSS VALIDATION ####

# test inner cv loop for hyperparameter tuning
k = 25
mus, vars, y_true, prams_test = GP.cv_param_tuning_CDRd62(X_train, y_train, k)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig=True, out_file =str(dir_outSubMa+'B62_corr_CV_simple.png'))


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True,method = '\nCDRdist BLOSUM62 ',
                 R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=True, out_file=str(dir_outSubMa+'B62_corr_variance_CV.png'))




# PAM40

## CROSS VALIDATION ####

# test inner cv loop for hyperparameter tuning
k = 25
mus, vars, y_true, prams_test = GP.cv_param_tuning_CDRdPAM40(X_train, y_train, k)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False, save_fig=True, out_file =str(dir_outSubMa+'P40_corr_CV_simple.png'))


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True, method = '\nCDRdist PAM40 ',
                 R2=r2, corr_coef=cor_coef, MSE = MSE, save_fig=True, out_file=str(dir_outSubMa+'P40_corr_variance_CV.png'))












