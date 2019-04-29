import random
import numpy as np
import pandas as pd
import stringdist as sd
from sklearn.model_selection import train_test_split, KFold
from scipy import optimize, linalg
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from GP_implementation import GP_fcts as GP




#### LOAD INPUT ####
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'
input_f_seq = input_dir + 'Final_49_AA_from_geneious.csv'
input_f_KD = input_dir + 'HC_KDvals.csv'

# Load sequence data
df_seq = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence'])

# Load KD values and add them together to a new dataframe (remove samples that w/o measured KD value)
KDs = pd.read_csv(input_f_KD, usecols=['SampleID', 'KD'], sep = ';')
data = pd.merge(df_seq,KDs, on='SampleID')

#### DATA PROCESSING ####

# normalize data
data['KD'] = GP.normalize_test_train_set(data['KD'])

# split into train and test data
X_train, X_test, y_train, y_test = GP.split_data(data, 5, r_state=123)


#### CROSS VALIDATION ####

# test inner cv loop for hyperparameter tuning
k = 30
mus, vars, y_true, prams_test = GP.cv_param_tuning(X_train, y_train, k)

# calculate and print scores
r2, cor_coef, MSE= GP.calc_print_scores(y_true, mus, k)

# draw simple correlation plot
GP.correlation_plot(y_true, mus, cor_line=False)


# draw correlation plot with standard deviation
GP.corr_var_plot(y_true, mus, vars, x_std=2, legend=True,
              R2=r2, corr_coef=cor_coef, MSE = MSE)




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
              R2=r2, corr_coef=cor_coef, MSE = MSE)

# correlation plot of test data
GP.corr_var_plot(y_test, mu_test, var_test, x_std=2, legend=True,
              R2=r2, corr_coef=cor_coef, MSE = MSE)

# correlation plot of both data sets
GP.corr_var_plot_highlighted(y_train, mu_train, var_train,  y_test, mu_test, var_test,
                        legend=True, R2=r2,cor_coef=cor_coef, MSE=MSE)




























###################### Calculation of the correlation coefficient
# according to gihub code of Bedbrook

### https://github.com/fhalab/channels/blob/master/regression/GP_matern_5_2_kernel.ipynb ###

# fit a linear function to the data points (least squares)
par = np.polyfit(measured, predicted, 1, full=True)
slope=par[0][0]
intercept=par[0][1]

variance = np.var(predicted)
residuals = np.var([(slope*xx + intercept - yy) for xx, yy in zip(measured, predicted)])
Rsqr = np.round(1-residuals/variance, decimals=2)
######################################################



