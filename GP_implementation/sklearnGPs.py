import pandas as pd
import os
from utils import GP_fcts as GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold

###### SET INPUT DIRECTORIES ######
input_dir = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/input'

input_f_seq = os.path.join(input_dir, 'input_HCs.csv')



## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/Plots/GP_model/sklearn'

f_out = os.path.join(dir_out, 'skRBFCorPlot.pdf')

f_out_rand = os.path.join(dir_out, 'skRBFCorPlot_randomlabels.pdf')

# If the output directories do not exist, then create it
if not os.path.exists(dir_out):
    os.makedirs(dir_out)



###### LOAD DATA #######
data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])


#### DATA PROCESSING ####
# normalize data
data['KD_norm'] = GP.normalize_test_train_set(data['KD'])

X = data['Sequence'].values
y = data['KD_norm'].values
# random labels
y_rand = data['KD_norm'].sample(frac=1).reset_index(drop = True).values

# one-hot encoding
X_OH = GP.one_hot_encode_matern(X)





###### GP implementation from sklearn with Matern #####
kernels = {'Matern': Matern(), 'RBF': RBF(),
           'M+W': Matern() + WhiteKernel(noise_level=1)} # doesn't really change compared to Matern
kernel = 'M+W'
### LOO CV
k = 35
kf = KFold(n_splits=k) # Define the n_split = number of folds

# initialize lists to store the predictions
mu_s = []
std_s = []
y_s = []
params_test = []
cycle_num = 1

# loop for the CV
for train_index, test_index in kf.split(X_OH):

    #split in train and test set
    X_train, X_test = X_OH[train_index], X_OH[test_index]
    y_train, y_test = y[train_index], y[test_index]

    gprMat = GaussianProcessRegressor(kernel = kernels[kernel],
                        n_restarts_optimizer=300, normalize_y=False, random_state=120).fit(X_train,y_train)

    gprMatpred = gprMat.predict(X = X_test, return_std = True)

    mu_s.append(gprMatpred[0][0])
    std_s.append(gprMatpred[1][0])
    y_s.append(y_test[0])
    params_test.append(gprMat.get_params)

    print(cycle_num)
    cycle_num += 1

vars_s = [x**2 for x in std_s]


r2, cor_coef, MSE= GP.calc_print_scores(y_s, mu_s, k)


GP.corr_var_plot(y_s, mu_s, vars_s, x_std=2, legend = True, method = "\n" + kernel + " kernel",
                  R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig = False, out_file=f_out)






###### GP implementation from sklearn with Matern RANDOMIZED LABELS #####

# initialize lists to store the predictions
mu_s = []
std_s = []
y_s = []
params_test = []
cycle_num = 1

# loop for the CV
for train_index, test_index in kf.split(X_OH):

    #split in train and test set
    X_train, X_test = X_OH[train_index], X_OH[test_index]
    y_train, y_test = y_rand[train_index], y_rand[test_index]

    gprMat = GaussianProcessRegressor(kernel = kernels[kernel],
                        n_restarts_optimizer=300, normalize_y=False, random_state=120).fit(X_train,y_train)

    gprMatpred = gprMat.predict(X = X_test, return_std = True)

    mu_s.append(gprMatpred[0][0])
    std_s.append(gprMatpred[1][0])
    y_s.append(y_test[0])
    params_test.append(gprMat.get_params)

    print(cycle_num)
    cycle_num += 1

vars_s = [x**2 for x in std_s]


r2, cor_coef, MSE= GP.calc_print_scores(y_s, mu_s, k)


GP.corr_var_plot(y_s, mu_s, vars_s, x_std=2, legend = True, method = "\n" + kernel + " kernel - randomized labels",
                  R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig = True, out_file=f_out_rand)



