import pandas as pd
import os
from utils import GP_fcts as GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.model_selection import KFold

###### SET INPUT DIRECTORIES ######
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'

input_f_seq = input_dir + 'input_HCs.csv'



## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Plots/GP_model/sklearn/'
f_out = dir_out + 'skMaternCorPlot.png'

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

# one-hot encoding
X_OH = GP.one_hot_encode_matern(X)







###### GP implementation from sklearn with Matern #####

kernels = [Matern(), RBF()]
const_kernel = Matern() + ConstantKernel()

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

    gprMat = GaussianProcessRegressor(kernel = kernels[0],
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


GP.corr_var_plot(y_s, mu_s, vars_s, x_std=2, legend = True, method = "\nRBF kernel",
                  R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig = True, out_file=f_out)
















