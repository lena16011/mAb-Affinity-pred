"""
Script to implement Ridge regression with sklearn
Should serve as a baseline comparison to GPs;
Lena Erlach
09.05.2023
"""


import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from utils import GP_fcts as GP
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV, cross_validate



def nested_param_tuning_eval(X_OH, y, reg, param_grid, k=10, k_i=5, metrics=['neg_mean_squared_error']):

    ### LOO CV
    kf_o = KFold(n_splits=k, shuffle=True, random_state=1) # Define the n_split = number of folds
    kf_i = KFold(n_splits=k_i, shuffle=True, random_state=2) # Define the n_split = number of folds
    cycle=0

    # Define the grid search object - non-nested
    grid_search = GridSearchCV(estimator=reg,
                               param_grid=param_grid,
                               scoring= metrics,
                               cv=kf_o,
                               refit=metrics[0])

    # Non_nested parameter search and scoring
    grid_search.fit(X_OH, y)
    non_nested_best_params = grid_search.best_params_

    # combine scores to a dataframe
    non_nested_cv_df = pd.DataFrame()
    cols = [match for match in grid_search.cv_results_.keys() if "param_" in match or "mean_test" in match or "std_test" in match]
    for c in cols:
        non_nested_cv_df[c] = grid_search.cv_results_[c].data

    # print non-nested CV score
    print('K value: %d'% (k))
    print('non-Nested CV score: %0.3f (+/- %0.3f)' % (np.mean(grid_search.best_score_), np.std(grid_search.best_score_)))
    print('Best parameters:', grid_search.best_params_)

    # Perform nested cross-validation
    outer_scores = []
    best_params = []

    # Define the grid search object - non-nested
    grid_search = GridSearchCV(estimator=reg,
                               param_grid=param_grid,
                               scoring= metrics,
                               cv=kf_i,
                               refit=metrics[0])

    # Perform nested cross-validation
    for train_index, test_index in kf_o.split(X):
        # split in train and test set
        X_train, X_test = X_OH[train_index], X_OH[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid_search.fit(X_train, y_train)
        outer_scores.append(grid_search.score(X_test, y_test))
        best_params.append(grid_search.best_params_)
        cycle += 1

    # Print the results
    print('Nested CV score: %0.3f (+/- %0.3f)' % (np.mean(outer_scores), np.std(outer_scores)))
    print('Best parameters:', best_params)

    # combine scores to a dataframe & get best performing parameters
    nested_cv_df = pd.DataFrame(data ={"MSE": outer_scores, "best_params": best_params})

    return non_nested_cv_df, nested_cv_df











###### SET INPUT DIRECTORIES ######
input_dir = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/input'
input_f_seq = os.path.join(input_dir, 'input_HCs.csv')


## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/Plots/Ridge_reg'
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



###### MODEL EVALUATION WITH NESTED CROSS VALIDATION #####
###### RIDGE REGRESSION IMPLEMENTATION FROM SKLEARN WITHOUT KERNEL hyper parameter tuning #####

# parameter grid
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf', 'polynomial'],
    'degree': [2, 3, 4],
    'gamma': [0.1, 1.0, 10.0]
}
model_name = "KernelRidge"
reg = KernelRidge()
k = 10
k_l = [2, 3, 5, 10,20, 35]
non_nested_scores = []
nested_scores = []
non_nested_mins = []
non_nested_maxs = []

# evaluate different ks
for k in k_l:
    non_nested_cv_df, nested_cv_df = nested_param_tuning_eval(X_OH, y, reg, param_grid, k=k, k_i=5, metrics=['neg_mean_squared_error', 'r2'])
    non_nested_scores.append(np.mean(non_nested_cv_df.mean_test_neg_mean_squared_error))
    non_nested_mins.append(min(non_nested_cv_df.mean_test_neg_mean_squared_error))
    non_nested_maxs.append(max(non_nested_cv_df.mean_test_neg_mean_squared_error))
    nested_scores.append(np.mean(nested_cv_df.MSE))

print('Non-nested scores per k: {}'.format(non_nested_scores))
print('Nested scores per k: {}'.format(nested_scores))
print('Mean non-nested scores over k: {}'.format(np.mean(non_nested_scores)))
print('Mean nested scores over k: {}'.format(np.mean(nested_scores)))


# line plot of k mean values with min/max error bars
plt.errorbar(np.array(k_l), np.array(non_nested_scores), yerr=[np.array(non_nested_mins), np.array(non_nested_maxs)], fmt='o')
plt.scatter(np.array(k_l), np.array(nested_scores), c='k')
# plot the ideal case in a separate color
plt.plot(np.array(k_l), [np.array(non_nested_scores)[-1] for _ in range(len(np.array(k_l)))], color='r')
plt.title("Nested CV (black) and non-nested (blue) - \n %s - per k vs. LOOCV (red bar)" % model_name)
plt.xlabel('k fold')
plt.ylabel('MSE')

plt.savefig(fname = os.path.join(dir_out, 'k_sensitivity_nestedCV_%s.pdf' % model_name))
# show the plot
plt.show()


###### HYPERPARAMETER TUNING AND LOO-CV WITH BEST PERFORMING PARAMETERS #####














########################################################################
#### NEXT TASK!! WHICH MODEL AND HYPERPARAMETER TO TAKE?? with this then proceed to the model below!
########################################################################


# initialize lists to store the predictions
mu_s = []
y_s = []
params_test = []

### LOO CV
k = len(X_OH)
kf = KFold(n_splits=k, shuffle=True, random_state=1) # Define the n_split = number of folds
cycle_num = 0
kernel = 'linear'


# loop for the CV
for train_index, test_index in kf.split(X_OH):

    #split in train and test set
    X_train, X_test = X_OH[train_index], X_OH[test_index]
    y_train, y_test = y[train_index], y[test_index]

    reg = KernelRidge(kernel = 'linear',
                      alpha=1.0).fit(X_train,y_train)

    gprMatpred = reg.predict(X = X_test)

    mu_s.append(gprMatpred)
    y_s.append(y_test)
    params_test.append(reg.get_params)

    print(cycle_num)
    cycle_num += 1

y_s = np.array(y_s).flatten()
mu_s = np.array(mu_s).flatten()

r2, cor_coef, MSE= calc_print_scores(y_s, mu_s, k)


# GP.corr_var_plot(y_s, mu_s, x_std=2, legend = True, method = "\n" + kernel + " kernel",
#                   R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig = False, out_file=f_out)
#







####################################
# program as class!!
####################################
class KD_Ridge_reg:
    def __init__(self, ):




# try to return something like that
# initialize lists to store the predictions
mu_s = []
std_s = []
y_s = []
params_test = []


# to be able to use the plot
r2, cor_coef, MSE= GP.calc_print_scores(y_s, mu_s, k)

GP.corr_var_plot(y_s, mu_s, vars_s, x_std=2, legend = True, method = "\n" + kernel + " kernel",
                  R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig = False, out_file=f_out)




def main():



if __name__ == '__main__':
    main()


