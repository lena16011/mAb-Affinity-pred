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
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


class regression_model_evaluation:
    def __init__(self, X_OH, y, reg, model_name, metrics):
        self.X_OH = X_OH
        self.y = y
        self.reg = reg
        self.model_name = model_name
        self.metrics = metrics
        self.k_l = None
        self.param_grid = None
        self.k_i = 3


    def nested_param_tuning_eval(self, param_grid, k_o=10, k_i=5,
                                 verbose=0):
        self.param_grid = param_grid
        # k_o = 16
        # k_i = 5
        # metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
        ### LOO CV
        kf_o = KFold(n_splits=k_o, shuffle=True, random_state=1)  # Define the n_split = number of folds
        kf_i = KFold(n_splits=k_i, shuffle=True, random_state=2)  # Define the n_split = number of folds

        # Define the grid search object - non-nested
        grid_search_nn = GridSearchCV(estimator=self.reg,
                                      param_grid=param_grid,
                                      scoring=self.metrics,
                                      cv=kf_o,
                                      refit=list(self.metrics.keys())[0])

        # Non_nested parameter search and scoring
        grid_search_nn.fit(self.X_OH, self.y)

        # combine non-nested scores to dataframe (and add to the summary scores
        non_nested_cv_df = pd.DataFrame()
        cols = [match for match in grid_search_nn.cv_results_.keys() if "param_" in match or "mean_test" in match or "std_test" in match]
        for c in cols:
            non_nested_cv_df[c] = grid_search_nn.cv_results_[c].data
        # add params
        non_nested_cv_df['params'] = grid_search_nn.cv_results_['params']

        # adjust negative scores (like neg. MSE)
        grid_search_nn.best_score_ = -1*grid_search_nn.best_score_
        cols = [match for match in non_nested_cv_df.columns if "mean_test" in match]
        for c in cols:
            if all(non_nested_cv_df[c] < 0):
                non_nested_cv_df[c] = non_nested_cv_df[c]*-1

        best_score_n = "mean_test_"+list(self.metrics.keys())[0]
        second_score = "mean_test_"+list(self.metrics.keys())[1]
        if verbose > 0:
            # print non-nested CV score
            print('K value: %d' % (k_o))
            print('non-Nested CV best mse score: %0.3f' % grid_search_nn.best_score_)
            #print(np.unique(grid_search_nn.cv_results_[second_score][grid_search_nn.cv_results_[best_score_n] == -1*grid_search_nn.best_score_]))
            print('non-Nested CV best r2 score: %0.3f' % (np.unique(grid_search_nn.cv_results_[second_score][grid_search_nn.cv_results_[best_score_n] == -1*grid_search_nn.best_score_])))
            print('----------------')


        # Perform nested cross-validation
        mse_scores = []
        r2_scores = []
        best_params = []

        # Define the grid search object - non-nested
        grid_search = GridSearchCV(estimator=self.reg,
                                   param_grid=param_grid,
                                   scoring=self.metrics,
                                   cv=kf_i,
                                   refit=list(self.metrics.keys())[0])

        # Perform nested cross-validation
        for train_index, test_index in kf_o.split(self.X_OH):
            # split in train and test set
            X_train, X_test = self.X_OH[train_index], self.X_OH[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # perform inner CV (param tuning) and predict on test data with the best model
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)

            mse_scores.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
            best_params.append(grid_search.best_params_)


        if verbose > 0:
            # Print the results
            print('Nested CV mse score mean: %0.3f (+/- %0.3f)' % (np.mean(mse_scores), np.std(mse_scores)))
            print('Nested CV r2 score mean: %0.3f (+/- %0.3f)' % (np.mean(r2_scores), np.std(r2_scores)))
            print('Best parameters:', best_params)
            print('----------------------------------\n')
        # combine scores to a dataframe & get best performing parameters
        nested_cv_df = pd.DataFrame()
        scores = [mse_scores, r2_scores]
        for i, c in enumerate(cols):
            nested_cv_df[c] = scores[i]
        nested_cv_df["best_params"] = best_params

        # summarize and save the scores per model (nested and non-nested)
        b_score = min(non_nested_cv_df.mean_test_neg_MSE)

        # summarize in dataframe
        scores_df = pd.DataFrame()
        scores_df["model_name"] = [self.model_name]
        scores_df["MSE_nonnested"] = [b_score]
        scores_df["R2_nonnested"] = non_nested_cv_df["mean_test_r2"][non_nested_cv_df.mean_test_neg_MSE == b_score].values[0]
        scores_df["best_params_nonnested"] = non_nested_cv_df["params"][non_nested_cv_df.mean_test_neg_MSE == b_score]
        scores_df["MSE_nested"] = np.mean(nested_cv_df.mean_test_neg_MSE)
        scores_df["R2_nested"] = np.mean(nested_cv_df.mean_test_r2)
        scores_df["k_outer"] = [k_outer]
        scores_df["k_inner"] = [5]
        nested_cv_df.best_params = nested_cv_df.best_params.astype("string")
        scores_df["mostselected_params_nested"] = nested_cv_df.best_params[max(nested_cv_df.best_params.value_counts())]

        return non_nested_cv_df, nested_cv_df, scores_df


    def evaluate_k(self, param_grid, k_in, k_l, verbose=0):
        self.k_l = k_l
        self.param_grid = param_grid
        if k_in != self.k_i:
            self.k_i = k_in

        # initialize scores lists
        non_nested_scores = []
        nested_scores = []
        non_nested_mins = []
        non_nested_maxs = []
        # evaluate different ks
        for k_s in self.k_l:
            # run nested CV loops for different ks
            non_nested_cv_df, nested_cv_df, scores_df = self.nested_param_tuning_eval(self.param_grid, k_o=k_s, k_i = self.k_i, verbose=verbose)

            # convert neg_meansquared error to mean_squared error in non-nested df
            non_nested_cv_df.mean_test_neg_MSE = non_nested_cv_df.mean_test_neg_MSE * -1
            non_nested_scores.append(np.mean(non_nested_cv_df.mean_test_neg_MSE))
            non_nested_mins.append(min(non_nested_cv_df.mean_test_neg_MSE))
            non_nested_maxs.append(max(non_nested_cv_df.mean_test_neg_MSE))
            nested_scores.append(np.mean(nested_cv_df.mean_test_neg_MSE))

        print('Non-nested scores per k: {}'.format(non_nested_scores))
        print('Nested scores per k: {}'.format(nested_scores))
        print('Mean non-nested scores over k: {}, std: {}'.format(np.mean(non_nested_scores), np.std(non_nested_scores)))
        print('Mean nested scores over k: {}, std: {}'.format(np.mean(nested_scores), np.std(nested_scores)))


def k_eval_plot(non_nested_scores, non_nested_mins, non_nested_maxs, nested_scores, k_l, model_name, save_fig=False, save_path=False):
    # line plot of k mean values with min/max error bars
    plt.errorbar(np.array(k_l), np.array(non_nested_scores),
                 yerr=[np.array(non_nested_mins), np.array(non_nested_maxs)], fmt='o')
    plt.scatter(np.array(k_l), np.array(nested_scores), c='k')
    # plot the ideal case in a separate color
    plt.plot(np.array(k_l), [np.array(non_nested_scores)[-1] for _ in range(len(np.array(k_l)))], color='r')
    plt.title("Nested CV (black) and non-nested (blue) - \n %s - per k vs. LOOCV (red bar)" % model_name)
    plt.xlabel('k fold')
    plt.ylabel('MSE')

    if save_fig == True:
        plt.savefig(fname=save_path)
    # show the plot
    plt.show()
    plt.close()






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
###### RIDGE REGRESSION IMPLEMENTATION FROM SKLEARN hyper parameter tuning #####

# parameter grid
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf', 'polynomial'],
    'degree': [2, 3, 4],
    'gamma': [0.1, 1.0, 10.0]
}
model_name = "KernelRidge"
reg = KernelRidge()
k_outer=5
k_l = list(range(2,31,1))
k_l.append(len(X_OH))
metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}

# #########################################################################################################
# ###### EVALUATION OF Ks; NESTED CV #####
# #########################################################################################################

# define class model
kernel = regression_model_evaluation(X_OH, y, reg, model_name, metrics)


######### DEBUG THAT FROM HERE ON WHERE THE 2 RED DOTS ARE; MORE VALUES RETURNED!
# # evaluate k for CV
kernel.evaluate_k(param_grid, k_in=3, k_l=k_l, verbose=0)

# run nested CV with set k values
non_nested_cv_df, nested_cv_df, scores_df = kernel.nested_param_tuning_eval(param_grid, k_o=k_outer, k_i=5, verbose=1)
# save scores dataframe
scores_df.to_csv(os.path.join('/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation', model_name+'_CV_scores.csv'))


# k_eval
# k_eval_plot(non_nested_scores, non_nested_mins, non_nested_maxs, nested_scores, k_l, model_name, save_fig = False)

# #########################################################################################################




###### HYPERPARAMETER TUNING AND LOO-CV WITH BEST PERFORMING PARAMETERS #####
### LOO CV
k=35

kf = KFold(n_splits=k, shuffle=True, random_state=1) # Define the n_split = number of folds

# Define the grid search object - non-nested
grid_search = GridSearchCV(estimator=reg,
                           param_grid=param_grid,
                           scoring= metrics,
                           cv=kf,
                           refit=list(metrics.keys())[0])

# Non_nested parameter search and scoring
grid_search.fit(X_OH, y)
best_model = grid_search.best_estimator_
best_score = -grid_search.best_score_
print('Best model: ', best_model)
print('Best score: ', best_score)


# Obtain the predicted values using cross-validation
y_pred = cross_val_predict(best_model, X_OH, y, cv=kf, verbose=1)

r2, cor_coef, MSE= GP.calc_print_scores(y, y_pred, k)

GP.corr_var_plot(y, y_pred, vars=False, x_std=2, legend = True, method = "\n" + model_name + " regression",
                  R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig = True, out_file=os.path.join(dir_out, 'KernelRidge_corr_plot.pdf'))



########################################################################
#### NEXT TASK!! WHICH MODEL AND HYPERPARAMETER TO TAKE?? with this then proceed to the model below!
########################################################################









####################################
# program as class!!
####################################


def main():



if __name__ == '__main__':
    main()


