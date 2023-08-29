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
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings, random
# Suppress the warning from sklearn.metrics module
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)#, module='sklearn.metrics')

def k_eval_plot(non_nested_scores: list, non_nested_mins: list, non_nested_maxs: list,
                nested_scores: list, k_l: list, model_name: str, save_fig: list=False, save_path=False):
    """ Plot of CV scores (nested/non-nested) across choice of k

                        Args:
                            non_nested_scores: list of scores in the non-nested (parameter tuning) CV
                            non_nested_mins: list of min scores in the non-nested (parameter tuning) CV
                            non_nested_maxs: list of max scores in the non-nested (parameter tuning) CV
                            nested_scores: list of scores in the nested/outer (after inner parameter tuning) CV
                            k_l: list of ints
                                list of ks to be tested in k-fold in outer CV (model evaluation);
                            model_name: str
                                name of the model type as a prefix for saving files/plots;
                            save_fig: boolean default=False
                                if True, plot will be saved in save_path;
                            save_path: str or os.Path default=False
                                if save_path is a string or a os.Path, the plot will be saved in the indicated location;
                                if False, ignored;


                           Returns:
                               plt: matplotlib figure of the plot
                           """

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
    return plt


class regression_model_evaluation:
    """ Evaluation of regression models with nested cross-validation based on sklearn;

        This class can be defned with various models, which will be evaluated for:
         - the value of k in nested CV; will output a plot with nested Cross validation per k
         - outputs/saves nested and non-nested (within param-tuning) accuracy metrics; currently the
         class works only with 2 max. metrics for regression (in the best case, MSE as the main metric
         and R2 value as secondary metric (the metric isn't well defined for < 2 test datapoints, so the
         model will be fit on MSE.


        Attributes:
            X_OH: numpy.ndarray of shape (n_samples, p_length * 20)
                protein sequence data, one-hot encoded;
            y: numpy.ndarray of shape (n_samples, 1)
                log-standardized kD value to be predicted from the protein sequences;
            reg: instance of a sklearn regressor
                sklearn estimator for regression; e.g. reg = sklearn.kernel_ridge.KernelRidge() with or without parameters
            model_name: str
                name of the model type as a prefix for saving files/plots;
            metrics: dict
                metric names (str) as keys and strings as values of the corresponding sklearn metrics for fitting the models;
                currently up to 2 metrics are supported in this class; the first one in the dictionary will be used as the main
                metric for refitting the model; the second will be only reported; optimized for MSE and R2 value;
                for default set to: {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
            param_grid: dict
                parameters names (str) as keys and lists of parameter settings to try as values in the parameter tuning of the model
            k_i: int, default 3;
                number of k-folds for the inner nested CV (parameter tuning)
            k_eval_plot: boolean
                if True, a plot for the k evaluation will be generated
        """

    def __init__(self, X_OH: np.ndarray, y: np.ndarray, reg, model_name: str, metrics: dict):
        """Initializes the instance based on regression model, data, model name and metrics.
        """
        self.X_OH = X_OH
        self.y = y
        self.reg = reg
        self.model_name = model_name
        self.metrics = metrics
        self.k_l = None
        self.param_grid = None
        self.k_i = 3
        self.k_eval_plot = None
        self.y_pred = None
        self.vars = None
        self.scaler = StandardScaler()
        self.best_model = None # best model from non-nested CV
        self.best_score = None # score from best model from non-nested CV

    def nested_param_tuning_eval(self, param_grid: dict, k_o: int =10, k_i: int =5,
                                 n_jobs : int = 1, verbose: int =0,):
        """ Implementation of nested CV andparameter tuning

                Performs nested CV and returns outer (non-nested) and nested (outer and inner, separately) and summarized
                scores (defined by metrics dict); train/test data are split within an inner/outer folds (defined by k_o, k_i);
                param_grid defines the parameter grid to be tested in GridSearchCV from sklearn;

                Args:
                    self.X_OH, self.y, self.reg, self.metrics
                        additional arguments inherited from class instantiation;
                   param_grid: dict
                        parameters names (str) as keys and lists of parameter settings to try as values in the parameter
                        tuning of the model
                    k_o: int
                        number of k-folds in outer CV (model evaluation); the k_o also used for the non-nested CV;
                    k_i: int
                        number of k-folds in inner CV (parameter tuning)


                Returns:
                    non_nested_cv_df: pd.DataFrame
                        non-nested {metrics} scores with all the different parameter combinations tested, best score and
                        best parameter combination will be in the scores_df summarized;
                    nested_cv_df: pd.DataFrame
                        scores from the outer nested CV; average of the reported scores in that dataframe will bein the
                        scores_df summarized; is the summarized score, since these models were optimized in the inner CV)
                    scores_df: pd.DataFrame
                        summary of the nested (average score over outer CV scores) and non-nested scores (best score in
                        parameter tuning CV);
                """

        self.param_grid = param_grid
        # k_o = 16
        # k_i = 5
        # metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
        ### LOO CV
        kf_o = KFold(n_splits=k_o, shuffle=True, random_state=1)  # Define the n_split = number of folds
        kf_i = KFold(n_splits=k_i, shuffle=True, random_state=2)  # Define the n_split = number of folds

        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('regressor', self.reg)
        ])
        # Define the grid search object - non-nested
        grid_search_nn = GridSearchCV(estimator=pipeline,  ######## ADJUSTED FOR PIPELINE
                                      param_grid=param_grid,
                                      scoring=self.metrics,
                                      cv=kf_o,
                                      refit=list(self.metrics.keys())[0],
                                      n_jobs=n_jobs)

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
        b_score_nn = grid_search_nn.best_score_
        best_params_nn = str(grid_search_nn.best_params_)

        if verbose > 0:
            # print non-nested CV score
            print('K value: %d' % (k_o))
            print('non-Nested CV best mse score: %0.3f' % grid_search_nn.best_score_)
            print('non-Nested CV best params: %s' % best_params_nn)
            print('non-Nested CV best r2 score: %0.3f' % np.unique(grid_search_nn.cv_results_[second_score][grid_search_nn.cv_results_[best_score_n] == -1*b_score_nn])[0])
            print('----------------')


        # Perform nested cross-validation
        mse_scores = []
        r2_scores = []
        best_params = []

        # Define the grid search object - non-nested
        grid_search = GridSearchCV(estimator=pipeline,
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

        # summarize in dataframe
        scores_df = pd.DataFrame()
        scores_df["model_name"] = pd.Series(self.model_name)
        scores_df["MSE_nonnested"] = pd.Series(b_score_nn)
        scores_df["R2_nonnested"] = non_nested_cv_df[second_score][non_nested_cv_df[best_score_n] == b_score_nn].values[0]
        scores_df["best_params_nonnested"] = pd.Series(best_params_nn)
        scores_df["MSE_nested"] = np.mean(nested_cv_df.mean_test_neg_MSE)
        scores_df["R2_nested"] = np.mean(nested_cv_df.mean_test_r2)
        scores_df["k_outer"] = pd.Series(k_o)
        scores_df["k_inner"] = pd.Series(k_i)
        nested_cv_df.best_params = nested_cv_df.best_params.astype("string")
        #nested_cv_df.best_params[np.argmax(nested_cv_df.best_params.value_counts())]
        scores_df["mostselected_params_nested"] = nested_cv_df.best_params[np.argmax(nested_cv_df.best_params.value_counts())]

        return non_nested_cv_df, nested_cv_df, scores_df


    def evaluate_k(self, param_grid: dict, k_in: int, k_l: list, plot: bool = True,
                   n_jobs : int = 1, save_fig: bool=False, save_path=False, verbose: bool=0):
        """ Evaluation of ks for consitency of CV scoring across choice of k

                Performs nested CV for a list of k valuse and summarizes scores (defined by metrics dict);

                    Args:
                        self.X_OH, self.y, self.reg, self.metrics
                            additional arguments inherited from class instantiation;
                        param_grid: dict
                            parameters names (str) as keys and lists of parameter settings to try as values in the parameter
                            tuning of the model
                        k_l: list of ints
                            list of ks to be tested in k-fold in outer CV (model evaluation);
                        k_in: int
                            number of k-folds in inner CV (parameter tuning)
                        plot: boolean default=True
                            if True, plot will be returend to visualize nested/non-nested scores vs. k-values and
                            can be saved;
                        save_fig: boolean default=False
                            if True, plot will be saved in save_path;
                        save_path: str or os.Path default=False
                            if save_path is a string or a os.Path, the plot will be saved in the indicated location;
                            if False, ignored;


                       Returns:
                           non_nested_cv_df: pd.DataFrame
                               non-nested {metrics} scores with all the different parameter combinations tested, best score and
                               best parameter combination will be in the scores_df summarized;
                           nested_cv_df: pd.DataFrame
                               scores from the outer nested CV; average of the reported scores in that dataframe will bein the
                               scores_df summarized; is the summarized score, since these models were optimized in the inner CV)
                           scores_df: pd.DataFrame
                               summary of the nested (average score over outer CV scores) and non-nested scores (best score in
                               parameter tuning CV);
                       """
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
            non_nested_cv_df, nested_cv_df, scores_df = self.nested_param_tuning_eval(self.param_grid, n_jobs = n_jobs,
                                                                                      k_o=k_s, k_i = self.k_i, verbose=verbose)

            # convert neg_meansquared error to mean_squared error in non-nested df
            non_nested_cv_df.mean_test_neg_MSE = non_nested_cv_df.mean_test_neg_MSE
            non_nested_scores.append(np.mean(non_nested_cv_df.mean_test_neg_MSE))
            non_nested_mins.append(min(non_nested_cv_df.mean_test_neg_MSE))
            non_nested_maxs.append(max(non_nested_cv_df.mean_test_neg_MSE))
            nested_scores.append(np.mean(nested_cv_df.mean_test_neg_MSE))

        print('Non-nested scores per k: {}'.format(non_nested_scores))
        print('Nested scores per k: {}'.format(nested_scores))
        print('Mean non-nested scores over k: {}, std: {}'.format(np.mean(non_nested_scores), np.std(non_nested_scores)))
        print('Mean nested scores over k: {}, std: {}'.format(np.mean(nested_scores), np.std(nested_scores)))

        # plot the results
        if plot == True:
            self.k_eval_plot = k_eval_plot(non_nested_scores, non_nested_mins, non_nested_maxs, nested_scores, k_l, self.model_name,
                                save_fig=save_fig, save_path=save_path)

            self.k_eval_plot.show()
            self.k_eval_plot.close()

    def k_CV_and_plot(self, param_grid: dict, k: int, plot: bool = True, x_lim=[-0.2,2.5], y_lim=[-0.2,2.5], x_std = 2,
                      save_fig: bool=False, w_vars = False, save_path=None) -> pd.DataFrame:

        kf = KFold(n_splits=k, shuffle=True, random_state=1)  # Define the n_split = number of folds

        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('regressor', self.reg)
        ])

        # Define the grid search object - non-nested
        grid_search = GridSearchCV(estimator= pipeline,
                                   param_grid=param_grid,
                                   scoring=self.metrics,
                                   cv=kf,
                                   refit=list(self.metrics.keys())[0])

        # Non_nested parameter search and scoring
        grid_search.fit(self.X_OH, self.y)
        self.best_model = grid_search.best_estimator_
        self.best_score = -grid_search.best_score_
        print('Best model: ', self.best_model)
        print('Best score: ', self.best_score)

        # Obtain the predicted values using cross-validation
        self.y_pred = cross_val_predict(self.best_model, self.X_OH, self.y, cv=kf, verbose=1)
        # reshape
        self.y = self.y.reshape(-1, )
        self.y_pred = self.y_pred.reshape(-1, )

        if w_vars is True:
            # compute std separately
            self.vars = np.empty_like(self.y_pred)
            for i, (train_index, test_index) in enumerate(kf.split(self.X_OH)):
                self.best_model.fit(self.X_OH[train_index], self.y[train_index])
                _, self.vars[test_index] = self.best_model.predict(self.X_OH[test_index], return_std=True)

            # square for returning the variance
            self.vars = self.vars.reshape(-1, )**2


        elif w_vars is False:
            self.vars = False


        # print scores
        r2, cor_coef, MSE = GP.calc_print_scores(self.y, self.y_pred, k)

        # plot the results
        if plot == True:
            GP.corr_var_plot(self.y, self.y_pred, vars=self.vars, x_std=x_std,
                             legend=True, method="\n" + self.model_name + " regression", x_lim=x_lim, y_lim=y_lim,
                             R2=r2, corr_coef=cor_coef, MSE=MSE, save_fig=save_fig, out_file=save_path)

        # summarize in df
        model_score_df = pd.DataFrame(data = {'Model': [self.model_name], 'R2': [r2], 'Corr_coef': [cor_coef],
                                              'MSE': [MSE], 'params': [str(self.best_model)]}, index=[0])

        return model_score_df




def run():
    # #########################################################################################################
    # ###### SET MODEL AND PARAMETERS #####
    random.seed(123)
    randomized = False # set true to run the models with randomized labels for evaluation
    log_transform = True # test if the models predict differently

    k_inner = 10
    k_outer = 10
    verbose = 1
    n_jobs = -1

    # model names for saving files etc.
    model_names = ["GaussianProcess_RBF", "GaussianProcess_Matern",
                  "KernelRidge" , "RandomForestRegression", "OrdinalLinearRegression"
                   ]
    # list of parameters to test per model (take care of order!)
    param_list = [
                  {'regressor__kernel': [RBF(l) for l in np.logspace(-1, 1, 3)],
                   'regressor__alpha': [1e-10, 1e-3, 0.1]},
                  {'regressor__kernel': [Matern(l) for l in np.logspace(-1, 1, 3)],
                   'regressor__alpha': [1e-10, 1e-3, 0.1]},
                  {'regressor__alpha': [1e-3, 0.1, 1.0, 10.0],
                   'regressor__kernel': ['linear', 'rbf', 'polynomial'],
                   'regressor__degree': [2, 3],
                   'regressor__gamma': [0.1, 1.0, 10.0]},
                  {'regressor__n_estimators': [10, 100, 200],
                   'regressor__max_depth': [2, 5, 10]},
                  {'regressor__fit_intercept': [True, False]}
                  ]
    # set True for GPs, False for other models; (again order!)
    vars_list = [True, True, False, False, False]

    model_list = [
        GaussianProcessRegressor(random_state=1),
                  GaussianProcessRegressor(random_state=1),
                  KernelRidge(),
                  RandomForestRegressor(random_state=1),
                  LinearRegression()
                  ]
    # define the metrics to be used; so far it only works with 2 max, and the first one will be the one the model will
    # be optimized for
    metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}

    if randomized is True:
        model_names = [n+'_randomized' for n in model_names]

    ###### SET INPUT DIRECTORIES ######
    input_dir = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/input'
    input_f_seq = os.path.join(input_dir, 'input_HCs.csv')


    ###### LOAD DATA #######
    data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])

    # data will be scaled in the
    X = data['Sequence'].values
    y = data['KD'].values
    # log transform
    if log_transform is True:
        y = np.log(y + 1)

    # random labels
    if randomized is True:
        y = data['KD_norm'].sample(frac=1).reset_index(drop = True).values

    # one-hot encoding
    X_OH = GP.one_hot_encode_matern(X)

    # list of ks to evaluate
    k_l = list(range(2,len(X_OH),1))
    k_l.append(len(X_OH))

    # Dataframes to be saved
    Nested_Scores_df= pd.DataFrame(columns=['model_name', 'MSE_nonnested', 'R2_nonnested', 'best_params_nonnested',
                                          'MSE_nested','R2_nested', 'k_outer', 'k_inner', 'mostselected_params_nested'])
    LOO_Scores_df = pd.DataFrame(columns=['Model', 'R2', 'Corr_coef', 'MSE', 'params'])

    #########################################################################################################
    ###### EVALUATION OF Ks; NESTED CV #####
    #########################################################################################################
    for i, model_name in enumerate(model_names):

        print('\nStart model evaluation: '+ model_name)
        param_grid = param_list[i]
        reg = model_list[i]
        w_vars = vars_list[i]

        ## SET OUTPUT DIRECTORIES (for plots to save)
        dir_out = os.path.join('/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/Plots/log_transformed_input', model_name)
        dir_out_eval = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/model_evaluation/log_transformed_input'
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        if not os.path.exists(dir_out_eval):
            os.makedirs(dir_out_eval)

        # define class model
        kernel = regression_model_evaluation(X_OH, y, reg, model_name, metrics)

        ### EVALUATE Ks FOR CV
        print('Evaluate k')
        kernel.evaluate_k(param_grid, k_in=k_inner, k_l=k_l, plot = True, save_fig=True, n_jobs= n_jobs,
                          save_path=os.path.join(dir_out, model_name+'_k_sensitivity_nestedCV_test_ki'+ str(k_inner) + '.pdf'), verbose=0)

        ### MODEL EVALUATION WITH NESTED CV (set k values)
        print('\nPerform nested CV')
        non_nested_cv_df, nested_cv_df, scores_df = kernel.nested_param_tuning_eval(param_grid, k_o=k_outer, k_i=k_inner,
                                                                                    n_jobs = n_jobs, verbose=verbose)
        # save scores dataframe
        Nested_Scores_df = Nested_Scores_df.append(scores_df)

        ###### HYPERPARAMETER TUNING AND LOO-CV WITH BEST PERFORMING PARAMETERS #####
        # ### LOO CV
        # print('\nPerform LOO-CV and make correlation plot')
        # model_score_df = kernel.k_CV_and_plot(param_grid, k=len(X_OH), plot = True, save_fig=True, x_lim=[-0.5,2.5],
        #                                       y_lim=[-0.5,2.5], w_vars = w_vars,
        #                                       save_path=os.path.join(dir_out, model_name+'_corr_plot.pdf'))
        #
        # LOO_Scores_df = LOO_Scores_df.append(model_score_df)
        ##########################################################################################################
        print("--------------------------------------------------------")
        print("DONE with " + model_name)
        print("---------------------++++++++++++-----------------------")

    # save dataframes
    Nested_Scores_df.to_csv(os.path.join(dir_out_eval, 'Nested_CV_scores_ki'+str(k_inner)+'_ko'+str(k_outer)+'.csv'))
    # LOO_Scores_df.to_csv(os.path.join(dir_out_eval, 'Param_tuned_LOOCV_scores.csv'))



def main():
    run()




if __name__ == '__main__':
    main()


