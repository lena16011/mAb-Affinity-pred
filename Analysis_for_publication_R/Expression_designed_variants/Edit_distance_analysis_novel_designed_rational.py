"""
Script to analyse the Edit distances or training data, novel designed variants and
natural repertoire variants


Lena Erlach
30.12.2023
"""


from Levenshtein import ratio as norm_dist
import Levenshtein
import matplotlib.pyplot as plt
import GP_implementation.Regression_Evaluation_framework.Regression_evaluation_paramTuning as REV


# Calculate levenshtein distance (normalized) matrix
def calc_norm_levens_dist(seqs: list, verbose=1):
    sim_matrix = np.ndarray((len(seqs), len(seqs)))
    for j in range(len(seqs)):

        if verbose > 0:
            if (j % 100 == 0):  print(j)

        LD_arr = []
        for i in range(len(seqs)):
            LD_arr.append(Levenshtein.distance(seqs[j], seqs[i]))

        # store distances in matrix
        sim_matrix[j, :] = LD_arr

    # return distance matrix
    # dist_matrix = 1 - sim_matrix
    return sim_matrix



def prep_native_seqs(data_natvars, data_natvars_total):

    '''
    Function to match the selected, incomplete sequences in data_natvars and match with the mixcr aligned sequences
    in data_natvars_total. Function also trimms off the trailing underscore.

    Args:
        data_natvars: pd.DataFrame of incomplete sequences which are in the column named VDJ_aaSeq
        data_natvars_total: pd.DataFrame of the mixcr aligned dataframe with all the sequences in the full
                            immune repertoire.

    Returns:
        trimmed_seqs: np.array of the mixcr aligned sequences that were matched with the input sequences; also
                      trimmed of the trailing underscore
    '''



    ### Match sequences that are in the dataset
    full_sequences = []
    invalid_id = []
    for i, seq in enumerate(data_natvars):
        unique_seqs = np.unique(data_natvars_total['VDJ_aaSeq'].loc[[seq in s for s in data_natvars_total.VDJ_aaSeq]])

        # skip if it doesn't exist
        if len(unique_seqs) == 0:
            print(f'ID: {i}: {len(unique_seqs)} seqs in mixcr data')
            # full_sequences.append(np.nan)
            invalid_id.append(i)

        elif len(unique_seqs) == 1:
            # Check if existent in data already
            if unique_seqs[0] in full_sequences:
                print(f'not unique! {i}')

            full_sequences.append(unique_seqs[0])
        # if there are > 1 matches, add first match
        elif len(unique_seqs) > 1:
            # Check if existent in data already
            if unique_seqs[0] in full_sequences:
                print(f'not unique! {i}')

            print(f'ID: {i}: {len(unique_seqs)} seqs in mixcr data')
            [full_sequences.append(s) for s in unique_seqs]

            invalid_id.append(i)


    # trim the sequences of the trailing
    trimmed_seqs = np.array([s.rstrip('_') for s in full_sequences])
    trimmed_seqs_u = np.unique(trimmed_seqs)

    return trimmed_seqs_u



#########################################################################################################
###### SET INPUT/OUTPUT DIRECTORIES ######
#########################################################################################################

ROOT_DIR = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/'
input_train_seq = os.path.join(ROOT_DIR, 'GP_implementation/data/input/input_HCs.csv')
input_test_seq = os.path.join(ROOT_DIR, 'GP_implementation/data/final_validation/novel_variants_AA_KDs.csv')
input_test_seqs_pred = os.path.join(ROOT_DIR, 'GP_implementation/data/final_validation/novel_seqs_predicted.csv')
input_VDJ_seq = os.path.join(ROOT_DIR, 'Predict_natural_vars/data/TEMP_mixcr_VDJ_HEL_clonotyped_80CDR3sim.csv')
# the raw data misses the first part of the mixcr file that's for matching the ones from the mixcr
VDJ_raw = os.path.join(ROOT_DIR, 'VDJ_Sequence_Selection/data/VDJ_selection/original_data/uniq_VDJs_from_Ann_Table_data_AP_simfilt80.txt')


# output
dir_out = os.path.join(ROOT_DIR, 'Analysis_for_publication_R/Expression_designed_variants')


#########################################################################################################
############ LOAD DATA #############
#########################################################################################################

data_train = pd.read_csv(input_train_seq, usecols=['SampleID', 'Sequence', 'KD'])
data_test_all = pd.read_csv(input_test_seq)
# train set of original training data and newly designed test set
X_train = np.concatenate((data_train['Sequence'].values, data_test['VDJ_AA'].values))

# drop the variant with bad performance
data_test = data_test_all.iloc[:8, :].drop(4).reset_index(drop = True)



############ LOAD NATIVE SEQS #############
data_natvars_total = pd.read_csv(input_VDJ_seq, sep=',')
data_natvars = pd.read_csv(VDJ_raw, sep='\t')['VDJ_AA']

# prepare the sequences
native_sel_seqs = prep_native_seqs(data_natvars, data_natvars_total)
# remove original, selected variants from test set
duplicated_id = [i for i, s in enumerate(native_sel_seqs) if s in X_train]
native_sel_seqs_u = np.delete(native_sel_seqs, duplicated_id)



#########################################################################################################
###### EDIT DISTANCE ANALYSIS #####
#########################################################################################################

l_t = len(data_train['SampleID'].values)

####### native variants
list_seqs = np.concatenate((data_train['Sequence'].values, native_sel_seqs_u))

dist_matrix_train_native = calc_norm_levens_dist(list_seqs, verbose=1)
min_dist_nat = np.min(dist_matrix_train_native[l_t:, :l_t], axis=1)


# plot a histogram of distances
plt.hist(min_dist_nat)
plt.title("Edit distances training data - natural repertoire")
plt.show()



####### rationally designed variants
list_labels = np.concatenate((data_train['SampleID'].values, data_test['IDs'].values))
list_seqs = np.concatenate((data_train['Sequence'].values, data_test['VDJ_AA'].values))


dist_matrix_train_novel = calc_norm_levens_dist(list_seqs, verbose=1)
min_dist_nov = np.min(dist_matrix_train_novel[l_t:, :l_t], axis=1)



####### randomly designed variants
random_des_df = data_test_all.loc[9:, ]
random_des_vars = random_des_df.drop([12,13,17,18,22,23])
random_des_loLD = random_des_df.drop([9,10,11,14, 15, 16, 19, 20,21])

list_seqs = np.concatenate((data_train['Sequence'].values, random_des_vars.VDJ_AA))

dist_matrix_train_native = calc_norm_levens_dist(list_seqs, verbose=1)
min_dist_rd = np.min(dist_matrix_train_native[l_t:, :l_t], axis=1)




######## loLD
list_seqs = np.concatenate((data_train['Sequence'].values, random_des_loLD.VDJ_AA))

dist_matrix_train_native = calc_norm_levens_dist(list_seqs, verbose=1)
min_dist_rd_lo = np.min(dist_matrix_train_native[l_t:, :l_t], axis=1)



##### plot a histogram of distances

cols = ['#1874cd', '#cd0000', '#228B22']
plt.hist([min_dist_nov, min_dist_rd, min_dist_rd_lo], color=cols,
         label=['rational', 'random', 'rand_loLD'])

# plt.bar(0, 3, color='red', edgecolor='black', hatch='/')
plt.title("Edit distances training data - rational, random & loLD design")
plt.legend()
plt.ylabel('count')
plt.xlabel('min LD to training dataset')
plt.savefig(os.path.join(dir_out, 'minLD_to_trainingset_histogram.pdf'))
plt.show()






####### plot distinguishing expressed and not expressed

####### randomly designed variants all not expressed!
####### rationally designed variants all expressed!

dist_rand = []
loLD_ex = random_des_loLD['VDJ_AA'][~random_des_loLD['KD_nM'].isna()]
loLD_nex = random_des_loLD['VDJ_AA'][random_des_loLD['KD_nM'].isna()]

for s in [loLD_ex, loLD_nex]:
    list_seqs = np.concatenate((data_train['Sequence'].values, s))
    print(len(list_seqs))
    dist_matrix = calc_norm_levens_dist(list_seqs, verbose=1)
    dist_rand.append(np.min(dist_matrix[l_t:, :l_t], axis=1))


### Histogram
pltls = [min_dist_nov, min_dist_rd] + dist_rand
cols = ['#1874cd', '#ff4d4d', '#0e5d0e', '#90ee90'] #'#228B22']

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.hist(pltls, color=cols, stacked=True, width = 1,
         label=['rational_expr', 'random_notexpressed', 'rand_loLD_expr', 'rand_loLD_notexpr'])

# ax1.bar set_hatch('/')

# plt.bar(0, 3, color='red', edgecolor='black', hatch='/')
plt.title("Edit distances training data - rational, random & loLD design")
plt.legend()
plt.ylabel('count')
plt.xlabel('min LD to training dataset')
plt.savefig(os.path.join(dir_out, 'minLD_to_trainingset_histogram_expressed.pdf'))
plt.show()







####### All variants labelled expressed and not expressed

####### randomly designed variants all not expressed!
####### rationally designed variants all expressed!

dist_rand = []
all_vars = data_test_all.drop(8)

loLD_ex = all_vars['VDJ_AA'][~all_vars['KD_nM'].isna()]
loLD_nex = all_vars['VDJ_AA'][all_vars['KD_nM'].isna()]

for s in [loLD_ex, loLD_nex]:
    list_seqs = np.concatenate((data_train['Sequence'].values, s))
    print(len(list_seqs))
    dist_matrix = calc_norm_levens_dist(list_seqs, verbose=1)
    dist_rand.append(np.min(dist_matrix[l_t:, :l_t], axis=1))


### Histogram
pltls = dist_rand
cols = ['#000000', '#A9A9A9']
# cols = ['#1874cd', '#ff4d4d', '#0e5d0e', '#90ee90'] #'#228B22']

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.hist(pltls, color=cols, stacked=True, width = 1,
         label=['expressed', 'not_expressed'])

# ax1.bar set_hatch('/')

# plt.bar(0, 3, color='red', edgecolor='black', hatch='/')
plt.title("Edit distances to training data - novel designed variants")
plt.legend()
plt.ylabel('count')
plt.xlabel('min LD to training dataset')
plt.savefig(os.path.join(dir_out, 'minLD_to_trainingset_histogram_onlyexpressed_label.pdf'))
plt.show()



#########################################################################################################
###### kD ANALYSIS #####
#########################################################################################################
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
# get predicted KDs

X_train = data_train['Sequence'].values
y_train = data_train['KD'].values

# one-hot encoding
X_train_OH = GP.one_hot_encode_matern(X_train)
y_train = np.log(y_train + 1)

reg = GaussianProcessRegressor(random_state=1)
model_name = 'GP_RBF'
params =[{'regressor__kernel': [RBF(l) for l in np.logspace(-1, 1, 3)],
                   'regressor__alpha': [1e-10, 1e-3, 0.1]}]

metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}


# define class model
regressor = REV.regression_model_evaluation(X_train_OH, y_train, reg, model_name, metrics)


metric_dict = regressor.k_CV_and_plot(param_grid=params, k=5, plot = True,
                      save_fig=False, w_vars = True, save_path=None)


model_trained = regressor.best_model.fit(X_train_OH, y_train)

X_test_OH = GP.one_hot_encode_matern(data_test_all['VDJ_AA'])

y_pred = model_trained.predict(X_test_OH, return_std=True)











data_test_pred = pd.read_csv(input_test_seqs_pred)
data_test_pred['y_pred'] = y_pred[0]

kds_expr = data_test_pred.y_pred[~data_test_pred.KD_nM.isna()]
kds_expr_m = data_test_pred.KD_nM[~data_test_pred.KD_nM.isna()]
kds_nonexpr = data_test_pred.y_pred[data_test_pred.KD_nM.isna()]


# ### Histogram
# pltls = [kds_expr, kds_nonexpr]
# cols = ['#000000', '#A9A9A9']
# # cols = ['#1874cd', '#ff4d4d', '#0e5d0e', '#90ee90'] #'#228B22']
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# ax1.hist(pltls, color=cols,
#          label=['expressed', 'not_expressed'])
#
# # ax1.bar set_hatch('/')
#
# # plt.bar(0, 3, color='red', edgecolor='black', hatch='/')
# plt.title("kD predicted - novel designed variants")
# plt.legend()
# plt.ylabel('count')
# plt.xlabel('KDs')
# # plt.savefig(os.path.join(dir_out, '!!!!.pdf'))
# plt.show()




plt.scatter(kds_expr, kds_expr_m)
plt.ylabel("measrued")
plt.xlabel("predicted")
plt.show()






kds_expr[4]
kds_expr_m[4]

list_seqs = np.concatenate((data_train['Sequence'].values, loLD_ex))
dist_matrix = calc_norm_levens_dist(list_seqs, verbose=1)

np.min(dist_matrix[l_t:, :l_t], axis=1)

