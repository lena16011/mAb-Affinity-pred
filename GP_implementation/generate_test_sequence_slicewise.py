import os
import numpy as np
import pandas as pd
import itertools as it
from GP_implementation import GP_fcts as GP
from GP_implementation import lazyCartProduct as catprod
import random
import matplotlib.pyplot as plt




###### SET INPUT DIRECTORIES ######
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'

input_f_seq = input_dir + 'input_HCs.csv'
input_train_seq = input_dir + 'Final_49_AA_from_geneious.csv'
input_train_KD = input_dir + 'HC_KDvals.csv'


## SET OUTPUT DIRECTORIES (for plots to save)
dir_outLD = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/intermediate_testmut/'

# If the output directories do not exist, then create it
if not os.path.exists(dir_outLD):
    os.makedirs(dir_outLD)





###### LOAD DATA #######
df_seq = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence'])

# Load KD values and add them together to a new dataframe (remove samples that w/o measured KD value)
KDs = pd.read_csv(input_train_KD, usecols=['SampleID', 'KD'])
data = pd.merge(df_seq,KDs, on='SampleID')

KDs_unnormalized = data

###### LOAD DATA ######
df_seq = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence'])



###### GET INFOS ######
## about amino acid variations in given sequences


# initialize dataframe with rows as sequences and AAs at each position as columns
AAs = pd.DataFrame()
# get a matrix of characters (122 cols are position of Sequence [:121],
# 49 rows are sequences [:48])
for i in range(len(df_seq.Sequence[0])):
    AAs[i] = [str(seq[i]) for seq in df_seq.Sequence]



# initialize list of positions where mutations occur
id_ls = []

# initialize dictionary of AAs at positions (for creation of new sequences
mut_dict = {}
for i in range(len(df_seq.Sequence[0])):
    if len(set(AAs.loc[:, i])) == 1:
        mut_dict[i] = list(set(AAs.loc[:, i]))
    else:
        id_ls.append(i)
        mut_dict[i] = list(set(AAs.loc[:, i]))



# create a list of ints of how many AA occur in variation
mut_ls = [len(mut_dict[key]) for key in mut_dict.keys() if len(mut_dict[key]) != 1]
# number of new sequences for the initialization of the dataframe
num_seqs = np.prod([len(mut_dict[key]) for key in mut_dict.keys() if len(mut_dict[key]) != 1])


###### PRINT INFOS ######
# print the positions where mutations occur
print("Mutations occurr at positions: {}\n{} positions have variations in AAs".format(id_ls, len(id_ls)))

# print number of maximal variation
print("max. variation per position {} at position {}".format(max(mut_ls), np.argmax([len(mut_dict[key]) for key in mut_dict.keys()])))

print("possible sequences from these combination {0:.1f} * 10^13".format(num_seqs/10**13))



################ CREATE NEW SEQUENCES #################

# # store keys
# positions = sorted(mut_dict)
# # create object of combinations
# combis = it.product(*(mut_dict[pos] for pos in positions))
#
# # get slice of the combinations
# slice = it.islice(combis, 1000)
#
# sl_list = list(slice)
# # new_seq = np.asarray([''.join(x) for x in sl_list])

########################################################






#### PREDICT AFFINITIES ####

###### LOAD DATA #######
df_train_seq = pd.read_csv(input_train_seq, usecols=['SampleID', 'Sequence'])

# Load KD values and add them together to a new dataframe (remove samples that w/o measured KD value)
KDs = pd.read_csv(input_train_KD, usecols=['SampleID', 'KD'])
data = pd.merge(df_train_seq, KDs, on='SampleID')

KDs_unnormalized = data

# normalize data
data['KD'] = GP.normalize_test_train_set(data['KD'])

# defien test and train data
X_train = data['Sequence'].values
y_train = data['KD'].values
# X_test = new_seq



########################### PREDICT LD KERNEL #######################
#### make a loop of testing the LD kernel
# get optimal noise parameter
# opt_param = GP.get_params(X_train, y_train, init_param=1)
#
# # predict the test set and training set
# # mu_test, var_test = GP.predict_GP(X_train, y_train, X_test, opt_param)
#
# mu_all = []
# var_all = []
#
# # store keys
# positions = sorted(mut_dict)
# # create object of combinations
# combis = it.product(*(mut_dict[pos] for pos in positions))
#
#
# for i in range(0, 50000, 500):
#
#     slice = it.islice(combis, i, i + 500)
#
#     sl_list = list(slice)
#     X_test = np.asarray([''.join(x) for x in sl_list])
#
#
#     mu_test, var_test = GP.predict_GP(X_train, y_train, X_test, opt_param)
#     mu_all.append(mu_test)
#     var_all.append(var_test)
#     print("samples processed: {}".format(i+500))
#     # print("cycles left: {}".format((10000 - i + 500)/500)
#
# # save intermediate mus
#
#
# df_muvar = pd.DataFrame()
# df_muvar['mu'] = np.concatenate(mu_all)
# df_muvar['var'] = np.concatenate(var_all)
#
# df_muvar.to_csv(dir_outLD + 'intermed_muvars.csv', sep=',')


###########################################################################################




########################### PREDICT MATERN KERNEL #######################
#### make a loop of testing the Matern kernel

#### TEST TEST SET ####

# get optimal noise parameter

X_train_OH = GP.one_hot_encode_matern(X_train)
opt_param = GP.get_params_mat(X_train_OH, y_train)
# opt_param = [0.1,10]


mu_all_mat = []
var_all_mat = []
steps = 1000


# store keys
positions = sorted(mut_dict)
# create object of combinations
combis = it.product(*(mut_dict[pos] for pos in positions))



####################### SLICE POSSIBLE SEQUENCES  ######################


for i in range(0, 1000000, steps):

    slice = it.islice(combis, i, i + steps)

    sl_list = list(slice)
    X_test = np.asarray([''.join(x) for x in sl_list])
    X_test_OH = GP.one_hot_encode_matern(X_test)

    mu_test, var_test = GP.predict_GP_mat(X_train_OH, y_train, X_test_OH, opt_param)
    mu_all_mat.append(mu_test)
    var_all_mat.append(var_test)
    print("samples processed: {}".format(i+steps))
    # print("cycles left: {}".format((10000 - i + 500)/500)

# save intermediate mus

df_muvar = pd.DataFrame()
df_muvar['mu'] = np.concatenate(mu_all_mat)
df_muvar['var'] = np.concatenate(var_all_mat)

df_muvar.to_csv(dir_outLD + 'intermed_muvars10_6.csv', sep=',')




####################### LAZY CARTESIAN PRODUCT ######################

sets = [mut_dict[x] for x in mut_dict]

# initialize lazy cartesina product object
cp = catprod.LazyCartesianProduct(sets)

# res1 = cp.entryAt(121)
# res2 = cp.entryAt(121)
# res3 = cp.entryAt(12524124635133)


# generate 1000 random numbers
random.seed(124)

random_idx = random.sample(range(num_seqs), 10000)

# get optimal noise parameter
X_train_OH = GP.one_hot_encode_matern(X_train)
opt_param = GP.get_params_mat(X_train_OH, y_train)
# opt_param = [0.1,10]


mu_all_mat_rsamp = []
var_all_mat_rsamp = []
X_test = np.asarray([''.join(cp.entryAt(x)) for x in random_idx])
X_test_OH = GP.one_hot_encode_matern(X_test)


mu_test, var_test = GP.predict_GP_mat(X_train_OH, y_train, X_test_OH, opt_param)


#### Plot the distribution of predicted values
fig, axs = plt.subplots(1, 2, tight_layout=True)

axs[0].hist(mu_test, bins = 30)
axs[1].hist(y_train, bins = 30)
axs[0].title.set_text('Distribution of predictions')
axs[1].title.set_text('Distribution of true KDs')

# axs[0].set_xlim([-3, 3])
# axs[1].set_xlim([-3, 3])
plt.show()







