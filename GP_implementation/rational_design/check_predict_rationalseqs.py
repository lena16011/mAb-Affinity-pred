'''
Script to predict the newly, rationally designed sequences
'''


import pandas as pd
from utils import GP_fcts as GP

in_dir1 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/rational_design/'
in_f1 = in_dir1 + 'rational_de_seqs.csv'

in_dir2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/VDJ_selection/VDJ_final_data/'
in_f2 = in_dir2 + 'VDJs_Selection_with_CDR3.txt'

VDJ_tot = pd.read_csv(in_f2, usecols= ['VDJ_AA', 'VDJ_NT'], delimiter = '\t')
rat_seqs = pd.read_csv(in_f1, delimiter = ',')


# if existent, remove duplicates of original sequences
# crop sequences from rational design to compare to original dataset
crp_seqs = pd.Series([x[7:] for x in rat_seqs.AA])
print("Sequences identical to original {} variants: {}".format(len(VDJ_tot),
                                                               any(crp_seqs.isin(VDJ_tot.VDJ_NT)) == True))
# if any(rat_seqs.AA.isin(VDJ_tot.VDJ_AA)) == True:
#     rat_seqs = rat_seqs[~rat_seqs.NT.isin(VDJ_tot.VDJ_NT)]



#################### PREDICT SEQUENCES KDS #################

###### SET INPUT DIRECTORIES ######
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'

input_f_seq = input_dir + 'input_HCs.csv'


###### LOAD DATA #######
data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])






#### DATA PROCESSING ####
# normalize data
data['KD_norm'] = GP.normalize_test_train_set(data['KD'])

X_train = data['Sequence'].values
y_train = data['KD_norm'].values

X_test = rat_seqs['AA'].values

# one-hot encoding
X_train_OH = GP.one_hot_encode_matern(X_train)
X_test_OH = GP.one_hot_encode_matern(X_test)

# get optimal parameters
opt_param = GP.get_params_mat(X_train_OH, y_train, init_param=(0.1,10))
print("optimal parameters: {}".format(opt_param))

# predict X_test
mu_test, var_test = GP.predict_GP_mat(X_train_OH, y_train, X_test_OH, opt_param)

rat_seqs.insert(loc=1, column='pred_KD', value=mu_test)
rat_seqs.insert(loc=2, column='pred_var', value=var_test)

rat_seqs.to_csv(in_dir1+'rational_de_seqs_predicted.csv')
















