

import numpy as np
import pandas as pd

from GP_implementation import lazyCartProduct as catprod



# Set input directories
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'
in_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/gen_seqs_muvar/'
in_dir_seq = input_dir + 'input_HCs.csv'

out_dir = in_dir + 'gen_pred_seqs_thresh/output/'


# load the indeces

new_seqs = pd.DataFrame()
new_seqs = pd.read_csv(in_dir+'gen_pred_seqs_thresh/output/muvars_sliced_10_7.txt', delimiter='\t', skiprows=1)

idx = list(new_seqs.random_idx.values)


# load the 35 sequences and generate AA dict


###### sequence data #######
data = pd.read_csv(in_dir_seq, usecols=['Sequence'])


###### GET INFOS ######
## about amino acid variations in given sequences


# initialize dataframe with rows as sequences and AAs at each position as columns
AAs = pd.DataFrame()
# get a matrix of characters (122 cols are position of Sequence [:121],
# 49 rows are sequences [:48])
for i in range(len(data.Sequence[0])):
    AAs[i] = [str(seq[i]) for seq in data.Sequence]




# initialize dictionary of AAs at positions (for creation of new sequences
mut_dict = {}
for i in range(len(data.Sequence[0])):
    if len(set(AAs.loc[:, i])) == 1:
        mut_dict[i] = list(set(AAs.loc[:, i]))
    else:
        mut_dict[i] = list(set(AAs.loc[:, i]))


# get the seuqence according to random index

sets = [mut_dict[x] for x in mut_dict]

# initialize lazy cartesina product object
cp = catprod.LazyCartesianProduct(sets)


# save new seqs

new_seqs['Sequences'] = np.asarray([''.join(cp.entryAt(x)) for x in idx])

new_seqs.to_csv(out_dir+'new_seq_gen_pos.csv')

























