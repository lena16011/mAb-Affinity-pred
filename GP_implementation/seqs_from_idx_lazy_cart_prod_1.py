'''
Script to extract the indices of the random samples and generate the VDJ sequences
with an AA dictionary of the possible combinations. Sort and get the top or lowest
1000 sequences.
'''


import numpy as np
import pandas as pd
from GP_implementation import lazyCartProduct as catprod
import math


def generate_AA_dict(seqs):
    '''
    Function to generate a dictionary according to seqs (input); keys are positions and values are AAs that
    occur on this position across the sequences.
    :param seqs: list of sequences
    :return: dictionary of positions and AAs occuring at positions within the sequences
    '''

    # initialize dataframe with rows as sequences and AAs at each position as columns
    AAs = pd.DataFrame()
    # get a matrix of characters (122 cols are position of Sequence [:121],
    # 49 rows are sequences [:48])
    for i in range(len(seqs[0])):
        AAs[i] = [str(seq[i]) for seq in seqs]

    # initialize dictionary of AAs at positions (for creation of new sequences
    mut_dict = {}
    for i in range(len(seqs[0])):
        if len(set(AAs.loc[:, i])) == 1:
            mut_dict[i] = list(set(AAs.loc[:, i]))
        else:
            mut_dict[i] = list(set(AAs.loc[:, i]))


    return mut_dict

def generate_new_seqs(seqs, index):
    '''
    Function to generate the new sequences using the lazy cartesian product.
    :param seqs sequences that are used to generate mutation dictionary
    :param index: index of sequences to be generated;
    '''

    # generate mutation dictionary
    mut_dict = generate_AA_dict(seqs)

    # get the seuqence according to random index
    sets = [mut_dict[x] for x in mut_dict]

    # initialize lazy cartesina product object
    cp = catprod.LazyCartesianProduct(sets)

    # sort and save top 1000 new seqs
    new = np.asarray([''.join(cp.entryAt(x)) for x in list(index)])

    return new





# define input mu/var file
# number of sequences that were generated
num_gen = 10**8


#define number of sequences to store in file
save_seqs = 1000


# Set input directories
input_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/input/'
in_dir_seq = input_dir + 'input_HCs.csv'

in_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/gen_seqs_muvar/10_8/pos/'
file_n = 'muvars_sliced_10_' + str(round(math.log(num_gen, 10))) + '.txt'

out_dir = in_dir






##### LOAD DATA #####
# load sequences to generate sequences of
data = pd.read_csv(in_dir_seq, usecols=['Sequence'])

# load the random drawn indeces from the gen_pred script generated file
new_seqs = pd.read_csv(in_dir + file_n, delimiter='\t', skiprows=1)



##### GENERATE SEQUENCES WITH LAZY CARTESIAN PRODUCT ####
new_seqs['Sequences'] = generate_new_seqs(seqs=data.Sequence, index=new_seqs.random_idx)


# if existent, remove duplicates of original sequences
print("Sequences identical to original 35 variants: {}".format(any(new_seqs.Sequences.isin(data.Sequence)) == True))
if any(new_seqs.Sequences.isin(data.Sequence)) == True:
    new_seqs = new_seqs[~new_seqs.Sequences.isin(data.Sequence)]


# sort the dataframe (ascending = True for neg and midrange; False for positive)
new_seqs.sort_values(by='mus', inplace = True, ascending=False)
new_seqs = new_seqs.reset_index(drop = True)

# get 1000 sequences in the middle for midrange
# new_seqs10 = new_seqs.iloc[int(len(new_seqs)/2 - 5): int(len(new_seqs)/2 + 5),]


#### SAVE SEQUENCES ###
new_seqs10 = new_seqs.iloc[:save_seqs,]

new_seqs10.to_csv(out_dir+'top' + str(save_seqs) + '_new_seq_gen_pos.csv')

























