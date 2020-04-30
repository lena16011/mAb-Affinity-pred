'''
Script to convert the generated, predicted AA sequences according to the 35 known sequences
codon usage to NT sequences
'''


import pandas as pd
import numpy as np

def get_mut_dict(seqs, consensus_seq):
    '''
    Function compares two sequences and returns a dictionary with the differing positions
    :param seq: sequence to be compared to the consensus sequence
    :param consensus_seq: Consensus sequence to compare AA sequence to.
    :return: mut_dict : keys are positions of varying mutations and tuple (AA in seq, AA in con)
    '''
    mut_dict = {}
    for seq in seqs:
        for i, con_AA in enumerate(consensus_seq):
            if con_AA == seq[i]:
                pass
            elif con_AA != seq[i]:
                if i not in mut_dict:
                    mut_dict[i] = [con_AA]
                if seq[i] not in mut_dict[i]:
                    mut_dict[i].append(seq[i])

    return mut_dict



def AA_to_NT(seqs, consensus_AA, consensus_NT, mut_df):
    '''
    Function that converts AA sequence to NT sequence.
    :param seqs: list of sequences to convert
    :param consensus_AA:
    :param consensus_NT:
    :param mut_df: a manually created dataframe with
    :return: list of NT sequences
    '''
    # make a consensus NT dataframe,

    NTs = [consensus_NT[x * 3:x * 3 + 3] for x in range(len(consensus_AA))]
    # AAs = [cons_seq[x] for x in range(len(cons_seq))]


    # for i, seq in enumerate(seqs):
    NT_ls = []
    for i, seq in enumerate(seqs):
        NT_string = str()
        for j, AA in enumerate(seq):

            if AA == cons_seq[j]:
                NT_string = NT_string + NTs[j]
                pass
            elif AA == mut_df.loc[j, 'AA1']:
                NT_string = NT_string + mut_df.loc[j, 'NT1']
                pass
            elif AA == mut_df.loc[j, 'AA2']:
                NT_string = NT_string + mut_df.loc[j, 'NT2']
                pass
            elif AA != mut_df.loc[j, 'in_seq']:
                print('Sep : pos. {}, {} ;AA in sequence not found'.format(i, j, AA))

        NT_ls.append(NT_string)


    return NT_ls



# input directorie
in_dir = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/gen_seqs_muvar/'
in_file = in_dir + '15_selected_seq_data.csv'


# define seuqences
cons_seq = 'QVQLQQSGAELVRPGASVTLSCKASGYTFTDYEMHWVKQTPVHGLEWIGAIDPETGGTAYNQKFKGKATLTADKSSSTAYMELRSLTSEDSAVYYCTRDYYGSNYLAWFAYWGQGTLVTVSA'
cons_NT = 'CAGGTTCAACTGCAGCAGTCTGGGGCTGAGCTGGTGAGGCCTGGGGCTTCAGTGACGCTGTCCTGCAAGGCTTCGGGCTACACATTTACTGACTATGAAATGCACTGGGTGAAGCAGACACCTGTGCATGGCCTGGAATGGATTGGAGCTATTGATCCTGAAACTGGTGGTACTGCCTACAATCAGAAGTTCAAGGGCAAGGCCACACTGACTGCAGACAAATCCTCCAGCACAGCCTACATGGAGCTCCGCAGCCTGACATCTGAGGACTCTGCCGTCTATTACTGTACAAGAGATTACTACGGTAGTAACTACCTGGCCTGGTTTGCTTACTGGGGCCAAGGGACTCTGGTCACTGTCTCTGCA'

data = pd.read_csv(in_file, index_col=0)


mut_dict = get_mut_dict(list(data.Sequences.values), cons_seq)



######################## can be loaded after it was saved and manually NTs of mutated positions were added
df_mut = pd.DataFrame.from_dict(mut_dict, orient='index')


df_mut.reset_index(inplace=True)
df_mut.columns = ['position', 'cons_seq', 'AA1', 'AA2']
df_mut.sort_values(by='position', inplace=True)
df_mut.reset_index(drop=True, inplace=True)


# df_mut.to_csv('/media/lena/LENOVO/Dokumente/Masterarbeit/data/GP/gen_seqs_muvar/mutation_dict2.csv')
#######################

# read in a df with mutated AAs and NTs: (created manually!!!)
mut_df = pd.read_csv(in_dir + 'mutation_dict.csv', index_col=0, usecols=range(1, 8))



NT_ls = AA_to_NT(data['Sequences'], cons_seq, cons_NT, mut_df)

data['NT'] = NT_ls

data.to_csv(in_dir + 'pred_seqs_data.csv')















