'''
Script to filter sequences similar to the consensus sequence of the original 35 sequences.
'''


import pandas as pd
import stringdist
import numpy as np


## Set input directories
abs_path = 'D:/Dokumente/Masterarbeit/Lena/GP_implementation'

in_dir = [abs_path + '/data/gen_seqs_muvar/10_8/pos/',
          abs_path + '/data/gen_seqs_muvar/10_8/neg/',
          abs_path + '/data/gen_seqs_muvar/midr/']

files = ['all_new_seq_gen_pos.csv', 'all_new_seq_gen_neg.csv', 'all_new_seq_gen_mid.csv']


# consensus sequence
cons_seq = 'QVQLQQSGAELVRPGASVTLSCKASGYTFTDYEMHWVKQTPVHGLEWIGAIDPETGGTAYNQKFKGKATLTADKSSSTAYMELRSLTSEDSAVYYCTRDYYGSNYLAWFAYWGQGTLVTVSA'

# load data from 1000 seq file


seqs_neg = pd.read_csv(in_dir[1] + files[1], index_col=0)
seqs_mid = pd.read_csv(in_dir[2] + files[2], index_col=0)
seqs_pos = pd.read_csv(in_dir[0] + files[0], index_col=0)


seqs_neg['LD_cons'] = seqs_neg['Sequences'].iloc[:].apply(stringdist.levenshtein, args=(cons_seq,))
seqs_mid['LD_cons'] = seqs_mid['Sequences'].iloc[:].apply(stringdist.levenshtein, args=(cons_seq,))
seqs_pos['LD_cons'] = seqs_pos['Sequences'].iloc[:].apply(stringdist.levenshtein, args=(cons_seq,))

# print indeces of smallest LD to consensus sequence

print("# sequence (neg) {} with lowest LD ({}) to consensus sequence".format(seqs_neg.LD_cons.values.argmin(),
                                                                          seqs_neg.LD_cons.values.min()))

print("# sequence (pos) {} with lowest LD ({}) to consensus sequence".format(seqs_pos.LD_cons.values.argmin(),
                                                                          seqs_pos.LD_cons.values.min()))

print("# sequence (mid) {} with lowest LD ({}) to consensus sequence".format(seqs_mid.LD_cons.values.argmin(),
                                                                          seqs_mid.LD_cons.values.min()))




lo_LD_neg = seqs_neg[seqs_neg.LD_cons.values == seqs_neg.LD_cons.values.min()]
lo_LD_neg.to_csv(in_dir[1]+'loLD_seq_neg.csv')

lo_LD_pos = seqs_pos[seqs_pos.LD_cons.values == seqs_pos.LD_cons.values.min()]
lo_LD_pos.to_csv(in_dir[0]+'loLD_seq_pos.csv')

lo_LD_mid = seqs_mid[seqs_mid.LD_cons.values == seqs_mid.LD_cons.values.min()]
lo_LD_mid.to_csv(in_dir[2]+'loLD_seq_mid.csv')

