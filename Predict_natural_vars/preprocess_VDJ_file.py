#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to preprocess the VDJ file in

Should serve as a baseline comparison to GPs;
Lena Erlach
09.05.2023
"""


import pandas as pd
import numpy as np
import os
import subprocess

# save fasta files of sequences
def save_fasta_file(sequence_df, col_name = "VDJ_aaSeq", id_name = "barcode", n_seq = 500, subdirectory = "data/", file_prefix = "Seq"):
    """
    Function that writes fasta files from a protein sequences pd.DataFrame; number of sequences per file can be set (in case for the Language
    model embeddings) that will be set to how many sequences can be embedded in one job;
        Args:
            sequence_df (pd.DataFrame): dataframe that contains the sequences in the col_name column and sequence fasta ids in id_name column;

            col_name (string): the name of the dataframe column that stores the protein sequences;

            id_name (string): the name of the dataframe column that stores the sequence ids that are to be used as identifier in the fasta file;

            n_seq (int): number of sequeces to be written per fasta file; (depends on how many can be embedded in one ESM embedding job)

            subdirectory (string): path leading to the folder where fasta files should be stored;

            file_prefix (string): file prefix for the fasta file names;

        """
    import math
    n_start = 0
    num_rounds = math.ceil(len(sequence_df) / n_seq)

    for r in range(num_rounds):
        print(r)
        if r < num_rounds - 1:
            # Downlsample OVA sequences
            OVA_VDJs = sequence_df.loc[:, col_name].tolist()[n_start:n_start + n_seq]
            barcodes = sequence_df.loc[:, id_name].tolist()[n_start:n_start + n_seq]
            n_start += n_seq
            # save fasta files
            ofile = open(os.path.join(subdirectory, file_prefix + "fasta_" + str(r) + ".fasta"), "w")
            for i, bc in enumerate(barcodes):
                ofile.write(">" + str(bc) + "\n" + OVA_VDJs[i][0:-1] + "\n")
            ofile.close()
            print("file saved:" + str(r))

        elif r == num_rounds - 1:
            OVA_VDJs = sequence_df.loc[:, col_name].tolist()[n_start:]
            barcodes = sequence_df.loc[:, id_name].tolist()[n_start:]
            # print("last round")
            # save fasta files
            ofile = open(os.path.join(subdirectory, file_prefix + "fasta_" + str(r) + ".fasta"), "w")
            for i, bc in enumerate(barcodes):
                #print(">" + bc)
                ofile.write(">" + str(bc) + "\n" + OVA_VDJs[i][0:-1] + "\n")
            ofile.close()
            print("last file saved")



########################################################################
# Load input
########################################################################
input_file = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/VDJ_Sequence_Selection/data/Filtered_files/ess_HEL_all_merged.txt'
fasta_dir = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Predict_natural_vars/data/imgt_alignment'


VDJ_file = pd.read_csv(input_file, delimiter= '\t', index_col=0).drop(columns=['Functionality', 'Table_CDR3'])
VDJ_file.reset_index(inplace = True)




########################################################################
# save fastas
########################################################################
# save_fasta_file(VDJ_file, col_name = "VDJ_NT", id_name = "index", n_seq = len(VDJ_file), subdirectory =fasta_dir, file_prefix = "VDJ_HEL_all")



########################################################################
# read mixcr aligned
########################################################################

ROOT_DIR = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Predict_natural_vars/data/imgt_alignment/'

mixcr_path_raw = os.path.join(ROOT_DIR, 'mixcr_VDJ_HEL.tsv')
# updated path for
mixcr_path = os.path.join(ROOT_DIR, 'TEMP_PROCESSED_mixcr_VDJ_HEL.tsv')
mixcr_file = pd.read_csv(mixcr_path, delimiter= '\t')
mixcr_file_raw = pd.read_csv(mixcr_path_raw, delimiter= '\t')

i = 1

# sequence is complete, but the imputed part is lower case & the missing part of the FR1 is imputed here and it seems to not have any differences!!
''.join([mixcr_file.aaSeqFR1[i], mixcr_file.aaSeqCDR1[i], mixcr_file.aaSeqFR2[i], mixcr_file.aaSeqCDR2[i],
         mixcr_file.aaSeqFR3[i], mixcr_file.aaSeqCDR3[i], mixcr_file.aaSeqFR4[i]])

all(mixcr_file.aaSeqImputedCDR1[i:500] == mixcr_file.aaSeqCDR1[i:500])
all(mixcr_file.aaSeqImputedCDR2[i:500] == mixcr_file.aaSeqCDR2[i:500])
all(mixcr_file.aaSeqImputedCDR3[i:500] == mixcr_file.aaSeqCDR3[i:500])




###### concat the sequences and upper case everything!
cols_to_concat = ['aaSeqImputedFR1', 'aaSeqImputedCDR1', 'aaSeqImputedFR2', 'aaSeqImputedCDR2', 'aaSeqImputedFR3', 'aaSeqImputedCDR3', 'aaSeqImputedFR4']

from tqdm import tqdm

ids_dropped = []
for id in tqdm(range(len(mixcr_file_raw)), total=len(mixcr_file_raw)):
    try:
        mixcr_file_raw.loc[id, "VDJ_aaSeq"] = ''.join([mixcr_file_raw.loc[id, c] for c in cols_to_concat])
    except:
        ids_dropped.append(id)

print(ids_dropped)
mixcr_file_raw['productive'] = airr_file.productive
mixcr_file = mixcr_file_raw.drop(ids_dropped).copy()

mixcr_file.reset_index(inplace=True, drop=True)
mixcr_file.VDJ_aaSeq = mixcr_file.VDJ_aaSeq.str.upper()


airr_file.drop(ids_dropped, inplace=True)
airr_file.reset_index(inplace=True, drop=True)



# # save dataframe since the for loop takes long
mixcr_file.to_csv('/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Predict_natural_vars/data/imgt_alignment/TEMP_PROCESSED_mixcr_VDJ_HEL.tsv',
                  index=False)


########################################################################
# Clonotyping MIXCR
########################################################################
# mixcr_file['allVHitsWithScore'][0]

##### don't work with mixcr --> doesn't work for some reason.

# # run mixcr in the command line -- exportAirr
# vdjc_input_path = os.path.join(ROOT_DIR, 'results.vdjca')
# cmd = ['mixcr', 'exportAirr', '-a', vdjc_input_path, 'airr_output.tsv']
# result = subprocess.run(cmd, capture_output=True, text=True, check=True)
# print(result.stdout)
# airr_file = pd.read_csv(os.path.join(ROOT_DIR, 'airr_output.tsv'), sep='\t')
#
# clns_input_path = os.path.join(ROOT_DIR, 'results.clns')
# cmd = ['mixcr', 'exportClones', '--impute-germline-on-export', clns_input_path, 'clns_output.tsv']
# result = subprocess.run(cmd, capture_output=True, text=True, check=True)
# print(result.stdout)
#
#
# clones_file = pd.read_csv(os.path.join(ROOT_DIR, 'clns_output_IGH.tsv'), sep='\t')
# clones_file2 = pd.read_csv(os.path.join(ROOT_DIR, 'clones_cdr3_IGH.tsv'), sep='\t')
#
# clones_file2 == clones_file

### airr_file just for comparison!
airr_file['Vgene'] = [airr_file.v_call[i].split(',')[0] for i in range(len(airr_file))]
airr_file['Jgene'] = [airr_file.j_call[i].split(',')[0] for i in range(len(airr_file))]
airr_file['VJgeneID'] = ['_'.join([airr_file.loc[i, 'Vgene'], airr_file.loc[i, 'Jgene']]) for i in range(len(airr_file))]


# cluster by identical VDJ germline hits - work with mixcr file!!
mixcr_file['Vgene'] = [mixcr_file['allVHitsWithScore'][i].split(',')[0].split(r'(')[0] for i in range(len(mixcr_file))]
mixcr_file['Jgene'] = [mixcr_file['allJHitsWithScore'][i].split(',')[0].split(r'(')[0] for i in range(len(mixcr_file))]
mixcr_file['seq_id'] = mixcr_file.index.values

### group by V and J gene columns
mixcr_file['VJgeneID'] = ['_'.join([mixcr_file.loc[i, 'Vgene'], mixcr_file.loc[i, 'Jgene']]) for i in range(len(mixcr_file))]

# drop not productive sequences!
mixcr_file_prod = mixcr_file[mixcr_file['productive'] != 'F']
len(mixcr_file)




########################################################################
# Clonotyping based on V and J gene identity and 80% CDR3 similarity
########################################################################


from Levenshtein import ratio as norm_dist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# Calculate levenshtein distance (normalized) matrix
def calc_norm_levens_dist(seqs: list, verbose=1):
    sim_matrix = np.ndarray((len(seqs), len(seqs)))
    for j in range(len(seqs)):

        if verbose > 0:
            if (j % 100 == 0):  print(j)

        LD_arr = []
        for i in range(len(seqs)):
            LD_arr.append(norm_dist(seqs[j], seqs[i]))

        # store distances in matrix
        sim_matrix[j, :] = LD_arr

    # return distance matrix
    dist_matrix = 1 - sim_matrix
    return dist_matrix




VJgene_combs = np.unique(mixcr_file_prod.VJgeneID)
cluster_thresh = 0.2
# assign clone_ids
mixcr_file_prod = mixcr_file_prod.copy()
mixcr_file_prod.loc[:,'clone_id'] = 0
n_clones = 0

for vf in tqdm(VJgene_combs, total=len(VJgene_combs)):
    # if CDRs are identical in VJgene combi
    vf_df = mixcr_file_prod[mixcr_file_prod.VJgeneID == vf]
    if len(np.unique(vf_df.aaSeqCDR3)) == 1:
        print('identical cdr3!')
        mixcr_file_prod.loc[mixcr_file_prod.VJgeneID == vf, 'clone_id'] = np.max(mixcr_file_prod['clone_id']) + 1
        n_clones += 1
    else:
        seqs = vf_df.aaSeqCDR3.values

        dist_matrix = calc_norm_levens_dist(seqs, verbose=0)
        # cluster CDR3s
        linked = linkage(squareform(dist_matrix), 'single')  # You can also use 'complete', 'average', etc.
        # Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t
        clusters = pd.Series(fcluster(linked, t=cluster_thresh, criterion='distance'), vf_df.index)
        n_clones += len(np.unique(clusters))
        #print('num clusters: ', len(np.unique(clusters)))
        # assign clusters
        mixcr_file_prod.loc[clusters.index, 'clone_id'] = clusters.values + np.max(mixcr_file_prod['clone_id'])


# # # save dataframe since the for loop takes long
# mixcr_file.to_csv('/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Predict_natural_vars/data/imgt_alignment/TEMP_PROCESSED_mixcr_VDJ_HEL_clonotyped_80CDR3sim.tsv',
#                   index=False)


########################################################################
# Clonotyping based on V and J gene identity and 80% CDR3 similarity
########################################################################

len(mixcr_file_prod.VDJ_aaSeq[mixcr_file_prod.clone_id == 3])
len(np.unique(mixcr_file_prod.VDJ_aaSeq[mixcr_file_prod.clone_id == 3]))



