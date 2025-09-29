'''
Script to generate heatmap-like plots from AA sequences for comparison


'''

# seaborn cluster map with dendogram
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc




def sequence_heatmap(seqs, seq_labels, cls_method = 'average',col_map="mako", fig_size=(30, 5), save_fig=False, out_dir=None):
    '''
    Function to create a heatmap like plot for sequence comparison with included dendogram acc. to seaborn.clustermap().
    :param seqs: data series or array with the sequences of to plot
    :param seq_labels: labels of each sequence
    :param col_map: matplotlib colormap
    :param save_fig: bool, if figure should be saved
    :param out_dir: destination and file name, if save_fig == True
    :return:
    '''
    # convert sequences to 2-D array of numbers
    ls = []
    for seq in seqs:
        l = [ord(x) - 64 for x in seq]
        ls.append(np.asarray(l))
    ls = np.asarray(ls)

    # create array for amino acid annotation
    annots = np.asarray([list(x) for x in seqs])

    ax = sns.clustermap(ls, figsize=fig_size,
                        #standard_scale=1,
                        cbar_pos=None,
                        col_cluster=False,
                        cmap=col_map, # colormap
                        alpha=0.8, #
                        col_colors=ref_seq_enc_norm,#.values,#.to_numpy(),
                        vmin= np.argmin(ls), vmax = np.argmax(ls), # doesn't really change much
                        yticklabels=seq_labels,
                        xticklabels=2,
                        dendrogram_ratio=(.07, .0),
                        linewidths=.7, linecolor='black',
                        fmt='',
                        annot=annots,
                        method=cls_method)

    # move x labels a bit; ax.xaxis.set_label_coords(x0, y0)
    #plt.xlabel("", labelpad=50)

    # make dendogram lines thicker
    plt.yticks(rotation=0)
    for a in ax.ax_row_dendrogram.collections:
        a.set_linewidth(3)

    plt.tick_params(labelsize=15, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.tick_params(axis='x', pad=15)

    if save_fig == True:
        plt.savefig(out_dir, bbox_inches='tight')

    plt.show()

sequence_heatmap(seqs, seq_labels, cls_method='average' ,col_map="mako", fig_size=(55, 10), save_fig=False, out_dir=out_dir)






# set absolute path to where Code folder is
abs_path = '/Users/lerlach/Documents/GPPaper/Sonstiges/GP_publication/Code'

## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = abs_path + '/VDJ_Sequence_Selection/data/Plots/Sequence_clusterheatmaps/'


# Load novel variants data
# data_novvars = pd.read_csv(abs_path+"/GP_implementation/data/final_validation/novel_variants_KD_vs_preds.csv")
#
# seqs = data_novvars['Sequences']
# kd_vals = data_novvars['KD value [nM]']
# seq_labels = ["{}, KD: {:.3f} nM".format(y, kd_vals[i]) for i, y in enumerate([" ".join(["tHC", str(x+1)]) for x in range(len(seqs))])]
# out_dir = dir_out + 'Sequence_heatmap_novel_vars.png'
#
#
# sequence_heatmap(seqs, seq_labels, col_map="mako", fig_size=(30, 5), save_fig=True, out_dir=out_dir)



###### LOAD DATA #######
data = pd.read_csv(abs_path+"/GP_implementation/data/input/input_HCs.csv", usecols=['SampleID', 'Sequence', 'KD'])
seqs = data['Sequence']
kd_vals = data['KD']
seq_labels = ["{}, KD: {:.3f} nM".format(y, kd_vals[i]) for i, y in enumerate(data["SampleID"])]
out_dir = dir_out + 'Sequence_heatmap_orig_vars_av.png'


sequence_heatmap(seqs, seq_labels, cls_method='average' ,col_map="mako", fig_size=(35, 10), save_fig=False, out_dir=out_dir)





# dict for ref seq
ref_seq = pd.read_csv(abs_path+"/GP_implementation/data/final_validation/germline_KD.csv", usecols=['SampleID', 'Sequence', 'KD'])


cmap = plt.cm.get_cmap('mako')
ref_seq_enc = pd.Series([ord(x) - 64 for x in ref_seq.Sequence[0]])
norm = mc.Normalize(vmin = np.argmin(ref_seq_enc), vmax = np.argmax(ref_seq_enc))

ref = [cmap(x) for x in norm(ref_seq_enc)]
ref_seq_enc_norm = pd.DataFrame(index=['Germline sequence'], columns=range(0,122))
ref_seq_enc_norm.loc['Germline sequence', ] = ref

#ref_seq_enc_norm = ref_seq_enc_norm.rename("Germline sequence")
#ref_seq_enc_norm = pd.DataFrame(data = {"Germline sequence":norm(ref_seq_enc)})







########################### adjust figure

fig_size=(35, 10)

# convert sequences to 2-D array of numbers
    ls = []
    for seq in seqs:
        l = [ord(x) - 64 for x in seq]
        ls.append(np.asarray(l))
    ls = np.asarray(ls)

    # create array for amino acid annotation
    annots = np.asarray([list(x) for x in seqs])

    f, ax = plt.subplots(figsize=fig_size)
    ax = sns.heatmap(ls,
                    cbar=None,
                    #col_cluster=False,
                    #cmap=col_map, # colormap
                    alpha=0.9, # Transparency of the colours
                    #col_colors=ref_seq_enc_norm.values,#.to_numpy(),
                    vmin= np.argmin(ls), vmax = np.argmax(ls), # doesn't really change much
                    yticklabels=seq_labels,
                    xticklabels=2,
                    # dendrogram_ratio=(.07, .0),
                    #linewidths=.7,# line width between the squares
                    #linecolor='black', # line color between the squares
                    fmt='', # format of the annotations;
                    annot=annots#,
                    #method=cls_method
                    )


    # make dendogram lines thicker
    # plt.yticks(rotation=0)
    # for a in ax.ax_row_dendrogram.collections:
    #     a.set_linewidth(3)

    # make horizontal lines thicker
    for i in range(ls.shape[1]+1):
        ax.axhline(i, color='white', lw=2.5)

    # set axis and labels
    plt.tick_params(labelsize=15, labelbottom=False, bottom=False, top=True, labeltop=True)

    # move x labels a bit; ax.xaxis.set_label_coords(x0, y0)
    # plt.tick_params(axis='x', pad=15)

    # if save_fig == True:
    #     plt.savefig(out_dir, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
