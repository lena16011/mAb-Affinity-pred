'''
Script to generate heatmap-like plots from AA sequences for comparison


'''

# seaborn cluster map with dendogram
import seaborn as sns; sns.set(color_codes=True)


def sequence_heatmap(seqs, seq_labels, col_map="mako", fig_size=(30, 5), save_fig=False, out_dir=None):
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

    ax = sns.clustermap(ls, figsize=fig_size,
                        cbar_pos=None,
                        col_cluster=False,
                        cmap=col_map,
                        yticklabels=seq_labels,
                        xticklabels=2,
                        dendrogram_ratio=(.07, .0),
                        linewidths=.7, linecolor='black')

    # make dendogram lines thicker
    plt.yticks(rotation=0)
    for a in ax.ax_row_dendrogram.collections:
        a.set_linewidth(3)

    plt.tick_params(labelsize=15, labelbottom = False, bottom=False, top = True, labeltop=True)

    if save_fig == True:
        plt.savefig(out_dir, bbox_inches='tight')

    plt.show()



## SET OUTPUT DIRECTORIES (for plots to save)
dir_out = abs_path + '/data/Plots/GP_model/CV_correlation/Matern_kernel/'


# Load novel variants data
data_novvars = pd.read_csv(abs_path+"/data/final_validation/novel_variants_KD_vs_preds.csv")

seqs = data_novvars['Sequences']
kd_vals = data_novvars['KD value [nM]']
seq_labels = ["{}, KD: {:.3f} nM".format(y, kd_vals[i]) for i, y in enumerate([" ".join(["tHC", str(x+1)]) for x in range(len(seqs))])]
out_dir = dir_out + 'Sequence_heatmap_novel_vars.png'


sequence_heatmap(seqs, seq_labels, col_map="mako", fig_size=(30, 5), save_fig=True, out_dir=out_dir)



###### LOAD DATA #######
data = pd.read_csv(input_f_seq, usecols=['SampleID', 'Sequence', 'KD'])
seqs = data['Sequence']
kd_vals = data['KD']
seq_labels = ["{}, KD: {:.3f} nM".format(y, kd_vals[i]) for i, y in enumerate(data["SampleID"])]
out_dir = dir_out + 'Sequence_heatmap_orig_vars.svg'


sequence_heatmap(seqs, seq_labels, col_map="mako", fig_size=(30, 10), save_fig=True, out_dir=out_dir)

