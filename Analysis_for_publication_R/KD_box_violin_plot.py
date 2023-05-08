'''
Script to visualize the measured KD values from the mAb variants (VDJ sequences selected/filtered);
that occur with our selected CDR3s; Network plots will be created with:

 - Box plot of KDs
 - Violin plot of KDs
 - scatter plot of KDs

'''


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

###### BOX PLOT - FUNCTION ######
def box_plot_KDs(data, x = None, y = None, groups=None, color_palette = 'Paired', fig_size = (6, 8), save_fig = None):
    '''
    Function to plot the KD values as a box plot with whiskers, no fliers
    params:
        data (pd.DataFrame()) : dataframe with the KD values and column as for the groups, if t-test should be performed;
        x (string) : column name of the x values (groups column), e.g. "dataset", or None
        y (string): column name of the y values (KD val column), e.g. "KD_nM"
        groups (None, list of strings): if None, no t-test will be performed, else list of entries in the x (e.g. group) column of
        the dataframe, comparison of these 2 groups in the t-test;
        color_palette (list of strings, string): HEX values of the colors to be used in the plot or name of the palette (Matplotlib, Seaborn);
        save_fig (None, os.Path) : if None, plot will only be shown, if os.Path is provided, the plot will be saved in the os.Path file name;

    returns: None
    '''
    if groups != None:
        # statistical test btw KD values
        group1 = data[y][data[x] == groups[0]]
        group2 = data[y][data[x] == groups[1]]
        # Run the t-test
        t = ttest_ind(group1, group2)
        # The t-test returns 2 values: the test statistic and the pvalue
        #print(np.round(t[1], 5))


    # setup a box plot with the original sequences
    sns.set(
            style="ticks",                   # The 'ticks' style
            rc={"figure.figsize": fig_size,      # width = 6, height = 9
                })  # Axes colour

    # custom palette
    if len(color_palette) > 1:
        c_pal = sns.color_palette(color_palette)
    else:
        c_pal = color_palette

    b = sns.boxplot(data = data, x = x, y = y,
                    width=0.6,  # The width of the boxes
                    palette=c_pal,
                    showfliers = False)  # Stop showing the fliers)
    b = sns.stripplot(data = data, x = x, y = y,
                      color = "black",)

    # Set the y axis and font size
    b.set_ylabel("kD [nM]", fontsize = 14)
    # Set the x axis label and font size
    b.set_xlabel("selected mAb variants", fontsize = 14)

    if groups != None:
        # Set the plot title and font size
        plt.suptitle("kD values of selected & novel mAb variants", fontsize=16)
        # subtitle
        b.set_title("p-value = {}".format(np.round(t[1], 5)), fontsize = 14)
    else:
        # Set the plot title and font size
        b.set_title("kD values of the selected mAb variants", fontsize=16)
    # Remove axis spines
    sns.despine(offset = 5, trim = True)

    if save_fig != None:
        plt.savefig(save_fig)
    plt.show()


###### SET INPUT DIRECTORIES & LOAD DATA ######
# Set paths to originally selected and novel sequences
in_dir1 = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/final_validation/'
in_f1 = os.path.join(in_dir1, 'novel_variants_AA_KDs.csv')

in_dir2 = '/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/GP_implementation/data/input'
in_f2 = os.path.join(in_dir2, 'input_HCs.csv')

# set path to output folder
save_path = "/Users/lerlach/Documents/current_work/GP_publication/code_git/Lena/Analysis_for_publication_R/KD_plots"

# load selected sequences
KD_seqs_orig = pd.read_csv(in_f2, usecols=['SampleID', 'Sequence', 'KD'])
# rename columns a bit
KD_seqs_orig.columns = ['SampleID', 'VDJ_AA', 'KD_nM']
KD_seqs_orig["label"] = "originally_selected"
KD_seqs_orig["dataset"] = "originally_selected"

# load KD valuse
KD_seqs_novel = pd.read_csv(in_f1, delimiter = ',')
KD_seqs_novel.columns = ['SampleID', 'IDs', 'KD_M', 'KD_nM', 'label', 'VDJ_AA']
KD_seqs_novel["dataset"] = "novel_selected"

# merge the dataframes
KD_merge = pd.concat([KD_seqs_orig, KD_seqs_novel])
KD_merge = KD_merge.dropna(subset=['KD_nM'])


###### BOX PLOT - COMBINED SEQUENCES ######
color_palette = ['#1874cd','#cd0000']
box_plot_KDs(KD_merge, x = "dataset", y = "KD_nM", groups=["originally_selected", "novel_selected"],
             color_palette = color_palette, save_fig = os.path.join(save_path, 'KD_originally_novel_variants_Ttest.pdf'))

###### BOX PLOT - ORIGINAL SEQUENCES ######
box_plot_KDs(KD_seqs_orig, x = None, y = "KD_nM", groups=None,
             color_palette = ['#1874cd'], fig_size = (5,8), save_fig = os.path.join(save_path, 'KD_original_variants.pdf'))

###### BOX PLOT - NOVEL SEQUENCES ######
box_plot_KDs(KD_seqs_novel, x = None, y = "KD_nM", groups=None,
             color_palette = ['#cd0000'], fig_size = (5.5,8), save_fig = os.path.join(save_path, 'KD_novel_variants.pdf'))

