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

###### BOX PLOT - ORIGINAL SEQUENCES ######
# setup a box plot with the original sequences
sns.set(
        style="ticks",                   # The 'ticks' style
        rc={"figure.figsize": (6, 8),      # width = 6, height = 9
            })  # Axes colour


b = sns.boxplot(data = KD_seqs_orig, y = "KD_nM",
                width=0.4,  # The width of the boxes
                color='#1874cd',
                #palette="Paired",
                showfliers = False)  # Sop showing the fliers)
b = sns.stripplot(data = KD_seqs_orig,
                  y = "KD_nM",
                  color = "black",)

# Set the y axis and font size
b.set_ylabel("kD [nM]", fontsize = 14)
# Set the x axis label and font size
b.set_xlabel("selected mAb variants", fontsize = 14)
# Set the plot title and font size
b.set_title("kD values of the selected mAb variants", fontsize = 16)
# Remove axis spines
sns.despine(offset = 5, trim = True)

plt.show()




###### BOX PLOT - NOVEL SEQUENCES ######
sns.set(
        style="ticks",                   # The 'ticks' style
        rc={"figure.figsize": (6, 8),      # width = 6, height = 9
            })  # Axes colour

b = sns.boxplot(data = KD_seqs_novel, y = "KD_nM",
                width=0.4,  # The width of the boxes
                color='#cd0000',
                #palette="Paired",
                showfliers = False)  # Sop showing the fliers)
b = sns.stripplot(data = KD_seqs_novel,
                  y = "KD_nM",
                  color = "black",)

# Set the y axis and font size
b.set_ylabel("kD [nM]", fontsize = 14)
# Set the x axis label and font size
b.set_xlabel("selected mAb variants", fontsize = 14)
# Set the plot title and font size
b.set_title("kD values of the selected mAb variants", fontsize = 16)
# Remove axis spines
sns.despine(offset = 5, trim = True)

plt.show()



###### BOX PLOT - COMBINED SEQUENCES ######
# merge the dataframes
KD_merge = pd.concat([KD_seqs_orig, KD_seqs_novel])
KD_merge = KD_merge.dropna(subset=['KD_nM'])

# statistical test btw KD values
group1 = KD_merge['KD_nM'][KD_merge['dataset'] == "originally_selected"]
group2 = KD_merge['KD_nM'][KD_merge['dataset'] == "novel_selected"]
# Run the t-test
t = ttest_ind(group1, group2)
# The t-test returns 2 values: the test statistic and the pvalue
print(np.round(t[1], 5))


# setup a box plot with the original sequences
sns.set(
        style="ticks",                   # The 'ticks' style
        rc={"figure.figsize": (6, 8),      # width = 6, height = 9
            })  # Axes colour

# custom palette
c_pal = sns.color_palette(['#1874cd','#cd0000'])

b = sns.boxplot(data = KD_merge, x = 'dataset', y = "KD_nM",
                width=0.6,  # The width of the boxes
                #color=['#1874cd','#cd0000'],
                palette=c_pal,
                showfliers = False)  # Sop showing the fliers)
b = sns.stripplot(data = KD_merge, x = 'dataset',
                  y = "KD_nM",
                  color = "black",)

# Set the y axis and font size
b.set_ylabel("kD [nM]", fontsize = 14)
# Set the x axis label and font size
b.set_xlabel("selected mAb variants", fontsize = 14)
# Set the plot title and font size
plt.suptitle("kD values of the selected mAb variants", fontsize=16)
# subtitle
b.set_title("p-value = {}".format(np.round(t[1], 5)), fontsize = 14)
# Remove axis spines
sns.despine(offset = 5, trim = True)
#plt.show()
plt.savefig(os.path.join(save_path, 'KD_originally_novel_variants_Ttest.pdf'))



