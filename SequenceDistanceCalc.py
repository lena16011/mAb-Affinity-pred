import numpy as np
import pandas as pd
import stringdist

# Load in the data
dir_name = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/'

files = []
list = ['A', 'B', 'C']
i=0
for i in range(4):
    for j in range(3):
    files.append(str(dir_name+"filteredFiles/ess_HEL"+str(i)+"BM"+list[j]+"Annot_filtered.txt"))

dataNames = []
for i in range(4):
    for j in range(3):
        dataNames.append(str("data" + str(i) + list[j]))


for i in range(12):
    locals()[dataNames[i]] = pd.read_csv(files[i], sep='\t')

with open(dir_name+"sequences.fasta", 'r') as targetFile:
lines= targetFile.readlines()
targetVDJ = lines[1].strip()
targetCDR3 = lines[3].strip()

## calculate Levenshtein distance, but it is much slower than the stringdist.levenshtein fct;
#def levenshtein(s1, s2):
#    if len(s1) < len(s2):
#        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
#    if len(s2) == 0:
#        return len(s1)
#
#    previous_row = range(len(s2) + 1)
#    for i, c1 in enumerate(s1):
#        current_row = [i + 1]
#        for j, c2 in enumerate(s2):
#            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
#            deletions = current_row[j] + 1       # than s2
#            substitutions = previous_row[j] + (c1 != c2) # elongates for +1 if true;
#            current_row.append(min(insertions, deletions, substitutions))
#        previous_row = current_row
#
#    return previous_row[-1]

# test the calculation for the dataset A3 with the previosly defined function
#levDist3A = {}
#for i in range(len(data3A.Corrected_NucleotideSeq)):
#    levDist3A[data3A.loc[i, 'Read#']] = levenshtein(data3A.Corrected_NucleotideSeq[i], targetVDJ)

# use stringdist package
#levDist2A = {}
#for i in range(len(data2A.Corrected_NucleotideSeq)):
#    levDist2A[data2A.loc[i, 'Read#']] = stringdist.levenshtein(data2A.Corrected_NucleotideSeq[i], targetVDJ)

# loop for all the data; Calculation of the Levenshtein distance (normalized)
levNames = []
for i in range(4):
    for j in range(3):
        levNames.append(str("levDist" + str(i) + list[j]))

for j in range(len(dataNames)):
    locals()[levNames[j]] = {}
    for i in range(len(locals()[dataNames[j]])):
        lev = stringdist.levenshtein_norm(locals()[dataNames[j]].Corrected_NucleotideSeq[i], targetVDJ)
        locals()[levNames[j]][locals()[dataNames[j]].loc[i, 'Read#']] = lev


# Calculate the damerau distance with pyxdameraulevenshtein package --->
# pretty slow compared to levenshtein, therefore use stringdistpackage
#import pyxdameraulevenshtein as dlev
dlevNames = []
for i in range(4):
    for j in range(3):
        dlevNames.append(str("DlevDist" + str(i) + list[j]))

for j in range(12):
    locals()[dlevNames[j]] = {}
    for i in range(len(locals()[dataNames[j]])):
        lev = stringdist.rdlevenshtein_norm(locals()[dataNames[j]].Corrected_NucleotideSeq[i], targetVDJ)
        locals()[dlevNames[j]][locals()[dataNames[j]].loc[i, 'Read#']] = lev

# Calculate also the longest common substring --> takes also quite a long time;

# Dynamic Programming implementation of LCS problem
#def lcs(X, Y):
    # find the length of the strings
#    m = len(X)
#    n = len(Y)

    # declaring the array for storing the dp values
#    L = [[None] * (n + 1) for i in range(m + 1)]

#    """Following steps build L[m+1][n+1] in bottom up fashion
#    Note: L[i][j] contains length of LCS of X[0..i-1]
#    and Y[0..j-1]"""
#    for i in range(m + 1):
#        for j in range(n + 1):
#            if i == 0 or j == 0:
#                L[i][j] = 0
#            elif X[i - 1] == Y[j - 1]:
#                L[i][j] = L[i - 1][j - 1] + 1
#            else:
#                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
#    return L[m][n]
    # end of function lcs

#
#lcsNames = []
#for i in range(4):
#    for j in range(3):
#        lcsNames.append(str("lcsDist" + str(i) + list[j]))

#for j in range(len(dataNames)):
#    locals()[lcsNames[j]] = {}
#    for i in range(len(locals()[dataNames[j]])):
#        lev = lcs(locals()[dataNames[j]].Corrected_NucleotideSeq[i], targetVDJ)
#        locals()[lcsNames[j]][locals()[dataNames[j]].loc[i, 'Read#']] = lev

### safe the dictionaries in txt files

filePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/distances/'

for i in range(len(levNames)):
    file = open(str(filePath + str(levNames[i]) + ".txt"), 'w')
    file.write(str(locals()[levNames[i]]))
    file.close()

for i in range(len(dlevNames)):
    file = open(str(filePath + str(dlevNames[i]) + ".txt"), 'w')
    file.write(str(locals()[dlevNames[i]]))
    file.close()

# delete the values that have a similarity that is lower than 80%
# Therefore the dicts are filtered for distances < 0.2

def filter_dist(dict, similarity):
    return {k:v for (k,v) in dict.items() if 1-v > similarity}

filtlevNames = []
for i in range(4):
    for j in range(3):
        filtlevNames.append(str("filtlevDist" + str(i) + list[j]))

for i in range(len(dataNames)):
    locals()[filtlevNames[i]] = filter_dist(locals()[levNames[i]], 0.8)


# filter the original data for the found similar sequences and create new dataframes
filtdataNames = []
for i in range(4):
    for j in range(3):
        filtdataNames.append(str("filtData" + str(i) + list[j]))

for j in range(len(dataNames)):
    locals()[filtdataNames[j]] = pd.DataFrame(columns = data0A.columns)
    for i in range(len(locals()[dataNames[j]][['Read#']])):
        if locals()[dataNames[j]].loc[i, 'Read#'] in locals()[filtlevNames[j]].keys():
            locals()[filtdataNames[j]].loc[i] = locals()[dataNames[j]].loc[i]

# Template;
#filtData3A = pd.DataFrame(columns = data3A.columns)
#for i in range(len(data3A[['Read#']])):
#    if data3A.loc[i, 'Read#'] in filtlevDist3A.keys():
#        filtData3A.loc[i] = data3A.loc[i]

# write the filtered sequences to files
filtFilePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/similarity_80Files/'

sim80Names = []
for i in range(4):
    for j in range(3):
        sim80Names.append(str("simfilt80Data" + str(i) + list[j]))
for i in range(len(sim80Names)):
    locals()[filtdataNames[i]].to_csv(str(filtFilePath + sim80Names[i] + '.txt'),
                                      sep = '\t', index = False)
# Template;
#filtData0A.to_csv(str(filtFilePath+'simfilt80Data0A'), sep = '\t', index = False)





# hopeless try to make list comprehension...
#filtData3A = [data3A.iloc[i, :] for i in data3A.loc[i:, 'Read#'] if data3A.loc[i:, 'Read#'] == filtlevDist3A.keys()]
#iltData3A = [data3A.iloc[i, :] for i in range(len(data3A)) if data3A.loc[i:, 'Read#'] in filtlevDist3A.keys()]