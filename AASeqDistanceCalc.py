import pandas as pd
import stringdist

# Load in the data sets
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
    locals()[dataNames[i]] = pd.read_csv(files[i], sep='\t', header=None, names=['ReadID', 'Functionality',
                                                'CDR3_AA', 'Table_CDR3', 'VDJ_AA', 'VDJ_NT'], skiprows=1)

# load in the target sequence
with open(dir_name+"sequences.fasta", 'r') as targetFile:
lines= targetFile.readlines()
#targetVDJ = lines[1].strip()
targetCDR3 = lines[7].strip()


##### Calculate the Levenshtein distance to the target AA sequence
levNames = []
for i in range(4):
    for j in range(3):
        levNames.append(str("levDistAA" + str(i) + list[j]))

for j in range(len(dataNames)):
    locals()[levNames[j]] = {}
    for i in range(len(locals()[dataNames[j]])):
        lev = stringdist.levenshtein_norm(locals()[dataNames[j]].CDR3_AA[i], targetCDR3)
        locals()[levNames[j]][locals()[dataNames[j]].loc[i, 'ReadID']] = lev


######### Similarity filtering #####
# define filter filter function;
def filter_dist(dict, similarity):
    return {k:v for (k,v) in dict.items() if 1-v > similarity}

# filter for 80% similarity and store new data sets
filtlevNames = []
for i in range(4):
    for j in range(3):
        filtlevNames.append(str("filtlevDist80" + str(i) + list[j]))

for i in range(len(dataNames)):
    locals()[filtlevNames[i]] = filter_dist(locals()[levNames[i]], 0.8)
# print number of sequences
for i in range(len(filtlevNames)):
    print(str(filtlevNames[i]), len(locals()[filtlevNames[i]]))

# filter the original data for the found similar sequences and create new dataframes
filtdataNames = []
for i in range(4):
    for j in range(3):
        filtdataNames.append(str("filtData80AA" + str(i) + list[j]))

for j in range(len(dataNames)):
    locals()[filtdataNames[j]] = pd.DataFrame(columns = data0A.columns)
    for i in range(len(locals()[dataNames[j]][['ReadID']])):
        if locals()[dataNames[j]].loc[i, 'ReadID'] in locals()[filtlevNames[j]].keys():
            locals()[filtdataNames[j]].loc[i] = locals()[dataNames[j]].loc[i]

#save the filtered sequences
filtFilePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/similarity80_FilesAA/'

sim80Names = []
for i in range(4):
    for j in range(3):
        sim80Names.append(str("simfilt80DataAA" + str(i) + list[j]))
for i in range(len(sim80Names)):
    locals()[filtdataNames[i]].to_csv(str(filtFilePath + sim80Names[i] + '.txt'),
                                      sep = '\t', index = False)

# filter for 70% similarity
filtlevNames70 = []
for i in range(4):
    for j in range(3):
        filtlevNames70.append(str("filtlevDist70" + str(i) + list[j]))

for i in range(len(dataNames)):
    locals()[filtlevNames70[i]] = filter_dist(locals()[levNames[i]], 0.7)
# print number of sequences
for i in range(len(filtlevNames70)):
    print(str(filtlevNames70[i]), len(locals()[filtlevNames70[i]]))

# filter the original data for the found similar sequences and create new dataframes
filtdataNames = []
for i in range(4):
    for j in range(3):
        filtdataNames.append(str("filtData70AA" + str(i) + list[j]))

for j in range(len(dataNames)):
    locals()[filtdataNames[j]] = pd.DataFrame(columns = data0A.columns)
    for i in range(len(locals()[dataNames[j]][['ReadID']])):
        if locals()[dataNames[j]].loc[i, 'ReadID'] in locals()[filtlevNames70[j]].keys():
            locals()[filtdataNames[j]].loc[i] = locals()[dataNames[j]].loc[i]

#save the filtered sequences
filtFilePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/similarity70_FilesAA/'

sim80Names = []
for i in range(4):
    for j in range(3):
        sim80Names.append(str("simfilt70DataAA" + str(i) + list[j]))
for i in range(len(sim80Names)):
    locals()[filtdataNames[i]].to_csv(str(filtFilePath + sim80Names[i] + '.txt'),
                                      sep = '\t', index = False)

########## Preparations for distance matrix
# add a column to the datasets to mark the sequences with boost and Dataset
listLet = ['A', 'B', 'C']
listBoost = ['0', '1', '2', '3']

for i in range(len(dataNames)):
    if dataNames[i][5] == "A":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Dataset = ['A']*len(locals()[dataNames[i]]))
    elif dataNames[i][5] == "B":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Dataset=['B']*len(locals()[dataNames[i]]))
    elif dataNames[i][5] == "C":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Dataset=['C']*len(locals()[dataNames[i]]))

for i in range(len(dataNames)):
    if dataNames[i][4] == "0":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['0']*len(locals()[dataNames[i]]))
    elif dataNames[i][4] == "1":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['1']*len(locals()[dataNames[i]]))
    elif dataNames[i][4] == "2":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['2']*len(locals()[dataNames[i]]))
    elif dataNames[i][4] == "3":
        locals()[dataNames[i]] = locals()[dataNames[i]].assign(Boost = ['3']*len(locals()[dataNames[i]]))

# merge the A, B, C datasets
mFiles = ['data0', 'data1', 'data2', 'data3']
count = 0
for i in range(len(mFiles)):
    merge1 = locals()[dataNames[count]]
    merge2 = locals()[dataNames[count+1]]
    merge3 = locals()[dataNames[count+2]]
    merge = merge1.append(merge2, ignore_index=True)
    merge = merge.append(merge3, ignore_index=True)
    locals()[mFiles[i]] = merge
    count = count + 3
# merge the data of all mice
data_all = data0.append(data1, ignore_index=True)
data_all = data_all.append(data2, ignore_index=True)
data_all = data_all.append(data3, ignore_index=True)

# drop the duplicate sequences
data_uniqCDR3 = data_all.drop_duplicates(['CDR3_AA'])
data_uniqVDJ = data_all.drop_duplicates(['VDJ_AA'])
#reset the index of the unique entries
data_uniqCDR3 = data_uniqCDR3.reset_index(drop=True)
data_uniqVDJ = data_uniqVDJ.reset_index(drop=True)

#### Note: here the Index is connected to the ReadID and thus to the information about which
# read comes from which mouse;
# the index, not the ReadID is used for the calculation of the distance matrices;

###### Calculate the distance matrix

col=range(len(data_uniqCDR3.CDR3_AA))
dist_matrix_CDR3 = pd.DataFrame(columns=col)
data_ser = pd.DataFrame(columns=col)
for i in range(len(data_uniqCDR3.CDR3_AA)):
    data_ser = data_uniqCDR3.CDR3_AA[:i+1].apply(stringdist.levenshtein_norm,
                                              args=(data_uniqCDR3.CDR3_AA[i],))
    dist_matrix_CDR3.loc[i] = data_ser.T

# save the distance matrix in a txt file
filtFilePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
dist_matrix_CDR3.to_csv(str(filtFilePath + 'uniqCDR3_DistMatrix.txt'),
                                      sep = '\t')


