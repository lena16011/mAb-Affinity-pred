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
target_CDR3_AA = lines[7].strip()

##### (1) Calculate the Levenshtein distance to the target AA sequence #### of EACH DATASET
# save the ReadID and the levenshtein distance as a dictionary
levNames = []
for i in range(4):
    for j in range(3):
        levNames.append(str("levDistAA" + str(i) + list[j]))

for j in range(len(dataNames)):
    locals()[levNames[j]] = {}
    for i in range(len(locals()[dataNames[j]])):
        lev = stringdist.levenshtein_norm(locals()[dataNames[j]].CDR3_AA[i], target_CDR3_AA)
        locals()[levNames[j]][locals()[dataNames[j]].loc[i, 'ReadID']] = lev


################################ Similarity filtering
# define filter filter function;
def filter_dist(dict, similarity):
    return {k:v for (k,v) in dict.items() if 1-v > similarity}

#### filter for 80% similarity and store new data sets
filtlevNames = []
for i in range(4):
    for j in range(3):
        filtlevNames.append(str("filtlevDist80" + str(i) + list[j]))

for i in range(len(dataNames)):
    locals()[filtlevNames[i]] = filter_dist(locals()[levNames[i]], 0.8)
# print number of sequences
for i in range(len(filtlevNames)):
    print(str(filtlevNames[i]), len(locals()[filtlevNames[i]]))
# there is only in the dataset 3A, 3B, 3C sequences that have 80% similarity




# add a column to the datasets to mark the sequences with boost and Dataset

list_data = ['A', 'B', 'C']
list_boost = ['0', '1', '2', '3']

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

# merge the filtered files
mFiles = ['f_data0', 'f_data1', 'f_data2', 'f_data3']
count = 0
for i in range(len(mFiles)):
    fmerge1 = locals()[filtdataNames[count]]
    fmerge2 = locals()[filtdataNames[count+1]]
    fmerge3 = locals()[filtdataNames[count+2]]
    fmerge = fmerge1.append(fmerge2, ignore_index=True)
    fmerge = fmerge.append(fmerge3, ignore_index=True)
    locals()[mFiles[i]] = fmerge
    count = count + 3
# merge the data of all mice
filt_data_all = f_data0.append(f_data1, ignore_index=True)
filt_data_all = filt_data_all.append(f_data2, ignore_index=True)
filt_data_all = filt_data_all.append(f_data3, ignore_index=True)






#save the filtered sequences
filtFilePath = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/similarity80_FilesAA/'

sim80Names = []
for i in range(4):
    for j in range(3):
        sim80Names.append(str("simfilt80DataAA" + str(i) + list[j]))
for i in range(len(sim80Names)):
    locals()[filtdataNames[i]].to_csv(str(filtFilePath + sim80Names[i] + '.txt'),
                                      sep = '\t', index = False)

#### filter for 70% similarity
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
#data_uniqVDJ = data_all.drop_duplicates(['VDJ_AA'])
#reset the index of the unique entries
data_uniqCDR3 = data_uniqCDR3.reset_index(drop=True)
#data_uniqVDJ = data_uniqVDJ.reset_index(drop=True)

# save this file for later
filt_file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
data_uniqCDR3.to_csv(str(filt_file_path + 'data_uniqCDR3.txt'), sep='\t')

#### Note: here the Index is connected to the ReadID and thus to the information about which
# read comes from which mouse;
# the index, not the ReadID is used for the calculation of the distance matrices;
# our target sequence has the following entry:
target_entry = data_uniqCDR3[data_uniqCDR3['CDR3_AA']==targetCDR3]
# 17358      34    productive  ...        A     3
# so our sequence is in row 17358

### Calculate levenshtein distance for merged, unique dataset
# save the ReadID and the levenshtein distance as a dictionary
#lev_dist_all = {}
#for i in range(len(data_uniqCDR3)):
#    lev = stringdist.levenshtein_norm(data_uniqCDR3.CDR3_AA[i], target_CDR3_AA)
#    lev_dist_all[data_uniqCDR3.loc[i, 'ReadID']] = lev
######## use another approach to take the already filtered files;

# filter the distance matrix
#filt_dist_all_80 = filter_dist(lev_dist_all, 0.8)
##################################################### !!!!!!!!!!!!! there ares no unique ReadIDs! solve that!


###### (1) Calculate the distance matrix for ALL sequences
col=range(len(data_uniqCDR3.CDR3_AA))
dist_matrix_CDR3 = pd.DataFrame(columns=col)
data_ser = pd.DataFrame(columns=col)
for i in range(len(data_uniqCDR3.CDR3_AA)):
    data_ser = data_uniqCDR3.CDR3_AA[:i+1].apply(stringdist.levenshtein_norm,
                                              args=(data_uniqCDR3.CDR3_AA[i],))
    dist_matrix_CDR3.loc[i] = data_ser.T

# save the distance matrix in a txt file
filt_file_path = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/'
dist_matrix_CDR3.to_csv(str(filt_file_path + 'uniqCDR3_DistMatrix.txt'),
                                      sep = '\t', float_format=np.float32, na_rep='0')


##### (2) Calculate the distance matrix for sequences of same CDR3 length
# filter the data set for CDR3 length
data_uniq_length = pd.DataFrame(columns=data_uniqCDR3.columns)
for i in range(len(data_uniqCDR3.CDR3_AA)):
    if len(data_uniqCDR3.CDR3_AA[i]) == len(target_CDR3):
        uniq_length = data_uniqCDR3.iloc[i,]
        data_uniq_length = data_uniq_length.append(uniq_length)
# reset the index
data_uniq_length.reset_index(drop=True, inplace=True)
# save this file for later
filt_file_path2 = '/media/lena/LENOVO/Dokumente/Masterarbeit/data/Clustering/Clustering_uniq_length_dist_matrix/'
data_uniq_length.to_csv(str(filt_file_path2 + 'data_uniq_length_CDR3.txt'), sep='\t')

#Calculate a new distance matrix for the new sequences
import stringdist
col=range(len(data_uniq_length.CDR3_AA))
dist_matrix_uniq_length = pd.DataFrame(columns=col)
data_ser = pd.DataFrame(columns=col)
for i in range(len(data_uniq_length.CDR3_AA)):
    data_ser = data_uniq_length.CDR3_AA[:i+1].apply(stringdist.levenshtein_norm,
                                              args=(data_uniq_length.CDR3_AA[i],))
    dist_matrix_uniq_length.loc[i] = data_ser.T
# save dataframe for clustering
dist_matrix_uniq_length.to_csv(str(filt_file_path2 + 'uniq_length_dist_matrix.txt'), sep='\t')
# where in the matrix is the target sequence
target_entry2 = data_uniq_length[data_uniq_length['CDR3_AA']==targetCDR3]
print(target_entry2)
#      ReadID Functionality  ...  Dataset Boost
# 1069     34    productive  ...        A     3