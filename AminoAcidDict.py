import numpy as np
from collections import OrderedDict
import pandas as pd

AminoAcids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "L", "I", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
DNA = ["A", "C", "G", "T"]

uniSizeFeat = ["Micro", "Big"]
uniSizeFeat.sort()
uniPolarityFeat = ["Non", "Basic", "Polar", "Acidic"]
uniPolarityFeat.sort()


def charToIndex(stateSpace):
    result = dict()
    spaceSize = len(stateSpace)
    for index in range(spaceSize):
        state = stateSpace[index]
        result[state] = index

    return result

def indexToChar(stateSpace):
    result = dict()
    spaceSize = len(stateSpace)
    for index in range(spaceSize):
        result[index] = stateSpace[index]
    return result

def aminoAcidToIndex():
    result = charToIndex(AminoAcids)
    return result
def indexToAminoAcids():
    result = indexToChar(AminoAcids)
    return result

def aminoAcidsVecToIndex(aminoAcidsVector):
    return list(map(lambda x: aminoAcidToIndex()[x], aminoAcidsVector))

def sizeToAminoAcids():
    result = {}
    result['Micro'] = ["A", "G", "S", "P", "C", "T", "N", "D"]
    result['Big'] = ["R", "E", "Q", "H", "I", "L", "K", "M", "F", "W", "Y", "V"]
    return result

def aminoAcidToSize():
    result = {}
    sizeToAminoAcidsTemplate = sizeToAminoAcids()
    for key in sizeToAminoAcidsTemplate.keys():
        for value in sizeToAminoAcidsTemplate[key]:
            result[value] = key
    return result

def polarityToAminoAcids():
    result = {}
    result['Non'] = ["A", "C", "G", "I", "L", "M", "F", "P", "W", "V"]
    result['Basic'] = ["R", "H", "K"]
    result['Polar'] = ["N", "Q", "S", "T", "Y"]
    result['Acidic'] = ["D", "E"]
    return result
def aminoAcidToPolarity():
    result = {}
    polarityToAminoAcidsTemplate = polarityToAminoAcids()
    for key in polarityToAminoAcidsTemplate.keys():
        for value in polarityToAminoAcidsTemplate[key]:
            result[value] = key
    return result


def pairAminoAcidToFeats(aminoAcid0, aminoAcid1):
    ## get univariate size and polarity features for each amino acid
    size0 = aminoAcidToSize()[aminoAcid0]
    size1 = aminoAcidToSize()[aminoAcid1]

    polarity0 = aminoAcidToPolarity()[aminoAcid0]
    polarity1 = aminoAcidToPolarity()[aminoAcid1]

    ## create bivariate features
    ## see which the first letter of the size or polarity appears earlier in alphabetical order,
    ## then that feature should appears first when we concatenate the two strings
    if size0[0] <= size1[0]:
        sizeFeat = size0 + size1
    else:
        sizeFeat = size1 + size0

    if polarity0[0] <= polarity1[0]:
        polarityFeat = polarity0 + polarity1
    else:
        polarityFeat = polarity1 + polarity0

    result = {}
    result['sizeFeat'] = sizeFeat
    result['polarityFeat'] = polarityFeat

    return result


## create a list to include all the bivariate features
def sizeAndPolarityFeatList():
    uniSize = uniSizeFeat
    uniPolarity = uniPolarityFeat

    bivariateFeat = list()
    for i in range(len(uniSize)):
        for j in np.arange(i, len(uniSize), 1):
            strFeat = uniSize[i] + uniSize[j]
            bivariateFeat.append(strFeat)

    for i in range(len(uniPolarity)):
        for j in np.arange(i, len(uniPolarityFeat), 1):
            strFeat = uniPolarity[i] + uniPolarity[j]
            bivariateFeat.append(strFeat)

    return bivariateFeat


allBivariateFeat = sizeAndPolarityFeatList()

## given a bivariate feature, finding its location
## in the list of all bivariate features from sizeAndPolarityFeatList()
def findLocationIndexBivariateFeat(bivariateFeatStr):
    ## bivariateFeatStr represents the bivariate feature string
    return allBivariateFeat.index(bivariateFeatStr)


## given a pair of states, get all the bivariate feature index
## out of all the bivariate feature list
def getBivaraiteFeatGivenPairStates(aminoAcid0, aminoAcid1):
    featDict = pairAminoAcidToFeats(aminoAcid0, aminoAcid1)
    sizeFeat = featDict['sizeFeat']
    polarityFeat = featDict['polarityFeat']

    featureIndex = list()
    sizeIndex = findLocationIndexBivariateFeat(sizeFeat)
    featureIndex.append(sizeIndex)

    polarityIndex = findLocationIndexBivariateFeat(polarityFeat)
    featureIndex.append(polarityIndex)

    return featureIndex

## loop over all amino acids and cache the results
def getBivariateFeatGivenAllAminoAcidPairs():

    nStates = len(AminoAcids)
    HardCodedDict = OrderedDict()

    for i in range(nStates):
        for j in np.arange((i+1), nStates, 1):
            state0 = AminoAcids[i]
            state1 = AminoAcids[j]
            ## add nPriorUniFeatures to each element of feature index
            featureIndices = np.array(getBivaraiteFeatGivenPairStates(state0, state1), dtype=np.int)
            HardCodedDict[(i, j)] = featureIndices
            HardCodedDict[(j, i)] = featureIndices

    return HardCodedDict



## create all the helper functions to rank pairs of amino acids according to their distances
## read amino acids pairs distances into a csv file
# AminoAcidsDist = pd.read_excel("/Users/crystal/Dropbox/rejfreePy_main_results/ReadDataset/EuclideanDistanceForAminoAcids.xlsx",sheet_name="Sheet1",header=0)
AminoAcidsDist = pd.read_excel("/home/tingtingzhao/rejfreePy_main_results/ReadDataset/EuclideanDistanceForAminoAcids.xlsx", sheet_name="Sheet1", header=0)
AminoAcidsDist.set_index("Amino Acids", inplace=True)
AminoAcidsDist.head(3)


## transforming this distance matrix into a dictionary where given a pair of
## amino acids index, eg. (0, 1) corresponds to A and R, the index of amino acids
## is determined by the order of characters we provide in "Amino Acids" in the first line
def getDistForPairsOfAminoAcids():
    ## transform the row names of Amino Acids into number index
    ## get the row names of AminoAcidsDist
    ## the row names are a list of amino acids
    rowNames = AminoAcidsDist.index
    colNames = AminoAcidsDist.columns.values
    ## transfer rowNames and colNames into number index
    charToIndexDict = charToIndex(AminoAcids)
    ## get the corresponding index
    rowIndex = [charToIndexDict[x] for x in rowNames]
    colIndex = [charToIndexDict[x] for x in colNames]
    ## the method is correct since np.take(AminoAcids, rowIndex) is the same as rowNames and colNames
    ## create a distance dictionary where the key of the dictionary is a tuple consist of the index
    ## of the corresponding amino acid index pairs. For example, result[(18, 8)] = result[(Y, H)] = 83
    result = dict()
    for i in range(len(rowIndex)):
        for j in range(len(colIndex)):
            if i != j:
                key = (i, j)
                result[key] = AminoAcidsDist.iloc[i, j]

    return result

def rankAminoAcidsPairs():
    ## find the amino acid pair with the smallest distance.
    ## create a copy of AminoAcidsDist
    AminoAcidsDistCopy = AminoAcidsDist.copy(deep=True)
    ## first assign positive infinity to diagonal elements in the distance matrix
    AminoAcidsDistCopy.values[[np.arange(AminoAcidsDistCopy.shape[0])]*2] = 100000
    ## find the row and column index for the element which is the minimum in the distance matrix
    ArgMinRowAndCol = np.unravel_index(np.argmin(AminoAcidsDistCopy.values), AminoAcidsDistCopy.shape)[0]
    argRowIndex = ArgMinRowAndCol[0]
    argColIndex = ArgMinRowAndCol[1]

    ## given an initial row and column index, find the index of the element within the same row or column
    ## that has the minimum distance, this is a recursion algorithm.
    




    ## create support for each row and column






    ## create a counter to count the number of existing features






















#               Y   H   Q   R   T    N   K    D    E    G    F    L    A    S  \
# Amino Acids
# Y             0  83  99  77  92  143  85  160  122  147   22   36  112  144
# H            83   0  24  29  47   68  32   81   40   98  100   99   86   89
# Q            99  24   0  43  42   46  53   61   29   87  116  113   91   68
#                P    I    M   V    C    W
# Amino Acids
# Y            110   33   36  55  194   37
# H             77   94   87  84  174  115
# Q             76  109  101  96  154  130




def test():
    ## compose a test case for function getBivariateFeatGivenAllAminoAcidPairs()
    ## randomly given two pairs of states of amino acids, and we check if the obtained bivariate feature index
    ## are correct or not.
    nStates = len(AminoAcids)
    cachedAminoAcidResult = getBivariateFeatGivenAllAminoAcidPairs()

    nRep = 100
    seeds = np.arange(1, nRep, 1)
    ## change this later to a for loop and check if the result is correct for each replicaiton
    for seed in seeds:
        np.random.seed(seed)
        aminoAcidIndex = np.random.choice(np.arange(0, 20, 1), 2, replace=False)
        print(aminoAcidIndex)
        statePair = (aminoAcidIndex[0], aminoAcidIndex[1])
        aminoAcid0 = AminoAcids[aminoAcidIndex[0]]
        aminoAcid1 = AminoAcids[aminoAcidIndex[1]]
        print(aminoAcid0)
        print(aminoAcid1)
        bivariateFeatures = pairAminoAcidToFeats(aminoAcid0, aminoAcid1)
        sizeFeat = bivariateFeatures["sizeFeat"]
        polarityFeat = bivariateFeatures["polarityFeat"]
        index0 = findLocationIndexBivariateFeat(sizeFeat)
        index1 = findLocationIndexBivariateFeat(polarityFeat)
        print(index0)
        print(index1)

        featIndex = cachedAminoAcidResult[statePair]
        print(featIndex)
        if not index0 in featIndex:
            raise ValueError("The index of the feature is not equal to the index obtained from our algorithm")

        if not index1 in featIndex:
            raise ValueError("The index of the feature is not equal to the index obtained from our algorithm")










def main():
    test()

if __name__ == "__main__": main()





























