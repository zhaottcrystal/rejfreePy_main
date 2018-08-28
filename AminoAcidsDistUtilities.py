AminoAcids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "L", "I", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

from AminoAcidDict import charToIndex
from AminoAcidDict import indexToChar
from AminoAcidDict import indexToAminoAcids
from AminoAcidDict import aminoAcidToIndex
from AminoAcidDict import aminoAcidsVecToIndex
import pandas as pd
import numpy as np
from copy import deepcopy

AminoAcidsDist = pd.read_excel("/Users/crystal/Dropbox/rejfreePy_main_results/ReadDataset/EuclideanDistanceForAminoAcids.xlsx",
                               sheet_name="Sheet1", header=0)
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
                key = (rowIndex[i], colIndex[j])
                result[key] = AminoAcidsDist.iloc[i, j]

    return result


class AminoAcidsUtilities:
    def __init__(self, aminoAcidDistPath=None):
        if aminoAcidDistPath is not None:
            self.aminoAcidDist = pd.read_excel(aminoAcidDistPath, sheet_name="Sheet1", header=0)
        else:
            self.aminoAcidDist = pd.read_excel("/Users/crystal/Dropbox/rejfreePy_main_results/ReadDataset/EuclideanDistanceForAminoAcids.xlsx",
                               sheet_name="Sheet1", header=0)

        ## get the row names of the data set, which is the 20 amino acids, the order of the amino acid in row names may be different from the ones in
        ## field AminoAcid at the beginning of the file
        self.rowNames = AminoAcidsDist.index
        self.colNames = AminoAcidsDist.columns.values
        charToIndexDict = charToIndex(AminoAcids)
        self.rowIndex = [charToIndexDict[x] for x in self.rowNames]
        self.colIndex = [charToIndexDict[x] for x in self.colNames]

        ## create all 380 amino acid pairs excluding the pairs where the two amino acids are the same like "AA", "RR"
        self.NonIdenticalStatePairs = list()
        for state0Ind in range(len(AminoAcids)):
            for state1Ind in range(len(AminoAcids)):
                if state0Ind != state1Ind:
                    self.NonIdenticalStatePairs.append((AminoAcids[state0Ind] + AminoAcids[state1Ind]))
        if len(self.NonIdenticalStatePairs) != len(AminoAcids) * (len(AminoAcids)-1):
            raise ValueError("The number of all non-identical amino acid pairs is not right")

        self.wholeState = np.arange(0, len(AminoAcids), 1)
        self.supportForEachAminoAcid = dict()
        for i in range(len(AminoAcids)):
            self.supportForEachAminoAcid[i] = list(set(self.wholeState)-set([i]))

        self.pairDist = getDistForPairsOfAminoAcids()
        self.orderDist = self.organizeAminoAcidDistMatrix()


    def organizeAminoAcidDistMatrix(self):
        ## we reorganize the AminoAcidDistance according to the order of amino acids in field AminoAcids
        numAminoAcids = len(AminoAcids)
        distMtx = np.zeros((numAminoAcids, numAminoAcids))
        np.fill_diagonal(distMtx, 10000)
        ## Todo: assign values to each element of the matrix using the dictionary
        for i in range(numAminoAcids):
            for j in range(numAminoAcids):
                if i != j:
                    distMtx[(i,j)] = self.pairDist[(i, j)]
        return distMtx



    def rankAminoAcidsPairs(self):
        ## find the amino acid pair with the smallest distance.
        ## create a copy of AminoAcidsDist

        OrderedAminoAcidsDist = deepcopy(self.orderDist)
        allPairs = deepcopy(self.NonIdenticalStatePairs)

        ## we will loop over this until allPairs is empty
        ArgMinRowAndCol = np.unravel_index(np.argmin(OrderedAminoAcidsDist), OrderedAminoAcidsDist.shape)

        counter = 0
        AminoAcidsPairRank = dict()

        ## create a dictionary, with the pair of index for corresponding amino acids
        argRowIndex = ArgMinRowAndCol[0]
        argColIndex = ArgMinRowAndCol[1]

        AminoAcidsPairRank[(argRowIndex, argColIndex)] = counter
        AminoAcidsPairRank[(argColIndex, argRowIndex)] = counter

        pairStates0 = AminoAcids[argRowIndex] + AminoAcids[argColIndex]
        pairStates1 = AminoAcids[argColIndex] + AminoAcids[argRowIndex]
        allPairs.remove(pairStates0)
        allPairs.remove(pairStates1)

        dynamicSupportForEachAminoAcid = deepcopy(self.supportForEachAminoAcid)
        dynamicSupportForEachAminoAcid[argRowIndex].remove(argColIndex)
        dynamicSupportForEachAminoAcid[argColIndex].remove(argRowIndex)

        ## these two indices will be updated recursively in the while loop
        index0 = argRowIndex
        index1 = argColIndex

        OrderedAminoAcidsDist[(index0, index1)] = 10000
        OrderedAminoAcidsDist[(index1, index0)] = 10000

        while len(allPairs) > 0:
            # print("The number of left pairs is")
            # print(len(allPairs))## find the next smallest distance from the support of bot the row index and the column index
            counter = counter + 1
            supportOfRow = dynamicSupportForEachAminoAcid[index0]
            supportOfCol = dynamicSupportForEachAminoAcid[index1]
            if len(supportOfRow) > 0 and len(supportOfCol) > 0:
                rowDist = np.take(OrderedAminoAcidsDist[index0,:], supportOfRow)
                colDist = np.take(OrderedAminoAcidsDist[:, index1], supportOfCol)
                rowMinInd = supportOfRow[np.argmin(rowDist)] # if there are ties, it will still take the index of the minimum
                # which appears first in the array
                colMinInd = supportOfCol[np.argmin(colDist)]
                ## take the index as the minimum of rowDist and colDist
                if OrderedAminoAcidsDist[(index0, rowMinInd)] <= OrderedAminoAcidsDist[(colMinInd, index1)]:
                    index1 = rowMinInd
                else:
                    index0 = colMinInd

            elif len(supportOfRow) ==0 and len(supportOfCol) >0:
                colDist = np.take(OrderedAminoAcidsDist[:, index1], supportOfCol)
                colMinInd = supportOfCol[np.argmin(colDist)]
                rowMinInd = index1
                index0 = colMinInd

            elif len(supportOfRow) >0 and len(supportOfCol) ==0:
                rowDist = np.take(OrderedAminoAcidsDist[index0, :], supportOfRow)
                rowMinInd = supportOfRow[np.argmin(rowDist)]
                colMinInd = index0
                index1 = rowMinInd

            elif len(supportOfRow)==0 and len(supportOfCol) ==0:
                ## find the next minimum distance from the rest of pairs
                ArgMinRowAndCol = np.unravel_index(np.argmin(OrderedAminoAcidsDist), OrderedAminoAcidsDist.shape)
                argRowIndex = ArgMinRowAndCol[0]
                argColIndex = ArgMinRowAndCol[1]
                index0 = argRowIndex
                index1 = argColIndex



            AminoAcidsPairRank[(index0, index1)] = counter
            AminoAcidsPairRank[(index1, index0)] = counter
            OrderedAminoAcidsDist[(index0, index1)] = 10000
            OrderedAminoAcidsDist[(index1, index0)] = 10000

            ## remove individual support for row and column
            if index1 in dynamicSupportForEachAminoAcid[index0]:
                dynamicSupportForEachAminoAcid[index0].remove(index1)
            if index0 in dynamicSupportForEachAminoAcid[index1]:
                dynamicSupportForEachAminoAcid[index1].remove(index0)

            ## remove the pairs of index in allPairs
            pairStates0 = AminoAcids[index0] + AminoAcids[index1]
            pairStates1 = AminoAcids[index1] + AminoAcids[index0]
            # print(pairStates0)
            # print(pairStates1)
            if pairStates0 in allPairs:
                allPairs.remove(pairStates0)
            if pairStates1 in allPairs:
                allPairs.remove(pairStates1)

        return AminoAcidsPairRank


def main():
    ## the main functions serve as test functions
    aminoAcidUtilities = AminoAcidsUtilities()
    result = aminoAcidUtilities.rankAminoAcidsPairs()

    ## check if the first pair is "IL"
    charToIndices = charToIndex(AminoAcids)
    stateA = charToIndices["A"]
    stateR = charToIndices["R"]
    if result[(stateA, stateR)] != 128:
        raise ValueError("Sth wrong with the code since the 128th smallest pair should be 'AR'")

    state0 = charToIndices["I"]
    state1 = charToIndices["L"]
    if result[(state0, state1)] != 0 or result[(state1, state0)] != 0:
        raise ValueError("Sth wrong with the code since the smallest pair should be 'IL'")
    ## check if the second pair is "IM"
    stateM = charToIndices["M"]
    if result[(state0, stateM)] != 1 or result[(stateM, state0)] != 1:
        raise ValueError("Sth wrong with the code since the second smallest pair should be 'IM'")

    ## check if the whole rank matrix is symmetric


if __name__ == '__main__':
    main()










