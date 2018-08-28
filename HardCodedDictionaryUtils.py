import numpy as np
from collections import OrderedDict
from AminoAcidDict import getBivariateFeatGivenAllAminoAcidPairs
from AminoAcidsDistUtilities import AminoAcidsUtilities

def getHardCodedDict():
    HardCodedDict = OrderedDict()
    HardCodedDict[(0, 1)] = np.array((0, 1), dtype=np.int)
    HardCodedDict[(0, 2)] = np.array((0, 1, 2), dtype=np.int)
    HardCodedDict[(0, 3)] = np.array((1, 2), dtype=np.int)
    HardCodedDict[(0, 4)] = np.array((0, 2), dtype=np.int)
    HardCodedDict[(0, 5)] = np.array((3, 4), dtype=np.int)
    HardCodedDict[(1, 0)] = np.array((0, 1), dtype=np.int)
    HardCodedDict[(1, 2)] = np.array((3, 4, 5), dtype=np.int)
    HardCodedDict[(1, 3)] = np.array((3, 5), dtype=np.int)
    HardCodedDict[(1, 4)] = np.array((4, 5), dtype=np.int)
    HardCodedDict[(1, 5)] = np.array((6, 7), dtype=np.int)
    HardCodedDict[(2, 0)] = np.array((0, 1, 2), dtype=np.int)
    HardCodedDict[(2, 1)] = np.array((3, 4, 5), dtype=np.int)
    HardCodedDict[(2, 3)] = np.array((6, 7, 8), dtype=np.int)
    HardCodedDict[(2, 4)] = np.array((7, 8), dtype=np.int)
    HardCodedDict[(2, 5)] = np.array((6, 8), dtype=np.int)
    HardCodedDict[(3, 0)] = np.array((1, 2), dtype=np.int)
    HardCodedDict[(3, 1)] = np.array((3, 5), dtype=np.int)
    HardCodedDict[(3, 2)] = np.array((6, 7, 8), dtype=np.int)
    HardCodedDict[(3, 4)] = np.array((9, 10), dtype=np.int)
    HardCodedDict[(3, 5)] = np.array((9, 11), dtype=np.int)
    HardCodedDict[(4, 0)] = np.array((0, 2), dtype=np.int)
    HardCodedDict[(4, 1)] = np.array((4, 5), dtype=np.int)
    HardCodedDict[(4, 2)] = np.array((7, 8), dtype=np.int)
    HardCodedDict[(4, 3)] = np.array((9, 10), dtype=np.int)
    HardCodedDict[(4, 5)] = np.array((10, 11), dtype=np.int)
    HardCodedDict[(5, 0)] = np.array((3, 4), dtype=np.int)
    HardCodedDict[(5, 1)] = np.array((6, 7), dtype=np.int)
    HardCodedDict[(5, 2)] = np.array((6, 8), dtype=np.int)
    HardCodedDict[(5, 3)] = np.array((9, 11), dtype=np.int)
    HardCodedDict[(5, 4)] = np.array((10, 11), dtype=np.int)
    return HardCodedDict

def getHardCodedDict10States():
    HardCodedDict = OrderedDict()
    HardCodedDict[(0, 1)] = np.array((0, 1), dtype=np.int)
    HardCodedDict[(0, 2)] = np.array((0, 1, 2), dtype=np.int)
    HardCodedDict[(0, 3)] = np.array((1, 2), dtype=np.int)
    HardCodedDict[(0, 4)] = np.array((0, 2), dtype=np.int)
    HardCodedDict[(0, 5)] = np.array((3, 4), dtype=np.int)
    HardCodedDict[(0, 6)] = np.array((12, 13), dtype=np.int)
    HardCodedDict[(0, 7)] = np.array((12, 13, 14), dtype=np.int)
    HardCodedDict[(0, 8)] = np.array((12, 14), dtype=np.int)
    HardCodedDict[(0, 9)] = np.array((13, 14), dtype=np.int)
    HardCodedDict[(1, 0)] = np.array((0, 1), dtype=np.int)
    HardCodedDict[(1, 2)] = np.array((3, 4, 5), dtype=np.int)
    HardCodedDict[(1, 3)] = np.array((3, 5), dtype=np.int)
    HardCodedDict[(1, 4)] = np.array((4, 5), dtype=np.int)
    HardCodedDict[(1, 5)] = np.array((6, 7), dtype=np.int)

    HardCodedDict[(1, 6)] = np.array((15, 16), dtype=np.int)
    HardCodedDict[(1, 7)] = np.array((15, 16, 17), dtype=np.int)
    HardCodedDict[(1, 8)] = np.array((15, 17), dtype=np.int)
    HardCodedDict[(1, 9)] = np.array((16, 17), dtype=np.int)

    HardCodedDict[(2, 0)] = np.array((0, 1, 2), dtype=np.int)
    HardCodedDict[(2, 1)] = np.array((3, 4, 5), dtype=np.int)
    HardCodedDict[(2, 3)] = np.array((6, 7, 8), dtype=np.int)
    HardCodedDict[(2, 4)] = np.array((7, 8), dtype=np.int)
    HardCodedDict[(2, 5)] = np.array((6, 8), dtype=np.int)

    HardCodedDict[(2, 6)] = np.array((18, 19), dtype=np.int)
    HardCodedDict[(2, 7)] = np.array((18, 19, 20), dtype=np.int)
    HardCodedDict[(2, 8)] = np.array((18, 20), dtype=np.int)
    HardCodedDict[(2, 9)] = np.array((19, 20), dtype=np.int)

    HardCodedDict[(3, 0)] = np.array((1, 2), dtype=np.int)
    HardCodedDict[(3, 1)] = np.array((3, 5), dtype=np.int)
    HardCodedDict[(3, 2)] = np.array((6, 7, 8), dtype=np.int)
    HardCodedDict[(3, 4)] = np.array((9, 10), dtype=np.int)
    HardCodedDict[(3, 5)] = np.array((9, 11), dtype=np.int)

    HardCodedDict[(3, 6)] = np.array((21, 22), dtype=np.int)
    HardCodedDict[(3, 7)] = np.array((21, 22, 23), dtype=np.int)
    HardCodedDict[(3, 8)] = np.array((21, 23), dtype=np.int)
    HardCodedDict[(3, 9)] = np.array((22, 23), dtype=np.int)

    HardCodedDict[(4, 0)] = np.array((0, 2), dtype=np.int)
    HardCodedDict[(4, 1)] = np.array((4, 5), dtype=np.int)
    HardCodedDict[(4, 2)] = np.array((7, 8), dtype=np.int)
    HardCodedDict[(4, 3)] = np.array((9, 10), dtype=np.int)
    HardCodedDict[(4, 5)] = np.array((10, 11), dtype=np.int)

    HardCodedDict[(4, 6)] = np.array((24, 25), dtype=np.int)
    HardCodedDict[(4, 7)] = np.array((24, 25, 26), dtype=np.int)
    HardCodedDict[(4, 8)] = np.array((24, 26), dtype=np.int)
    HardCodedDict[(4, 9)] = np.array((25, 26), dtype=np.int)

    HardCodedDict[(5, 0)] = np.array((3, 4), dtype=np.int)
    HardCodedDict[(5, 1)] = np.array((6, 7), dtype=np.int)
    HardCodedDict[(5, 2)] = np.array((6, 8), dtype=np.int)
    HardCodedDict[(5, 3)] = np.array((9, 11), dtype=np.int)
    HardCodedDict[(5, 4)] = np.array((10, 11), dtype=np.int)

    HardCodedDict[(5, 6)] = np.array((27, 28), dtype=np.int)
    HardCodedDict[(5, 7)] = np.array((27, 28, 29), dtype=np.int)
    HardCodedDict[(5, 8)] = np.array((27, 29), dtype=np.int)
    HardCodedDict[(5, 9)] = np.array((28, 29), dtype=np.int)

    HardCodedDict[(6, 0)] = np.array((12, 13), dtype=np.int)
    HardCodedDict[(6, 1)] = np.array((15, 16), dtype=np.int)
    HardCodedDict[(6, 2)] = np.array((18, 19), dtype=np.int)
    HardCodedDict[(6, 3)] = np.array((21, 22), dtype=np.int)
    HardCodedDict[(6, 4)] = np.array((24, 25), dtype=np.int)

    HardCodedDict[(6, 5)] = np.array((27, 28), dtype=np.int)
    HardCodedDict[(6, 7)] = np.array((30, 31), dtype=np.int)
    HardCodedDict[(6, 8)] = np.array((30, 31, 32), dtype=np.int)
    HardCodedDict[(6, 9)] = np.array((30, 32), dtype=np.int)

    HardCodedDict[(7, 0)] = np.array((12, 13), dtype=np.int)
    HardCodedDict[(7, 1)] = np.array((15, 16), dtype=np.int)
    HardCodedDict[(7, 2)] = np.array((18, 19), dtype=np.int)
    HardCodedDict[(7, 3)] = np.array((21, 22), dtype=np.int)
    HardCodedDict[(7, 4)] = np.array((24, 25), dtype=np.int)

    HardCodedDict[(7, 5)] = np.array((27, 28), dtype=np.int)
    HardCodedDict[(7, 6)] = np.array((30, 31), dtype=np.int)
    HardCodedDict[(7, 8)] = np.array((31, 32), dtype=np.int)
    HardCodedDict[(7, 9)] = np.array((33, 34), dtype=np.int)

    HardCodedDict[(8, 0)] = np.array((12, 14), dtype=np.int)
    HardCodedDict[(8, 1)] = np.array((15, 17), dtype=np.int)
    HardCodedDict[(8, 2)] = np.array((18, 20), dtype=np.int)
    HardCodedDict[(8, 3)] = np.array((21, 23), dtype=np.int)
    HardCodedDict[(8, 4)] = np.array((24, 26), dtype=np.int)

    HardCodedDict[(8, 5)] = np.array((27, 29), dtype=np.int)
    HardCodedDict[(8, 6)] = np.array((30, 31, 32), dtype=np.int)
    HardCodedDict[(8, 7)] = np.array((31, 32), dtype=np.int)
    HardCodedDict[(8, 9)] = np.array((33, 34, 35), dtype=np.int)

    HardCodedDict[(9, 0)] = np.array((13, 14), dtype=np.int)
    HardCodedDict[(9, 1)] = np.array((16, 17), dtype=np.int)
    HardCodedDict[(9, 2)] = np.array((19, 20), dtype=np.int)
    HardCodedDict[(9, 3)] = np.array((22, 23), dtype=np.int)
    HardCodedDict[(9, 4)] = np.array((25, 26), dtype=np.int)

    HardCodedDict[(9, 5)] = np.array((28, 29), dtype=np.int)
    HardCodedDict[(9, 6)] = np.array((30, 32), dtype=np.int)
    HardCodedDict[(9, 7)] = np.array((33, 34), dtype=np.int)
    HardCodedDict[(9, 8)] = np.array((33, 34, 35), dtype=np.int)
    return HardCodedDict

def getHardCodedDictChainGraph(nStates):
    HardCodedDict = OrderedDict()

    ## create a sequence of number starting from (nStates-1) and ending with 1, with step size 1
    numOfPrevElements = np.flip(np.arange(1, nStates), axis=0)
    for firstState in range(nStates-1):
        for index, secondState in enumerate(np.arange((firstState+1), nStates)):
            statePair = (firstState, secondState)
            flipStatePair = (secondState, firstState)
            if firstState == 0:
                featIndex = index
                if index == 0:
                    HardCodedDict[statePair] = np.array(featIndex, dtype=np.int)
                    HardCodedDict[flipStatePair] = np.array(featIndex, dtype= np.int)
                if index > 0:
                    HardCodedDict[statePair] = np.array(((featIndex-1), featIndex), dtype= np.int)
                    HardCodedDict[flipStatePair] = np.array(((featIndex-1), featIndex), dtype= np.int)

            else:
                featIndex =  sum(numOfPrevElements[0:firstState]) + index
                HardCodedDict[statePair] = np.array(((featIndex - 1), featIndex), dtype=np.int)
                HardCodedDict[flipStatePair] = HardCodedDict[statePair]
    return HardCodedDict


def getHardCodedDictPolaritySizeGraph():
    return getBivariateFeatGivenAllAminoAcidPairs()


def getHardCodedDictFromAminoAcidDistRankPairs():
    aminoAcidUtilities = AminoAcidsUtilities()
    result = aminoAcidUtilities.rankAminoAcidsPairs()
    HardCodedDict = OrderedDict()
    ## loop over all the keys
    for k in result.keys():

        featIndex = result[k]
        if featIndex ==0:
            HardCodedDict[k] = np.array(featIndex, dtype= np.int)
        else:
            HardCodedDict[k] = np.array(((featIndex-1), featIndex), dtype= np.int)

    return HardCodedDict



def test():
    ## test the correctness of getHardCodedDictChainGraph()
    dictAccordingToAminoAcidDistOrder = getHardCodedDictFromAminoAcidDistRankPairs()
    print(dictAccordingToAminoAcidDistOrder[(9, 10)])
    print(dictAccordingToAminoAcidDistOrder)

    dictFor15States = getHardCodedDictChainGraph(15)
    key1 = (0, 14)
    print(dictFor15States[key1])

    key2 = (1, 13)
    print(dictFor15States[key2])

    key3 = (5, 10)
    print(dictFor15States[key3])

    key4 = (14, 0)
    print(dictFor15States[key4])

    key5 = (13, 1)
    print(dictFor15States[key5])

    key6 = (10, 5)
    print(dictFor15States[key6])

    key7 = (13, 14)
    print(dictFor15States[key7])

    key8 = (0, 1)
    print(dictFor15States[key8])

    ## test the symmetric property of getHardCodedDict10States()
    nStates = 10
    tenStatesDict = getHardCodedDict10States()
    for firstState in range(nStates):
        for index, secondState in enumerate(np.arange((firstState + 1), nStates)):
            statePair = (firstState, secondState)
            flipStatePair = (secondState, firstState)
            if tenStatesDict[statePair].all() != tenStatesDict[flipStatePair].all():
                print(statePair)
                print(flipStatePair)
                break

    print("The test is passed")

def main():
    test()

if __name__ == "__main__": main()






