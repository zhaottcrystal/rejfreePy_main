import numpy as np
from collections import OrderedDict

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
    HardCodedDict[(4, 0)] = np.array((2, 0), dtype=np.int)
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


def test():
    ## test the correctness of getHardCodedDictChainGraph()

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









