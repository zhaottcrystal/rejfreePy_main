import numpy as np


def getEmpiricalInitialStatesCount(seqList):
    nSeq = len(seqList)
    empiricalInitialStates = np.zeros(nSeq)
    for i in range(nSeq):
        empiricalInitialStates[i] = seqList[i]['states'][0]
    # get the frequency count of each different states out of the vector
    unique, counts = np.unique(empiricalInitialStates, return_counts=True)
    result = np.asarray((unique, counts)).T
    return result

def getTransitionCounts(seqList):
    nStates = seqList[0]['transitCount'].shape[0]
    counts = np.zeros((nStates, nStates))
    nSeq = len(seqList)
    for i in range(nSeq):
        counts = counts + seqList[i]['transitCount']
    return counts

def getSojournTime(seqList):
    nStates = len(seqList[0]['sojourn'])
    result = np.zeros(nStates)
    nSeq = len(seqList)
    for i in range(nSeq):
        result = result + seqList[i]['sojourn']
    return result