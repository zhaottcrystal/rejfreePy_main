import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import EndPointSampler
import PathStatistics
import Path
import ReversibleRateMtx
import SimuSeq

# from main.EndPointSampler import EndPointSampler
# from main.PathStatistics import PathStatistics
# from main.Path import Path
# from main.ReversibleRateMtx import ReversibleRateMtx
# from  main.SimuSeq import ForwardSimulation
import numpy as np
import bisect
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from Bio import SeqIO
from numpy.random import RandomState


def generateFullPath(nSeq, nstates, prng, weights, bt, isNormalized):
    """Return the full trajectory of each sequence, where initial state is sampled
    from the stationary distribution. Save all generated sequences in a list"""
    rateMtxQ = ReversibleRateMtx.ReversibleRateMtx(nstates, weights)
    if isNormalized:
        Q = rateMtxQ.getNormalizedRateMtx()
    else:
        Q = rateMtxQ.getRateMtx()
        
    stationary = rateMtxQ.getStationaryDist()
    ## sample the initial states for each sequence
    initialStateSeq = prng.choice(nstates, nSeq, replace=True, p=stationary)
    ## given the initial state, we sample the full path and save each sequence in a list
    seqList = []
    simulator = SimuSeq.ForwardSimulation(bt, Q)
    for i in range(0, nSeq):
        seq = simulator.sampleStateTimeSeq(prng, initialStateSeq[i])
        seqList.append(seq)
    ## get the keys for each sequence seqList[0].keys()    
    return seqList

def generateInitialStateSeqUsingStationaryDist(prng, nStates, nSeq, stationaryDist):
    initialStateSeq = prng.choice(nStates, nSeq, replace=True, p=stationaryDist)
    return initialStateSeq

def generateFullPathUsingRateMtxAndStationaryDist(prng, nSeq, nstates, rateMtx, stationaryDist, bt, initialStateSeq=None):

    Q = rateMtx
    if initialStateSeq is None:
        initialStateSeq = prng.choice(nstates, nSeq, replace=True, p=stationaryDist)
    ## given the initial state, we sample the full path and save each sequence in a list
    seqList = []
    simulator = SimuSeq.ForwardSimulation(bt, Q)
    for i in range(0, nSeq):
        prng = RandomState(i)
        seq = simulator.sampleStateTimeSeq(prng, initialStateSeq[i])
        seqList.append(seq)
    ## get the keys for each sequence seqList[0].keys()
    return seqList



def getFirstAndLastStateOfSeq(seq, nStates):
    firstState = seq[0]
    lastState = seq[(len(seq)-1)]
    if firstState >= nStates or lastState >= nStates:
        raise ValueError("The first state or the last state is outside the state space")
    else:
        result = {'firstState': firstState, 'lastState': lastState}
        return result

def extractSeqFromSeqList(seqDictionary):
    """
    :param seqDictionary: the output from function sampleStateTimeSeq() method of SimuSeq.py
    :return: the first and last state of the the sequence of the states 
    """
    sequences = seqDictionary['states']
    seqResult = np.zeros(2)
    seqResult[0] = sequences[0]
    seqResult[1] = sequences[(len(sequences)-1)]
    return seqResult

def extractBtFromSeqList(seqDictionary):
    """
    :param seqDictionary: the output from function sampleStateTimeSeq() method of SimuSeq.py
    :return: the branch length and time interval of the sequence
    """
    tlist = seqDictionary['time']
    bt = tlist[(len(tlist)-1)]
    return bt



def getFirstAndLastStateOfListOfSeq(seqList):
    """ 
    :param seqList: it represents the output from function generateFullPath()
    :return: a dictionary consists of two elements: the first element is a nSeq * 2 dimentional array
             nSeq: the number of rows represents the number of sequences in the seqList
             Within the 2 columns, the first column represents the first state of the sequences in the list
             and the second column represents the last state of the sequences
             the second element provides the branch lengths 
    """
    nSeq = len(seqList)
    seqResult = np.zeros((nSeq, 2))
    btSeq = np.zeros(nSeq)
    for i in range(nSeq):
        seqResult[i, :] = extractSeqFromSeqList(seqList[i])
        btSeq[i] = extractBtFromSeqList(seqList[i])
    result = {'firstLastState': seqResult, 'btSeq': btSeq}
    return result

@staticmethod
def test():
    nSeq = 100
    nstates = 4
    seedNum = 123
    np.random.seed(seedNum)
    weights = np.random.uniform(0, 1, 10)
    bt = 10
    seqList = generateFullPath(nSeq, nstates, seedNum, weights, bt, False)

    result = getFirstAndLastStateOfListOfSeq(seqList)
    firstLastStatesSeq = result['firstLastState']
    btSeq = result['btSeq']

    # select 10 random indices from 0 to nSeq
    # check if the initial and end states are consistent with seqList
    indices = np.random.randint(low=0, high=len(seqList), size=10)
    for i in range(len(indices)):
        sequences = seqList[indices[i]]['states']
        if sequences[0] != firstLastStatesSeq[indices[i], 0] or sequences[(len(sequences)-1)]!= firstLastStatesSeq[indices[i], 1]:
            print(indices[i])
            raise ValueError("Something wrong with function getFirstAndLastStateOfListOfSeq")
    
    print("The getFirstAndLastStateOfListOfSeq is correct")

    if bt != btSeq[0]:
        raise ValueError("The extracted branch length is not consistent as the true one")

def endPointSamplerSummarizeStatisticsOneBt(cached, prng, rateMatrix, firstLastStatesArray, bt):
    """
    :param cached: if true, we use cached uniformization in the enpointsampler, default value for cached should be true
    :param rateMtx: the rateMtx we use to simulate the path
    :param firstLastStatesArray: a nSeq*2 dimensional array, the first column stores the initial states and
     the second column stores the lasts states of all sequences
    :param bt: a single scalar, if all sequences share the same branch length
    :return: the summary statistics of nInit, nTransitions and nSojournTime
    """
    ## use endPointSampler to sample the full path between each pair of initial and end states
    ## and summarize the sufficient statistics for that
    EndPointSampler.cached = cached
    ## initialize path and path statistics
    nStates = rateMatrix.shape[0]
    pathStat2 = PathStatistics.PathStatistics(nStates=nStates)
    postSampler = EndPointSampler.EndPointSampler(rateMtx=rateMatrix, totalTime=bt)
    nSeq = firstLastStatesArray.shape[0]
    
    unique, counts = np.unique(firstLastStatesArray[:, 0], return_counts=True)
    nInitCount = np.asarray((unique, counts)).T
    nInit = nInitCount[:, 1]

    for i in range(nSeq):
        p2 = Path.Path()
        postSampler.sample(prng, int(firstLastStatesArray[i, 0]), int(firstLastStatesArray[i, 1]), bt, pathStat2, p2)

    ## extract the diagonal elements from
    holdTimes = np.diag(pathStat2.counts)
    nTrans = np.copy(pathStat2.counts)
    ## make diagonal elements of nTrans into zero
    np.fill_diagonal(nTrans, 0)
    result = {'holdTimes': holdTimes, 'nInit': nInit, 'nTrans': nTrans}
    #result = ExpectedCompleteReversibleObjective(holdTimes=holdTimes, nInit=nInit, nTrans=nTrans)
    return result


@staticmethod
def testEndPointSamplerSummarizeStatisticsOneBt():
    nSeq = 100
    nStates = 4
    seedNum = 123
    np.random.seed(seedNum)
    weights = np.random.uniform(0, 1, 10)
    bt = 10
    prng = RandomState(123)
    
    ## Todo: make replication process and take average, this is more precise
    ## but looking at the printed values from the forward sampler and the backward sampler
    ## the result should be correct
    
    seqList = generateFullPath(nSeq, nstates, prng, weights, bt, False)
    result = getFirstAndLastStateOfListOfSeq(seqList)
    firstLastStatesArray = result['firstLastState']

    cached = True
    rateMatrix = ReversibleRateMtx.ReversibleRateMtx(nStates, weights).getRateMtx()

    resultEndPointSampler = endPointSamplerSummarizeStatisticsOneBt(cached, prng, rateMatrix, firstLastStatesArray, bt)
    nInit = resultEndPointSampler.nInit
    nTrans = resultEndPointSampler.nTrans
    holdTimes = resultEndPointSampler.holdTimes

    ## compare the sufficient statistics obtained from the endpointsampler and the ones from the ForwardSampler
    transitCount = np.zeros((nStates, nStates))
    sojournTime = np.zeros(nStates)

    for i in range(len(seqList)):
        transitCount += seqList[i]['transitCount']
        sojournTime += seqList[i]['sojourn']

    #print(transitCount)
    #print(sojournTime)
    #print(nTrans)
    #print(holdTimes)
    











def generateFullPathBtArray(nSeq, nstates, seedNum, weights, btArray, isNormalized):
    """Return the full trajectory of each sequence, where initial state is sampled
    from the stationary distribution. Save all generated sequences in a list"""
    rateMtxQ = ReversibleRateMtx.ReversibleRateMtx(nstates, weights)
    if isNormalized:
        Q = rateMtxQ.getNormalizedRateMtx()
    else:
        Q = rateMtxQ.getRateMtx()

    stationary = rateMtxQ.getStationaryDist()
    ## sample the initial states for each sequence
    np.random.seed(seedNum)
    initialStateSeq = np.random.choice(nstates, nSeq, replace=True, p=stationary)
    ## given the initial state, we sample the full path and save each sequence in a list
    seqList = []

    for i in range(0, nSeq):
        simulator = SimuSeq.ForwardSimulation(btArray[i], Q)
        seq = simulator.sampleStateTimeSeq(initialStateSeq[i])
        seqList.append(seq)
    ## get the keys for each sequence seqList[0].keys()
    return seqList


def endPointSamplerSummarizeStatisticsBtArray(cached, rateMatrix, firstLastStatesArray, btArray):
    """
    :param cached: if true, we use cached uniformization in the enpointsampler, default value for cached should be true
    :param rateMtx: the rateMtx we use to simulate the path
    :param firstLastStatesArray: a nSeq*2 dimensional array, the first column stores the initial states and
     the second column stores the lasts states of all sequences
    :param bt: a single scalar, if all sequences share the same branch length
    :return: the summary statistics of nInit, nTransitions and nSojournTime
    """
    ## use endPointSampler to sample the full path between each pair of initial and end states
    ## and summarize the sufficient statistics for that
    EndPointSampler.cached = cached
    ## initialize path and path statistics
    nStates = rateMatrix.shape[0]
    pathStat2 = PathStatistics(nStates=nStates)
    p2 = Path()
    nSeq = firstLastStatesArray.shape[0]

    unique, counts = np.unique(firstLastStatesArray[:, 0], return_counts=True)
    nInitCount = np.asarray((unique, counts)).T
    nInit = nInitCount[:, 1]

    for i in range(nSeq):
        postSampler = EndPointSampler(rateMtx= rateMatrix, totalTime=btArray[i])
        postSampler.sample(firstLastStatesArray[i, 0], firstLastStatesArray[i, 1], btArray[i], pathStat2, p2)

    ## extract the diagonal elements from
    holdTimes = np.diag(pathStat2.counts)
    nTrans = pathStat2.counts
    result = {'holdTimes':holdTimes, 'nInit': nInit, 'nTrans': nTrans}
    #result = ExpectedCompleteReversibleObjective(holdTimes=holdTimes, nInit=nInit, nTrans=nTrans)
    return result


def getAverageNumOfObsFullPath(seqList):
    """Return the average number observations across all sequences"""
    result = []
    result = [len(seqList[i]["states"]) for i in range(0, len(seqList))]
    avgNumObs = round(sum(result)/len(result))
    return avgNumObs

def getActualNumObs(avgNumObs, proportion):
    """Return the number of observations, result = avgNumObs * proportion
    avgNumObs is the average number of observations in the full trajectory"""
    result = round(avgNumObs*proportion)
    return result
    
def generateSeqObservationTimes(bt, numTimePoints):
    """Return an increasing order of observation time points with first element 0
    and last element bt"""
    ## numTimePoints corresponds to actualObsNum
    ## generate numTimePoints uniformly distributed observation times from [0, bt]
    timeSeries = np.random.uniform(0, bt, numTimePoints)
    ## see if the initial time point is zero or not. If not, add zero as the first element
    ## of the array
    if timeSeries[0]!=0:
        timeSeries = np.insert(timeSeries, 0, 0)
    ## add bt as the last time point of timeSeries
    timeSeries = np.insert(timeSeries, len(timeSeries), bt)
    timeSeries.sort()
    return timeSeries
    
def getObsAtGivenTimes(singleSeq, timePoints):
    """Input: singleSeq is a sequence that saves the states at a finite number of timepoints
    timePoints is a sequence of actual observation times, can be different from the observation
    times of a full trajectory. This function returns the states of the sequence at timePoints"""
    
    fullPathTime = singleSeq["time"]
    fullPathStates = singleSeq["states"]
    actualStates = []
    ## for each element in timePoints, find the right most index in sorted list with an increasing
    ## order of singleSeq that is smaller than timePoints[i]
    for i in range(0, len(timePoints)):
        'Find rightmost value less than or equal to x'
        ind = bisect.bisect(fullPathTime, timePoints[i])
        ind = ind -1
        if ind < 0 or ind > len(fullPathTime):
            raise ValueError
        actualStates.append(fullPathStates[ind])
    result = {"time": timePoints, "actualSeq": actualStates}
    return result

def getObsArrayAtSameGivenTimes(seqList, timePoints):
    """Input: a list of sequences, each sequence element is like one singleSeq in getObsAtGivenTimes(),
    Returns a numpy array with first row as the timePoints and the rest rows, each row represent a
    sequence of observations given at these timePoints"""
    nCols = len(timePoints)
    nRows = len(seqList)+1
    result = np.zeros((nRows, nCols))
    result[0,:] = timePoints
    for i in range(1, nRows):
        result[i,:] = getObsAtGivenTimes(seqList[i-1], timePoints)["actualSeq"]
    return result    
    
## organize observed sequences of states observed at the same time points into sequences of unrooted trees 
## with two leaves
def organizeSeqOfTimeSeriesIntoSeqForTrees(seqAndTimePointsArray):
    """seqAndTimePointsArray is the result from function getObsArrayAtSameGivenTimes(seqList, timePoints)"""
    ## in  seqAndTimePointsArray, the first row saves the times of the observations
    ## the rest of the rows saves the states at those observation times for each sequence
    ## since we assume for all sequences they are observed at the same time, when we form the data 
    ## for the trees, the total branch length for the first tree is the interval length between the first
    ## and second observation of time points and the first observation across all sequences of the time series
    ## forms the first sequence of all sites for the first tree and the second observations across all time 
    ## series sequences forms the second sequence of all sites as the second tip for the first tree
    nRows = seqAndTimePointsArray.shape[0]
    data = seqAndTimePointsArray[1:nRows, :]
    nCol = data.shape[1] # get the number of column
    seqSites = []
    for i in range(0, (nCol-1)):
        dictionary = {}
        dictionary["t1"] = data[:, i]
        dictionary["t2"] = data[:, (i+1)]
        seqSites.append(dictionary)
    return seqSites
  

def createFileNames(nFileNames, seqPrefix):
    """Based on the number of files we want to create, obtain file names for each file seperately
    paste seqPrefix and file numbers together"""
    nameList = []
    nameList = [seqPrefix+str(i)+".txt" for i in range(0, nFileNames)]
    return nameList
    
## example use of createFileNames function
#filenames = createFileNames(5, "alignment")

def transformNumberToDNA(inputState):
    """the inputFile saves the state sequence as 0, 1, 2, 3, we transform it into A,C,G,T"""
    if inputState == 0:
        result = "A"
    elif inputState == 1:
        result = "C"
    elif inputState == 2:
        result = "G"
    elif inputState == 3:
        result = "T"
    else:
        raise ValueError("The input state is not valid as 0,1,2 or 3") 
    return result
    

def transformNumSeqToCharacter(seqNumSites):
    """Transform sequences of multiple sites using 0,1,2,3 to A,C,G,T representation"""
    ## the input seqNumSites is a sequence consist of states 0, 1, 2, 3, saved as a np.array format
    tmp = seqNumSites.tolist()
    result = np.array(list(map(transformNumberToDNA, tmp)))
    return result

def transformNumSeqListDicToCharacter(seqNumSitesDicList):
    """seqNumSitesDicList is a list of dictionary, each dictionary has two keys: t1 and t2, 
    the values of t1 and t2 are two sequences in the format of 0, 1, 2, 3. This function returns
    each sequence of 0, 1, 2, 3 to A, C, G, T"""
    result = []
    for i in range(0, len(seqNumSitesDicList)):
        dictionary = {}
        dictionary["t1"] = transformNumSeqToCharacter(seqNumSitesDicList[i]["t1"])
        dictionary["t2"] = transformNumSeqToCharacter(seqNumSitesDicList[i]["t2"])
        result.append(dictionary)
    return result    

def generator_of_sequences(sequenceSets):
    for (index,string_seq) in enumerate(sequenceSets):
        yield SeqRecord(Seq(string_seq,  IUPAC.unambiguous_dna), id= ("t"+str(index+1)))   
        
def writeSeqSitesToFiles(path, filenames, seqCharSitesDicList):
    """Write sequences of sites for trees to files in specific paths"""
    ## filenames is a list of file anmes, get the number of total files
    nFiles = len(filenames)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(0, nFiles):
        filepath = os.path.join(path, filenames[i])
        align_file = open(filepath, "w")
        sequenceSet = []
        sequenceSet.append(''.join(seqCharSitesDicList[i]["t1"]))
        sequenceSet.append(''.join(seqCharSitesDicList[i]["t2"]))
        records =[]
        for (index,seq) in enumerate(sequenceSet):
            records.append(SeqRecord(Seq(seq, IUPAC.unambiguous_dna), id= ("t"+str(index+1)), description=''))
        SeqIO.write(records, open(os.path.join(path, filenames[i]), "w"), "fasta") 
        align_file.close()
        
    
## use these interval times to create trees    
def getIntervalTimeBetween2Obs(timeSeries):
    """Input: a sequence of observation times
       Output: the interval time between each of the two observations"""
    result = []
    result = [timeSeries[i+1]-timeSeries[i] for i in range(len(timeSeries)-1)]
    return result
    
## based on these interval times, create trees, each interval on a time series is split into a tree
## where the delta interval time is split into half and each half is the branch lengh of the tree with 
## nodes, provide path and tree names, write these trees with different names into a folder
def getTreesFromDeltaIntervalTimeSeries(deltaTimeSeries, path):
    numTrees = len(deltaTimeSeries)
    ## create tree names for the files which store the tree 
    nameList = []
    nameList= ["tree"+str(i)+".nwk.txt" for i in range(0, numTrees)]
    if not os.path.exists(path):
        os.makedirs(path)
    ## write trees to txt files
    for i in range(0, numTrees):
        filepath =  os.path.join(path, nameList[i])
        tree_file = open(filepath, "w")
        tree_file.write("(t1:"+str(deltaTimeSeries[i]/2.0)+", t2:"+ str(deltaTimeSeries[i]/2.0)+");")
        tree_file.close()

## try to see if function getTreesFromDeltaIntervalTimeSeries() works
#path = "/Users/crystal/Dropbox/rejfree/rejfreePy/data"
#deltaTimeSeries = np.arange(3)
#deltaTimeSeries[0]=2.2
#getTreesFromDeltaIntervalTimeSeries(deltaTimeSeries, path)

def getTreesFromObsTimeSeries(timeSeries, path):
    deltaTimeSeries = getIntervalTimeBetween2Obs(timeSeries)
    getTreesFromDeltaIntervalTimeSeries(deltaTimeSeries, path)
    
############################################################
#### create automatic code based on the number of trees and sequence files    

    

################################################################
# nSeq =100
# nstates = 4
# seedNum = 123
# np.random.seed(seedNum)
# weights = np.random.uniform(0,1,10)
# bt = 10
# prng = RandomState(seedNum)
# seqList = generateFullPath(nSeq, nstates, prng, weights, bt, False)
# avgObs = getAverageNumOfObsFullPath(seqList)
# actualObsNum = getActualNumObs(avgObs, proportion=0.7)
# timePoints = generateSeqObservationTimes(bt, actualObsNum)
# seqArray = getObsArrayAtSameGivenTimes(seqList, timePoints)
# ## check the correctness of the result
# #seqArray[0]
# #seqArray[1]
# #timePoints
# #seqList[0]["states"]
# #seqList[0]["time"]
#
# ## write tree files to a specific path
# path = "/Users/crystal/Dropbox/rejfree/rejfreePy/data"
# getTreesFromObsTimeSeries(timePoints, path)
# ## write sequence files to a specific path
# ## create list of file names
# filenames = createFileNames((len(timePoints)-1), "alignment")
# ##get sequence data in the format of 0, 1, 2, 3
# seqNumSitesDicList = organizeSeqOfTimeSeriesIntoSeqForTrees(seqArray)
# ## transform data into A C G T
# seqCharSitesDicList = transformNumSeqListDicToCharacter(seqNumSitesDicList)
# ## write sequence to files according to a specified path
# ## writeSeqSitesToFiles(path, filenames, seqNumSitesCharacterDicList)
  
    
    
    
    
    
    











