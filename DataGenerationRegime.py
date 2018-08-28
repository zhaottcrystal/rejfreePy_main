import os
import sys

#sys.path.append("/Users/crystal/Dropbox/rejfreePy_main/")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy_main/")
import numpy as np
import FullTrajectorGeneration
import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
# from main.FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
# from main.ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
# from main.FullTrajectorGeneration import getObsArrayAtSameGivenTimes
from numpy.random import RandomState
import warnings

defaultSeed = 1234567890

class WeightGenerationRegime:
    def __init__(self, nStates, nBivariateFeat, prng=None, seed=None, stationaryWeights=None, bivariateWeights=None):
        self.nStates = nStates
        self.nBivariateFeat = nBivariateFeat
        if prng is not None:
            self.prng = prng
        else:
            if seed is not None:
                self.prng = RandomState(seed)
            else:
                self.defaultRandomSeed = defaultSeed
                prng = RandomState(1234567890)

        if prng is not None and seed is not None:
            warnings.warn("both prng and seed are provided but we use the provided prng", DeprecationWarning)

        self.stationaryWeights = stationaryWeights
        self.bivariateWeights = bivariateWeights


    def generateStationaryWeightsFromNormal(self):
        self.stationaryWeights = self.prng.normal(0, 1, self.nStates)
        return self.stationaryWeights

    def generateBivariateWeightsFromNormal(self):
        self.bivariateWeights = self.prng.normal(0, 1, self.nBivariateFeat)
        return self.bivariateWeights

    def generateStationaryWeightsFromUniform(self):
        self.stationaryWeights = self.prng.uniform(0, 1, self.nStates)
        return self.stationaryWeights

    def generateBivariateWeightsFromUniform(self):
        self.bivariateWeights = self.prng.uniform(0, 1, self.nBivariateFeat)
        return self.bivariateWeights





class DataObsPairSeqs:
    def __init__(self, nStates, bivariateFeatIndexDictionary, nBivariateFeat, btLength, nSeq, data = None, weightGenerationRegime = None, interLength=1.0):
        self.nStates = nStates
        self.bivariateFeatIndexDictionary = bivariateFeatIndexDictionary
        self.bt = btLength
        self.interLength = interLength
        self.nSeq = nSeq
        self.data = data
        self.nPairSeq = 1
        self.weightGenerationRegime = weightGenerationRegime
        self.nBivariateFeat = nBivariateFeat
        self.prng = RandomState(defaultSeed)







class DataGenerationRegime:

    def __init__(self, nStates,  bivariateFeatIndexDictionary, btLength, nSeq, weightGenerationRegime = None, stationaryWeights=None, bivariateWeights=None, prng=None, seed = None, interLength=1.0):

        if weightGenerationRegime is None and stationaryWeights is None and bivariateWeights is None:
            raise Exception("Either weightGenerationRegime or a combination of stationaryWeights and bivariateWeights should be provided as the weights to generate data")

        if weightGenerationRegime is not None:
            self.stationaryWeights = weightGenerationRegime.stationaryWeights
            self.bivariateWeights = weightGenerationRegime.bivariateWeights
        else:
            if stationaryWeights is not None and bivariateWeights is not None:
                self.stationaryWeights = stationaryWeights
                self.bivariateWeights = bivariateWeights


        if prng is not None:
            self.prng = prng
        else:
            if seed is not None:
                self.prng = RandomState(seed)
            else:
                self.prng = RandomState(defaultSeed)

        if prng is not None and seed is not None:
            warnings.warn("both prng and seed are provided but we use the provided prng", DeprecationWarning)

        self.nStates = nStates
        self.bivariateFeatIndexDictionary = bivariateFeatIndexDictionary
        self.bt = btLength
        self.interLength = interLength
        self.nSeq = nSeq
        self.rateMtxObj = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure.ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(self.nStates, self.stationaryWeights,
                                                                            self.bivariateWeights,
                                                                                self.bivariateFeatIndexDictionary)
        self.stationaryDist = self.rateMtxObj.getStationaryDist()
        self.exchangeCoef = self.rateMtxObj.getExchangeCoef()
        self.rateMtx = self.rateMtxObj.getRateMtx()
        self.data = None
        self.suffStat = None
        self.nPairSeq = None
        self.nBivariateFeat = len(self.bivariateWeights)


    def generatingInitialStateSeq(self):
        initialStateSeq = FullTrajectorGeneration.generateInitialStateSeqUsingStationaryDist(self.prng, self.nStates,
                                                                                             self.nSeq,
                                                                                             self.stationaryDist)
        return initialStateSeq


    def generatingSeq(self, initialStateSeq=None):
        if initialStateSeq is None:
            initialStateSeq = FullTrajectorGeneration.generateInitialStateSeqUsingStationaryDist(self.prng,
                                                                                                 self.nStates,
                                                                                                 self.nSeq,
                                                                                                 self.stationaryDist)
        seqList = FullTrajectorGeneration.generateFullPathUsingRateMtxAndStationaryDist(self.prng, self.nSeq, self.nStates, self.rateMtx,
                                                                                 self.stationaryDist, self.bt, initialStateSeq)
        return seqList

    def getSufficientStatFromSeq(self, seqList):
        nSeq = len(seqList)
        nStates = len(seqList[0]['sojourn'])
        totalSojournTime = np.zeros(nStates)
        totalTransitionCount = np.zeros((nStates, nStates))

        for i in range(0, nSeq):
            totalSojournTime = totalSojournTime + seqList[i]['sojourn']
            totalTransitionCount = totalTransitionCount + seqList[i]['transitCount']

        result = {}
        result['sojourn'] = totalSojournTime
        result['transitCount'] = totalTransitionCount
        self.suffStat = result

        return result




    def generatingSeqGivenRateMtxAndBtInterval(self, seqList):

        observedTimePoints = np.arange(0, (self.bt + self.interLength*0.5), self.interLength)
        observedSeqList = FullTrajectorGeneration.getObsArrayAtSameGivenTimes(seqList, observedTimePoints)
        observedAllSequences = observedSeqList[1:observedSeqList.shape[0], :]

        firstLastStatesArrayAll = list()
        nPairSeq = int(len(observedTimePoints) - 1)
        self.nPairSeq = nPairSeq

        for i in range(nPairSeq):
            pairSeq = observedAllSequences[:, i:(i + 2)]
            firstLastStatesArrayAll.append(pairSeq)

        self.data = firstLastStatesArrayAll

        return firstLastStatesArrayAll



    def summaryFirstLastStatesArrayIntoMatrix(self, firstLastStateArrayAll, nStates):
        """

        :param firstLastStateArrayAll: the observation time series stored in a list,
                each element of the list is an array, two columns, number of rows
                represents the number of sequences. For each row, the first element
                represents the starting state and the last element represents the
                observed states after a unit time interval length

        :param nStates: the number of states
        :return: a dictionary with two elements: nInit records the initial count of the states
                                                 count records the counts of all pairs starting and ending states
        """
        count = np.zeros((nStates, nStates))
        ## loop over all columns and all rows to get the count matrix
        ## check the form of firstLastStatesArrayAll to see how to write the loop

        nInit = np.zeros(self.nStates)
        unique, counts = np.unique(firstLastStateArrayAll[0][:, 0], return_counts=True)
        nInitCount = np.asarray((unique, counts)).T
        nInit[nInitCount[:, 0].astype(int)] = nInitCount[:, 1]

        nSeq = firstLastStateArrayAll[0].shape[0]
        for i in range(len(firstLastStateArrayAll)):
            pairSeq = firstLastStateArrayAll[i]
            for j in range(nSeq):
                startState = pairSeq[j][0]
                endState = pairSeq[j][1]
                count[int(startState)][int(endState)] = count[int(startState)][int(endState)] + 1

        result = dict()
        result['nInit'] = nInit
        result['count'] = count
        return result







