import os
import sys

sys.path.append("/Users/crystal/Dropbox/rejfreePy_main/")
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy_main/")
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
    def __init__(self, nStates, nBivariateFeat, prng=None, seed=None):
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

        self.stationaryWeights = None
        self.bivariateWeights = None


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
        self.nPairSeq = None


    def generatingSeqGivenRateMtxAndBtInterval(self):

        seqList = FullTrajectorGeneration.generateFullPathUsingRateMtxAndStationaryDist(self.nSeq, self.nStates, self.prng, self.rateMtx, self.stationaryDist, self.bt)
        observedTimePoints = np.arange(0, (self.bt + 1), self.interLength)
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


