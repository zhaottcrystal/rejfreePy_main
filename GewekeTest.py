import numpy as np
import HardCodedDictionaryUtils
from scipy import stats
from DataGenerationRegime import DataGenerationRegime
from DataGenerationRegime import WeightGenerationRegime
import sys
from numpy.random import RandomState
from MCMCRunningRegime import MCMCRunningRegime
from ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from HMC import HMC
from ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
from LocalRFSamplerForBinaryWeights import  LocalRFSamplerForBinaryWeights
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from PhyloLocalRFMove import PhyloLocalRFMove
from OptionClasses import RFSamplerOptions
from OptionClasses import MCMCOptions

bivariateFeatDictionary = HardCodedDictionaryUtils.getHardCodedDictChainGraph(4)
nLeapFrogSteps = 40
stepSize = 0.002
nItersPerPathAuxVar = 1000
trajectoryLength = 0.125
refreshmentMethod = "GLOBAL"
rfOptions = RFSamplerOptions(trajectoryLength=trajectoryLength, refreshmentMethod=refreshmentMethod)
nMCMCIters = int(1)
mcmcOptions = MCMCOptions(nMCMCIters,1,0)

def getExchangeCoef(nStates, binaryWeights, bivariateFeatDictionary):
    exchangeList = list()
    wholeStates = np.arange(0, nStates)
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            if state1 > state0:
                keyPair = (state0, state1)
                # print(keyPair)
                exchangeList.append(np.exp(np.sum(np.take(binaryWeights, bivariateFeatDictionary[keyPair]))))
    return exchangeList



class GewekeAndExactInvarianceTest:

    def __init__(self, nPriorSamples, nPosteriorSamples, nParam, nRep, K=1000):
        ## nParam represents the number of parameters

        self.nPriorSamples = nPriorSamples
        self.nPosteriorSamples = nPosteriorSamples
        self.nParam = nParam
        self.nRep = nRep
        self.priorSeed = np.arange(0, nRep, 1)
        self.K = K



    def getPriorSamples(self, seed):
        weightsSamples = np.zeros((self.nPriorSamples, self.nParam))
        np.random.seed(seed)
        ## ToDo: check the values from different rows are totally different
        for i in range(self.nPriorSamples):
            weightsSamples[i, :] = np.random.normal(0, 1, self.nParam)

        return weightsSamples



    def gFunc(self, weightSamples, nStates, nBivariateFeat, bivariateFeatDictionary):
        stationaryWeights = weightSamples[0:nStates, :]
        bivariateWeights = weightSamples[nStates:(nStates + nBivariateFeat), :]
        rateMatrix = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights,
                                                                                   bivariateWeights,
                                                                                   bivariateFeatIndexDictionary=bivariateFeatDictionary)
        stationaryDist = rateMatrix.getStationaryDist()
        exchangeCoef = rateMatrix.getExchangeCoef()

        result = {}
        result['weights'] = weightSamples
        result['stationaryDist'] = stationaryDist
        result['exchangeCoef'] = exchangeCoef
        return result

    def gFuncMSamples(self, nPriorSamples, nStates, nBivariateFeat, bivariateFeatDictionary, seed=None, priorWeights=None ):

        priorWeights = priorWeights
        np.random.seed(seed)

        stationaryDistSamples = np.zeros((nPriorSamples, nStates))
        exchangeDim = int(nStates * (nStates-1)/2)
        exchangeCoefSamples = np.zeros((nPriorSamples, exchangeDim))

        if priorWeights is None:
            dim = int(nStates + nBivariateFeat)
            priorWeights = np.zeros((nPriorSamples, dim))

            for i in range(nPriorSamples):
                priorWeights[i, :] = np.random.normal(0, 1, dim)

        for i in range(nPriorSamples):
            result = self.gFunc(priorWeights[i, :], nStates, nBivariateFeat, bivariateFeatDictionary)
            stationaryDistSamples[i, :] = result['stationaryDist']
            exchangeCoefSamples[i, :] = result['exchangeCoef']

        output = {}
        output['stationaryDist'] = stationaryDistSamples
        output['exchangeCoef'] = exchangeCoefSamples
        return output



    def gFuncMean(self, weightSamples, nStates, nBivariateFeat, bivariateFeatDictionary):
        ## this function returns us the stationary weight distribution
        ## and exchangeable parameters
        stationaryWeights = weightSamples[0:nStates, :]
        bivariateWeights = weightSamples[nStates:(nStates+nBivariateFeat), :]

        M = weightSamples.shape[0]

        ## how to apply a function to each row of a numpy array
        ## get the stationary distribution of stationary weights for each row
        unnormalizedStationaryDist = np.exp(stationaryWeights)
        stationaryDist = unnormalizedStationaryDist/unnormalizedStationaryDist.sum(axis=1, keepdims=True)

        ## get the exchangeable parameters for each row of the numpy array
        ## apply getExchangeCoef to each row of the numpy array

        exchangeCoef = np.apply_along_axis(getExchangeCoef, 1, bivariateWeights, nStates, bivariateFeatDictionary)

        ## get the column mean and sd of stationary dist and
        exchangeCoefColMean = np.mean(exchangeCoef, axis=0)
        stationaryDistColMean = np.mean(stationaryDist, axis=0)

        sigma2MStationary = 1/M * np.sum(np.square(stationaryDist), axis=0)-np.square(stationaryDistColMean)
        sigma2MExchangeCoef = 1/M * np.sum(np.square(exchangeCoef), axis=0)-np.square(exchangeCoefColMean)

        result = {}
        result['stationaryDistMean'] = stationaryDistColMean
        result['stationaryDistSigma2'] = sigma2MStationary
        result['exchangeCoefMean'] = exchangeCoefColMean
        result['exchangeCoefSigma2'] = sigma2MExchangeCoef
        return result

    def getReplicateMeanSamples(self, resultsFromGFunc, nStates, nBivariateFeat, bivariateFeatDictionary):
        stationaryMeanSamples = np.zeros((self.nRep, nStates))
        stationarySigma2Samples = np.zeros((self.nRep, nStates))
        exchangeCoefMeanSamples = np.zeros((self.nRep, nBivariateFeat))
        exchangeCoefSigma2Samples = np.zeros((self.nRep, nBivariateFeat))
        M = self.getPriorSamples(self.priorSeed[0]).shape[0]

        for i in range(self.nRep):
            weightSamples = self.getPriorSamples(self.priorSeed[i])
            result = self.gFunc(weightSamples, nStates, nBivariateFeat, bivariateFeatDictionary)
            stationaryMeanSamples[i, :] = result['stationaryDistMean']
            stationarySigma2Samples[i, :] = result['stationaryDistSigma2']
            exchangeCoefMeanSamples[i, :] = result['exchangeCoefMean']
            exchangeCoefSigma2Samples[i, :] = result['exchangeCoefSigma2']

        stationaryMeanEst = np.mean(stationaryMeanSamples, axis=0)
        stationarySigma2Est = np.mean(stationarySigma2Samples, axis=0)

        result = {}
        result['stationaryDistMeanSamples'] = stationaryMeanSamples
        result['stationaryDistSigma2Samples'] = stationarySigma2Samples
        result['exchangeCoefSigma2Samples'] = exchangeCoefSigma2Samples
        result['exchangeCoefMeanSamples'] = exchangeCoefMeanSamples
        result['stationaryMeanEst'] = stationaryMeanEst
        result['stationarySigma2Est'] = stationarySigma2Est/M

        ## we should test if result['stationaryDistMeanSamples'] follows a Normal distribution with mean stationaryMeanEst
        ## and standard deviation stationarySigma2Est/M

        return result

    def successiveConditionalSimulatorLBPS(self, K, nStates, nBivariateFeat, bivariateFeatDictionary, seed, bt, nSeq, interLength, HMCPlusBPS, onlyHMC):
        np.random.seed(seed)
        theta0 = np.random.normal(0, 1, (nStates+nBivariateFeat))
        theta0Stationary = theta0[0:nStates]
        theta0BinaryWeights = theta0[nStates:(nStates+nBivariateFeat)]

        ## create arrays to store the samples
        theta0StationarySamples = np.zeros((K, nStates))
        theta0BinaryWeightsSamples = np.zeros((K, nBivariateFeat))

        ## for debuging reasons, we ave the stationary distribution and exchangeable coef
        ## actually, we only need the last sample at the last iteration K
        theta0StationaryDistSamples = np.zeros((K, nStates))
        theta0ExchangeableCoefSamples = np.zeros((K, int(nStates * (nStates-1)/2)))


        thetaStationaryWeights = theta0Stationary
        thetaBinaryWeights = theta0BinaryWeights

        sample = None

        if onlyHMC:
            sample = np.zeros((nStates+nBivariateFeat))
            sample[0:nStates] = thetaStationaryWeights
            sample[nStates:(nStates+nBivariateFeat)] = thetaBinaryWeights

        if HMCPlusBPS:
            sample = thetaStationaryWeights

        ## based on the current values, generate the first observation
        for i in np.arange(0, K, 1):
            ## generates the next theta based on the current obseration
            ## generate observation based on the current values of the parameters
            ## randomly generate a seed number
            theta0StationarySamples[i, :] = thetaStationaryWeights
            theta0BinaryWeightsSamples[i, :] = thetaBinaryWeights

            weightGenerationRegime = WeightGenerationRegime(nStates=nStates, nBivariateFeat=nBivariateFeat,
                                                            stationaryWeights=thetaStationaryWeights,
                                                            bivariateWeights=thetaBinaryWeights)
            prng = RandomState(np.random.choice(sys.maxsize, 1))

            dataRegime = DataGenerationRegime(nStates=nStates,
                        bivariateFeatIndexDictionary=bivariateFeatDictionary, btLength=bt, nSeq=nSeq,
                                                                   weightGenerationRegime=weightGenerationRegime,
                                                                   prng=prng, interLength=interLength)
            ## generate the sequences data
            seqList = dataRegime.generatingSeq()
            suffStat = dataRegime.getSufficientStatFromSeq(seqList)
            firstLastStatesArrayAll = dataRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)


            # replicate K iterations to get new parameters
            nTrans = suffStat["transitCount'"]
            holdTime = suffStat["sojourn"]

            unique, counts = np.unique(firstLastStatesArrayAll[0][:, 0], return_counts=True)
            nInitCount = np.asarray((unique, counts)).T
            nInit = nInitCount[:, 1]

            initialExchangeCoef = getExchangeCoef(nStates, thetaBinaryWeights, bivariateFeatDictionary)
            expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, initialExchangeCoef)

            ## given the current data and previous theta

            if HMCPlusBPS:
                expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0,
                                                                                          initialExchangeCoef)
            if onlyHMC:
                expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, nBivariateFeatWeightsDictionary=bivariateFeatDictionary)

            #####################################
            hmc = HMC(nLeapFrogSteps, stepSize, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
            lastSample = sample
            for k in range(nItersPerPathAuxVar):
                 hmcResult = hmc.doIter(nLeapFrogSteps, stepSize, lastSample, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective, True)
                 lastSample = hmcResult.next_q

            sample = lastSample

            if onlyHMC:
                thetaStationaryWeights = sample[0:nStates]
                thetaBinaryWeights = sample[nStates:(nStates + nBivariateFeat)]

            # sample stationary distribution elements using HMC
            if HMCPlusBPS:
                thetaStationaryWeights = sample
                # update stationary distribution elements to the latest value
                thetaStationaryDist = np.exp(sample) / np.sum(np.exp(sample))
                # sample exchangeable coefficients using local bouncy particle sampler
                ## define the model
                model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates,
                                                                       thetaBinaryWeights,  thetaStationaryDist,
                                                                         bivariateFeatDictionary)
                ## define the sampler to use
                ## local sampler to use

                localSampler =LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates,
                                                            bivariateFeatDictionary)
                phyloLocalRFMove = PhyloLocalRFMove(model=model, sampler=localSampler, initialPoints=thetaBinaryWeights, options=rfOptions, prng=RandomState(i))
                thetaBinaryWeights = phyloLocalRFMove.execute()

            initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, thetaStationaryWeights,
                                                                                           thetaBinaryWeights,
                                                                                           bivariateFeatIndexDictionary=bivariateFeatDictionary)

            ## save stationary distribution samples
            ## save exchangeable coef samples

            initialStationaryDist = initialRateMtx.getStationaryDist()
            initialExchangeCoef = initialRateMtx.getExchangeCoef()
            theta0StationaryDistSamples[i, :] = initialStationaryDist
            theta0ExchangeableCoefSamples[i, :] = initialExchangeCoef

        result = {}
        result['stationaryDist'] = theta0StationaryDistSamples[(K-1), :]
        result['exchangeCoef'] = theta0ExchangeableCoefSamples[(K-1), :]

        return result

















