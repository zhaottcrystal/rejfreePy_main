import numpy as np
import HardCodedDictionaryUtils
from DataGenerationRegime import DataGenerationRegime
from DataGenerationRegime import WeightGenerationRegime
import sys
from numpy.random import RandomState
from ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from HMC import HMC
from ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
from LocalRFSamplerForBinaryWeights import  LocalRFSamplerForBinaryWeights
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from PhyloLocalRFMove import PhyloLocalRFMove
from OptionClasses import RFSamplerOptions
from OptionClasses import MCMCOptions
from OptionClasses import RefreshmentMethod

import argparse



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


class ExactInvarianceTestHMC:

    def __init__(self, nPriorSamples,nParam, K=1000):
        ## nParam represents the number of parameters
        ## nParam represents the total number univariate and bivariate features
        self.nPriorSamples = nPriorSamples
        self.nParam = nParam
        self.priorSeed = np.arange(0, nPriorSamples, 1)
        self.K = K



    def getPriorSamples(self, seed):
        weightsSamples = np.zeros((self.nPriorSamples, self.nParam))
        np.random.seed(seed)
        ## ToDo: check the values from different rows are totally different
        for i in range(self.nPriorSamples):
            weightsSamples[i, :] = np.random.normal(0, 1, self.nParam)

        return weightsSamples



    def gFunc(self, weightSamples, nStates, nBivariateFeat, bivariateFeatDictionary):
        stationaryWeights = weightSamples[0:nStates]
        bivariateWeights = weightSamples[nStates:(nStates + nBivariateFeat)]
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

    def gFuncMSamples(self, nStates, nBivariateFeat, bivariateFeatDictionary, seed=1, priorWeights=None):

        priorWeights = priorWeights
        np.random.seed(seed)

        stationaryDistSamples = np.zeros((self.nPriorSamples, nStates))
        exchangeDim = int(nStates * (nStates-1)/2)
        exchangeCoefSamples = np.zeros((self.nPriorSamples, exchangeDim))

        if priorWeights is None:
            dim = int(nStates + nBivariateFeat)
            priorWeights = np.zeros((self.nPriorSamples, dim))

            for i in range(self.nPriorSamples):
                priorWeights[i, :] = np.random.normal(0, 1, dim)

        for i in range(self.nPriorSamples):
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


    def sameInitialWeightsAndDataGeneration(self, nStates, nBivariateFeat, bivariateFeatDictionary, seed, bt, nSeq, interLength):
        np.random.seed(seed)
        theta0 = np.random.normal(0, 1, (nStates + nBivariateFeat))
        stationaryWeights = theta0[0:nStates]
        binaryWeights = theta0[nStates:(nStates + nBivariateFeat)]

        ## based on the current values, generate the first observation
        ## generates the next theta based on the current obseration
        ## generate observation based on the current values of the parameters
        ## randomly generate a seed number
        weightGenerationRegime = WeightGenerationRegime(nStates=nStates, nBivariateFeat=nBivariateFeat,
                                                            stationaryWeights=stationaryWeights,
                                                            bivariateWeights=binaryWeights)
        prng = RandomState(np.random.choice(2 ** 32 - 1, 1))

        dataRegime = DataGenerationRegime(nStates=nStates,
                                        bivariateFeatIndexDictionary=bivariateFeatDictionary, btLength=bt,
                                        nSeq=nSeq,
                                        weightGenerationRegime=weightGenerationRegime,
                                        prng=prng, interLength=interLength)
        ## generate the sequences data
        seqList = dataRegime.generatingSeq()
        suffStat = dataRegime.getSufficientStatFromSeq(seqList)
        firstLastStatesArrayAll = dataRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)

        # replicate K iterations to get new parameters
        nTrans = suffStat["transitCount"]
        holdTime = suffStat["sojourn"]

        nInit = np.zeros(nStates)
        unique, counts = np.unique(firstLastStatesArrayAll[0][:, 0], return_counts=True)
        nInitCount = np.asarray((unique, counts)).T
        nInit[nInitCount[:, 0].astype(int)] = nInitCount[:, 1]


        suffStatDict = {}
        suffStatDict['nTrans'] = nTrans
        suffStatDict['sojourn'] = holdTime
        suffStatDict['nInit'] = nInit

        result = {}
        result['stationaryWeights'] = stationaryWeights
        result['binaryWeights'] = binaryWeights
        result['seqList'] = seqList
        result['suffStat'] = suffStatDict

        return result


    def succesiveConditionalSimulatorWithInitialWeightAndData(self, initialParamAndData, K, nStates, nBivariateFeat, bivariateFeatDictionary, seed, bt, nSeq,
                                       interLength, HMCPlusBPS, onlyHMC, nLeapFrogSteps, stepSize, nItersPerPathAuxVar, rfOptions, mcmcOptions):

        thetaStationaryWeights = initialParamAndData['stationaryWeights']
        thetaBinaryWeights = initialParamAndData['binaryWeights']

        theta0StationarySamples = np.zeros((K, nStates))
        theta0BinaryWeightsSamples = np.zeros((K, nBivariateFeat))

        ## for debuging reasons, we ave the stationary distribution and exchangeable coef
        ## actually, we only need the last sample at the last iteration K
        theta0StationaryDistSamples = np.zeros((K, nStates))
        theta0ExchangeableCoefSamples = np.zeros((K, int(nStates * (nStates - 1) / 2)))

        theta0StationarySamples[0,:] = thetaStationaryWeights
        theta0BinaryWeightsSamples[0, :] = thetaBinaryWeights
        theta0StationaryDistSamples[0,:] = np.exp(thetaStationaryWeights)/np.sum(np.exp(thetaStationaryWeights))
        initialExchangeCoef =  getExchangeCoef(nStates, thetaBinaryWeights, bivariateFeatDictionary)
        theta0ExchangeableCoefSamples[0, :] = initialExchangeCoef

        sample = None

        if onlyHMC:
            sample = np.zeros((nStates+nBivariateFeat))
            sample[0:nStates] = thetaStationaryWeights
            sample[nStates:(nStates+nBivariateFeat)] = thetaBinaryWeights

        if HMCPlusBPS:
            sample = thetaStationaryWeights

        suffStat =   initialParamAndData['suffStat']
        nInit = suffStat['nInit']
        nTrans = suffStat['nTrans']
        holdTime = suffStat['sojourn']

        expectedCompleteReversibleObjective = None
        for i in np.arange(1, K, 1):

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
                theta0StationarySamples[i, :] = thetaStationaryWeights
                theta0BinaryWeightsSamples[i, :] = thetaBinaryWeights

            if HMCPlusBPS:
                thetaStationaryWeights = sample
                theta0StationarySamples[i, :] = thetaStationaryWeights
                # update stationary distribution elements to the latest value
                thetaStationaryDist = np.exp(sample) / np.sum(np.exp(sample))
                # sample exchangeable coefficients using local bouncy particle sampler
                ## define the model
                model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates,
                                                                       thetaBinaryWeights,  thetaStationaryDist,
                                                                         bivariateFeatDictionary)
                ## define the sampler to use
                ## local sampler to use

                localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates,
                                                            bivariateFeatDictionary)
                phyloLocalRFMove = PhyloLocalRFMove(model=model, sampler=localSampler, initialPoints=thetaBinaryWeights, options=rfOptions, prng=RandomState(i))
                thetaBinaryWeights = phyloLocalRFMove.execute()
                theta0BinaryWeightsSamples[i, :] = thetaBinaryWeights

            initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, thetaStationaryWeights,
                                                                                       thetaBinaryWeights,
                                                                                       bivariateFeatIndexDictionary=bivariateFeatDictionary)

            initialStationaryDist = initialRateMtx.getStationaryDist()
            initialExchangeCoef = initialRateMtx.getExchangeCoef()
            theta0StationaryDistSamples[i, :] = initialStationaryDist
            theta0ExchangeableCoefSamples[i, :] = initialExchangeCoef

            weightGenerationRegime = WeightGenerationRegime(nStates=nStates, nBivariateFeat=nBivariateFeat,
                                                            stationaryWeights=thetaStationaryWeights,
                                                            bivariateWeights=thetaBinaryWeights)

            prng = RandomState(np.random.choice(2 ** 32 - 1, 1))

            dataRegime = DataGenerationRegime(nStates=nStates,
                                              bivariateFeatIndexDictionary=bivariateFeatDictionary, btLength=bt,
                                              nSeq=nSeq,
                                              weightGenerationRegime=weightGenerationRegime,
                                              prng=prng, interLength=interLength)
            ## generate the sequences data
            seqList = dataRegime.generatingSeq()
            suffStat = dataRegime.getSufficientStatFromSeq(seqList)
            firstLastStatesArrayAll = dataRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)

            # replicate K iterations to get new parameters
            nTrans = suffStat["transitCount"]
            holdTime = suffStat["sojourn"]

            nInit = np.zeros(nStates)
            unique, counts = np.unique(firstLastStatesArrayAll[0][:, 0], return_counts=True)
            nInitCount = np.asarray((unique, counts)).T
            nInit[nInitCount[:, 0].astype(int)] = nInitCount[:, 1]


        result = {}
        result['stationaryDist'] = theta0StationaryDistSamples[(K - 1), :]
        result['exchangeCoef'] = theta0ExchangeableCoefSamples[(K - 1), :]
        result['stationaryWeights'] = thetaStationaryWeights
        result['binaryWeights'] = thetaBinaryWeights
        ## after testing the code is right, the following two lines should be removed
        #result['StationaryDistSamples'] = theta0StationaryDistSamples
        #result['ExchangeableCoefSamples'] = theta0ExchangeableCoefSamples
        return result

    def getMSuccessiveConditionalSamples(self, M, K, nStates, nBivariateFeat, bivariateFeatDictionary, seed, bt, nSeq, interLength, HMCPlusBPS, onlyHMC, nLeapFrogSteps, stepSize, nItersPerPathAuxVar, rfOptions, mcmcOptions):

        mStationarySamples = np.zeros((M, nStates))
        mStationaryWeightsSamples = np.zeros((M, nStates))
        exchangeCoefDim = int(nStates * (nStates-1)/2)
        mExchangeCoefSamples = np.zeros((M, exchangeCoefDim))
        mBinaryWeightsSamples = np.zeros((M, nBivariateFeat))


        for i in range(M):
            paramAndData = self.sameInitialWeightsAndDataGeneration(nStates, nBivariateFeat, bivariateFeatDictionary, i, bt,
                                                nSeq, interLength)

            result = self.succesiveConditionalSimulatorWithInitialWeightAndData(paramAndData, K, nStates, nBivariateFeat, bivariateFeatDictionary, i, bt, nSeq, interLength, HMCPlusBPS, onlyHMC, nLeapFrogSteps, stepSize, nItersPerPathAuxVar, rfOptions, mcmcOptions)
            mStationarySamples[i, :] = result['stationaryDist']
            mStationaryWeightsSamples[i, :] = result['stationaryWeights']
            mExchangeCoefSamples[i, :] = result['exchangeCoef']
            mBinaryWeightsSamples[i, :] = result['binaryWeights']
            print(i)

        result = {}
        result['StationaryDistSamples'] = mStationarySamples
        result['ExchangeableCoefSamples'] = mExchangeCoefSamples
        result['stationaryWeightsSamples'] = mStationaryWeightsSamples
        result['binaryWeightsSamples'] = mBinaryWeightsSamples

        return result


    @staticmethod
    def main():
        ## test some correctness of this class

        argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('-nStates', action="store", type=int, dest='nStates', default=2,
                            help='save the number of states in the ctmc')
        ## add boolean variable to indicate whether we only use hmc or we use a combination of hmc and local bps
        parser.add_argument('--onlyHMC', action="store_true",
                            help='HMC flag, the existence of the argument indicates HMC is used.')
        ## add boolean variable to indicate whether we use the local bps algorithm
        parser.add_argument('--HMCPlusBPS', action='store_true',
                            help='BPS flag, the existence of the argument indicates a combination of HMC and local BPS is used.')

        parser.add_argument('-dir_name', action='store', dest='dir_name', type=str,
                            help='store the directory name to save the csv files')

        bivariateFeatDictionary = HardCodedDictionaryUtils.getHardCodedDictChainGraph(3)
        nLeapFrogSteps = 40
        stepSize = 0.002
        nItersPerPathAuxVar = 30
        trajectoryLength = 0.125
        refreshmentMethod = RefreshmentMethod.LOCAL
        rfOptions = RFSamplerOptions(trajectoryLength=trajectoryLength, refreshmentMethod=refreshmentMethod)
        nMCMCIters = int(1)
        mcmcOptions = MCMCOptions(nMCMCIters, 1, 0)

        M = 150
        K = 100
        EIT3by3 = ExactInvarianceTestHMC(M, 10, K)

        ## save prior samples
        fWeightSamples = EIT3by3.getPriorSamples(123456789)
        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/fWeightshmc.csv", fWeightSamples, fmt='%.3f', delimiter=',')


        fGFuncSamples= EIT3by3.gFuncMSamples(4, 6, bivariateFeatDictionary, seed=2, priorWeights=fWeightSamples)

        fStationary = fGFuncSamples['stationaryDist']
        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/fStationaryhmc.csv", fStationary, fmt='%.3f', delimiter=',')
        fExchangeCoef = fGFuncSamples['exchangeCoef']
        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/fExchangehmc.csv", fExchangeCoef, fmt='%.3f', delimiter=',')

        HTransitionSampleHMC = EIT3by3.getMSuccessiveConditionalSamples(M=M, K=K, nStates=4, nBivariateFeat=6,
                                                                        bivariateFeatDictionary=bivariateFeatDictionary,
                                                                        seed=3, bt=3, nSeq=50,
                                                                        interLength=0.5, HMCPlusBPS=False, onlyHMC=True,
                                                                        nLeapFrogSteps=nLeapFrogSteps,
                                                                        stepSize=stepSize,
                                                                        nItersPerPathAuxVar=nItersPerPathAuxVar,
                                                                        rfOptions=rfOptions, mcmcOptions=mcmcOptions)

        stationaryWeightsHMC = HTransitionSampleHMC['stationaryWeightsSamples']
        exchangeCoefHMC = HTransitionSampleHMC['ExchangeableCoefSamples']
        stationaryDistHMC = HTransitionSampleHMC['StationaryDistSamples']
        binaryWeightsHMC = HTransitionSampleHMC['binaryWeightsSamples']

        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/hmcStationaryWeights.csv", stationaryWeightsHMC, fmt='%.3f',
                   delimiter=',')
        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/hmcStationaryDist.csv", stationaryDistHMC, fmt='%.3f', delimiter=',')
        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/hmcBinaryWeights.csv", binaryWeightsHMC, fmt='%.3f', delimiter=',')
        np.savetxt("/home/zhaott/project/zhaott/rejfreePy_main/EIT/hmcExchangeCoef.csv", exchangeCoefHMC, fmt='%.3f', delimiter=',')







        ## test every function of this class
        ## test getExchangeCoef
        # prng = np.random.RandomState(1)
        # binaryWeights = prng.normal(0, 1, 12)
        # bivariateFeatDictionary6 = HardCodedDictionaryUtils.getHardCodedDict()
        # exchangeCoef = getExchangeCoef(6, binaryWeights, bivariateFeatDictionary6)
        # print(np.round(exchangeCoef, 3))
        # ## calculate the exchange coef mannually and compare it with exchangeCoef
        # exchangeCoefManually = np.zeros(15)
        # featIndexlist = [None] * 15
        # featIndexlist[0] = np.array((0, 1), dtype=np.int)
        # featIndexlist[1] = np.array((0, 1, 2), dtype=np.int)
        # featIndexlist[2] = np.array((1, 2), dtype=np.int)
        # featIndexlist[3] = np.array((0, 2), dtype=np.int)
        # featIndexlist[4] = np.array((3, 4), dtype=np.int)
        # featIndexlist[5] = np.array((3, 4, 5), dtype=np.int)
        # featIndexlist[6] = np.array((3, 5), dtype=np.int)
        # featIndexlist[7] = np.array((4, 5), dtype=np.int)
        # featIndexlist[8] = np.array((6, 7), dtype=np.int)
        # featIndexlist[9] = np.array((6, 7, 8), dtype=np.int)
        # featIndexlist[10] = np.array((7, 8), dtype=np.int)
        # featIndexlist[11] = np.array((6, 8), dtype=np.int)
        # featIndexlist[12] = np.array((9, 10), dtype=np.int)
        # featIndexlist[13] = np.array((9, 11), dtype=np.int)
        # featIndexlist[14] = np.array((10, 11), dtype=np.int)
        # for i in range(len(exchangeCoefManually)):
        #     exchangeCoefManually[i] = np.exp(np.sum(np.take(binaryWeights, featIndexlist[i])))
        # print(np.round(exchangeCoefManually, 3))

        ## the exchangeable parameters are calculated correctly



if __name__ == "__main__":
    ExactInvarianceTestHMC.main()












