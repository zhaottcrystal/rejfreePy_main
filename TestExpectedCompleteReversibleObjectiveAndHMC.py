import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

from ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from HMC import HMC
import numpy as np
from ReversibleRateMtxPiAndExchangeGTR import ReversibleRateMtxPiAndExchangeGTR
from FullTrajectorGeneration import getFirstAndLastStateOfListOfSeq
from FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from numpy.random import RandomState
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import  ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from HardCodedDictionaryUtils import getHardCodedDictChainGraph
from ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
from LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
from PhyloLocalRFMove import  PhyloLocalRFMove
from OptionClasses import RFSamplerOptions
from OptionClasses import MCMCOptions
import matplotlib.pyplot as plt

import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def obtainSufficientStatisticsForOneRateMtx(nStates):

    ## we generate the sufficient statistics for a large number of replications first
    ## and then we summarize the sufficient statistics for the forward sampler and
    ## then we use this data to run HMC and local BPS algorithms separately to see
    ## if we can obtain reasonable estimates of the exchangeable parameters

    nStates = 2
    nRep = 10000
    seedNum = np.arange(0, nRep)
    bivariateDictionary = getHardCodedDictChainGraph(nStates)

    bt = 3.0
    nSeq = 100

    prng = RandomState(1234567890)
    stationaryWeights = prng.uniform(0, 1, nStates)
    bivariateWeights = prng.normal(0, 1, int((nStates) * (nStates - 1) / 2))
    testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights,
                                                                            bivariateWeights, bivariateDictionary)
    stationaryDist = testRateMtx.getStationaryDist()
    rateMatrix = testRateMtx.getRateMtx()
    print(rateMatrix)
    print(stationaryDist)
    print(testRateMtx.getExchangeCoef())

    nInit = np.zeros(nStates)
    holdTimes = np.zeros(nStates)
    nTrans = np.zeros((nStates, nStates))

    for i in seedNum:

        seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, RandomState(i), rateMatrix,
                                                                stationaryDist, bt)
        ## summarize the sufficient statistics
        ## extract first state from sequences
        firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
        unique, counts = np.unique(firstStates, return_counts=True)
        nInitCount = np.asarray((unique, counts)).T
        nInit = nInit + nInitCount[:, 1]

        for j in range(nSeq):
            sequences = seqList[j]
            holdTimes = holdTimes + sequences['sojourn']
            nTrans = nTrans + sequences['transitCount']
        print(i)

    avgNTrans = nTrans / nRep
    avgHoldTimes = holdTimes / nRep
    avgNInit = nInit / nRep

    result = {}
    result['stationaryWeights'] = stationaryWeights
    result['bivariateDictionary'] = bivariateDictionary
    result['bivariateWeights'] = bivariateWeights
    result['rateMatrix'] = rateMatrix
    result['stationaryDist'] = stationaryDist
    result['exchangeableCoef'] = testRateMtx.getExchangeCoef()
    result['transitCount'] = avgNTrans
    result['sojourn'] = avgHoldTimes
    result['nInit'] = avgNInit
    return result


## define a global variable
# data = obtainSufficientStatisticsForOneRateMtx()

def testLocalBPSForStationaryAndBivariateWeights(data):
    avgNTrans = data['transitCount']
    avgHoldTimes = data['sojourn']
    avgNInit = data['nInit']
    bivariateDictionary = data['bivariateDictionary']

    prng = np.random.RandomState(1)
    ## run HMC to estimate the stationary distribution
    initialExchangeCoef = [np.exp(prng.uniform(0, 1, 1))]
    ## construct expected complete reversible model objective
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(avgHoldTimes, avgNInit, avgNTrans, 1.0, nBivariateFeatWeightsDictionary=bivariateDictionary)
    nStates = data['rateMatrix'].shape[0]

    initialWeights = np.array((0.61879477, 0.59162363))
    stationaryDistEst = np.exp(initialWeights) / np.sum(np.exp(initialWeights))
    # update stationary distribution elements to the latest value
    initialStationaryDist = stationaryDistEst

    # sample exchangeable coefficients using local bouncy particle sampler
    ## define the model
    model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates,
                                                             [np.log(initialExchangeCoef)], initialStationaryDist,
                                                             bivariateDictionary)

    ## define the sampler to use
    ## local sampler to use
    rfOptions = RFSamplerOptions(trajectoryLength=0.1)
    mcmcOptions = MCMCOptions(3000,1,0)
    nBivariateFeat = int(nStates *(nStates-1)/2)
    initialBinaryWeights = prng.normal(0, 1, nBivariateFeat)
    seed = 3
    nMCMCIters = 1000

    stationarySamples = np.zeros((nMCMCIters, nStates))
    binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
    exchangeableSamples = np.zeros((nMCMCIters, len(initialExchangeCoef)))

    ####### below is the older version of the sampler
    for i in range(nMCMCIters):
        # save the samples of the parameters
        stationarySamples[i, :] = initialStationaryDist
        binaryWeightsSamples[i, :] = initialBinaryWeights
        exchangeableSamples[i, :] = initialExchangeCoef
        localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates, bivariateDictionary)
        phyloLocalRFMove = PhyloLocalRFMove(model, localSampler, initialBinaryWeights, options=rfOptions, randomSeed=i)
        initialBinaryWeights = phyloLocalRFMove.execute()
        print("The initial estimates of the binary weights are:")
        # print(initialBinaryWeights)

        # localSamplerOld = LocalRFSamplerForBinaryWeightsOldVersion(model, rfOptions, mcmcOptions, nStates,
        #                                                          bivariateFeatIndexDictionary)
        # phyloLocalRFMove = PhyloLocalRFMove(seed, model, localSamplerOld, initialBinaryWeights)
        # initialBinaryWeightsOld = phyloLocalRFMove.execute()

        initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights,
                                                                               initialBinaryWeights,
                                                                               bivariateDictionary)

        initialStationaryDist = np.round(initialRateMtx.getStationaryDist(), 3)
        initialExchangeCoef = np.round(initialRateMtx.getExchangeCoef(), 3)
        print(i)


    result = {}
    result['stationaryDist'] = stationarySamples
    result['binaryWeights'] = binaryWeightsSamples
    result['exchangeableCoef'] = exchangeableSamples
    return result













def testHMCForStationaryAndBivariateWeights(data):
    avgNTrans = data['transitCount']
    avgHoldTimes = data['sojourn']
    avgNInit = data['nInit']
    bivariateDictionary = data['bivariateDictionary']

    prng = np.random.RandomState(1)
    ## run HMC to estimate the stationary distribution                                          nBivariateFeatWeightsDictionary=bivariateDictionary)
    nStates = data['rateMatrix'].shape[0]
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTimes=avgHoldTimes, nInit=avgNInit,
                                                                              nTrans=avgNTrans, kappa=1,
                                                                              nBivariateFeatWeightsDictionary=bivariateDictionary)

    hmc = HMC(prng, 40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = prng.uniform(0, 1, int(nStates + nStates * (nStates - 1) / 2))
    samples = hmc.run(0, 10000, sample)

    ## from the weights, obtain stationary distribution and exchangeable parameters
    avgStationaryWeights = np.sum(samples[:, 0:nStates], axis=0) / samples.shape[0]
    avgBinaryWeights = np.sum(samples[:, nStates:(nStates + int(nStates * (nStates - 1) / 2))]) / samples.shape[0]
    newRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, avgStationaryWeights,
                                                                           avgBinaryWeights,
                                                                           bivariateFeatIndexDictionary=bivariateDictionary)
    exchangeableParam = newRateMtx.getExchangeCoef()
    stationaryDistEst = newRateMtx.getStationaryDist()
    print(exchangeableParam)
    print(stationaryDistEst)





def test():
    ## Provided the weights for the stationary distribution and the exchangeable coefficients
    ## we use them to generate the true rate matrix and the sequences to get the averaged
    ## sufficient statistics for the sequences throughout a larger number of replications
    ## Based on the sufficient statistics and we fix the exchangeable coefficients to
    ## its true values, we use HMC to estimate the weights for the stationary distribution
    ## The rate matrix is an un-normalized version
    
    ## The correctness of HMC and ExpectedCompleteReversibleObjective has been tested
    ## The estimated stationary distribution 'stationaryDistEst' is very close to stationaryDist

    nStates = 4
    nRep = 1000
    seedNum = np.arange(0, nRep)
    np.random.seed(123)
    weights = np.random.uniform(0, 1, nStates)
    print(weights)
    exchangeCoef = np.array((1, 2, 3, 4, 5, 6))

    ## get the rate matrix
    testRateMtx = ReversibleRateMtxPiAndExchangeGTR(nStates, weights, exchangeCoef)
    stationaryDist = testRateMtx.getStationaryDist()
    rateMtx = testRateMtx.getRateMtx()
    bt = 5.0
    nSeq = 100

    nInit = np.zeros(nStates)
    holdTimes = np.zeros(nStates)
    nTrans = np.zeros((nStates, nStates))

    for j in range(nRep):
        ## do forward sampling
        seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seedNum[j], rateMtx, stationaryDist, bt)
        ## summarize the sufficient statistics
        ## extract first state from sequences
        firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
        unique, counts = np.unique(firstStates, return_counts=True)
        nInitCount = np.asarray((unique, counts)).T
        nInit = nInit + nInitCount[:, 1]

        for i in range(nSeq):
            sequences = seqList[i]
            holdTimes = holdTimes + sequences['sojourn']
            nTrans = nTrans + sequences['transitCount']

    avgNTrans = nTrans / nRep
    avgHoldTimes = holdTimes / nRep
    avgNInit = nInit / nRep

    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTimes=avgHoldTimes, nInit=avgNInit, nTrans=avgNTrans, kappa=1, exchangeCoef=exchangeCoef)
    prng = np.random.RandomState(1)
    hmc = HMC(prng, 40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = prng.uniform(0, 1, nStates)
    samples = hmc.run(0, 5000, sample)
    avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    stationaryDistEst = np.exp(avgWeights)/np.sum(np.exp(avgWeights))
    print(weights)
    print(avgWeights)
    print(stationaryDist)
    print(stationaryDistEst)


def main():
    # TestForwardAndEndPointSamplers.test()
    # testHMCForStationaryAndBivariateWeights()
    # data = obtainSufficientStatisticsForOneRateMtx()
    nStates = 2
    prng = RandomState(1234567890)
    stationaryWeights = prng.uniform(0, 1, nStates)
    bivariateWeights = prng.normal(0, 1, int((nStates) * (nStates - 1) / 2))
    testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights,
                                                                            bivariateWeights, getHardCodedDictChainGraph(2))
    stationaryDist = testRateMtx.getStationaryDist()
    rateMatrix = testRateMtx.getRateMtx()
    data = {}
    data['bivariateWeights'] = 1.49898834
    data['exchangeableCoef'] = 4.47715743666
    data['nInit'] = np.array((50.7113, 49.2887))
    data['sojourn'] = np.array((152.02389579, 147.97610421))
    data['stationaryDist'] = stationaryDist
    data['stationaryWeights'] = np.array((0.61879477, 0.59162363))
    data['transitCount'] = np.zeros((2, 2))
    data['transitCount'][0, 0 ] = data['transitCount'][1, 1] =0
    data['transitCount'][0, 1] = 335.5429
    data['transitCount'][1, 0] = 335.5562
    data['bivariateDictionary'] = getHardCodedDictChainGraph(2)
    data['rateMatrix'] = rateMatrix
    result = testLocalBPSForStationaryAndBivariateWeights(data=data)
    #np.savetxt(trueStationaryDistFileName, self.dataGenerationRegime.stationaryDist, fmt='%.3f', delimiter=',')
    np.savetxt("/Users/crystal/Desktop/binaryWeightsRunningtestLocalBPSForStationaryAndBivariateWeightsForBinaryWeightsSamplers.csv", result['binaryWeights'], fmt='%.3f', delimiter=',')
    np.savetxt("/Users/crystal/Desktop/binaryWeightsRunningtestLocalBPSForStationaryAndBivariateWeightsForExchangeableCoef.csv", result['exchangeableCoef'], fmt='%.3f', delimiter=',')
    plt.plot(np.arange(100, 1000, 1), result['exchangeableCoef'][100:1000])
    plt.show()
    plt.hist(result['exchangeableCoef'])



    # N = 500
    # random_x = np.linspace(0, 1, N)
    # random_y = np.random.randn(N)
    #
    # # Create a trace
    # trace = go.Scatter(
    #     x=random_x,
    #     y=random_y
    # )
    #
    # data = [trace]
    #
    # py.iplot(data, filename='basic-line')

if __name__ == "__main__": main()