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



def obtainSufficientStatisticsForChainGraphRateMtx(nStates, nRep=5000, bt=5.0, nSeq=100, SDOfBiWeights=0.5, bivariateDictionary=None):

    ## we generate the sufficient statistics for a large number of replications first
    ## and then we summarize the sufficient statistics for the forward sampler and
    ## then we use this data to run HMC and local BPS algorithms separately to see
    ## if we can obtain reasonable estimates of the exchangeable parameters

    seedNum = np.arange(0, nRep)

    if bivariateDictionary is None:
        bivariateDictionary = getHardCodedDictChainGraph(nStates)

    prng = RandomState(1234567890)
    stationaryWeights = prng.uniform(0, 1, nStates)
    bivariateWeights = prng.normal(0, SDOfBiWeights, int((nStates) * (nStates - 1) / 2))
    testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights,
                                                                            bivariateWeights, bivariateDictionary)
    stationaryDist = testRateMtx.getStationaryDist()
    rateMatrix = testRateMtx.getRateMtx()

    nInit = np.zeros(nStates)
    holdTimes = np.zeros(nStates)
    nTrans = np.zeros((nStates, nStates))

    for i in seedNum:

        seqList = generateFullPathUsingRateMtxAndStationaryDist(RandomState(i), nSeq, nStates,  rateMatrix,
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

def obtainSufficientStatisticsForOneRateMtx(nStates):

    ## we generate the sufficient statistics for a large number of replications first
    ## and then we summarize the sufficient statistics for the forward sampler and
    ## then we use this data to run HMC and local BPS algorithms separately to see
    ## if we can obtain reasonable estimates of the exchangeable parameters

    nRep = 5000
    seedNum = np.arange(0, nRep)
    bivariateDictionary = getHardCodedDictChainGraph(nStates)

    bt = 5.0
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

def testLocalBPSForStationaryAndBivariateWeights(data, nMCMCIter = 500, trajectoryLength=0.1):
    avgNTrans = data['transitCount']
    avgHoldTimes = data['sojourn']
    avgNInit = data['nInit']
    bivariateDictionary = data['bivariateDictionary']

    prng = np.random.RandomState(1)
    nStates = len(data['stationaryWeights'])
    ## run HMC to estimate the stationary distribution
    initialExchangeCoef = np.exp(prng.uniform(0, 1, int(nStates *(nStates-1)/2)))
    ## construct expected complete reversible model objective
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(avgHoldTimes, avgNInit, avgNTrans, 1.0,  initialExchangeCoef)


    initialWeights = data['stationaryWeights']
    stationaryDistEst = np.exp(initialWeights) / np.sum(np.exp(initialWeights))
    # update stationary distribution elements to the latest value
    initialStationaryDist = stationaryDistEst

    # sample exchangeable coefficients using local bouncy particle sampler
    ## define the model
    if isinstance(initialExchangeCoef, float):
        initialExchangeCoef = [initialExchangeCoef]
    model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates,
                                                             np.log(initialExchangeCoef),
                                                             initialStationaryDist,
                                                             bivariateDictionary)

    ## define the sampler to use
    ## local sampler to use
    rfOptions = RFSamplerOptions(trajectoryLength=trajectoryLength)
    mcmcOptions = MCMCOptions(nMCMCIter,1,0)
    nBivariateFeat = int(nStates *(nStates-1)/2)
    initialBinaryWeights = prng.normal(0, 1, nBivariateFeat)
    seed = 3

    stationarySamples = np.zeros((nMCMCIter, nStates))
    binaryWeightsSamples = np.zeros((nMCMCIter, nBivariateFeat))
    exchangeableSamples = np.zeros((nMCMCIter, len(initialExchangeCoef)))

    ####### below is the older version of the sampler
    for i in range(nMCMCIter):
        # save the samples of the parameters
        stationarySamples[i, :] = initialStationaryDist
        binaryWeightsSamples[i, :] = initialBinaryWeights
        exchangeableSamples[i, :] = initialExchangeCoef
        localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates, bivariateDictionary)
        phyloLocalRFMove = PhyloLocalRFMove(model, localSampler, initialBinaryWeights, options=rfOptions, randomSeed=i)
        initialBinaryWeights = phyloLocalRFMove.execute()
        print("The initial estimates of the exchangeable param are:")
        print(initialExchangeCoef)
        # print(initialBinaryWeights)

        # localSamplerOld = LocalRFSamplerForBinaryWeightsOldVersion(model, rfOptions, mcmcOptions, nStates,
        #                                                          bivariateFeatIndexDictionary)
        # phyloLocalRFMove = PhyloLocalRFMove(seed, model, localSamplerOld, initialBinaryWeights)
        # initialBinaryWeightsOld = phyloLocalRFMove.execute()

        initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights,
                                                                               initialBinaryWeights,
                                                                               bivariateDictionary)

        initialStationaryDist = initialRateMtx.getStationaryDist()
        initialExchangeCoef = initialRateMtx.getExchangeCoef()
        print(i)


    result = {}
    result['stationaryWeights'] = stationaryWeightsSamples
    result['stationaryDist'] = stationarySamples
    result['binaryWeights'] = binaryWeightsSamples
    result['exchangeableCoef'] = exchangeableSamples
    return result


def testHMCForStationaryAndBivariateWeights(data, nMCMCIter = 500, nLeapFrogSteps=40, stepSize=0.002):
    avgNTrans = data['transitCount']
    avgHoldTimes = data['sojourn']
    avgNInit = data['nInit']
    bivariateDictionary = data['bivariateDictionary']

    ## run HMC to estimate the stationary distribution                                          nBivariateFeatWeightsDictionary=bivariateDictionary)
    nStates = len(data['stationaryWeights'])
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTimes=avgHoldTimes, nInit=avgNInit,
                                                                              nTrans=avgNTrans, kappa=1,
                                                                              nBivariateFeatWeightsDictionary=bivariateDictionary)

    hmc = HMC(nLeapFrogSteps, stepSize, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = np.random.uniform(0, 1, int(nStates + nStates * (nStates - 1) / 2))
    lastSample = sample
    exchangeableParamSamples = np.zeros((nMCMCIter, int(nStates + nStates * (nStates - 1) / 2)))
    binaryWeightsSamples = np.zeros((nMCMCIter, int(nStates + nStates * (nStates - 1) / 2)))
    stationaryDistEstSamples = np.zeros((nMCMCIter, nStates))
    for i in range(nMCMCIter):

        for k in range(1000):
            hmcResult = hmc.doIter(nLeapFrogSteps, stepSize, lastSample, expectedCompleteReversibleObjective,
                               expectedCompleteReversibleObjective, True)
            lastSample = hmcResult.next_q

        sample = lastSample
        initialStationaryWeights = sample[nStates]
        initialBinaryWeights = sample[nStates:(nStates + int(nStates * (nStates - 1) / 2))]


        newRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialStationaryWeights,
                                                                           initialBinaryWeights,
                                                                           bivariateFeatIndexDictionary=bivariateDictionary)
        exchangeableParam = newRateMtx.getExchangeCoef()
        stationaryDistEst = newRateMtx.getStationaryDist()
        exchangeableParamSamples[i, :] = exchangeableParam
        stationaryDistEstSamples[i, :] = stationaryDistEst
        binaryWeightsSamples[i, :] = initialBinaryWeights
        print(exchangeableParam)
        print(i)
    result = {}
    result['exchangeableCoef'] = exchangeableParamSamples
    result['stationaryDist'] = stationaryDistEstSamples
    result['binaryWeights'] = binaryWeightsSamples
    return result




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
    hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = prng.uniform(0, 1, nStates)
    samples = hmc.run(0, 5000, sample)
    avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    stationaryDistEst = np.exp(avgWeights)/np.sum(np.exp(avgWeights))
    print(weights)
    print(avgWeights)
    print(stationaryDist)
    print(stationaryDistEst)


def main():
    ## test the 4-by-4 rate matrix settings
    #####################################################################################################################################################
    ## Generate data for a high dimensional context of a rate matrix larger than 2-by-2, we use a 4-by-4 rate matrix to generate the data and compare the
    ## posterior samples
    #bigData = obtainSufficientStatisticsForChainGraphRateMtx(4)
    bigData = {}
    bigData['stationaryWeights'] = np.array((0.61879477,  0.59162363,  0.88868359,  0.8916548 ))
    bigData['sojourn'] = np.array((108.74163598,  105.89295418,  142.50513143,  142.8602784))
    bigData['bivariateWeights'] = np.array((0.74949417, -0.11432167,  1.015517  ,  1.08516247,  0.79822632, -0.38472578))
    bigData['stationaryDist'] = np.array((0.21754596,  0.21171457,  0.28494579,  0.28579368))
    bigData['exchangeCoef'] = np.array((2.1159294498305594, 1.8873476920348198, 2.4625449140713962, 8.1717204849059897, 6.5757510365330916, 1.512101713646661))
    bigData['nInit'] = np.array(( 21.7718,  21.1846,  28.4604,  28.5832))
    bigData['transitCount'] = np.zeros((4, 4))
    bigData['transitCount'][0,:] = np.array(( 0.    ,   48.8434,   58.3048,   76.444 ))
    bigData['transitCount'][1, :]= np.array((  48.6902,    0.    ,  246.9   ,  199.029 ))
    bigData['transitCount'][2, :]= np.array((   58.5   ,  246.6048,    0.    ,   61.7518))
    bigData['transitCount'][3, :]= np.array((76.4762,  199.1218,   61.6802,    0.  ))
    nStates=4
    bigData['bivariateDictionary'] = getHardCodedDictChainGraph(nStates)
    result = testLocalBPSForStationaryAndBivariateWeights(bigData, nMCMCIter=3000)
    np.savetxt(
        "/Users/crystal/Desktop/binaryWeightsRunningtestLocalBPSForStationaryAndBivariateWeightsForBinaryWeightsSamplers4states.csv",
        result['binaryWeights'], fmt='%.3f', delimiter=',')
    np.savetxt(
        "/Users/crystal/Desktop/binaryWeightsRunningtestLocalBPSForStationaryAndBivariateWeightsForExchangeableCoef4states.csv",
        result['exchangeableCoef'], fmt='%.3f', delimiter=',')
    ## assignValues to bigData2 like data so that next time, we don't need to run it again






    # TestForwardAndEndPointSamplers.test()
    # testHMCForStationaryAndBivariateWeights()
    # data = obtainSufficientStatisticsForOneRateMtx()

    ############################################################
    ## this part is the data generation part, once it has been run and we summarized the sufficient statistics, we don't need to
    ## rerun it so that we comment the following code
    # nStates = 2
    # prng = RandomState(1234567890)
    # stationaryWeights = prng.uniform(0, 1, nStates)
    # bivariateWeights = prng.normal(0, 1, int((nStates) * (nStates - 1) / 2))
    # testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights,
    #                                                                        bivariateWeights, getHardCodedDictChainGraph(2))
    # stationaryDist = testRateMtx.getStationaryDist()
    # rateMatrix = testRateMtx.getRateMtx()
    data = {}
    data['bivariateWeights'] = 1.49898834
    data['exchangeableCoef'] = 4.47715743666
    data['nInit'] = np.array((50.7113, 49.2887))
    data['sojourn'] = np.array((152.02389579, 147.97610421))
    data['stationaryDist'] = np.array((np.exp(0.61879477)/(np.exp(0.61879477)+np.exp(0.59162363)), np.exp(0.61879477)/(np.exp(0.61879477)+np.exp(0.59162363))))
    data['stationaryWeights'] = np.array((0.61879477, 0.59162363))
    data['transitCount'] = np.zeros((2, 2))
    data['transitCount'][0, 0 ] = data['transitCount'][1, 1] =0
    data['transitCount'][0, 1] = 335.5429
    data['transitCount'][1, 0] = 335.5562
    data['bivariateDictionary'] = getHardCodedDictChainGraph(2)

    nMCMCIter = 1000
    result2 = testLocalBPSForStationaryAndBivariateWeights(data=data, nMCMCIter = 1000)
    # np.savetxt(trueStationaryDistFileName, self.dataGenerationRegime.stationaryDist, fmt='%.3f', delimiter=',')
    np.savetxt(
        "/Users/crystal/Desktop/binaryWeightsRunningtestLocalBPSForStationaryAndBivariateWeightsForBinaryWeightsSamplers.csv",
        result2['binaryWeights'], fmt='%.3f', delimiter=',')
    np.savetxt(
        "/Users/crystal/Desktop/binaryWeightsRunningtestLocalBPSForStationaryAndBivariateWeightsForExchangeableCoef.csv",
        result2['exchangeableCoef'], fmt='%.3f', delimiter=',')
    print(np.arange(0, int(nMCMCIter/2), 1))
    print(result2['exchangeableCoef'][int(nMCMCIter/2):nMCMCIter])
    print(result2['exchangeableCoef'][int(nMCMCIter/2):nMCMCIter].shape[0])

    plt.plot(np.arange(0, int(nMCMCIter/2), 1), result2['exchangeableCoef'][int(nMCMCIter/2):nMCMCIter])
    plt.show()
    plt.hist(result2['exchangeableCoef'])


    #data['rateMatrix'] = rateMatrix
    #result1 = testHMCForStationaryAndBivariateWeights(data)
    #plt.plot(np.arange(0, nMCMCIter, 1), result1['exchangeableCoef'][0:nMCMCIter])
    #plt.show()
    #plt.hist(result1['exchangeableCoef'])



    ## TODO: check the two histogram of the two algorithms are similar or not






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