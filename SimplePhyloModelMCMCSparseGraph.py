import os
import sys

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
from ReversibleRateMtxPiAndExchangeGTR import ReversibleRateMtxPiAndExchangeGTR
from FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from OptionClasses import MCMCOptions
from OptionClasses import RFSamplerOptions
from Utils import summarizeSuffStatUsingEndPoint
from ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
from HMC import HMC
from LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
from PhyloLocalRFMove import PhyloLocalRFMove
from Utils import generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure

nStates = 4
## generate the exchangeable coefficients
## set the seed so that we can reproduce generating the
seed = 234
np.random.seed(seed)
nBivariateFeat = 10
nChoosenFeatureRatio = 0.3

bivariateWeights = np.random.normal(0, 1, nBivariateFeat)

np.random.seed(seed)
stationaryWeights = np.random.normal(0, 1, nStates)

bivariateFeatIndexDictionary = generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat(nStates, nBivariateFeat, nChoosenFeatureRatio)
print(bivariateFeatIndexDictionary)

print("The true binary weights are:")
print(bivariateWeights)
#[ 0.81879162 -1.04355064  0.3509007   0.92157829 -0.08738186 -3.12888464
# -0.96973267  0.93466579  0.04386634  1.4252155 ]

rfOptions = RFSamplerOptions()
mcmcOptions = MCMCOptions(10000,1,0)

## create the rate matrix based on the sparse graphical structure
testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights, bivariateWeights, bivariateFeatIndexDictionary)
stationaryDist = testRateMtx.getStationaryDist()
print("The true stationary distribution is")
print(stationaryDist)

rateMtx = testRateMtx.getRateMtx()
print("The true rate matrix is ")
print(rateMtx)


print("The true exchangeable parameters are ")
trueExchangeCoef = testRateMtx.getExchangeCoef()
print(trueExchangeCoef)


## generate data sequences of a CTMC with an un-normalized rate matrix
bt = 5.0
nSeq = 5000
seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seed, rateMtx, stationaryDist, bt)

## initial guess of the parameters
newSeed = 456
np.random.seed(newSeed)
initialWeights = np.random.normal(0, 1, nStates)
print(initialWeights)
initialBinaryWeights = np.random.normal(0, 1, nBivariateFeat)
print("The initial binary feature weights at 0th iteration are: ")
print(initialBinaryWeights)

### get the rate matrix with initialWeights for the stationary distribution combined with the true binary weights
#RateMtxWithFixedCoef = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, bivariateWeights, bivariateFeatIndexDictionary)
#initialRateMtx = RateMtxWithFixedCoef
initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, initialBinaryWeights, bivariateFeatIndexDictionary)
initialStationaryDist = initialRateMtx.getStationaryDist()
print("The initial stationary distribution is ")
print(initialStationaryDist)

initialRateMatrix = initialRateMtx.getRateMtx()
print("The initial exchangeable parameters at 0th iteration are")
initialExchangeCoef = initialRateMtx.getExchangeCoef()
print(initialExchangeCoef)

## obtain the sufficient statistics based on the current values of the parameters and perform MCMC sampling scheme
nMCMCIters = mcmcOptions.nMCMCSweeps
thinningPeriod = MCMCOptions().thinningPeriod
burnIn = MCMCOptions().burnIn

stationarySamples = np.zeros((nMCMCIters, nStates))
stationaryWeightsSamples = np.zeros((nMCMCIters, nStates))
binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
exchangeableSamples = np.zeros((nMCMCIters, len(initialExchangeCoef)))

# to debug code, set nMCMCIters=1 temporarily
nMCMCIters= 200

for i in range(nMCMCIters):

    # save the samples of the parameters
    stationaryWeightsSamples = initialWeights
    stationarySamples[i, :] = initialStationaryDist
    binaryWeightsSamples[i, :] = initialBinaryWeights
    exchangeableSamples[i, :] = initialExchangeCoef

    # use endpointSampler to collect sufficient statistics of the ctmc given the current values of the parameters
    suffStat = summarizeSuffStatUsingEndPoint(seqList, bt, initialRateMatrix)

    # get each sufficient statistics element
    nInit = suffStat['nInit']
    holdTime = suffStat['holdTimes']
    nTrans = suffStat['nTrans']

    # construct expected complete reversible model objective
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, initialExchangeCoef)

    # sample stationary distribution elements using HMC
    hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = np.random.uniform(0, 1, nStates)
    samples = hmc.run(0, 2000, sample)
    avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    initialWeights = avgWeights
    stationaryDistEst = np.exp(avgWeights) / np.sum(np.exp(avgWeights))
    # update stationary distribution elements to the latest value
    initialStationaryDist = stationaryDistEst

    # sample exchangeable coefficients using local bouncy particle sampler
    ## define the model
    model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates, initialBinaryWeights, initialStationaryDist, bivariateFeatIndexDictionary)

    ## define the sampler to use
    ## local sampler to use
    allFactors = model.localFactors
    localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates, bivariateFeatIndexDictionary)
    phyloLocalRFMove = PhyloLocalRFMove(model, localSampler, initialBinaryWeights)
    initialBinaryWeights = phyloLocalRFMove.execute()
    print("The initial estimates of the binary weights are:")
    print(initialBinaryWeights)

    initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, initialBinaryWeights,
                                                              bivariateFeatIndexDictionary)

    initialStationaryDist = initialRateMtx.getStationaryDist()
    initialRateMatrix = initialRateMtx.getRateMtx()
    initialExchangeCoef = initialRateMtx.getExchangeCoef()
    print("The initial estimates of the exchangeable parameters are:")
    print(initialExchangeCoef)
    print(initialRateMatrix)
    print(initialStationaryDist)
    print(i)












