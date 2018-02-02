import os
import sys


from scipy import linalg
sys.path.append("/Users/crystal/Dropbox/rejfree/main/rejfreePy/")
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
import numpy as np
from main.ReversibleRateMtxPiAndExchangeGTR import ReversibleRateMtxPiAndExchangeGTR
from main.FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from main.OptionClasses import MCMCOptions
from main.OptionClasses import RFSamplerOptions
from main.Utils import summarizeSuffStatUsingEndPoint
from main.ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from main.ExpectedCompleteReversibleModel import ExpectedCompleteReversibleModel
from main.HMC import HMC
from main.LocalRFSamplerForOnlyExchangeCoefParam import LocalRFSamplerForOnlyExchangeCoefParam
from main.PhyloLocalRFMove import PhyloLocalRFMove
from main.UtilsForCheckingGeneration import getEmpiricalInitialStatesCount
from main.UtilsForCheckingGeneration import getTransitionCounts
from main.UtilsForCheckingGeneration import getSojournTime

nStates = 4
nExchange = int(nStates*(nStates-1)/2)
## generate the exchangeable coefficients
## set the seed so that we can reproduce generating the
seed = 123
np.random.seed(seed)
exchangeCoef = np.random.gamma(1, 1, nExchange)
print(exchangeCoef)
# [ 1.19227214  0.33706748  0.25728408  0.80143368  1.27107094  0.55009754]

rfOptions = RFSamplerOptions()
mcmcOptions = MCMCOptions(10000,10,0)


## generate the stationary distribution
np.random.seed(456)
weights = np.random.normal(0, 1, nStates)
print(weights)
# [-1.0856306   0.99734545  0.2829785  -1.50629471]
## get the true rate matrix
testRateMtx = ReversibleRateMtxPiAndExchangeGTR(nStates, weights, exchangeCoef)
stationaryDist = testRateMtx.getStationaryDist()
print(stationaryDist)
#[ 0.07344939  0.58967563  0.28864734  0.04822765]
rateMtx = testRateMtx.getRateMtx()
print(rateMtx)
#[[-0.81275566  0.70305383  0.09729363  0.01240821]
# [ 0.08757166 -0.38020412  0.2313317   0.06130076]
# [ 0.0247574   0.47258591 -0.52387322  0.02652991]
# [ 0.01889736  0.74951955  0.15878419 -0.9272011 ]]


## generate data sequences of a CTMC with an un-normalized rate matrix
bt = 5.0
nSeq = 5000
seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seed, rateMtx, stationaryDist, bt)

## check the empirical frequency of the generated sequences to see if it matches the stationary distribution
###################################################################################
empiricalInitialStatesCount = getEmpiricalInitialStatesCount(seqList)
print(empiricalInitialStatesCount)
theoreticalCount = stationaryDist * 5000
print(theoreticalCount)   ## the counts are very close, so that the generation for the initial counts are right
###################################################################################
## check the correctness of the transition counts of the generated sequences
transitionCount = getTransitionCounts(seqList)
## check the estimates of exchangeable coefficients
sojournTime = getSojournTime(seqList)
theta01 = (transitionCount[1, 3] + transitionCount[3, 1])/(stationaryDist[1] * sojournTime[3] + stationaryDist[3]* sojournTime[1])
print(theta01)
print(exchangeCoef[4])
# the correctness of the generating process is correct
#################################################################################

## initial guess of the parameters
newSeed = 234
np.random.seed(234)
initialWeights = np.random.normal(0, 1, nStates)
print(initialWeights)
print(exchangeCoef)

initialRateMtx = ReversibleRateMtxPiAndExchangeGTR(nStates, initialWeights, exchangeCoef)
initialStationaryDist = initialRateMtx.getStationaryDist()
# array([ 0.3460345 ,  0.05374208,  0.21672897,  0.38349445])
initialRateMtx = initialRateMtx.getRateMtx()

## obtain the sufficient statistics based on the current values of the parameters and perform MCMC sampling scheme
nMCMCIters = mcmcOptions.nMCMCSweeps
thinningPeriod = MCMCOptions().thinningPeriod
burnIn = MCMCOptions().burnIn

stationarySamples = np.zeros((nMCMCIters, nStates))
exchangeCoefSamples = np.zeros((nMCMCIters, nExchange))

# to debug code, set nMCMCIters=1 temporarily
nMCMCIters= 500

for i in range(nMCMCIters):

    # save the samples of the parameters
    stationarySamples[i, :] = initialStationaryDist
    #exchangeCoefSamples[i, :] = initialExchangeCoef

    # use endpointSampler to collect sufficient statistics of the ctmc given the current values of the parameters
    suffStat = summarizeSuffStatUsingEndPoint(seqList, bt, initialRateMtx)

    # get each sufficient statistics element
    nInit = suffStat['nInit']
    holdTime = suffStat['holdTimes']
    nTrans = suffStat['nTrans']

    # construct expected complete reversible model objective
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, exchangeCoef)

    # sample stationary distribution elements using HMC
    hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = np.random.normal(0, 1, nStates)
    samples = hmc.run(0, 2000, sample)
    avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    initialWeights = avgWeights
    stationaryDistEst = np.exp(avgWeights) / np.sum(np.exp(avgWeights))
    # update stationary distribution elements to the latest value
    initialStationaryDist = stationaryDistEst

    # sample exchangeable coefficients using local bouncy particle sampler
    ## define the model
    #model = ExpectedCompleteReversibleModel(expectedCompleteReversibleObjective, nStates, True, None, initialExchangeCoef, stationaryDistEst)
    ## define the sampler to use
    ## local sampler to use
    #allFactors = model.localFactors
    #localSampler = LocalRFSamplerForOnlyExchangeCoefParam(model, rfOptions, mcmcOptions, allFactors, nStates)
    #phyloLocalRFMove = PhyloLocalRFMove(model, localSampler, initialExchangeCoef)
    #initialExchangeCoef = phyloLocalRFMove.execute()
    initialRateMtx = ReversibleRateMtxPiAndExchangeGTR(nStates, initialWeights, exchangeCoef)
    initialRateMtx = initialRateMtx.getRateMtx()
    #print(initialRateMtx)
    print(initialStationaryDist)
    print(i)












