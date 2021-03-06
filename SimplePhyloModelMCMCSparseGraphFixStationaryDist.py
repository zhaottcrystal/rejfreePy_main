#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:00:22 2017

@author: crystal
"""

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
mcmcOptions = MCMCOptions(10000,10,0)

## create the rate matrix based on the sparse graphical structure
testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights, bivariateWeights, bivariateFeatIndexDictionary)
stationaryDist = testRateMtx.getStationaryDist()
print("The true stationary distribution is")
print(stationaryDist)
# [ 0.3460345   0.05374208  0.21672897  0.38349445]
rateMtx = testRateMtx.getRateMtx()
print("The true rate matrix is ")
print(rateMtx)
#[[-0.78028091  0.01019415  0.63967359  0.13041317]
# [ 0.06563807 -0.27952517  0.03614709  0.17774   ]
# [ 1.02131772  0.00896336 -2.46060442  1.43032334]
# [ 0.11767434  0.0249081   0.80833633 -0.95091876]]


print("The true exchangeable parameters are ")
trueExchangeCoef = testRateMtx.getExchangeCoef()
print(trueExchangeCoef)

## generate data sequences of a CTMC with an un-normalized rate matrix
bt = 5.0
nSeq = 5000
seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seed, rateMtx, stationaryDist, bt)

## initial guess of the parameters
newSeed = 456
np.random.seed(456)
initialWeights = stationaryWeights
print(initialWeights)

## this is the weight at the 0th iteration
#initialBinaryWeights = np.array((0.586, 0.876, -0.1884, 0.8487, 0.5998, -1.35819,
#                                 -1.521, -0.6138, -0.58865, 0.19012))
#print("The initial binary feature weights at 0th iteration are: ")


# This is the weight after 250th iteration
# initialBinaryWeights = np.array((-0.26843204,  0.36688318, -0.97301064, 0.91227563,  1.20215414, -1.69677469,-0.95141154, -0.59620609, 0.8609998,  0.54728876))
# this is the weight after another 300 iteration after the 250 iterations above
initialBinaryWeights =  np.array((-0.78780265, 0.09349204, 1.33947872,  0.32352438,  0.12542144, -1.68445323,
 -2.53439988, 0.56992156, -0.10013338,  0.31902502))
print("The initial binary feature weights at 250th iteration are: ")
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
binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
exchangeableSamples = np.zeros((nMCMCIters, len(initialExchangeCoef)))
# to debug code, set nMCMCIters=1 temporarily
nMCMCIters= 300


#for i in range(nMCMCIters):
for i in np.arange(300, 500):

    # save the samples of the parameters
    # stationarySamples[i, :] = initialStationaryDist
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
    #hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    #sample = np.random.uniform(0, 1, nStates)
    #samples = hmc.run(0, 2000, sample)
    #avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    #initialWeights = avgWeights
    #stationaryDistEst = np.exp(avgWeights) / np.sum(np.exp(avgWeights))
    # update stationary distribution elements to the latest value
    #initialStationaryDist = stationaryDistEst

    # sample exchangeable coefficients using local bouncy particle sampler
    ## define the model
    model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates, initialBinaryWeights, stationaryDist , bivariateFeatIndexDictionary)

    ## define the sampler to use
    ## local sampler to use
    allFactors = model.localFactors
    localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates, bivariateFeatIndexDictionary)
    phyloLocalRFMove = PhyloLocalRFMove(model, localSampler, initialBinaryWeights)
    initialBinaryWeights = phyloLocalRFMove.execute()
    print("The initial estimates of the binary weights are:")
    print(initialBinaryWeights)

    initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights, initialBinaryWeights,
                                                              bivariateFeatIndexDictionary)

    #initialStationaryDist = initialRateMtx.getStationaryDist()
    initialRateMatrix = initialRateMtx.getRateMtx()
    initialExchangeCoef = initialRateMtx.getExchangeCoef()
    print("The initial estimates of the exchangeable parameters are:")
    print(initialExchangeCoef)
    print(initialRateMatrix)
    #print(initialStationaryDist)
    print(i)












