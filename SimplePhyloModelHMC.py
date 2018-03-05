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
from FullTrajectorGeneration import getObsArrayAtSameGivenTimes
from FullTrajectorGeneration import endPointSamplerSummarizeStatisticsOneBt
from HardCodedDictionaryUtils import getHardCodedDict
from datetime import datetime
from numpy.random import RandomState
import argparse
import OptionClasses
import MCMCRunningRegime
import DataGenerationRegime
import pickle


argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-nMCMCIter', action='store', type = int, default=2000, dest='nMCMCIter', help = 'store the number of MCMC iterations')
parser.add_argument('-dir_name', action='store', dest='dir_name', type=str, help='store the directory name to save the csv files')
parser.add_argument('--provideSeq', action="store_true", dest='provideSeq', help='tell the program if the sequences have been generated')
parser.add_argument('-bt', action='store', dest='bt', type=float, default=5.0, help='store the branch length, in other words, the total length of the time series')
## store the total number of generated time series sequences
parser.add_argument('-nSeq', action='store', dest='nSeq', type=int, default= 5000, help='store the number of sequences of the time series')
## store the time interval between two observation points
parser.add_argument('-interLength', action='store', dest='interLength', type=float, default=1.0, help='store the interval length of two observation points in the time series')

results = parser.parse_args()
nMCMCIters = results.nMCMCIter
dir_name = results.dir_name
bt=results.bt
nSeq = results.nSeq
interLength = results.interLength
provideSeq = results.provideSeq



nStates = 6
## generate the exchangeable coefficients
## set the seed so that we can reproduce generating the
seed = 1
np.random.seed(seed)
nBivariateFeat = 12
bivariateWeights = np.random.normal(0, 1, nBivariateFeat)

np.random.seed(2)
stationaryWeights = np.random.normal(0, 1, nStates)
print("The true stationary weights are")
# the 4th and fifth weights are so large and so small lead to very large and very small stationary distributions,
# so that we adjust the weight to have a more balanced stationary distributions
stationaryWeights[2] = -1.35
stationaryWeights[3] = 0.05
stationaryWeights[4] = -1.0
print(stationaryWeights)
## The true stationary weights are
## [-0.41675785 -0.05626683 -1.35        0.05       -1.         -0.84174737]

print("The true stationary distribution is")
print(np.exp(stationaryWeights)/np.sum(np.exp(stationaryWeights)))
# [ 0.17749417  0.25453257  0.0698043   0.2830704   0.09905702  0.11604154]

bivariateFeatIndexDictionary = getHardCodedDict()
print(bivariateFeatIndexDictionary)

print("The true binary weights are:")
print(bivariateWeights)
# [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763 -2.3015387
#  1.74481176 -0.7612069   0.3190391  -0.24937038  1.46210794 -2.06014071]



rfOptions = RFSamplerOptions()
mcmcOptions = MCMCOptions(nMCMCIters, 1, 0)

## create the rate matrix based on the sparse graphical structure
testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights, bivariateWeights,
                                                                        bivariateFeatIndexDictionary)
stationaryDist = testRateMtx.getStationaryDist()
print("The true stationary distribution is")
print(stationaryDist)
# [ 0.17749417  0.25453257  0.0698043   0.2830704   0.09905702  0.11604154]


rateMtx = testRateMtx.getRateMtx()
print("The true rate matrix is ")
print(rateMtx)
# The true rate matrix is
#[[-1.29524102  0.7006565   0.11330835  0.0905378   0.29644723  0.09429114]
# [ 0.48859148 -0.83782491  0.00567798  0.00969091  0.02356033  0.3103042 ]
# [ 0.28811364  0.02070405 -2.32786621  1.0414191   0.06365818  0.91397125]
# [ 0.05677009  0.00871392  0.25681079 -0.66691567  0.33309682  0.01152406]
# [ 0.53118555  0.0605396   0.04485916  0.95187449 -1.65226915  0.06381036]
# [ 0.14422532  0.68064009  0.54979555  0.02811166  0.0544707  -1.45724331]]

print("The true exchangeable parameters are ")
trueExchangeCoef = testRateMtx.getExchangeCoef()
print(trueExchangeCoef)
# [2.7527184481647362, 1.6232287117730273, 0.31984199654874601, 2.9926928816680904, 0.8125636797020549, 0.081341438819008474, 0.034234981080808163, 0.23784619170668767, 2.6740785756861607, 3.679010975414601, 0.64264178800599125, 7.8762417929474466, 3.3626776006407701, 0.099309793742461433, 0.54989233610573274]

## generate data sequences of a CTMC with an un-normalized rate matrix

if not provideSeq:
    ## Weight Generation
    prng = RandomState(seed)
    weightGenerationRegime = DataGenerationRegime.WeightGenerationRegime(nStates=nStates,
                                                                         nBivariateFeat=bivariateFeatIndexDictionary,
                                                                         stationaryWeights=stationaryWeights,
                                                                         bivariateWeights=bivariateWeights)
    dataRegime = DataGenerationRegime.DataGenerationRegime(nStates=nStates,
                                                           bivariateFeatIndexDictionary=bivariateFeatIndexDictionary,
                                                           btLength=bt, nSeq=nSeq,
                                                           weightGenerationRegime=weightGenerationRegime, prng=prng,
                                                           interLength=interLength)
    ## generate the sequences data
    initialStateSeq = dataRegime.generatingInitialStateSeq()
    seqList = dataRegime.generatingSeq(initialStateSeq)
    suffStat = dataRegime.getSufficientStatFromSeq(seqList)
    firstLastStatesArrayAll = dataRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)
    trueRateMtx = dataRegime.rateMtxObj.getRateMtx()

    ## try if using pickle library works
    ## serialize dataRegime so that when we compare HMC and BPS, they can have the same data frame
    dataFileDirName = "nStates" + str(nStates) + "seedGenData" + str(seed) + "bt" + str(bt) + "nSeq" + str(nSeq) + "interLength" + str(interLength)
    directory = os.path.join(dir_name, dataFileDirName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

    dataFileName = dataFileDirName + ".file"
    with open(dataFileName, "wb") as f:
        pickle.dump(dataRegime, f, pickle.HIGHEST_PROTOCOL)
    with open(dataFileName, "rb") as f:
        dataRegime = pickle.load(f)
else:
    dataFileDirName = "nStates" + str(nStates) + "seedGenData" + str(seed) + "bt" + str(bt) + "nSeq" + str(nSeq) + "interLength" + str(interLength)
    os.chdir(dir_name)
    directory = dataFileDirName
    if not os.path.exists(directory):
        raise ValueError("The directory of the provided sequences does not exist")
    else:
        os.chdir(os.path.join(dir_name, directory))

    dataFileName = dataFileDirName + ".file"
    with open(dataFileName, "rb") as f:
        dataRegime = pickle.load(f)



mcmcRegimeIteratively = MCMCRunningRegime.MCMCRunningRegime(dataRegime, nMCMCIter=nMCMCIters, thinning=1.0, burnIn=0, onlyHMC= True, HMCPlusBPS=False,
                                          nLeapFrogSteps=40, stepSize=0.02,  saveRateMtx=False, initialSampleSeed=3,
                                          rfOptions=OptionClasses.RFSamplerOptions(trajectoryLength=0.125), dumpResultIteratively=True,dumpResultIterations=10, dir_name=dir_name)


mcmcRegimeIteratively.run()



# seqList = generateFullPathUsingRateMtxAndStationaryDist(prng=prng, nSeq=nSeq, nstates=nStates, rateMtx=rateMtx, stationaryDist=stationaryDist, bt=bt)
# observedTimePoints = np.arange(0, (bt+1))
# observedSeqList = getObsArrayAtSameGivenTimes(seqList, observedTimePoints)
# observedAllSequences = observedSeqList[1:observedSeqList.shape[0], :]
#
#
#
# ## initial guess of the parameters
# newSeed = 3
# np.random.seed(newSeed)
# initialWeights = np.random.normal(0, 1, nStates)
# print("The weights for the initial stationary distirbution are")
# print(initialWeights)
# # [ 1.78862847  0.43650985  0.09649747 -1.8634927  -0.2773882  -0.35475898]
#
#
# ## this is the weight at the 0th iteration
# # initialBinaryWeights = np.array((0.586, 0.876, -0.1884, 0.8487, 0.5998, -1.35819,
# #                                 -1.521, -0.6138, -0.58865, 0.19012))
# # print("The initial binary feature weights at 0th iteration are: ")
#
#
# # This is the weight after 250th iteration
# # initialBinaryWeights = np.array((-0.26843204,  0.36688318, -0.97301064, 0.91227563,  1.20215414, -1.69677469,-0.95141154, -0.59620609, 0.8609998,  0.54728876))
# # this is the weight after another 300 iteration after the 250 iterations above
# initialBinaryWeights = np.random.normal(0, 1, nBivariateFeat)
# print("The initial binary feature weights at the 0th iteration are: ")
# print(initialBinaryWeights)
#
# ### get the rate matrix with initialWeights for the stationary distribution combined with the true binary weights
# # RateMtxWithFixedCoef = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, bivariateWeights, bivariateFeatIndexDictionary)
# # initialRateMtx = RateMtxWithFixedCoef
# initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights,
#                                                                            initialBinaryWeights,
#                                                                            bivariateFeatIndexDictionary)
# initialStationaryDist = initialRateMtx.getStationaryDist()
# print("The initial stationary distribution is ")
# print(initialStationaryDist)
#
#
# initialRateMatrix = initialRateMtx.getRateMtx()
# print("The initial exchangeable parameters at 0th iteration are")
# initialExchangeCoef = initialRateMtx.getExchangeCoef()
# print(initialExchangeCoef)
#
# ## obtain the sufficient statistics based on the current values of the parameters and perform MCMC sampling scheme
# thinningPeriod = MCMCOptions().thinningPeriod
# burnIn = MCMCOptions().burnIn
#
#
# weightSamples = np.zeros((nMCMCIters, (nStates+nBivariateFeat)))
# # to debug code, set nMCMCIters=1 temporarily
# avgWeights = np.zeros((nStates+nBivariateFeat))
# avgWeights[0:nStates] = initialWeights
# avgWeights[nStates: (nStates+nBivariateFeat)] = initialBinaryWeights
# weightSamples[0, :] =  avgWeights
# weightSamples[0, nStates:(nStates+nBivariateFeat)] = initialBinaryWeights
#
# stationarySamples = np.zeros((nMCMCIters, nStates))
# stationaryWeightsSamples = np.zeros((nMCMCIters, nStates))
# binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
# exchangeableSamples = np.zeros((nMCMCIters, len(initialExchangeCoef)))
#
# #stationarySamplesTmp = np.zeros((500, nStates))
# #stationaryWeightsSamplesTmp = np.zeros((500, nStates))
# #binaryWeightsSamplesTmp = np.zeros((500, nBivariateFeat))
# #exchangeableSamplesTmp = np.zeros((500, len(initialExchangeCoef)))
#
# #stationarySamplesTmp[0:200, :] = stationarySamples
# #stationaryWeightsSamplesTmp[0:200, :] = stationaryWeightsSamples
# #binaryWeightsSamplesTmp[0:200, :] = binaryWeightsSamples
# #exchangeableSamplesTmp[0:200, :] = exchangeableSamples
#
# #stationarySamples = stationarySamplesTmp
# #stationaryWeightsSamples = stationaryWeightsSamplesTmp
# #binaryWeightsSamples = binaryWeightsSamplesTmp
# #exchangeableSamples = exchangeableSamplesTmp
#
#
# firstLastStatesArrayAll = list()
# nPairSeq = int(len(observedTimePoints)-1)
#
# for i in range(nPairSeq):
#     pairSeq = observedAllSequences[:, i:(i+2)]
#     firstLastStatesArrayAll.append(pairSeq)
#
#
# rateMatrixSamples = np.zeros((nMCMCIters, nStates, nStates))
#
# startTime = datetime.now()
# print(startTime)
#
#
# for i in range(nMCMCIters):
#     # save the samples of the parameters
#     # stationarySamples[i, :] = initialStationaryDist
#     if i > 0:
#         weightSamples[i, :] = avgWeights
#
#     # save the samples of the parameters
#     stationarySamples[i, :] = initialStationaryDist
#     binaryWeightsSamples[i, :] = initialBinaryWeights
#     exchangeableSamples[i, :] = initialExchangeCoef
#     #rateMatrixSamples[i, :, :] = initialRateMatrix
#     stationaryWeightsSamples[i, :] = initialWeights
#
#     # use endpointSampler to collect sufficient statistics of the ctmc given the current values of the parameters
#     # summarize all observed sequences which is a two dimensional array, the number of rows represents the number of sequences
#     # the number of columns represents the number of finite number of time points at which the sequences are observed
#
#     nInit = np.zeros(nStates)
#     holdTime = np.zeros(nStates)
#     nTrans = np.zeros((nStates, nStates))
#
#     for j in range(nPairSeq):
#         suffStat =  endPointSamplerSummarizeStatisticsOneBt(True, RandomState(j), initialRateMatrix, firstLastStatesArrayAll[j], 1.0)
#         nInit = nInit + suffStat['nInit']
#         holdTime = holdTime + suffStat['holdTimes']
#         nTrans = nTrans + suffStat['nTrans']
#
#     # construct expected complete reversible model objective
#     expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, nBivariateFeatWeightsDictionary= bivariateFeatIndexDictionary)
#
#     # sample stationary distribution elements using HMC
#     hmc = HMC(40, 0.002, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
#     sample = np.random.uniform(0, 1, len(avgWeights))
#     samples = hmc.run(0, 2000, sample)
#     avgWeights = np.sum(samples, axis=0) / samples.shape[0]
#     stationaryDistEst = np.exp(avgWeights[0:nStates]) / np.sum(np.exp(avgWeights[0:nStates]))
#     initialStationaryWeights = avgWeights[0:nStates]
#     # update stationary distribution elements to the latest value
#     initialStationaryDist = stationaryDistEst
#
#     initialBinaryWeights = avgWeights[nStates:(nStates+nBivariateFeat)]
#
#
#     initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialStationaryWeights,
#                                                                                initialBinaryWeights,
#                                                                                bivariateFeatIndexDictionary)
#
#     initialStationaryDist = initialRateMtx.getStationaryDist()
#     #initialStationaryDist = np.round(initialStationaryDist, 3)
#     initialRateMatrix = initialRateMtx.getRateMtx()
#     initialExchangeCoef = initialRateMtx.getExchangeCoef()
#     #print("The initial estimates of the exchangeable parameters are:")
#     #print(initialExchangeCoef)
#     #print("The estimated stationary distribution is")
#     #print(stationaryDistEst)
#     #print("The estimated rate matrix is ")
#     #print(initialRateMatrix)
#     # print(initialStationaryDist)
#     print(i)
#
#
# endTime = datetime.now()
# timeElapsed = 'Duration: {}'.format(endTime - startTime)
# print("The elapsed time interval is ")
# print(timeElapsed)
#
# download_dir = "timeElapsedHMC" + str(nMCMCIters)+ ".csv" #where you want the file to be downloaded to
# csv = open(download_dir, "w")
# #"w" indicates that you're writing strings to the file
# columnTitleRow = "elapsedTime\n"
# csv.write(columnTitleRow)
# row = timeElapsed
# csv.write(str(row))
# csv.close()
#
#
# np.savetxt('stationaryDistributionHMC.csv', stationarySamples, fmt='%.3f', delimiter=',')
# np.savetxt('stationaryWeightHMC.csv', stationaryWeightsSamples, fmt='%.3f', delimiter=',')
# np.savetxt('exchangeableParametersHMC.csv', exchangeableSamples, fmt='%.3f', delimiter=',')
# np.savetxt('binaryWeightsHMC.csv', binaryWeightsSamples, fmt='%.3f', delimiter=',')
# #np.save('3dsaveHMC.npy', rateMatrixSamples)
#
#
#
#
#
#
#
#
#
#
#
#
