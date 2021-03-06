
import os
import sys
#sys.path.append("/Users/crystal/Dropbox/rejfreePy_main/")
#os.chdir("/Users/crystal/Dropbox/rejfreePy_main/")
import numpy as np

from FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from OptionClasses import MCMCOptions
from OptionClasses import RFSamplerOptions
from Utils import summarizeSuffStatUsingEndPoint
from ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
from HMC import HMC
from LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
from PhyloLocalRFMove import PhyloLocalRFMove
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from HardCodedDictionaryUtils import getHardCodedDict
from FullTrajectorGeneration import getObsArrayAtSameGivenTimes
from FullTrajectorGeneration import  endPointSamplerSummarizeStatisticsOneBt
from collections import OrderedDict
from datetime import datetime
from numpy.random import RandomState
import argparse
import DataGenerationRegime
import MCMCRunningRegime
import OptionClasses
import pickle
import cProfile


argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-nMCMCIter', action='store', dest='nMCMCIter', type=int, default =2000, help = 'store the number of MCMC iterations')
parser.add_argument('-dir_name', action='store', dest='dir_name', type=str, help='store the directory name to save the csv files')
parser.add_argument('-refreshmentMethod', action='store', dest='refreshmentMethod', default= "LOCAL", type=OptionClasses.RefreshmentMethod.from_string, choices=list(OptionClasses.RefreshmentMethod))
parser.add_argument('-trajectoryLength', action="store", dest='trajectoryLength', default = 0.125, help='save the trajectory length of the local bps sampler', type=float)
parser.add_argument('--provideSeq', action="store_true", dest='provideSeq', help='tell the program if the sequences have been generated')
parser.add_argument('-bt', action='store', dest='bt', type=float, default=5.0, help='store the branch length, in other words, the total length of the time series')
## store the total number of generated time series sequences
parser.add_argument('-nSeq', action='store', dest='nSeq', type=int, default= 5000, help='store the number of sequences of the time series')
## store the time interval between two observation points
parser.add_argument('-interLength', action='store', dest='interLength', type=float, default=1.0, help='store the interval length of two observation points in the time series')


results = parser.parse_args()
nMCMCIters = results.nMCMCIter
dir_name = results.dir_name
refreshmentMethod = results.refreshmentMethod
trajectoryLength = results.trajectoryLength
provideSeq = results.provideSeq
nSeq = results.nSeq
bt = results.bt
interLength = results.interLength

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

print(np.round(np.exp(stationaryWeights)/np.sum(np.exp(stationaryWeights)), 3))

# The true stationary distribution is
# [ 0.17749417  0.25453257  0.0698043   0.2830704   0.09905702  0.11604154]

bivariateFeatIndexDictionary = getHardCodedDict()
print(bivariateFeatIndexDictionary)

print("The true binary weights are:")
print(bivariateWeights)
## [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763 -2.3015387
##  1.74481176 -0.7612069   0.3190391  -0.24937038, 1.46210794, -2.06014071])



rfOptions = RFSamplerOptions(trajectoryLength=0.125)

mcmcOptions = MCMCOptions(nMCMCIters,1,0)

## create the rate matrix based on the sparse graphical structure
testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights, bivariateWeights, bivariateFeatIndexDictionary)
stationaryDist = testRateMtx.getStationaryDist()
print("The true stationary distribution is")
print(stationaryDist)
#The true stationary distribution is
#[ 0.17749417  0.25453257  0.0698043   0.2830704   0.09905702  0.11604154]

rateMtx = testRateMtx.getRateMtx()
print("The true rate matrix is ")
print(rateMtx)
# [[-1.29524102  0.7006565   0.11330835  0.0905378   0.29644723  0.09429114]
#  [ 0.48859148 -0.83782491  0.00567798  0.00969091  0.02356033  0.3103042 ]
#  [ 0.28811364  0.02070405 -2.32786621  1.0414191   0.06365818  0.91397125]
#  [ 0.05677009  0.00871392  0.25681079 -0.66691567  0.33309682  0.01152406]
#  [ 0.53118555  0.0605396   0.04485916  0.95187449 -1.65226915  0.06381036]
#  [ 0.14422532  0.68064009  0.54979555  0.02811166  0.0544707  -1.45724331]]




print("The true exchangeable parameters are ")
trueExchangeCoef = testRateMtx.getExchangeCoef()
print(trueExchangeCoef)

# The true exchangeable parameters are
# [2.7527184481647362, 1.6232287117730273, 0.31984199654874601, 2.9926928816680904, 0.8125636797020549, 0.081341438819008474, 0.034234981080808163,
# 0.23784619170668767, 2.6740785756861607, 3.679010975414601, 0.64264178800599125, 7.8762417929474466, 3.3626776006407701, 0.099309793742461433, 0.54989233610573274]

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


mcmcRegimeIteratively = MCMCRunningRegime.MCMCRunningRegime(dataRegime, nMCMCIter=nMCMCIters, thinning=1.0, burnIn=0, onlyHMC= False, HMCPlusBPS=True,
                                          nLeapFrogSteps=40, stepSize=0.02,  saveRateMtx=False, initialSampleSeed=3,
                                                            rfOptions=OptionClasses.RFSamplerOptions(
                                                                trajectoryLength=trajectoryLength,
                                                                refreshmentMethod=refreshmentMethod), dumpResultIteratively=True,dumpResultIterations=10, dir_name=dir_name)


mcmcRegimeIteratively.run()


# input = summarizeSuffStatUsingEndPoint(seqList, bt, rateMtx)

## initial guess of the parameters
# newSeed = 3
# np.random.seed(newSeed)
# initialWeights = np.random.normal(0, 1, nStates)
# print("The weights for the initial stationary distirbution are")
# print(initialWeights)
# # [ 1.78862847  0.43650985  0.09649747 -1.8634927  -0.2773882  -0.35475898]
#
#
# initialBinaryWeights = np.random.normal(0, 1, nBivariateFeat)
# print("The initial binary feature weights at 0th iteration are: ")
# print(initialBinaryWeights)
#
# ### get the rate matrix with initialWeights for the stationary distribution combined with the true binary weights
# #RateMtxWithFixedCoef = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, bivariateWeights, bivariateFeatIndexDictionary)
# #initialRateMtx = RateMtxWithFixedCoef
# initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, initialBinaryWeights, bivariateFeatIndexDictionary)
# initialStationaryDist = initialRateMtx.getStationaryDist()
# print("The initial stationary distribution is ")
# print(initialStationaryDist)
#
# initialRateMatrix = initialRateMtx.getRateMtx()
# print("The initial exchangeable parameters at 0th iteration are")
# initialExchangeCoef = initialRateMtx.getExchangeCoef()
# print(initialExchangeCoef)
#
#
# print("The initialRateMtx is")
# print(initialRateMatrix)
#
# ## obtain the sufficient statistics based on the current values of the parameters and perform MCMC sampling scheme
# nMCMCIters = mcmcOptions.nMCMCSweeps
# thinningPeriod = MCMCOptions().thinningPeriod
# burnIn = MCMCOptions().burnIn
#
# stationarySamples = np.zeros((nMCMCIters, nStates))
# stationaryWeightsSamples = np.zeros((nMCMCIters, nStates))
# binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
# exchangeableSamples = np.zeros((nMCMCIters, len(initialExchangeCoef)))
#
# # to debug code, set nMCMCIters=1 temporarily
# nMCMCIters= nMCMCIters
#
# nPairSeq = int(len(observedTimePoints)-1)
#
# #After 130th iteration
# #The initial estimates of the binary weights are:
# #[ 1.47329104 -0.49679277 -0.51178121 -0.4536991   0.37707951 -1.75517446
# #  1.50458647 -0.69647191  0.32232739 -0.1699001   1.16512585 -1.68608641]
# #The initial stationary distribution is
# #[ 0.172  0.253  0.074  0.278  0.103  0.12 ]
# #The initial estimates of the exchangeable parameters are:
# #[ 2.655  1.592  0.365  2.616  0.926  0.16   0.11   0.252  2.244  3.097
# #  0.688  6.215  2.705  0.156  0.594]
# #[[-1.272  0.672  0.117  0.101  0.27   0.112]
# # [ 0.457 -0.796  0.012  0.03   0.026  0.27 ]
# # [ 0.274  0.041 -1.994  0.86   0.071  0.749]
# # [ 0.063  0.028  0.228 -0.617  0.279  0.019]
# # [ 0.45   0.064  0.051  0.751 -1.387  0.072]
# # [ 0.159  0.568  0.458  0.043  0.061 -1.289]]
#
# # [ 0.172  0.253  0.074  0.278  0.103  0.12 ]
#
#
#
# ## create a three dimensional array to save the rate matrix elements
# rateMatrixSamples = np.zeros((nMCMCIters, nStates, nStates))
#
#
# startTime = datetime.now()
# print(startTime)
#
# for i in range(nMCMCIters):
#
#     # save the samples of the parameters
#     stationarySamples[i, :] = initialStationaryDist
#     binaryWeightsSamples[i, :] = initialBinaryWeights
#     exchangeableSamples[i, :] = initialExchangeCoef
#     #rateMatrixSamples[i, :, :] = initialRateMatrix
#     stationaryWeightsSamples[i, :] = initialWeights
#
#     # use endpointSampler to collect sufficient statistics of the ctmc given the current values of the parameters
#     # suffStat = summarizeSuffStatUsingEndPoint(seqList, bt, initialRateMatrix)
#     #
#     # # get each sufficient statistics element
#     # nInit = suffStat['nInit']
#     # holdTime = suffStat['holdTimes']
#     # nTrans = suffStat['nTrans']
#
#     nInit = np.zeros(nStates)
#     holdTime = np.zeros(nStates)
#     nTrans = np.zeros((nStates, nStates))
#
#     for j in range(nPairSeq):
#         suffStat = endPointSamplerSummarizeStatisticsOneBt(True, RandomState(int(i * nPairSeq+j)), initialRateMatrix, firstLastStatesArrayAll[j], 1.0)
#         nInit = nInit + suffStat['nInit']
#         holdTime = holdTime + suffStat['holdTimes']
#         nTrans = nTrans + suffStat['nTrans']
#
#
#     # construct expected complete reversible model objective
#     expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, initialExchangeCoef)
#
#     # sample stationary distribution elements using HMC
#     hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
#     sample = np.random.uniform(0, 1, nStates)
#     samples = hmc.run(0, 2000, sample)
#     avgWeights = np.sum(samples, axis=0) / samples.shape[0]
#     initialWeights = avgWeights
#     stationaryDistEst = np.exp(avgWeights) / np.sum(np.exp(avgWeights))
#     # update stationary distribution elements to the latest value
#     initialStationaryDist = stationaryDistEst
#
#     # sample exchangeable coefficients using local bouncy particle sampler
#     ## define the model
#     model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates, initialBinaryWeights, initialStationaryDist, bivariateFeatIndexDictionary)
#
#     ## define the sampler to use
#     ## local sampler to use
#     #allFactors = model.localFactors
#     localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates, bivariateFeatIndexDictionary)
#     ####### below is the older version of the sampler
#     phyloLocalRFMove = PhyloLocalRFMove(model, localSampler, initialBinaryWeights, options=rfOptions, prng=RandomState(i))
#     initialBinaryWeights = phyloLocalRFMove.execute()
#     #print("The initial estimates of the binary weights are:")
#     #print(initialBinaryWeights)
#
#     #localSamplerOld = LocalRFSamplerForBinaryWeightsOldVersion(model, rfOptions, mcmcOptions, nStates,
#     #                                                          bivariateFeatIndexDictionary)
#     #phyloLocalRFMove = PhyloLocalRFMove(seed, model, localSamplerOld, initialBinaryWeights)
#     #initialBinaryWeightsOld = phyloLocalRFMove.execute()
#
#     initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights, initialBinaryWeights,
#                                                               bivariateFeatIndexDictionary)
#
#     initialStationaryDist = initialRateMtx.getStationaryDist()
#     initialRateMatrix = initialRateMtx.getRateMtx()
#     initialExchangeCoef = initialRateMtx.getExchangeCoef()
#     print(i)
#     print(initialExchangeCoef)
#
# endTime = datetime.now()
# timeElapsed = 'Duration: {}'.format(endTime - startTime)
# print("The elapsed time interval is ")
# print(timeElapsed)
# # 21:10:49.600034
#
# download_dir = "timeElapsed.csv" #where you want the file to be downloaded to
# csv = open(download_dir, "w")
# #"w" indicates that you're writing strings to the file
# columnTitleRow = "elapsedTime\n"
# csv.write(columnTitleRow)
# row = timeElapsed
# csv.write(str(row))
# csv.close()
#
#
# stationaryDistName = "stationaryDistlbps" + str(nMCMCIters) + ".csv"
# stationaryWeightsName = "stationaryWeightlbps" + str(nMCMCIters) +".csv"
# exchangeableCoefName = "exchangeableParameterslbps" + str(nMCMCIters) +".csv"
# binaryWeightsName = "binaryWeightslbps" + str(nMCMCIters) +".csv"
# np.savetxt(stationaryDistName, stationarySamples, fmt='%.3f', delimiter=',')
# np.savetxt(stationaryWeightsName, stationaryWeightsSamples, fmt='%.3f', delimiter=',')
# np.savetxt(exchangeableCoefName, exchangeableSamples, fmt='%.3f', delimiter=',')
# np.savetxt(binaryWeightsName, binaryWeightsSamples, fmt='%.3f', delimiter=',')
# #np.save('3dsavelbps2000.npy', rateMatrixSamples)
#








