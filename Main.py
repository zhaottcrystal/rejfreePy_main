import sys
import os
# sys.path.append("/Users/crystal/PycharmProjects/rejfreePy_main")
# os.chdir("/Users/crystal/PycharmProjects/rejfreePy_main")
sys.path.append("/home/tingtingzhao/rejfreePy_main")
os.chdir("/home/tingtingzhao/rejfreePy_main")
from numpy.random import RandomState
import DataGenerationRegime
import MCMCRunningRegime
import HardCodedDictionaryUtils
import OptionClasses
import argparse
import FullTrajectorGeneration
import numpy as np
import pickle
import cProfile

## add command line argument
## list the arguments that we would like to provide to the code
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
## add the number of states as arguments
parser.add_argument('-nStates', action="store", type=int, dest='nStates', default=2, help='save the number of states in the ctmc')
## add boolean variable to indicate whether we only use hmc or we use a combination of hmc and local bps
parser.add_argument('--onlyHMC', action="store_true", help='HMC flag, the existence of the argument indicates HMC is used.')
## add boolean variable to indicate whether we use the local bps algorithm
parser.add_argument('--HMCPlusBPS', action='store_true', help='BPS flag, the existence of the argument indicates a combination of HMC and local BPS is used.')
## add standard output option
parser.add_argument('--logFile', action='store_true', help='flag to indicate the standard output will be saved to a file')
## add the trajectory length if we use local bps
parser.add_argument('-trajectoryLength', action="store", dest='trajectoryLength', default = 0.125, help='save the trajectory length of the local bps sampler', type=float)
## add indicator to indicate whether we will normalize the trajectory length in local bps
parser.add_argument('--normalizeTraj', action='store_true', help='Normalize trajectory length flag, the existence of the argument indicates the trajectory length will be normalized.')
## add the total number of mcmc iterations
parser.add_argument('-nMCMCIter', action="store", dest='nMCMCIter', default=2000, type=int, help='store the total number of posterior samples')
## add the burning period of the posterior samples
parser.add_argument('-burnIn', action='store', dest='burnIn', default=0, type=int, help='store the burnIn period of the posterior samples')
## store the total number of leapfrog steps in HMC
parser.add_argument('-nLeapFrogSteps', action='store', dest='nLeapFrogSteps', default=40, type= int , help='store the total number of leapfrog steps in HMC')
## store the leapfrog size of HMC
parser.add_argument('-stepSize', action='store', dest='stepSize', default=0.02, type=float, help='store the leapfrog step size in HMC.')
## store the number of HMC samples
parser.add_argument('-nItersPerPathAuxVar', action='store', dest='nItersPerPathAuxVar', default=500, type=int, help='store the number of HMC samples in HMC algorithm')
## add the boolean variable to indicate whether we store the result in the end or we write results to csv files
parser.add_argument('--dumpResultIteratively', action='store_true', help='flag indicating we write results to csv iteratively instead of in the end')
## the number of iterations we write results to disk
parser.add_argument('-dumpResultIterations', action='store', dest='dumpResultIterations', default=50, type=int, help='store the number of iteration interval that we write results to csv')
##store the directory that we would lie to save the result
parser.add_argument('-dir_name', action='store', dest='dir_name', type=str, help='store the directory name to save the csv files')
## store the seed to used to generate the data
parser.add_argument('-seedGenData', action='store', dest='seed', default = 1234567890, type=int, help='store the seed we use to generate the sequences')
## store the seed for Markov chain sampling
parser.add_argument('-samplingSeed', action='store', dest='initialSampleSeed', type=int, default=3, help='store the seed we use to do sampling')
## store the branch length of the time series of the generated data
parser.add_argument('-bt', action='store', dest='bt', type=float, default=5.0, help='store the branch length, in other words, the total length of the time series')
## store the total number of generated time series sequences
parser.add_argument('-nSeq', action='store', dest='nSeq', type=int, default= 5000, help='store the number of sequences of the time series')
## store the time interval between two observation points
parser.add_argument('-interLength', action='store', dest='interLength', type=float, default=1.0, help='store the interval length of two observation points in the time series')
## add the refreshment rate of lbps algorithm
parser.add_argument('-refreshmentRate', action='store', dest='interLength', type=float, default=1, help='store the refreshment rate for the lbps algorithm')
## add the method we use to generate the initial weights
parser.add_argument('-initialSamplesGenerateMethod', action='store', dest='initialSamplesMethod', default='Uniform', help='store the method used to generate the initial weights samples, the options include Uniform, Normal, Fixed, AssignedWeightsValues')
## add the initial univariate weights if we would like to provide initial weights
parser.add_argument('-univariateWeights', action='store', dest='uniWeights', help = 'store the univariate weights for the stationary distribution')
## add the initial bivariate weights if we would like to provide initial weights
parser.add_argument('-bivariateWeights', action='store', dest='biWeights', help = 'store the bivariate weights for the exchangeable parameters')
parser.add_argument('-refreshmentMethod', action='store', dest='refreshmentMethod', default= "LOCAL", type=OptionClasses.RefreshmentMethod.from_string, choices=list(OptionClasses.RefreshmentMethod))
parser.add_argument('--provideSeq', action="store_true", dest='provideSeq', help='tell the program if the sequences have been generated')
parser.add_argument('-batchSize', action='store', dest='batchSize', type=int, default=50, help='the batch size when updating the ergodic mean')
parser.add_argument('-bivariateDict', action='store', dest='bivariateDict', default = 'chain', choices=['customized10', 'customized6', 'chain'])
parser.add_argument('-bivariateFeatDist', action='store', dest='bivariateFeatDist', default='Normal', choices=['Normal', 'Unif'])
parser.add_argument('--unknownTrueRateMtx', action="store_true", help='unknown rate matrix flag, the existence of the argument indicates the rate matrix used to generate the data is unknown.')



results = parser.parse_args()
dir_name = results.dir_name
nSeq = results.nSeq
bt = results.bt
nStates = results.nStates
nMCMCIter = results.nMCMCIter
nLeapFrogSteps = results.nLeapFrogSteps
stepSize = results.stepSize
trajectoryLength = results.trajectoryLength
initialSampleSeed = results.initialSampleSeed
interLength = results.interLength

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logFileName = os.path.join(dir_name, 'log.txt')
if results.logFile:
    sys.stdout = open(logFileName, 'w')
###########################################
###### normalize the trajectory length
if results.normalizeTraj:
    trajectoryLength = 1/(nSeq * interLength * (bt/interLength + 1))
print(trajectoryLength)
##########################################

dumpResultIterations = results.dumpResultIterations
refreshmentMethod = results.refreshmentMethod
provideSeq = results.provideSeq
seedGenData = results.seed
nItersPerPathAuxVar = results.nItersPerPathAuxVar
batchSize = results.batchSize
bivariateDictStr = results.bivariateDict
bivariateFeatDist = results.bivariateFeatDist


if results.initialSamplesMethod is not None:
    initialWeightsDist = results.initialSamplesMethod
else:
    initialWeightsDist = 'Fixed'

bivariateFeatIndexDictionary = None
nBivariateFeat = None
if bivariateDictStr == 'customized10':
    bivariateFeatIndexDictionary = HardCodedDictionaryUtils.getHardCodedDict10States()
    nBivariateFeat = int(36)
elif bivariateDictStr == 'customized6':
    bivariateFeatIndexDictionary = HardCodedDictionaryUtils.getHardCodedDict()
    nBivariateFeat = 12
elif bivariateDictStr == 'chain':
    bivariateFeatIndexDictionary = HardCodedDictionaryUtils.getHardCodedDictChainGraph(nStates)
    nBivariateFeat = int(nStates * (nStates-1)/2)

####################################################
if not provideSeq:
    ## Weight Generation
    prng = RandomState(seedGenData)
    weightGenerationRegime = DataGenerationRegime.WeightGenerationRegime(nStates = nStates, nBivariateFeat= nBivariateFeat, prng=prng)
    weightGenerationRegime.generateStationaryWeightsFromUniform()
    if bivariateFeatDist == "Unif":
        weightGenerationRegime.generateBivariateWeightsFromUniform()
    else:
        weightGenerationRegime.generateBivariateWeightsFromNormal()

####################################################
    ## sequences data generation

    dataRegime = DataGenerationRegime.DataGenerationRegime(nStates=nStates,  bivariateFeatIndexDictionary=bivariateFeatIndexDictionary, btLength=bt, nSeq=nSeq, weightGenerationRegime=weightGenerationRegime, prng = prng, interLength=interLength)
    ## generate the sequences data
    initialStateSeq = dataRegime.generatingInitialStateSeq()
    seqList = dataRegime.generatingSeq(initialStateSeq)
    suffStat = dataRegime.getSufficientStatFromSeq(seqList)
    firstLastStatesArrayAll = dataRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)
    trueRateMtx = dataRegime.rateMtxObj.getRateMtx()
    print(trueRateMtx)

    ## try if using pickle library works
    ## serialize dataRegime so that when we compare HMC and BPS, they can have the same data frame
    dataFileDirName = "nStates" + str(nStates) + "seedGenData" + str(seedGenData) + "bt" + str(bt) + "nSeq" + str(nSeq) + "interLength" + str(interLength)
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
    dataFileDirName = "nStates" + str(nStates) + "seedGenData" + str(seedGenData) + "bt" + str(bt) + "nSeq" + str(nSeq) + "interLength" + str(interLength)
    os.chdir(dir_name)
    directory = dataFileDirName
    if not os.path.exists(directory):
        raise ValueError("The directory of the provided sequences does not exist")
    else:
        os.chdir(os.path.join(dir_name, directory))

    dataFileName = dataFileDirName + ".file"
    with open(dataFileName, "rb") as f:
        dataRegime = pickle.load(f)

####################################################
## validate the data generating process is correct via the sufficient statistics of the time series.
## get the rate matrix and see if the sufficient statistics is consistent with the rate matrix we use to generate the data
## get the initial count of each state
# sojournTime = suffStat['sojourn']
# transitCount = suffStat['transitCount']
# nInit = np.bincount(initialStateSeq)
# ii = np.nonzero(nInit)[0]
# nInitCount = np.vstack((ii, nInit[ii])).T
# empiricalStationary = nInit/nSeq
# ## compare it with stationary distribution
# print(empiricalStationary)
# print(dataRegime.stationaryDist)
# ## calculate the exchangeable coefficients
# theta = (transitCount[0, 1] + transitCount[1, 0])/(empiricalStationary[0]* sojournTime[1]+ empiricalStationary[1]*sojournTime[0])
# print(theta)

####################################################
# BPS algorithm
## run BPS plus HMC algorithm

#mcmcRegime = MCMCRunningRegime(dataRegime, nMCMCIter=nMCMCIter, thinning=1.0, burnIn=0, onlyHMC=True, HMCPlusBPS=False,  nLeapFrogSteps=40, stepSize=0.02, nHMCSamples=1000, saveRateMtx = False, initialSampleSeed=3, rfOptions=RFSamplerOptions(trajectoryLength=0.125))
#mcmcSamples = mcmcRegime.run()
# record Samples
#mcmcWriteToFile = mcmcRegime.recordResult(mcmcSamples, "/Users/crystal/Dropbox/try/", "wallTime", "HMC", nMCMCIter=str(nMCMCIter), saveRateMtx=False)

#mcmcWriteToFile = mcmcRegime.recordResult(mcmcSamples, "/Users/crystal/Dropbox/rejfree/rejfreePy/results/", "wallTime", "HMCPlusBPS", nMCMCIter=2000, saveRateMtx=False)

mcmcRegimeIteratively = MCMCRunningRegime.MCMCRunningRegime(dataRegime, nMCMCIter, thinning=1.0, burnIn=0, onlyHMC= results.onlyHMC, HMCPlusBPS=results.HMCPlusBPS,
                                          nLeapFrogSteps=nLeapFrogSteps, stepSize=stepSize, saveRateMtx=False, initialSampleSeed=initialSampleSeed,
                                          rfOptions=OptionClasses.RFSamplerOptions(trajectoryLength=trajectoryLength, refreshmentMethod=refreshmentMethod), dumpResultIteratively=True,
                                                            dumpResultIterations=dumpResultIterations, dir_name=dir_name, nItersPerPathAuxVar=nItersPerPathAuxVar, batchSize=batchSize)
if initialWeightsDist is not None:
    if initialWeightsDist == "AssignedWeightsValues":
        if results.uniWeights is not None and results.biWeights is not None:
            uniWeights = np.array(eval(results.uniWeights))
            biWeights = np.array(eval(results.biWeights))
            mcmcRegimeIteratively.run(uniWeightsValues=uniWeights, biWeightsValues=biWeights)
    else:
        mcmcRegimeIteratively.run()

else:
    cProfile.run(mcmcRegimeIteratively.run())


#"/Users/crystal/Dropbox/try/IterativelyTry"
################################################
# the 4th and fifth weights are so large and so small lead to very large and very small stationary distributions,
# so that we adjust the weight to have a more balanced stationary distributions if we use Normal distributions to generate those weight
# weightGenerationRegime = WeightGenerationRegime(nStates=6, nBivariateFeat=12, seed=1)
# stationaryWeights = weightGenerationRegime.generateStationaryWeights()
# stationaryWeights[2] = -1.35
# stationaryWeights[3] = 0.05
# stationaryWeights[4] = -1.0
# print(stationaryWeights)
# bivariateWeights = weightGenerationRegime.generateBivariateWeights()


#######################################################
## examples of how to add command line options
