import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy")
# import numpy as np
# from main.DataGenerationRegime import DataGenerationRegime
# from main.DataGenerationRegime import WeightGenerationRegime
# from main.MCMCRunningRegime import  MCMCRunningRegime
# from main.HardCodedDictionaryUtils import getHardCodedDict
# from main.HardCodedDictionaryUtils import getHardCodedDictChainGraph
# from main.OptionClasses import RFSamplerOptions
from numpy.random import RandomState
import DataGenerationRegime
import MCMCRunningRegime
import HardCodedDictionaryUtils
import OptionClasses
import argparse
import FullTrajectorGeneration
import numpy as np


## add command line argument
## list the arguments that we would like to provide to the code
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
## add the number of states as arguments
parser.add_argument('-nStates', action="store", type=int, dest='nStates', help='save the number of states in the ctmc')
## add boolean variable to indicate whether we only use hmc or we use a combination of hmc and local bps
parser.add_argument('--onlyHMC', action="store_true", help='HMC flag, the existence of the argument indicates HMC is used.')
## add boolean variable to indicate whether we use the local bps algorithm
parser.add_argument('--HMCPlusBPS', action='store_true', help='BPS flag, the existence of the argument indicates a combination of HMC and local BPS is used.')
## add the trajectory length if we use local bps
parser.add_argument('-trajectoryLength', action="store", dest='trajectoryLength', default = 0.125, help='save the trajectory length of the local bps sampler', type=float)
## add the total number of mcmc iterations
parser.add_argument('-nMCMCIter', action="store", dest='nMCMCIter', default=2000, type=int, help='store the total number of posterior samples')
## add the burning period of the posterior samples
parser.add_argument('-burnIn', action='store', dest='burnIn', default=0, type=int, help='store the burnIn period of the posterior samples')
## store the total number of leapfrog steps in HMC
parser.add_argument('-nLeapFrogSteps', action='store', dest='nLeapFrogSteps', default=40, type= int , help='store the total number of leapfrog steps in HMC')
## store the leapfrog size of HMC
parser.add_argument('-stepSize', action='store', dest='stepSize', default=0.02, type=float, help='store the leapfrog step size in HMC.')
## store the number of HMC samples
parser.add_argument('-nHMCSamples', action='store', dest='nHMCSamples', default=2000, type=int, help='store the number of HMC samples in HMC algorithm')
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
interLength= results.interLength
nHMCSamples = results.nHMCSamples
dumpResultIterations = results.dumpResultIterations

####################################################
## Weight Generation
seedGenData = results.seed
prng = RandomState(seedGenData)
weightGenerationRegime = DataGenerationRegime.WeightGenerationRegime(nStates = nStates, nBivariateFeat= int(nStates *(nStates-1)/2), prng=prng)
weightGenerationRegime.generateStationaryWeightsFromUniform()
weightGenerationRegime.generateBivariateWeightsFromNormal()

####################################################
## sequences data generation

dataRegime = DataGenerationRegime.DataGenerationRegime(nStates=nStates,  bivariateFeatIndexDictionary=HardCodedDictionaryUtils.getHardCodedDictChainGraph(nStates=nStates), btLength=bt, nSeq=nSeq, weightGenerationRegime=weightGenerationRegime, prng = prng, interLength=interLength)
## generate the sequences data
initialStateSeq = dataRegime.generatingInitialStateSeq()
seqList = dataRegime.generatingSeq(initialStateSeq)
suffStat = dataRegime.getSufficientStatFromSeq(seqList)
firstLastStatesArrayAll = dataRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)
trueRateMtx = dataRegime.rateMtxObj.getRateMtx()

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
                                          nLeapFrogSteps=nLeapFrogSteps, stepSize=stepSize, nHMCSamples=nHMCSamples, saveRateMtx=False, initialSampleSeed=initialSampleSeed,
                                          rfOptions=OptionClasses.RFSamplerOptions(trajectoryLength=trajectoryLength), dumpResultIteratively=True,
                                                            dumpResultIterations=dumpResultIterations, dir_name=dir_name)
mcmcRegimeIteratively.run()


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