import sys
import DataGenerationRegime
import MCMCRunningRegime
import HardCodedDictionaryUtils
import OptionClasses
import argparse
import FullTrajectorGeneration
import numpy as np
import os
import cProfile
from DataGenerationRegime import DataObsPairSeqs
from TransformAminoAcidsToIndexFromCSV import AminoAcidsCSV
import glob
import pandas as pd
from AminoAcidDict import sizeAndPolarityFeatList

## add command line argument
## list the arguments that we would like to provide to the code
argv = sys.argv[1:]
## ToDo: make sure fileNameIndex is a numeric number
## fileNameIndex = os.environ["fileNameIndex"]
fileNameIndex = 1
print(fileNameIndex)
fileNameIndex = int(int(float(fileNameIndex))-1)

parser = argparse.ArgumentParser()
## add the number of states as arguments
parser.add_argument('-nStates', action="store", type=int, dest='nStates', default=2, help='save the number of states in the ctmc')
## add boolean variable to indicate whether we only use hmc or we use a combination of hmc and local bps
parser.add_argument('--onlyHMC', action="store_true", help='HMC flag, the existence of the argument indicates HMC is used.')
parser.add_argument('--unknownTrueRateMtx', action="store_true", help='unknown rate matrix flag, the existence of the argument indicates the rate matrix used to generate the data is unknown.')
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
parser.add_argument('-nItersPerPathAuxVar', action='store', dest='nItersPerPathAuxVar', default=500, type=int, help='store the number of HMC samples in HMC algorithm')
## add the boolean variable to indicate whether we store the result in the end or we write results to csv files
parser.add_argument('--dumpResultIteratively', action='store_true', help='flag indicating we write results to csv iteratively instead of in the end')
## the number of iterations we write results to disk
parser.add_argument('-dumpResultIterations', action='store', dest='dumpResultIterations', default=50, type=int, help='store the number of iteration interval that we write results to csv')
##store the directory that we would lie to save the result
parser.add_argument('-dir_name', action='store', dest='dir_name', type=str, help='store the directory name to save the csv files')
## store the seed for Markov chain sampling
parser.add_argument('-samplingSeed', action='store', dest='initialSampleSeed', type=int, default=3, help='store the seed we use to do sampling')
## store the branch length of the time series of the generated data
parser.add_argument('-bt', action='store', dest='bt', type=float, default=5.0, help='store the branch length, in other words, the total length of the time series')
## store the total number of generated time series sequences
parser.add_argument('-nSeq', action='store', dest='nSeq', type=int, help='store the number of sequences of the time series')
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
parser.add_argument('-batchSize', action='store', dest='batchSize', type=int, default=50, help='the batch size when updating the ergodic mean')
parser.add_argument('-bivariateDict', action='store', dest='bivariateDict', default = 'chain', choices=['customized10', 'customized6', 'chain', 'polaritySize', 'AminoAcidDistRankPairs'])
parser.add_argument('-bivariateFeatDist', action='store', dest='bivariateFeatDist', default='Normal', choices=['Normal', 'Unif'])

parser.add_argument('-filePath', action='store', dest='filePath',type=str, help='the path of the directory which we store the real dataset sequences')
# parser.add_argument('-fileName', action='store',dest='filePath', type=str, help='the name of the input dataset')

results = parser.parse_args()
btLength = results.bt
nStates = results.nStates
nMCMCIter = results.nMCMCIter
nLeapFrogSteps = results.nLeapFrogSteps
stepSize = results.stepSize
trajectoryLength = results.trajectoryLength
initialSampleSeed = results.initialSampleSeed
interLength= results.interLength
dumpResultIterations = results.dumpResultIterations
refreshmentMethod = results.refreshmentMethod
nItersPerPathAuxVar = results.nItersPerPathAuxVar
batchSize = results.batchSize
bivariateDictStr = results.bivariateDict
bivariateFeatDist = results.bivariateFeatDist
unknownTrueRateMtx = results.unknownTrueRateMtx

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
elif bivariateDictStr == 'polaritySize':
    bivariateFeatIndexDictionary = HardCodedDictionaryUtils.getHardCodedDictPolaritySizeGraph()
    nBivariateFeat = len(sizeAndPolarityFeatList())
elif bivariateDictStr == 'AminoAcidDistRankPairs':
    bivariateFeatIndexDictionary = HardCodedDictionaryUtils.getHardCodedDictFromAminoAcidDistRankPairs()
    nBivariateFeat = int(nStates * (nStates-1)/2)

weightGenerationRegime = DataGenerationRegime.WeightGenerationRegime(nStates = nStates, nBivariateFeat= nBivariateFeat)

## get data from reading the csv file
## ToDo: need to create a loop for this later

# filePath = f
filePath = results.filePath
#filePath = "/Users/crystal/Dropbox/rejfreePy_main/ReadDataset"
extension = 'csv'
os.chdir(filePath)
# AllFileNames = [i for i in glob.glob('*.{}'.format(extension))]
# print(AllFileNames)
# allFileNamesDF = pd.DataFrame({'fileNames':AllFileNames})
# allFileNamesDF.to_csv("allFileNames.txt",index=False, header=False)
AllFileNames = pd.read_csv("allFileNames.txt", header=None)
fileName = AllFileNames.iloc[fileNameIndex, 0]

# for fileName in AllFileNames:
fileNameNoCSVExt = os.path.splitext(fileName)[0]
wholePath = os.path.join(filePath, fileNameNoCSVExt)
wholePath = os.path.join(wholePath, bivariateDictStr)
print(wholePath)
## check if the folder exists or not
if not os.path.exists(wholePath):
    os.mkdir(wholePath)

dir_name = wholePath
os.chdir(wholePath)
## read the data files
aminoAcidsData = AminoAcidsCSV(filePath=filePath, fileName=fileName)
aminoAcidsDataFilled = aminoAcidsData.combineIndexSeqIntoPD()
data = list()
data.append(aminoAcidsDataFilled.values)
nSeq = len(aminoAcidsDataFilled)
dataRegime = DataObsPairSeqs(nStates=nStates, bivariateFeatIndexDictionary=bivariateFeatIndexDictionary,
                                 nBivariateFeat=nBivariateFeat, btLength=btLength, interLength=interLength, nSeq=nSeq,
                                 data=data, weightGenerationRegime=weightGenerationRegime)

mcmcRegimeIteratively = MCMCRunningRegime.MCMCRunningRegime(dataRegime, nMCMCIter, thinning=1.0, burnIn=0,
                                                               onlyHMC=results.onlyHMC, HMCPlusBPS=results.HMCPlusBPS,
                                                               nLeapFrogSteps=nLeapFrogSteps, stepSize=stepSize,
                                                               saveRateMtx=False, initialSampleSeed=initialSampleSeed,
                                                               rfOptions=OptionClasses.RFSamplerOptions(
                                                                   trajectoryLength=trajectoryLength,
                                                                   refreshmentMethod=refreshmentMethod),
                                                               dumpResultIteratively=True,
                                                               dumpResultIterations=dumpResultIterations,
                                                               dir_name=dir_name,
                                                               nItersPerPathAuxVar=nItersPerPathAuxVar,
                                                              batchSize=batchSize,
                                                               unknownTrueRateMtx=unknownTrueRateMtx)
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

