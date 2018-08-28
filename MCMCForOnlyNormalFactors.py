
import os
import numpy as np
import OptionClasses

from datetime import datetime
from numpy.random import RandomState
import ExpectedCompleteReversibleModelBinaryFactors
import NormalFactor
from collections import OrderedDict
from LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
from PhyloLocalRFMove import PhyloLocalRFMove

def neighbourVariables(collisionFactor):
    result = list()
    if isinstance(collisionFactor, NormalFactor.NormalFactor):
        result.append(collisionFactor.gradientIndex)
        return result

def neighborVariableForAllFactors(allFactors):
    result = OrderedDict()
    ## change dictionary key from using factor to use the index of the factor
    for index, factor in enumerate(allFactors):
        result[index] = neighbourVariables(factor)
    return result


def neighbourVariblesAndFactorsAndExtendedNeighborsOfAllFactorsDict(allFactors):

    immediateNeighborVariablesIndexForAllFactors = OrderedDict()
    neighborFactorListForAllFactors = OrderedDict()
    extendedNeighborVariablesIndexForAllFactors = immediateNeighborVariablesIndexForAllFactors

    for indexOfFactor, factor in enumerate(allFactors):
        immediateNeighborVariablesIndexInTest = neighbourVariables(factor)
        immediateNeighborVariablesIndexForAllFactors[indexOfFactor] = immediateNeighborVariablesIndexInTest
        neighborFactorListForAllFactors[indexOfFactor] = indexOfFactor
    result = {}
    result['neighborVariablesAll'] = immediateNeighborVariablesIndexForAllFactors
    result['neighborFactorsAll'] = neighborFactorListForAllFactors
    result['extendedVariablesAll'] = extendedNeighborVariablesIndexForAllFactors
    return result


def getIndexOfNeighborFactorsForEachIndexOfBinaryFeature(dim):
    result = dict()
    for i in range(dim):
        result[i] = i

    return result



class MCMCForOnlyNormalFactors:
    def __init__(self,  nMCMCIter, thinning, burnIn, dim, prng, trajectoryLength, seed=1234567890, rfOptions=None,
                 dumpResultIteratively=False, dumpResultIterations = 50, dir_name=os.getcwd(),
                 initialSampleDist="Fixed", refreshmentMethod= OptionClasses.RefreshmentMethod.LOCAL,
                 batchSize=50, unknownTrueRateMtx = True):
        self.nMCMCIter = nMCMCIter
        self.thinning = thinning
        self.burnIn = burnIn
        self.mcmcOptions = OptionClasses.MCMCOptions(nMCMCIter, thinning, burnIn)

        self.rfOptions = rfOptions
        # TODO: should set-up the trajectory length here
        self.trajectoryLength = trajectoryLength

        self.dim = dim

        if prng is not None:
            self.prng = prng
        else:
            if seed is not None:
                self.prng = RandomState(seed)
            else:
                self.prng = RandomState(1234567890)


        if dumpResultIteratively:
            self.dumpResultIteratively = dumpResultIteratively
            self.dumpResultIterations = dumpResultIterations

        self.dir_name = dir_name
        self.initialWeightDist = initialSampleDist
        self.refreshmentMethod = refreshmentMethod
        self.batchSize = batchSize
        self.neighborVariablesForAllFactors = None
        self.variableAndFactorInfo = None
        self.indexOfFactorsForEachBivariateFeat = None
        self.unknownTrueRateMtx = unknownTrueRateMtx
        self.initialSampleSeed = seed

    def generateFixedInitialWeights(self):
        initialWeights = np.ones(self.dim) / self.dim
        return initialWeights

    def generateInitialWeightsFromUniformDist(self):

        initialWeights = self.prng.uniform(0, 1, self.dim)
        return initialWeights

    def generateInitialWeightsFromNormalDist(self):

        initialWeights = self.prng.normal(0, 1, self.dim)
        return initialWeights

    def generateFixedInitialWeightsValues(self, initialWeights):
        return initialWeights

    def generateInitialSamples(self, initialWeightsDist="Uniform", initialWeights=None):

        weights = self.generateInitialWeightsFromUniformDist()

        if initialWeightsDist == "Normal":
            weights = self.generateInitialWeightsFromNormalDist()
        if initialWeightsDist == "Fixed":
            weights = self.generateFixedInitialWeights()
        if initialWeightsDist == "AssignedWeightsValues":
            if initialWeights is None:
                raise Exception("The provided initial weights values can't be None")
            else:
                weights = self.generateFixedInitialWeightsValues(initialWeights)

        if initialWeightsDist is not None and not initialWeightsDist.__contains__("Uniform") \
                and not initialWeightsDist.__contains__("Normal") and not initialWeightsDist.__contains__("Fixed") \
                and not initialWeightsDist.__contains__("AssignedWeightsValues"):
            raise Exception(
                "The provided string for initial distribution of weights is not Uniform, Normal, fixed scalars or fixed provided initial values")

        return weights

    def generateArraysToSaveSamples(self):
        nMCMCIters = self.nMCMCIter
        samples = np.zeros((nMCMCIters, self.dim))
        return samples

    def getErgodicMean(self, nSamples, previousMean, newPartialSum, batchSize):
        ## this funcion uses the previous Mean vector and the number
        ## of previous samples to obtain the new cumulative mean after
        ## obtaining the batch of new vector values "newValues"
        newMean = (previousMean * (nSamples - batchSize) + newPartialSum) / nSamples
        return newMean

    def run(self, initialWeights=None):
        ## for every batchSize number of samples, we get the total sum
        ## of these samples and then refresh it to zero after we reach the
        ## batch size
        ## we only check the stationary distribution and exchangeable parameters
        weightsDistBatchSum = np.zeros((1, self.dim))
        initialWeights = self.generateInitialSamples(initialWeightsDist="Normal")


        # create arrays to save the posterior samples
        postSamples = self.generateArraysToSaveSamples()

        if self.dumpResultIteratively:
            allFileNames = self.createAllOutputFileNames(self.dir_name, self.nMCMCIter,
                                                            trajectoryLength=self.trajectoryLength,
                                                            mcmcSeed=self.initialSampleSeed)

        # this algorithm runs a combination of HMC and local BPS
        startTime = datetime.now()
        previousWeightsMean = np.zeros((1, self.dim))

        for i in range(self.nMCMCIter):

            postSamples[i, :] = initialWeights
            weightsDistBatchSum = weightsDistBatchSum + initialWeights

            if i > 0 and (i + 1) % self.batchSize == 0:
                ## When we reach the batch size, refresh the sum
                ## write the currrent file to csv and then refresh the vector to zeros
                weightsDistBatchMean = self.getErgodicMean(nSamples=int(i + 1),
                                                                  previousMean=previousWeightsMean,
                                                                  newPartialSum=weightsDistBatchSum,
                                                                  batchSize=self.batchSize)
                previousWeightsMean = weightsDistBatchMean
                self.dumpResult(weightsDistBatchMean[0, :], allFileNames['weightsErgodicMean'])

                weightsDistBatchSum = np.zeros(self.dim)


            if i > 0 and (i + 1) % self.dumpResultIterations == 0:
                ## record the results
                self.dumpResult(postSamples[(i + 1 - self.dumpResultIterations):(i + 1), :],
                                    allFileNames['weights'])


            model = ExpectedCompleteReversibleModelBinaryFactors.ModelWithNormalFactors(initialWeights)

            if i == 0:
                self.neighborVariablesForAllFactors = neighborVariableForAllFactors(model.localFactors)
                self.variableAndFactorInfo = neighbourVariblesAndFactorsAndExtendedNeighborsOfAllFactorsDict(
                        model.localFactors)
                self.indexOfFactorsForEachBivariateFeat = getIndexOfNeighborFactorsForEachIndexOfBinaryFeature(
                        self.dim)

            localSampler = LocalRFSamplerForBinaryWeights(model, self.rfOptions, self.mcmcOptions, 0,
                                                             self.neighborVariablesForAllFactors,
                                                             self.variableAndFactorInfo,
                                                             self.indexOfFactorsForEachBivariateFeat)

            phyloLocalRFMove = PhyloLocalRFMove(model=model, sampler=localSampler,
                                                    initialPoints=initialWeights, options=self.rfOptions,
                                                    prng=RandomState(i))
            initialWeights = phyloLocalRFMove.execute()

            print("The current iteration finished is:")
            print(i)

        endTime = datetime.now()
        timeElapsed = (endTime - startTime).total_seconds()

        if not self.dumpResultIteratively:
            result = {}
            result['postSamples'] = postSamples
            result['elapsingTime'] = timeElapsed
            return result
        else:
            self.outputRunningTime(actualRunningTimeInSec=timeElapsed, dir_name=self.dir_name,
                                    time_base_filename="wallTime", nMCMCIter=self.nMCMCIter)

    def dumpResult(self, posteriorSampleArray, fullOutputFileName):

        # import numpy as np
        # f = open('asd.dat', 'ab')
        # for iind in range(4):
        #    a = np.random.rand(10, 10)
        #    np.savetxt(f, a)
        # f.close()

        f = open(fullOutputFileName, 'ab')
        np.savetxt(f, posteriorSampleArray, fmt='%.3f', delimiter=',', newline='\r\n')
        f.close()

        # with open(fullOutputFileName, 'a') as outputFile:
        #    np.savetxt(outputFile, posteriorSampleArray, fmt='%.3f', delimiter=',')

    def outputRunningTime(self, actualRunningTimeInSec, dir_name,  nMCMCIter,
                              time_base_filename="wallTime"):
        format = ".csv"
        if not isinstance(actualRunningTimeInSec, str):
            actualRunningTimeInSec = str(actualRunningTimeInSec)

        if not isinstance(nMCMCIter, str):
            nMCMCIter = str(nMCMCIter)
        timeStr = time_base_filename +  nMCMCIter
        timeStr = timeStr + "trajectoryLength" + str(
                self.trajectoryLength) + "Dim"+ str(self.dim)+"refreshementMethod" + self.refreshmentMethod.name

        timeStr = timeStr + format
        timeFileName = os.path.join(dir_name, timeStr)
        csv = open(timeFileName, "w")
        columnTitleRow = "elapsedTime in seconds\n"
        csv.write(columnTitleRow)
        if not isinstance(actualRunningTimeInSec, str):
            actualRunningTimeInSec = str(actualRunningTimeInSec)
        csv.write(actualRunningTimeInSec)
        csv.close()

    def createAllOutputFileNames(self, dir_name,  nMCMCIter,
                                    trajectoryLength=None, mcmcSeed=None):

        if not isinstance(trajectoryLength, str):
            trajectoryLength = str(trajectoryLength)
        if not isinstance(mcmcSeed, str):
            mcmcSeed = str(mcmcSeed)
        if not isinstance(nMCMCIter, str):
            nMCMCIter = str(nMCMCIter)

        weightStr = "weight"

        weightStr = weightStr + "Dim" + str(self.dim)+ nMCMCIter + "initialWeight" + self.initialWeightDist + self.refreshmentMethod.name
        if trajectoryLength is not None:
            weightStr = weightStr+ "trajectoryLength" + str(trajectoryLength)

        weightsErgodicMean = weightStr +"Dim"+ str(self.dim)+ "ErgodicMean" + "batchSize" + str(self.batchSize)
        if mcmcSeed is not None:
            weightStr = weightStr + "mcmcSeed" + mcmcSeed + ".csv"

        weightStr = os.path.join(dir_name, weightStr)

        result = {}
        result['weights'] = weightStr
        result['weightsErgodicMean'] = weightsErgodicMean
        return result

    def recordResult(self, mcmcResult, dir_name, time_base_filename, nMCMCIter):
        # currentIteration represents the current Iteration of the MCMC iteration. It is only not None when recordIteratively is true
        # iterationSize represents the number of iterations that we record each time. For example, if we record result every 50 iterations, iterationSize=50.
        # saveRateMtx is a boolean variable, if it is true, we save the posterior samples for the whole rate matrix
        # mcmcResult is the result after running the MCMC algorithm
        format = '.csv'
        timeFileName = os.path.join(dir_name, time_base_filename + nMCMCIter + format)

        csv = open(timeFileName, "w")
        columnTitleRow = "elapsedTime\n"
        csv.write(columnTitleRow)
        row = mcmcResult['elapsingTime']
        csv.write(str(row))
        csv.close()

        weightSamples = mcmcResult['postSamples']
        timeElapsed = mcmcResult['elapsingTime']

        weightStr = "weights"
        trueWeights = "trueWeights"
        weightFileName = os.path.join(dir_name, weightStr + nMCMCIter + format)
        trueWeightName = os.path.join(dir_name, trueWeights)
        np.savetxt(weightFileName, weightSamples, fmt='%.3f', delimiter=',')
        np.savetxt(trueWeightName, self.data ,fmt='%.3f', delimiter=',')


