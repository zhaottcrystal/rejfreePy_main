import os
import sys

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
import OptionClasses
import ExpectedCompleteReversibleObjective
import ExpectedCompleteReversibleModelBinaryFactors
import HMC
from LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
from PhyloLocalRFMove import PhyloLocalRFMove
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
import FullTrajectorGeneration
import DataGenerationRegime
from datetime import datetime
# from main.DataGenerationRegime import WeightGenerationRegime

from numpy.random import RandomState
from LocalRFSamplerForBinaryWeights import neighborVariableForAllFactors
from LocalRFSamplerForBinaryWeights import neighbourVariblesAndFactorsAndExtendedNeighborsOfAllFactorsDict
from LocalRFSamplerForBinaryWeights import getIndexOfNeighborFactorsForEachIndexOfBinaryFeature
class MCMCRunningRegime:

    def __init__(self, dataGenerationRegime, nMCMCIter, thinning, burnIn, onlyHMC, HMCPlusBPS, prng=None, nLeapFrogSteps=40,
                 stepSize=0.02, saveRateMtx = False, initialSampleSeed=None, rfOptions=None,
                 dumpResultIteratively=False, dumpResultIterations = 50, dir_name=os.getcwd(), nItersPerPathAuxVar=1000,
                 initialSampleDist="Fixed", refreshmentMethod= OptionClasses.RefreshmentMethod.LOCAL, batchSize=50, unknownTrueRateMtx = True):
        if prng is None:
            self.prng = dataGenerationRegime.prng
        else:
            self.prng = prng

        # MCMC related fields
        self.nMCMCIter = nMCMCIter
        self.thinning = thinning
        self.burnIn = burnIn
        self.mcmcOptions = OptionClasses.MCMCOptions(nMCMCIter, thinning, burnIn)

        # HMC related options
        self.nLeapFrogSteps = nLeapFrogSteps
        self.stepSize = stepSize

        # onlyHMC and HMCPlusBPS are both boolean variables
        self.onlyHMC = onlyHMC

        # BPS related options if necessary
        # HMCPlusBPS should be a boolean variable to indicate whether we are using a combination of HMC and BPS algorithm
        self.HMCPlusBPS = HMCPlusBPS
        self.rfOptions = rfOptions
        # TODO: should set-up the trajectory length here
        self.trajectoryLength = rfOptions.trajectoryLength

        # data related information
        self.dataGenerationRegime = dataGenerationRegime
        self.data = dataGenerationRegime.data
        self.nStates = dataGenerationRegime.nStates
        self.bivariateFeatIndexDictionary = dataGenerationRegime.bivariateFeatIndexDictionary
        self.nBivariateFeat = dataGenerationRegime.nBivariateFeat
        if HMCPlusBPS:
            self.samplingMethod = "HMCPlusBPS"
        if onlyHMC:
            self.samplingMethod = "HMC"

        if initialSampleSeed is not None:
            self.initialSampleSeed = initialSampleSeed
        else:
            initialSampleSeed = 1
        if dumpResultIteratively:
            self.dumpResultIteratively = dumpResultIteratively
            self.dumpResultIterations = dumpResultIterations

        self.saveRateMtx = saveRateMtx
        self.dir_name = dir_name
        self.nItersPerPathAuxVar = nItersPerPathAuxVar
        self.initialWeightDist = "Fixed"
        self.refreshmentMethod = refreshmentMethod
        self.nExchange = int(self.nStates * (self.nStates-1)/2)
        self.batchSize = batchSize
        self.neighborVariablesForAllFactors = None
        self.variableAndFactorInfo = None
        self.indexOfFactorsForEachBivariateFeat = None
        self.unknownTrueRateMtx = unknownTrueRateMtx


    def generateFixedInitialWeights(self):

        initialStationaryWeights = np.ones(self.nStates)/self.nStates
        initialBinaryWeights = np.ones(self.nBivariateFeat) /self.nBivariateFeat
        result = {}
        result['initialStationaryWeights'] = initialStationaryWeights
        result['initialBinaryWeights'] = initialBinaryWeights
        return result


    def generateInitialWeightsFromUniformDist(self):
        weightGenerationRegime = DataGenerationRegime.WeightGenerationRegime(self.nStates, self.nBivariateFeat,self.prng)
        initialStationaryWeights = weightGenerationRegime.generateStationaryWeightsFromUniform()
        initialBinaryWeights = weightGenerationRegime.generateBivariateWeightsFromUniform()
        result = {}
        result['initialStationaryWeights'] = initialStationaryWeights
        result['initialBinaryWeights'] = initialBinaryWeights
        return result


    def generateInitialWeightsFromNormalDist(self):
        weightGenerationRegime = DataGenerationRegime.WeightGenerationRegime(self.nStates, self.nBivariateFeat,self.prng)
        initialStationaryWeights = weightGenerationRegime.generateStationaryWeightsFromNormal()
        initialBinaryWeights = weightGenerationRegime.generateBivariateWeightsFromNormal()
        result = {}
        result['initialStationaryWeights'] = initialStationaryWeights
        result['initialBinaryWeights'] = initialBinaryWeights
        return result

    def generateFixedInitialWeightsValues(self, uniWeightsValues, biWeightsValues):
        result = {}
        result['initialStationaryWeights'] = uniWeightsValues
        result['initialBinaryWeights'] = biWeightsValues
        return result

    def generateInitialSamples(self, initialWeightsDist="Uniform", uniWeightsValues=None, biWeightsValues=None):

        weights = self.generateInitialWeightsFromUniformDist()

        if initialWeightsDist == "Normal":
            weights = self.generateInitialWeightsFromNormalDist()
        if initialWeightsDist == "Fixed":
            weights = self.generateFixedInitialWeights()
        if initialWeightsDist == "AssignedWeightsValues":
            if uniWeightsValues is None or biWeightsValues is None:
                raise Exception("The provided initial weights values can't be None")
            else:
                weights = self.generateFixedInitialWeightsValues(uniWeightsValues, biWeightsValues)

        if initialWeightsDist is not None and not initialWeightsDist.__contains__("Uniform") \
                and not initialWeightsDist.__contains__("Normal") and not initialWeightsDist.__contains__("Fixed")\
                and not initialWeightsDist.__contains__("AssignedWeightsValues"):
            raise Exception("The provided string for initial distribution of weights is not Uniform, Normal, fixed scalars or fixed provided initial values")


        initialStationaryWeights = weights['initialStationaryWeights']
        initialBinaryWeights = weights['initialBinaryWeights']

        initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(self.nStates,
                                                                                   initialStationaryWeights,
                                                                                   initialBinaryWeights,
                                                                                   self.bivariateFeatIndexDictionary)
        initialRateMatrix = initialRateMtx.getRateMtx()
        initialStationaryDist = initialRateMtx.getStationaryDist()
        initialExchangeCoef = initialRateMtx.getExchangeCoef()

        result = {}
        result['initialStationaryWeights'] = initialStationaryWeights
        result['initialStationaryDist'] = initialStationaryDist
        result['initialBinaryWeights'] = initialBinaryWeights
        result['initialRateMatrix'] = initialRateMatrix
        result['initialExchangeCoef'] = initialExchangeCoef
        return result

    def generateArraysToSaveSamples(self):

        nMCMCIters = self.nMCMCIter
        nStates = self.nStates
        exchangeCoefDim = int(nStates*(nStates-1)/2)
        nBivariateFeat = self.nBivariateFeat
        result = {}

        if self.HMCPlusBPS:
            stationaryDistSamples = np.zeros((nMCMCIters, nStates))
            stationaryWeightsSamples = np.zeros((nMCMCIters, nStates))
            binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
            exchangeableSamples = np.zeros((nMCMCIters, exchangeCoefDim))
            rateMatrixSamples = np.zeros((nMCMCIters, nStates, nStates))

        else:
            weightSamples = np.zeros((nMCMCIters, (nStates + nBivariateFeat)))
            avgWeights = np.zeros((nStates + nBivariateFeat))
            stationaryDistSamples = np.zeros((nMCMCIters, nStates))
            stationaryWeightsSamples = np.zeros((nMCMCIters, nStates))
            binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))

            exchangeableSamples = np.zeros((nMCMCIters, int(self.dataGenerationRegime.nStates * (self.dataGenerationRegime.nStates-1)/2)))
            rateMatrixSamples = np.zeros((nMCMCIters, nStates, nStates))
            result['weightSamples'] = weightSamples
            result['avgWeights'] = avgWeights

        result['stationaryDistSamples'] = stationaryDistSamples
        result['stationaryWeightsSamples'] = stationaryWeightsSamples
        result['binaryWeightsSamples'] = binaryWeightsSamples
        result['exchangeableSamples'] = exchangeableSamples
        result['rateMatrixSamples'] = rateMatrixSamples
        return result


    def initializeHMCWeights(self, initialStationaryWeights, initialBinaryWeights):
        result = self.generateArraysToSaveSamples()
        weightSamples = result['weightSamples']
        avgWeights = result['avgWeights']
        avgWeights[0:self.nStates] = initialStationaryWeights
        avgWeights[self.nStates: (self.nStates + self.nBivariateFeat)] = initialBinaryWeights
        weightSamples[0, :] = avgWeights
        result['avgWeights'] = avgWeights
        result['weightSamples'] = weightSamples
        return result
    
    def getErgodicMean(self, nSamples, previousMean, newPartialSum, batchSize):
        ## this funcion uses the previous Mean vector and the number
        ## of previous samples to obtain the new cumulative mean after
        ## obtaining the batch of new vector values "newValues"
        newMean = (previousMean * (nSamples-batchSize) + newPartialSum)/nSamples
        return newMean
        
        

    def run(self,  uniWeightsValues=None, biWeightsValues=None):
        ## for every batchSize number of samples, we get the total sum
        ## of these samples and then refresh it to zero after we reach the 
        ## batch size
        ## we only check the stationary distribution and exchangeable parameters
        stationaryDistBatchSum = np.zeros((1, self.nStates))
        exchangeCoefBatchSum = np.zeros((1, self.nExchange))
        
        ## output the true stationary distribution and exchangeable parameters
        if self.unknownTrueRateMtx is False:
            self.outputTrueParameters(self.dir_name)

        # call methods to create initial samples
        initialSamples = self.generateInitialSamples(initialWeightsDist=self.initialWeightDist, uniWeightsValues=uniWeightsValues, biWeightsValues=biWeightsValues)
        # decompose the initial samples
        initialStationaryWeights = initialSamples['initialStationaryWeights']
        initialStationaryDist = initialSamples['initialStationaryDist']
        initialBinaryWeights = initialSamples['initialBinaryWeights']
        initialRateMatrix = initialSamples['initialRateMatrix']
        initialExchangeCoef = initialSamples['initialExchangeCoef']

        # create arrays to save the posterior samples
        posteriorSamples = self.generateArraysToSaveSamples()
        stationaryDistSamples = posteriorSamples['stationaryDistSamples']
        stationaryWeightsSamples = posteriorSamples['stationaryWeightsSamples']
        binaryWeightsSamples = posteriorSamples['binaryWeightsSamples']
        exchangeableSamples = posteriorSamples['exchangeableSamples']
        rateMatrixSamples = posteriorSamples['rateMatrixSamples']
        
        if self.dumpResultIteratively:
            allFileNames = self.createAllOutputFileNames(self.dir_name, self.samplingMethod, self.nMCMCIter, saveRateMtx=False,
                                     trajectoryLength=self.trajectoryLength, mcmcSeed=self.initialSampleSeed)


        # this algorithm runs a combination of HMC and local BPS
        startTime = datetime.now()
        if self.onlyHMC:
            initializedPosteriorSamples = self.initializeHMCWeights(initialStationaryWeights, initialBinaryWeights)
            weightSamples = initializedPosteriorSamples['weightSamples']
            sample = initializedPosteriorSamples['avgWeights']
        else:
            sample = initialStationaryWeights
        
        previousStationaryDistMean = np.zeros((1, self.nStates))
        previousExchangeCoefMean = np.zeros((1, self.nExchange))

        nInit = np.zeros(self.nStates)
        unique, counts = np.unique(self.data[0][:, 0], return_counts=True)
        nInitCount = np.asarray((unique, counts)).T
        nInit[nInitCount[:, 0].astype(int)] = nInitCount[:, 1]


        for i in range(self.nMCMCIter):
            
            stationaryWeightsSamples[i, :] = initialStationaryWeights
            binaryWeightsSamples[i, :] = initialBinaryWeights
            exchangeableSamples[i,:] = initialExchangeCoef
            if self.saveRateMtx:
                rateMatrixSamples[i, :] = initialRateMatrix
            stationaryDistSamples[i, :] = initialStationaryDist

            stationaryDistBatchSum = stationaryDistBatchSum + initialStationaryDist
            exchangeCoefBatchSum = exchangeCoefBatchSum + initialExchangeCoef

            if i > 0 and (i+1) % self.batchSize == 0:
                ## When we reach the batch size, refresh the sum
                ## write the currrent file to csv and then refresh the vector to zeros
                stationaryDistBatchMean = self.getErgodicMean(nSamples=int(i+1),
                                                              previousMean=previousStationaryDistMean,
                                                              newPartialSum=stationaryDistBatchSum,
                                                              batchSize = self.batchSize)
                exchangeCoefBatchMean = self.getErgodicMean(nSamples = int(i+1),
                                                            previousMean=previousExchangeCoefMean,
                                                            newPartialSum=exchangeCoefBatchSum,
                                                            batchSize=self.batchSize)
                previousStationaryDistMean = stationaryDistBatchMean
                previousExchangeCoefMean = exchangeCoefBatchMean
                self.dumpResult(stationaryDistBatchMean[0, :],allFileNames['stationaryDistErgodicMean'])
                self.dumpResult(exchangeCoefBatchMean[0, :],allFileNames['exchangeCoefErgodicMean'])
                stationaryDistBatchSum = np.zeros(self.nStates)
                exchangeCoefBatchSum = np.zeros(self.nExchange)


                                     
            if i > 0 and (i+1) % self.dumpResultIterations == 0:
                ## record the results
                self.dumpResult(stationaryDistSamples[(i+1-self.dumpResultIterations):(i+1), :],   allFileNames['stationaryDist'])
                self.dumpResult(exchangeableSamples[(i+1-self.dumpResultIterations):(i+1), :], allFileNames['exchangeableCoef'])
                self.dumpResult(binaryWeightsSamples[(i+1-self.dumpResultIterations):(i+1), :], allFileNames['binaryWeights'])
                self.dumpResult(stationaryWeightsSamples[(i+1-self.dumpResultIterations):(i+1), :], allFileNames['stationaryWeights'])




            holdTime = np.zeros(self.nStates)
            nTrans = np.zeros((self.nStates, self.nStates))

            for j in range(self.dataGenerationRegime.nPairSeq):
                ## change it to the true rate matrix and see if the sufficient statistics match
                #suffStat = FullTrajectorGeneration.endPointSamplerSummarizeStatisticsOneBt(True, self.prng,
                #                                                                           self.dataGenerationRegime.rateMtxObj.getRateMtx(),
                #                                                                           self.data[j],
                #                                                                           self.dataGenerationRegime.interLength)

                suffStat = FullTrajectorGeneration.endPointSamplerSummarizeStatisticsOneBt(True, RandomState(int(i * self.dataGenerationRegime.nSeq + j)), initialRateMatrix,
                                                                       self.data[j], self.dataGenerationRegime.interLength)
                # nInit = nInit + suffStat['nInit']
                holdTime = holdTime + suffStat['holdTimes']
                nTrans = nTrans + suffStat['nTrans']

                    # construct expected complete reversible model objective
            if self.HMCPlusBPS:
                expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective.ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0,
                                                                                          initialExchangeCoef)
            if self.onlyHMC:
                expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective.ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0, nBivariateFeatWeightsDictionary=self.bivariateFeatIndexDictionary)

            #####################################
            hmc = HMC.HMC(self.nLeapFrogSteps, self.stepSize, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
            lastSample = sample
            for k in range(self.nItersPerPathAuxVar):
                 hmcResult = hmc.doIter(self.nLeapFrogSteps, self.stepSize, lastSample, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective, True)
                 lastSample = hmcResult.next_q

            sample = lastSample                 

            if self.onlyHMC:
                initialStationaryWeights = sample[0:self.nStates]
                initialBinaryWeights = sample[self.nStates:(self.nStates + self.nBivariateFeat)]

            # sample stationary distribution elements using HMC
            if self.HMCPlusBPS:
                initialStationaryWeights = sample
                # update stationary distribution elements to the latest value
                initialStationaryDist = np.exp(sample) / np.sum(np.exp(sample))
                # sample exchangeable coefficients using local bouncy particle sampler
                ## define the model
                model = ExpectedCompleteReversibleModelBinaryFactors.ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, self.nStates,
                                                                       initialBinaryWeights,  initialStationaryDist,
                                                                         self.bivariateFeatIndexDictionary)
                ## define the sampler to use
                ## local sampler to use
                if i == 0:
                    self.neighborVariablesForAllFactors = neighborVariableForAllFactors(self.nStates, model.localFactors,
                                                                                   self.bivariateFeatIndexDictionary)
                    self.variableAndFactorInfo = neighbourVariblesAndFactorsAndExtendedNeighborsOfAllFactorsDict(self.nStates, model.localFactors, self.bivariateFeatIndexDictionary,self.nBivariateFeat)
                    self.indexOfFactorsForEachBivariateFeat = getIndexOfNeighborFactorsForEachIndexOfBinaryFeature(self.bivariateFeatIndexDictionary,
                                                         self.nBivariateFeat, model.localFactors)

                localSampler = LocalRFSamplerForBinaryWeights(model, self.rfOptions, self.mcmcOptions, self.nStates,
                                                              self.neighborVariablesForAllFactors, self.variableAndFactorInfo, self.indexOfFactorsForEachBivariateFeat)

                phyloLocalRFMove = PhyloLocalRFMove(model=model, sampler=localSampler, initialPoints=initialBinaryWeights, options=self.rfOptions, prng=RandomState(i))
                initialBinaryWeights = phyloLocalRFMove.execute()

            initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(self.nStates, initialStationaryWeights,
                                                                                           initialBinaryWeights,
                                                                                           bivariateFeatIndexDictionary=self.bivariateFeatIndexDictionary)

            initialStationaryDist = initialRateMtx.getStationaryDist()
            initialRateMatrix = initialRateMtx.getRateMtx()
            initialExchangeCoef = initialRateMtx.getExchangeCoef()
            initialBinaryWeights = initialBinaryWeights
            print("The current iteration finished is:")
            print(i)

        endTime = datetime.now()
        timeElapsed =(endTime - startTime).total_seconds()


        if not self.dumpResultIteratively:
            result = {}
            result['stationaryDistSamples'] = stationaryDistSamples
            result['stationaryWeightsSamples'] = stationaryWeightsSamples
            result['binaryWeightsSamples'] = binaryWeightsSamples
            result['exchangeableSamples'] = exchangeableSamples
            if self.saveRateMtx:
                result['rateMatrixSamples'] = rateMatrixSamples
            result['elapsingTime'] = timeElapsed
            return result
        else:
            self.outputRunningTime(actualRunningTimeInSec=timeElapsed, dir_name= self.dir_name, time_base_filename= "wallTime", samplingMethod=self.samplingMethod, nMCMCIter=self.nMCMCIter)

    def dumpResult(self, posteriorSampleArray, fullOutputFileName):

        #import numpy as np
        #f = open('asd.dat', 'ab')
        #for iind in range(4):
        #    a = np.random.rand(10, 10)
        #    np.savetxt(f, a)
        #f.close()

        f = open(fullOutputFileName, 'ab')
        np.savetxt(f, posteriorSampleArray, fmt= '%.3f', delimiter=',', newline='\r\n')
        f.close()

        #with open(fullOutputFileName, 'a') as outputFile:
        #    np.savetxt(outputFile, posteriorSampleArray, fmt='%.3f', delimiter=',')

    def outputRunningTime(self, actualRunningTimeInSec, dir_name, samplingMethod, nMCMCIter, time_base_filename="wallTime" ):
        format = ".csv"
        if not isinstance(actualRunningTimeInSec, str):
            actualRunningTimeInSec = str(actualRunningTimeInSec)
        if not isinstance(samplingMethod, str):
            samplingMethod = str(samplingMethod)
        if not isinstance(nMCMCIter, str):
            nMCMCIter = str(nMCMCIter)
        timeStr = time_base_filename+samplingMethod+nMCMCIter+str(self.nStates)
        if self.samplingMethod == "HMCPlusBPS":
            timeStr = timeStr + "trajectoryLength" + str(self.trajectoryLength) + "refreshementMethod" + self.refreshmentMethod.name
        else:
            timeStr = timeStr + "stepSize" + str(self.stepSize) + "nLeapFrogSteps" + str(self.nLeapFrogSteps)

        timeStr = timeStr + format
        timeFileName = os.path.join(dir_name, timeStr)
        csv = open(timeFileName, "w")
        columnTitleRow = "elapsedTime in seconds\n"
        csv.write(columnTitleRow)
        if not isinstance(actualRunningTimeInSec, str):
            actualRunningTimeInSec = str(actualRunningTimeInSec)
        csv.write(actualRunningTimeInSec)
        csv.close()

    def createAllOutputFileNames(self, dir_name, samplingMethod, nMCMCIter, saveRateMtx=False, trajectoryLength=None, mcmcSeed=None):

        if not isinstance(trajectoryLength, str):
            trajectoryLength = str(trajectoryLength)
        if not isinstance(mcmcSeed, str):
            mcmcSeed = str(mcmcSeed)
        if not isinstance(nMCMCIter, str):
            nMCMCIter = str(nMCMCIter)


        stationaryDistStr = "stationaryDistribution"
        stationaryWeightsStr = "stationaryWeights"
        exchangeableCoefStr = "exchangeableCoef"
        binaryWeightsStr = "binaryWeights"

        rateMtxStrName = "rateMatrix"
        if saveRateMtx:
            rateMtxStr = "rateMatrix"
            rateMtxStrName = rateMtxStr + samplingMethod + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist

        stationaryDistStrName = stationaryDistStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist
        stationaryWeightsStrName = stationaryWeightsStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist
        exchangeableCoefStrName = exchangeableCoefStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist
        binaryWeightsStrName = binaryWeightsStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist
        
        if self.HMCPlusBPS:
            if saveRateMtx:
                rateMtxStr = "rateMatrix"
                rateMtxStrName = rateMtxStr + samplingMethod + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist + self.refreshmentMethod.name

            stationaryDistStrName = stationaryDistStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist + self.refreshmentMethod.name
            stationaryWeightsStrName = stationaryWeightsStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist + self.refreshmentMethod.name
            exchangeableCoefStrName = exchangeableCoefStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist + self.refreshmentMethod.name
            binaryWeightsStrName = binaryWeightsStr + samplingMethod + nMCMCIter + "initialWeightDist" + self.initialWeightDist + self.refreshmentMethod.name


        if self.dataGenerationRegime.nSeq is not None:
            stationaryDistStrName = stationaryDistStrName + "nSeq" + str(self.dataGenerationRegime.nSeq)
            stationaryWeightsStrName = stationaryWeightsStrName + "nSeq" + str(self.dataGenerationRegime.nSeq)
            exchangeableCoefStrName = exchangeableCoefStrName + "nSeq" + str(self.dataGenerationRegime.nSeq)
            binaryWeightsStrName = binaryWeightsStrName + "nSeq" + str(self.dataGenerationRegime.nSeq)
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + "nSeq" + str(self.dataGenerationRegime.nSeq)


        if self.nStates is not None:
            stationaryDistStrName = stationaryDistStrName + "nStates" + str(self.nStates)
            stationaryWeightsStrName = stationaryWeightsStrName +"nStates" + str(self.nStates)
            exchangeableCoefStrName = exchangeableCoefStrName + "nStates" + str(self.nStates)
            binaryWeightsStrName = binaryWeightsStrName + "nStates" + str(self.nStates)
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + "nStates" + str(self.nStates)

        ## store the sampling seed of the program
        if self.initialSampleSeed is not None:
            stationaryDistStrName = stationaryDistStrName + "mcmcSamplingSeed" + str(self.initialSampleSeed)
            stationaryWeightsStrName = stationaryWeightsStrName + "mcmcSamplingSeed" + str(self.initialSampleSeed)
            exchangeableCoefStrName = exchangeableCoefStrName + "mcmcSamplingSeed" + str(self.initialSampleSeed)
            binaryWeightsStrName = binaryWeightsStrName + "mcmcSamplingSeed" + str(self.initialSampleSeed)
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + "mcmcSamplingSeed" + str(self.initialSampleSeed)


        if trajectoryLength is not None:
            stationaryDistStrName = stationaryDistStrName + "trajectoryLength"+ str(trajectoryLength)
            stationaryWeightsStrName = stationaryWeightsStrName + "trajectoryLength"+  str(trajectoryLength)
            exchangeableCoefStrName = exchangeableCoefStrName + "trajectoryLength"+ str(trajectoryLength)
            binaryWeightsStrName = binaryWeightsStrName + "trajectoryLength" + str(trajectoryLength)
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + "trajectoryLength"+ str(trajectoryLength)

        if self.nLeapFrogSteps is not None:
            stationaryDistStrName = stationaryDistStrName + "nLeapFrogSteps" + str(self.nLeapFrogSteps)
            stationaryWeightsStrName = stationaryWeightsStrName + "nLeapFrogSteps" + str(self.nLeapFrogSteps)
            exchangeableCoefStrName = exchangeableCoefStrName + "nLeapFrogSteps" + str(self.nLeapFrogSteps)
            binaryWeightsStrName = binaryWeightsStrName + "nLeapFrogSteps" + str(self.nLeapFrogSteps)
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + str(self.stepSize)

        if self.stepSize is not None:
            stationaryDistStrName = stationaryDistStrName + "stepSize" + str(self.stepSize)
            stationaryWeightsStrName = stationaryWeightsStrName + "stepSize" + str(self.stepSize)
            exchangeableCoefStrName = exchangeableCoefStrName + "stepSize" + str(self.stepSize)
            binaryWeightsStrName = binaryWeightsStrName + "stepSize" + str(self.stepSize)
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + str(self.stepSize)
        
        stationaryDistErgodicMean = stationaryDistStrName + "ErgodicMean" + "batchSize" + str(self.batchSize)
        exchangeCoefErgodicMean = exchangeableCoefStrName + "ErgodicMean" + "batchSize" + str(self.batchSize)

        if mcmcSeed is not None:
            stationaryDistStrName = stationaryDistStrName + "mcmcSeed" +mcmcSeed+ ".csv"
            stationaryWeightsStrName = stationaryWeightsStrName + "mcmcSeed" + mcmcSeed+ ".csv"
            exchangeableCoefStrName = exchangeableCoefStrName + "mcmcSeeed" + mcmcSeed+ ".csv"
            binaryWeightsStrName = binaryWeightsStrName + "mcmcSeed"+ mcmcSeed + ".csv"
            stationaryDistErgodicMean = stationaryDistErgodicMean + "mcmcSeed"+ mcmcSeed + ".csv"
            exchangeCoefErgodicMean = exchangeCoefErgodicMean + "mcmcSeed"+ mcmcSeed + ".csv"
            
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + mcmcSeed

        stationaryDistFileName = os.path.join(dir_name, stationaryDistStrName)
        stationaryWeightsFileName = os.path.join(dir_name, stationaryWeightsStrName)
        exchangeableCoefFileName = os.path.join(dir_name, exchangeableCoefStrName)
        binaryWeightsFileName = os.path.join(dir_name, binaryWeightsStrName)
        stationaryDistErgodicMeanFileName = os.path.join(dir_name,stationaryDistErgodicMean)
        exchangeCoefErgodicMeanFileName = os.path.join(dir_name, exchangeCoefErgodicMean)
        if saveRateMtx:
            rateMtxFileName = os.path.join(dir_name, rateMtxStrName)

        result = {}
        result['stationaryDist'] = stationaryDistFileName
        result['stationaryWeights'] = stationaryWeightsFileName
        result['exchangeableCoef'] = exchangeableCoefFileName
        result['binaryWeights'] = binaryWeightsFileName
        result['stationaryDistErgodicMean'] = stationaryDistErgodicMeanFileName
        result['exchangeCoefErgodicMean'] = exchangeCoefErgodicMeanFileName
        if saveRateMtx:
            result['rateMatrix'] =rateMtxFileName
        return result


    def outputTrueParameters(self, dir_name):
        trueStationaryDistStr = "trueStationaryDistribution" + str(self.nStates)
        trueStationaryWeightsStr = "trueStationaryWeights" + str(self.nStates)
        trueExchangeableCoefStr = "trueExchangeableCoef" + str(self.nStates)
        trueBinaryWeightsStr = "trueBinaryWeights" + str(self.nStates)
        format = '.csv'
        trueStationaryDistFileName = os.path.join(dir_name, trueStationaryDistStr + format)
        trueExchangeableCoefFileName = os.path.join(dir_name, trueExchangeableCoefStr + format)
        trueStationaryWeightsFileName = os.path.join(dir_name, trueStationaryWeightsStr + format)
        trueBinaryWeightsFileName = os.path.join(dir_name, trueBinaryWeightsStr + format)

        np.savetxt(trueStationaryDistFileName, self.dataGenerationRegime.stationaryDist, fmt='%.3f', delimiter=',')
        np.savetxt(trueExchangeableCoefFileName, self.dataGenerationRegime.exchangeCoef, fmt='%.3f', delimiter=',')
        np.savetxt(trueStationaryWeightsFileName, self.dataGenerationRegime.stationaryWeights, fmt='%.3f',
                   delimiter=',')
        np.savetxt(trueBinaryWeightsFileName, self.dataGenerationRegime.bivariateWeights, fmt='%.3f', delimiter=',')







    def recordResult(self, mcmcResult, dir_name, time_base_filename, samplingMethod, nMCMCIter, saveRateMtx=False):
        # currentIteration represents the current Iteration of the MCMC iteration. It is only not None when recordIteratively is true
        # iterationSize represents the number of iterations that we record each time. For example, if we record result every 50 iterations, iterationSize=50.
        # saveRateMtx is a boolean variable, if it is true, we save the posterior samples for the whole rate matrix
        # mcmcResult is the result after running the MCMC algorithm
        format = '.csv'
        timeFileName = os.path.join(dir_name, time_base_filename+samplingMethod+nMCMCIter+format)

        csv = open(timeFileName,"w")
        columnTitleRow = "elapsedTime\n"
        csv.write(columnTitleRow)
        row = mcmcResult['elapsingTime']
        csv.write(str(row))
        csv.close()

        stationarySamples = mcmcResult['stationaryDistSamples']
        stationaryWeightsSamples = mcmcResult['stationaryWeightsSamples']
        binaryWeightsSamples = mcmcResult['binaryWeightsSamples']
        exchangeableSamples = mcmcResult['exchangeableSamples']
        if self.saveRateMtx:
            rateMatrixSamples = mcmcResult['rateMatrixSamples']
        timeElapsed = mcmcResult['elapsingTime']

        stationaryDistStr = "stationaryDistribution"
        stationaryWeightsStr = "stationaryWeights"
        exchangeableCoefStr = "exchangeableCoef"
        binaryWeightsStr = "binaryWeights"

        trueStationaryDistStr = "trueStationaryDistribution"
        trueStationaryWeightsStr = "trueStationaryWeights"
        trueExchangeableCoefStr = "trueExchangeableCoef"
        trueBinaryWeightsStr = "trueBinaryWeights"


        stationaryDistFileName = os.path.join(dir_name, stationaryDistStr + samplingMethod + nMCMCIter + format)
        stationaryWeightsFileName = os.path.join(dir_name, stationaryWeightsStr + samplingMethod + nMCMCIter + format)
        exchangeableCoefFileName = os.path.join(dir_name, exchangeableCoefStr + samplingMethod + nMCMCIter + format)
        binaryWeightsFileName = os.path.join(dir_name, binaryWeightsStr + samplingMethod + nMCMCIter + format)

        trueStationaryDistFileName = os.path.join(dir_name, trueStationaryDistStr)
        trueExchangeableCoefFileName = os.path.join(dir_name, trueExchangeableCoefStr)
        trueStationaryWeightsFileName = os.path.join(dir_name, trueStationaryWeightsStr)
        trueBinaryWeightsFileName = os.path.join(dir_name, trueBinaryWeightsStr)


        np.savetxt(stationaryDistFileName, stationarySamples, fmt='%.3f', delimiter=',')
        np.savetxt(stationaryWeightsFileName, stationaryWeightsSamples, fmt='%.3f', delimiter=',')
        np.savetxt(exchangeableCoefFileName, exchangeableSamples, fmt='%.3f', delimiter=',')
        np.savetxt(binaryWeightsFileName, binaryWeightsSamples, fmt='%.3f', delimiter=',')

        ## save the stationary distribution and the exchangeable parameters
        np.savetxt(trueStationaryDistFileName, self.dataGenerationRegime.stationaryDist, fmt='%.3f', delimiter=',')
        np.savetxt(trueExchangeableCoefFileName, self.dataGenerationRegime.exchangeCoef, fmt='%.3f', delimiter=',')
        np.savetxt(trueStationaryWeightsFileName, self.dataGenerationRegime.stationaryWeights, fmt='%.3f', delimiter=',')
        np.savetxt(trueBinaryWeightsFileName, self.dataGenerationRegime.bivariateWeights, fmt='%.3f', delimiter=',')


        ## write the true stationary and bivariate weight to tiles

        if saveRateMtx:
            rateMtxStr = "rateMatrix"
            rateMtxFileName = os.path.join(dir_name, rateMtxStr + samplingMethod + nMCMCIter)
            np.save(rateMtxFileName, rateMatrixSamples)














