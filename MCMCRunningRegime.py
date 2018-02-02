import os
import sys

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
import OptionClasses
import ExpectedCompleteReversibleObjective
import ExpectedCompleteReversibleModelBinaryFactors
import HMC
import LocalRFSamplerForBinaryWeights
import PhyloLocalRFMove
import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
import FullTrajectorGeneration
import DataGenerationRegime
# from main.OptionClasses import MCMCOptions
# from main.OptionClasses import RFSamplerOptions
# from main.ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
# from main.ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
# from main.HMC import HMC
# from main.LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
# from main.PhyloLocalRFMove import PhyloLocalRFMove
# from main.ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import \
#     ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
# from main.FullTrajectorGeneration import endPointSamplerSummarizeStatisticsOneBt
from datetime import datetime
# from main.DataGenerationRegime import WeightGenerationRegime


class MCMCRunningRegime:

    def __init__(self, dataGenerationRegime, nMCMCIter, thinning, burnIn, onlyHMC, HMCPlusBPS, prng=None, nLeapFrogSteps=40,
                 stepSize=0.02, nHMCSamples=2000, saveRateMtx = False, initialSampleSeed=None, rfOptions=None,
                 dumpResultIteratively=False, dumpResultIterations = 50, dir_name=os.getcwd()):
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
        self.nHMCSamples = nHMCSamples
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
        self.nBivariateFeat = len(dataGenerationRegime.bivariateWeights)
        if HMCPlusBPS:
            self.samplingMethod = "HMCPlusBPS"
        if onlyHMC:
            self.samplingMethod = "HMC"

        self.trajectoryLength = self.rfOptions.trajectoryLength

        if initialSampleSeed is not None:
            self.initialSampleSeed = initialSampleSeed
        else:
            initialSampleSeed = 1
        if dumpResultIteratively:
            self.dumpResultIteratively = dumpResultIteratively
            self.dumpResultIterations = dumpResultIterations

        self.saveRateMtx = saveRateMtx
        self.dir_name = dir_name

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

    def generateInitialSamples(self, initialWeightsDist="uniform"):

        if initialWeightsDist == "Uniform":
            weights = self.generateInitialWeightsFromUniformDist()
        if initialWeightsDist == "Normal":
            weights = self.generateInitialWeightsFromNormalDist()
        if initialWeightsDist == "Fixed":
            weights = self.generateFixedInitialWeights()
        if initialWeightsDist is not None and not initialWeightsDist.__contains__("Uniform") and not initialWeightsDist.__contains__("Normal") and not initialWeightsDist.__contains__("Fixed"):
            raise Exception("The provided string for initial distribution of weights is not Uniform, Normal or fixed scalars")

        initialStationaryWeights = weights['initialStationaryWeights']
        initialBinaryWeights = weights['initialBinaryWeights']

        initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure.ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(self.nStates,
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
            exchangeableSamples = np.zeros((nMCMCIters, len(self.dataGenerationRegime.exchangeCoef)))
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


    def run(self, initialWeightDist ="Fixed"):

        # call methods to create initial samples
        initialSamples = self.generateInitialSamples(initialWeightsDist=initialWeightDist)
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

        stationaryTmp = np.zeros((self.dumpResultIterations, self.nStates))
        exchangeableTmp = np.zeros((self.dumpResultIterations, self.nBivariateFeat))

        for i in range(self.nMCMCIter):

            if self.onlyHMC:
                if i == 0:
                # initialize the posteriorSamples at the 0th iteration specifically for HMC
                    initializedPosteriorSamples = self.initializeHMCWeights(initialStationaryWeights, initialBinaryWeights)
                    weightSamples = initializedPosteriorSamples['weightSamples']
                    avgWeights = initializedPosteriorSamples['avgWeights']
                if i > 0:
                   weightSamples[i, :] = avgWeights

            stationaryWeightsSamples[i, :] = initialStationaryWeights
            binaryWeightsSamples[i, :] = initialBinaryWeights
            exchangeableSamples[i,:] = initialExchangeCoef
            if self.saveRateMtx:
                rateMatrixSamples[i, :] = initialRateMatrix
            stationaryDistSamples[i, :] = initialStationaryDist

            if i > 0 and (i+1) % self.dumpResultIterations == 0:
                ## record the results
                self.dumpResult(stationaryDistSamples[(i+1-self.dumpResultIterations):(i+1), :],   allFileNames['stationaryDist'])
                self.dumpResult(exchangeableSamples[(i+1-self.dumpResultIterations):(i+1), :], allFileNames['exchangeableCoef'])



            nInit = np.zeros(self.nStates)
            holdTime = np.zeros(self.nStates)
            nTrans = np.zeros((self.nStates, self.nStates))

            for j in range(self.dataGenerationRegime.nPairSeq):
                suffStat = FullTrajectorGeneration.endPointSamplerSummarizeStatisticsOneBt(True, self.prng, initialRateMatrix,
                                                                       self.data[j], self.dataGenerationRegime.interLength)
                nInit = nInit + suffStat['nInit']
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
            if self.onlyHMC:
                sample = self.prng.uniform(0, 1, len(avgWeights))
            if self.HMCPlusBPS:
                sample = self.prng.uniform(0, 1, self.nStates)

            samples = hmc.run(0, self.nHMCSamples, sample)
            avgWeights = np.sum(samples, axis=0) / samples.shape[0]

            if self.onlyHMC:
                initialStationaryWeights = avgWeights[0:self.nStates]
                initialBinaryWeights = avgWeights[self.nStates:(self.nStates + self.nBivariateFeat)]

            # sample stationary distribution elements using HMC
            if self.HMCPlusBPS:
                initialStationaryWeights = avgWeights
                # update stationary distribution elements to the latest value
                initialStationaryDist = np.exp(avgWeights) / np.sum(np.exp(avgWeights))
                # sample exchangeable coefficients using local bouncy particle sampler
                ## define the model
                model = ExpectedCompleteReversibleModelBinaryFactors.ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, self.nStates,
                                                                         initialBinaryWeights, initialStationaryDist,
                                                                         self.bivariateFeatIndexDictionary)
                ## define the sampler to use
                ## local sampler to use
                localSampler = LocalRFSamplerForBinaryWeights.LocalRFSamplerForBinaryWeights(model, self.rfOptions, self.mcmcOptions, self.nStates,
                                                              self.bivariateFeatIndexDictionary)
                phyloLocalRFMove = PhyloLocalRFMove(model=model, sampler=localSampler, initialPoints=initialBinaryWeights, options=self.rfOptions, randomSeed=self.initialSampleSeed)
                initialBinaryWeights = phyloLocalRFMove.execute()

            initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure.ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(self.nStates, initialStationaryWeights,
                                                                                           initialBinaryWeights,
                                                                                           bivariateFeatIndexDictionary=self.bivariateFeatIndexDictionary)

            initialStationaryDist = np.round(initialRateMtx.getStationaryDist(), 3)
            initialRateMatrix = np.round(initialRateMtx.getRateMtx(), 3)
            initialExchangeCoef = np.round(initialRateMtx.getExchangeCoef(), 3)
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
            nMCMCIter= str(nMCMCIter)
        timeFileName = os.path.join(dir_name, time_base_filename+samplingMethod+nMCMCIter+format)
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
        if saveRateMtx:
            rateMtxStr = "rateMatrix"
            rateMtxStrName = rateMtxStr + samplingMethod + samplingMethod + nMCMCIter

        stationaryDistStrName = stationaryDistStr + samplingMethod + nMCMCIter
        stationaryWeightsStrName = stationaryWeightsStr + samplingMethod + nMCMCIter
        exchangeableCoefStrName = exchangeableCoefStr + samplingMethod + nMCMCIter
        binaryWeightsStrName = binaryWeightsStr + samplingMethod + nMCMCIter

        if trajectoryLength is not None:
            stationaryDistStrName = stationaryDistStrName + "trajectoryLength"+ trajectoryLength
            stationaryWeightsStrName = stationaryWeightsStrName + "trajectoryLength"+  trajectoryLength
            exchangeableCoefStrName = exchangeableCoefStrName + "trajectoryLength"+ trajectoryLength
            binaryWeightsStrName = binaryWeightsStrName + "trajectoryLength" + trajectoryLength
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + "trajectoryLength"+ trajectoryLength


        if mcmcSeed is not None:
            stationaryDistStrName = stationaryDistStrName + "mcmcSeed" +mcmcSeed+ ".csv"
            stationaryWeightsStrName = stationaryWeightsStrName + "mcmcSeed" + mcmcSeed+ ".csv"
            exchangeableCoefStrName = exchangeableCoefStrName + "mcmcSeeed" + mcmcSeed+ ".csv"
            binaryWeightsStrName = binaryWeightsStrName +"mcmcSeed"+ mcmcSeed + ".csv"
            if saveRateMtx:
                rateMtxStrName = rateMtxStrName + mcmcSeed

        stationaryDistFileName = os.path.join(dir_name, stationaryDistStrName)
        stationaryWeightsFileName = os.path.join(dir_name, stationaryWeightsStrName)
        exchangeableCoefFileName = os.path.join(dir_name, exchangeableCoefStrName)
        binaryWeightsFileName = os.path.join(dir_name, binaryWeightsStrName)
        if saveRateMtx:
            rateMtxFileName = os.path.join(dir_name, rateMtxStrName)

        result = {}
        result['stationaryDist'] = stationaryDistFileName
        result['stationaryWeights'] = stationaryWeightsFileName
        result['exchangeableCoef'] = exchangeableCoefFileName
        result['binaryWeights'] = binaryWeightsFileName
        if saveRateMtx:
            result['rateMatrix'] =rateMtxFileName
        return result






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














