
import sys
import os
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
import datetime as dt
from collections import OrderedDict

## OrderedDict is equivalent to LinkedHashMap in java
import EventQueue
from TrajectoryRay import TrajectoryRay
from CollisionContext import CollisionContext
import Utils
import OptionClasses
import NormalFactor
import SojournTimeFactorWithoutPiEstWithBinaryFactors
import TransitionCountWithoutPiWithBinaryFactors

#from main.EventQueue import EventQueue
#from main.TrajectoryRay import TrajectoryRay
#from main.CollisionContext import CollisionContext
#from main import Utils
#from main.OptionClasses import RefreshmentMethod
import copy
#from main.Utils import getBivariateLocalFactorIndexDictOnlyExchangeCoef
#from main.NormalFactor import NormalFactor
#from main.SojournTimeFactorWithoutPiEstWithBinaryFactors import SojournTimeFactorWithoutPiEstWithBinaryFactors
#from main.TransitionCountWithoutPiWithBinaryFactors import TransitionCountFactorWithoutPiEstWithBinaryFactors

from numpy.random import RandomState


#from main.Utils import generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat
#from main.ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
#from main.ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors


epsilon = 1

def getBivariateLocalFactorIndexDictUsingStates(nStates, nBivariateWeights):

    nPrior = nBivariateWeights
    nSojourn = nStates * (nStates-1)
    nTransit = nStates * (nStates-1)
    result = Utils.getBivariateLocalFactorIndexDictOnlyExchangeCoef(nStates, nPrior, nSojourn, nTransit)
    return result

def getPairOfStatesConnectedToEachDimOfBivariateWeights(nBivariateFeatWeightsDictionary, nBivariateFeatures):

    result = OrderedDict()
    for i in range(nBivariateFeatures):
        ## loop over all pair of states
        keyList = list()
        for key in nBivariateFeatWeightsDictionary.keys():
            values = nBivariateFeatWeightsDictionary[key]
            if len(np.atleast_1d(values)) > 1:
                if i in values:
                    keyList.append(key)
                    result[i] = keyList
            else:
                if i ==0:
                    keyList.append(key)
                    result[np.asscalar(values)] = keyList

    return result

def getIndexOfBinaryFactorsGivenConnectedPairOfStates(allFactors, state0, state1):
    values = list()
    cleaned_values = list()

    for index, factor in enumerate(allFactors):
        if hasattr(factor, 'state0') and hasattr(factor, 'state1'):
            if factor.state0 == state0 and factor.state1 == state1:
                values.append(index)

    ## remove duplicate values in "values"
    if len(np.atleast_1d(values)) > 1:
        cleaned_values = list(set(values))
    else:
        cleaned_values.append(values)

    return cleaned_values

def getIndexOfNeighborFactorsGivenIndexOfBinaryFeature(nBivariateFeatIndex, nBivariateFeatWeightsDictionary, nBinaryFeatures, allFactors):

    ## get the pair of states connected to the current element of the weights for the bivariate features
    allPairOfStatesForAllFeatures = getPairOfStatesConnectedToEachDimOfBivariateWeights(nBivariateFeatWeightsDictionary, nBinaryFeatures)
    if nBivariateFeatIndex not in allPairOfStatesForAllFeatures.keys():
        print(allPairOfStatesForAllFeatures)
    allPairOfStates = allPairOfStatesForAllFeatures[nBivariateFeatIndex]

    ## get the factors connected to those pair of states and the univariate features

    indexList = list()
    for pairOfStates in allPairOfStates:
        state0 = pairOfStates[0]
        state1 = pairOfStates[1]
        indexList.extend(getIndexOfBinaryFactorsGivenConnectedPairOfStates(allFactors, state0, state1))
    indexList.append(nBivariateFeatIndex)

    return indexList





def getIndexOfBinaryFactorsGivenConnectedPairOfStatesGivenNumberOfStates(allFactors, nStates):

    wholeStates = np.arange(0, nStates)
    result = {}
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            keyPair = (state0, state1)
            result[keyPair] = getIndexOfBinaryFactorsGivenConnectedPairOfStates(allFactors, state0, state1)

    return result




def getAllIndexOfBinaryFactorsGivenConnectedPairOfStates(allFactors, nBivariateFeatWeightsDictionary, nBivariateFeatures, nStates):

    indToPairs = getPairOfStatesConnectedToEachDimOfBivariateWeights(nBivariateFeatWeightsDictionary, nBivariateFeatures)
    pairStatesToFactorIndex = getIndexOfBinaryFactorsGivenConnectedPairOfStatesGivenNumberOfStates(allFactors, nStates)

    result = OrderedDict()
    for i in range(nBivariateFeatures):
        values = list()
        pairList = indToPairs[i]
        for tuplePair in pairList:
            values.extend(pairStatesToFactorIndex[tuplePair])

        if len(np.atleast_1d(values)) > 1:
            cleaned_values = list(set(values))
        else:
            cleaned_values = list()
            cleaned_values.append(values)
        cleaned_values.append(i)
        result[i] = cleaned_values

    return result

def neighbourVariables(nStates, collisionFactor, nBivariateFeatWeightsDictionary):
    result = list()
    if isinstance(collisionFactor, NormalFactor.NormalFactor):
        result.append(collisionFactor.gradientIndex)
        return result
    if isinstance(collisionFactor, SojournTimeFactorWithoutPiEstWithBinaryFactors.SojournTimeFactorWithoutPiEstWithBinaryFactors) or isinstance(collisionFactor, TransitionCountWithoutPiWithBinaryFactors.TransitionCountFactorWithoutPiEstWithBinaryFactors):
        state0 = collisionFactor.state0
        state1 = collisionFactor.state1
        keyPair = (state0, state1)
        values = nBivariateFeatWeightsDictionary[keyPair]
        if len(np.atleast_1d(values)) > 1:
            result.extend(values)
        else:
            result.append(np.asscalar(values))
        return result



def neighborVariableForAllFactors(nStates, allFactors, nBivariateFeatWeightsDictionary):
    result = OrderedDict()
    ## change dictionary key from using factor to use the index of the factor
    for index, factor in enumerate(allFactors):
        result[index] = neighbourVariables(nStates, factor, nBivariateFeatWeightsDictionary)
    return result

def getIndexOfNeighborFactorsForEachIndexOfBinaryFeature(nBivariateFeatWeightsDictionary,
                                                         nBivariateFeatures, allFactors):
    result = dict()
    for i in range(nBivariateFeatures):
        values = getIndexOfNeighborFactorsGivenIndexOfBinaryFeature(i,
                                                                    nBivariateFeatWeightsDictionary,
                                                                    nBivariateFeatures, allFactors)
        result[i] = values
    return result



def neighbourVariblesAndFactorsAndExtendedNeighborsOfAllFactorsDict(nStates, allFactors, nBivariateFeatWeightsDictionary, nBivariateFeatures):

    immediateNeighborVariablesIndexForAllFactors = OrderedDict()
    neighborFactorListForAllFactors = OrderedDict()
    extendedNeighborVariablesIndexForAllFactors = OrderedDict()

    for indexOfFactor, factor in enumerate(allFactors):
        immediateNeighborVariablesIndexInTest = neighbourVariables(nStates, factor,
                                                                nBivariateFeatWeightsDictionary)
        immediateNeighborVariablesIndexForAllFactors[indexOfFactor] = immediateNeighborVariablesIndexInTest
        neighborFactorList = list()
        for immediateVariable in immediateNeighborVariablesIndexInTest:
            values = getIndexOfNeighborFactorsGivenIndexOfBinaryFeature(immediateVariable,
                                                                        nBivariateFeatWeightsDictionary,
                                                                        nBivariateFeatures, allFactors)
            if len(np.atleast_1d(values)) > 1:
                neighborFactorList.extend(values)
            else:
                neighborFactorList.append(np.asscalar(values))

        if len(np.atleast_1d(neighborFactorList)) > 1:
            neighborFactorList = list(set(neighborFactorList))

        neighborFactorListForAllFactors[indexOfFactor] = neighborFactorList

        extendedNeighborVariablesIndex = list()
        for index in neighborFactorList:
            currentNeighborVariables = neighbourVariables(nStates, allFactors[index],
                                                            nBivariateFeatWeightsDictionary)
            extendedNeighborVariablesIndex.extend(currentNeighborVariables)

        ## remove duplicate elements
        if len(extendedNeighborVariablesIndex) > 1:
            extendedNeighborVariablesIndex = list(set(extendedNeighborVariablesIndex))

        extendedNeighborVariablesIndexForAllFactors[indexOfFactor] = extendedNeighborVariablesIndex

    result = {}
    result['neighborVariablesAll'] = immediateNeighborVariablesIndexForAllFactors
    result['neighborFactorsAll'] = neighborFactorListForAllFactors
    result['extendedVariablesAll'] = extendedNeighborVariablesIndexForAllFactors
    return result





# ## check the correctness of all the previous methods
# nStates = 4
# nBivariateFeat = 10
# nChoosenFeatureRatio = 0.3
# bivariateWeights = np.random.normal(0, 1, nBivariateFeat)
# exchangeableCoef = np.exp(bivariateWeights)
# stationaryWeights = np.random.normal(0, 1, 4)
# stationaryDist = np.exp(stationaryWeights)/np.sum(np.exp(stationaryWeights))
# bivariateFeatIndexDictionary = generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat(nStates, nBivariateFeat, nChoosenFeatureRatio)
# print(bivariateFeatIndexDictionary)
# expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(5*np.ones(4), 5*np.ones(4), np.zeros((4,4)), exchangeableCoef)
# expectedCompleteReversibleObjectiveModel = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates, bivariateWeights, stationaryDist, bivariateFeatIndexDictionary)
# allFactors = expectedCompleteReversibleObjectiveModel.localFactors
# ## test the correctness of the following two functions
# result = getAllIndexOfBinaryFactorsGivenConnectedPairOfStates(allFactors, bivariateFeatIndexDictionary, nBivariateFeat, nStates)
# OrderedDict(bivariateFeatIndexDictionary)
# result
#
# ## select a Normal factor, a sojournTimeFactor and transitionCountFactor
# collisionFactor1 = allFactors[0]
# collisionFactor2 = allFactors[10]
# collisionFactor3 = allFactors[25]
# neighbourVariables1 = neighborVariables(nStates, collisionFactor1, bivariateFeatIndexDictionary)
# neighbourVariables1
#
# neighbourVariables2 = neighborVariables(nStates, collisionFactor2, bivariateFeatIndexDictionary)
# neighbourVariables2
# bivariateFeatIndexDictionary[(collisionFactor2.state0, collisionFactor2.state1)]
#
# neighbourVariables3 = neighborVariables(nStates, collisionFactor3, bivariateFeatIndexDictionary)
# neighbourVariables3



class LocalRFSamplerForBinaryWeights:
    """Make model a class of ExpectedCompleteReversibleModel"""

    def __init__(self, model, rfOptions, mcmcOptions, nStates,  neighborVariableForAllFactors, variableAndFactorInfo, indexOfFactorsForEachBivariateFeat):
        self.model = model
        self.rfOptions = rfOptions
        self.mcmcOptions = mcmcOptions
        self.allFactors = model.localFactors
        self.collisionQueue = EventQueue.EventQueue()
        self.isCollisionMap = OrderedDict()
        self.trajectories = {}
        self.nCollisions = 0
        self.nCollidedVariables = 0
        self.nRefreshments = 0
        self.nRefreshedVariables = 0
        ## this defines the global time
        self.currentTime = 0
        self.nStates = nStates

        self.nBivariateFeatWeightsDictionary = model.bivariateFeatIndexDictionary
        self.nBivariateFeatures = len(model.variables)
        self.neighborVariablesForAllFactors = neighborVariableForAllFactors

        ## neighborVariable, neighborFactors and extendedVariable information for all factors
        self.variableAndFactorInfo = variableAndFactorInfo
        self.neighborVariablesForAllFactors = variableAndFactorInfo['neighborVariablesAll']
        self.neighborFactorsAll = variableAndFactorInfo['neighborFactorsAll']
        self.extendedVariablesAll = variableAndFactorInfo['extendedVariablesAll']
        self.indexOfFactorsForEachBivariateFeat = indexOfFactorsForEachBivariateFeat



    def currentVelocity(self, variables):
        ## variables represent the parameters to be estimated
        dimensionality = len(np.atleast_1d(variables))
        result = np.zeros(dimensionality)
        for d in range(dimensionality):
            result[d] = self.trajectories[d].velocity_t
        return result

    def initTrajectory(self, refreshmentTime, variable, variableIndex, currentVelocity):
        ## variableIndex stores an integer, it is the index of the current variable in the vector of model.variables
        ## variable provides the value of one element of the parameter
        if variableIndex in self.trajectories.keys():
            raise ValueError("The current trajectory contains the variable as one of the key values")
        self.trajectories[variableIndex] = TrajectoryRay(refreshmentTime, variable, currentVelocity)

    def updateTrajectory(self, time, variableIndex, newVelocity):
        oldRay = self.trajectories[variableIndex]
        newPosition = oldRay.position(time)
        newRay = TrajectoryRay(time, newPosition, newVelocity)
        self.trajectories[variableIndex] = newRay
        # processRay(variable, oldRay, time)

    def getVelocityMatrix(self, collisionFactor):
        length = len(np.atleast_1d(collisionFactor.variables))
        result = np.zeros(length)
        for i in range(length):
            result[i] = self.trajectories[i].velocity_t
        return result

    def updateVariable(self, variableIndex, currentTime):
        ray = self.trajectories[variableIndex]
        variable = ray.position(currentTime)
        return variable

    def updateAllVariables(self, currentTime):
        for index in range(len(self.model.variables)):
            self.model.variables[index] = self.updateVariable(index, currentTime)
        return self.model.variables

    def collideTrajectories(self, collisionFactor, collisionTime):
        gradient = collisionFactor.gradient()
        oldVelocity = self.getVelocityMatrix(collisionFactor)
        newVelocity = Utils.bounce(oldVelocity, gradient)

        length = len(np.atleast_1d(newVelocity))
        for i in range(length):
            #variable = collisionFactor.getVariable(i)
            newVelocityCoordinate = newVelocity[i]
            self.updateTrajectory(collisionTime, i, newVelocityCoordinate)

    def updateCandidateCollision(self, prng,  collisionFactorIndex, currentTime):
        collisionFactor = self.allFactors[collisionFactorIndex]
        self.collisionQueue.remove(collisionFactor)
        context = CollisionContext(prng, self.getVelocityMatrix(collisionFactor))

        collisionInfo = collisionFactor.getLowerBoundForCollisionDeltaTime(context)
        deltaTime = collisionInfo['deltaTime']
        if isinstance(deltaTime, np.ndarray):
            deltaTime = np.asarray(deltaTime, dtype=np.float)[0]
        candidateCollisionTime = currentTime + deltaTime
        if isinstance(candidateCollisionTime, np.ndarray):
            candidateCollisionTime = np.asscalar(candidateCollisionTime[0])
        if candidateCollisionTime < 0:
            raise ValueError("The collision time can't be smaller than zero.")

        self.isCollisionMap[collisionFactor] = collisionInfo['collision']

        if self.collisionQueue.containsTime(candidateCollisionTime):
            print('The sampler has hit an event of probability zero: two collisions scheduled exactly at the same time.')
            print('Because of numerical precision, this could possibly happen, but very rarely.')
            print('For internal implementation reasons, one of the collisions at time' + ' ' + str(
                candidateCollisionTime) + ' was moved to' + ' ' + str(candidateCollisionTime + epsilon))
            candidateCollisionTime += epsilon

        self.collisionQueue.add(collisionFactorIndex, candidateCollisionTime)
        # self.collisionQueue.add(collisionFactor, candidateCollisionTime)

    def neighbourVariblesAndFactorsAndExtendedNeighborsOfAllFactorsDict(self):

        immediateNeighborVariablesIndexForAllFactors = OrderedDict()
        neighborFactorListForAllFactors = OrderedDict()
        extendedNeighborVariablesIndexForAllFactors = OrderedDict()

        for factor in self.allFactors:
            immediateNeighborVariablesIndexInTest = neighbourVariables(self.nStates, factor,
                                                                       self.nBivariateFeatWeightsDictionary)
            immediateNeighborVariablesIndexForAllFactors[factor] = immediateNeighborVariablesIndexInTest
            neighborFactorList = list()
            for immediateVariable in immediateNeighborVariablesIndexInTest:
                values = getIndexOfNeighborFactorsGivenIndexOfBinaryFeature(immediateVariable,
                                                                            self.nBivariateFeatWeightsDictionary,
                                                                            self.nBivariateFeatures, self.allFactors)
                if len(np.atleast_1d(values)) > 1:
                    neighborFactorList.extend(values)
                else:
                    neighborFactorList.append(np.asscalar(values))

            if len(np.atleast_1d(neighborFactorList)) > 1:
                neighborFactorList = list(set(neighborFactorList))

            neighborFactorListForAllFactors[factor] = neighborFactorList

            extendedNeighborVariablesIndex = list()
            for index in neighborFactorList:
                currentNeighborVariables = neighbourVariables(self.nStates, self.allFactors[index],
                                                              self.nBivariateFeatWeightsDictionary)
                extendedNeighborVariablesIndex.extend(currentNeighborVariables)

            ## remove duplicate elements
            if len(extendedNeighborVariablesIndex) > 1:
                extendedNeighborVariablesIndex = list(set(extendedNeighborVariablesIndex))

            extendedNeighborVariablesIndexForAllFactors[factor] = extendedNeighborVariablesIndex

        result = {}
        result['neighborVariablesAll'] = immediateNeighborVariablesIndexForAllFactors
        result['neighborFactorsAll'] = neighborFactorListForAllFactors
        result['extendedVariablesAll'] = extendedNeighborVariablesIndexForAllFactors
        return result


    def doCollision(self, prng):

        collision = self.collisionQueue.pollEvent()
        ## get the key values of the popitem of collisionQueue where collision is a pair with keys and values
        collisionTime = collision[0]
        collisionFactorIndex = collision[1]
        collisionFactor = self.allFactors[collisionFactorIndex]
        ## print("The collision factor is" + collisionFactor)

        isActualCollision = self.isCollisionMap[collisionFactor]

        ## get the variables connected to the current factor
        immediateNeighborVariablesIndex = self.neighborVariablesForAllFactors[collisionFactorIndex]
        neighborFactorList = self.neighborFactorsAll[collisionFactorIndex]
        extendedNeighborVariablesIndex = self.extendedVariablesAll[collisionFactorIndex]

        self.nCollisions += 1
        self.nCollidedVariables += len(np.atleast_1d(immediateNeighborVariablesIndex))

        for index in extendedNeighborVariablesIndex:
            self.model.variables[index] = self.updateVariable(index, collisionTime)

        if isActualCollision:
            self.collideTrajectories(collisionFactor, collisionTime)

        if isActualCollision:
            for i in neighborFactorList:
                self.updateCandidateCollision(prng, i, collisionTime)
        else:
            self.updateCandidateCollision(prng, collisionFactorIndex, np.asscalar(collisionTime))


    def localVelocityRefreshment(self, prng, refreshmentTime):
        ## sample a factor
        allFactors = self.allFactors
        ## sample the index of the factor
        #ind = int(np.asscalar(prng.randint(0, len(np.atleast_1d(allFactors)), 1)))
        ind = int(np.asscalar(np.random.randint(0, len(np.atleast_1d(allFactors)), 1)))
        f = allFactors[ind]  ## f is a collision factor

        ## using the cached results for all factors
        immediateNeighborVariablesIndex = self.neighborVariablesForAllFactors[f]

        if len(np.atleast_1d(immediateNeighborVariablesIndex)) == 1:
            increasedNeighborhood = set()
            increasedNeighborhood.update(immediateNeighborVariablesIndex)

            #f2 = allFactors[int(np.asscalar(prng.randint(0, len(allFactors), 1)))]
            f2 = allFactors[int(np.asscalar(np.random.randint(0, len(allFactors), 1)))]
            immediateNeighborVariablesIndex2 = self.neighborVariablesForAllFactors[f2]

            increasedNeighborhood.update(immediateNeighborVariablesIndex2)

            if len(increasedNeighborhood) > 1:
                immediateNeighborVariablesIndex = list(set(increasedNeighborhood))
            else:
                immediateNeighborVariablesIndex = list(increasedNeighborhood)

        neighborFactors = list()
        for immediateVariable in immediateNeighborVariablesIndex:
            ## ToDo: cache the results instead of evaluating it every time when updating the velocities
            values = self.indexOfFactorsForEachBivariateFeat[immediateVariable]

                #getIndexOfNeighborFactorsGivenIndexOfBinaryFeature(immediateVariable,
                #                                                        self.nBivariateFeatWeightsDictionary,
                #                                                        self.nBivariateFeatures, self.allFactors)
            neighborFactors.extend(values)
        if len(neighborFactors) > 1:
            neighborFactors = list(set(neighborFactors))

        ## get the variables connected to the neighbor factors
        extendedNeighborVariablesIndex = list()
        for index in neighborFactors:
            ## ToDo: cache the results, intead of evaluating it every time when updating the velocities
            neighborVariables = self.neighborVariablesForAllFactors[index]

                #neighbourVariables(self.nStates, self.allFactors[index],
                #                                   self.nBivariateFeatWeightsDictionary)
            extendedNeighborVariablesIndex.extend(neighborVariables)

        if len(extendedNeighborVariablesIndex) > 1:
            extendedNeighborVariablesIndex = list(set(extendedNeighborVariablesIndex))


        self.nRefreshments += 1
        self.nRefreshedVariables += len(np.atleast_1d(immediateNeighborVariablesIndex))

        ## sample new velocity vector
        #newVelocity = prng.normal(size=len(np.atleast_1d(immediateNeighborVariablesIndex)))
        newVelocity = np.random.normal(size=len(np.atleast_1d(immediateNeighborVariablesIndex)))
        if len(np.atleast_1d(immediateNeighborVariablesIndex))==1:
            newVelocity = np.array([newVelocity])



        for index, variable in enumerate(np.array(self.model.variables)[extendedNeighborVariablesIndex]):
            oldValue = variable
            variableIndex = extendedNeighborVariablesIndex[index]
            self.model.variables[variableIndex] = self.updateVariable(variableIndex, refreshmentTime)
            newKey = self.model.variables[variableIndex]

            ## update the trajectory rays
            if newKey != oldValue:
                self.trajectories[variableIndex].position_t = newKey
                self.trajectories[variableIndex].t = refreshmentTime


        ## 2-update rays for variables in immediate neighborhood (and process), update the velocity
        d = int(0)

        if len(np.atleast_1d(immediateNeighborVariablesIndex)) > 1:
            for variableIndex in immediateNeighborVariablesIndex:
                self.trajectories[variableIndex].velocity_t = newVelocity[int(d)]
                # self.updateTrajectory(refreshmentTime, variable, newVelocity[int(d)])
                d += 1
        else:
            self.trajectories[immediateNeighborVariablesIndex[0]].velocity_t = newVelocity[int(d)]
            d += 1



        ## 3-recompute the collisions for the other factors touching the variables (including the one we just popped)
        for element in neighborFactors:
            factor = self.allFactors[element]
            self.updateCandidateCollision(prng, element, np.asscalar(refreshmentTime))


    def globalVelocityRefreshment(self, prng,  refreshmentTime, initializing):
        ## initializing should be a boolean variable
        variables = self.model.variables
        dimensionality = len(variables)

        if initializing:
            if self.rfOptions.refreshmentMethod == OptionClasses.RefreshmentMethod.RESTRICTED:
                newVelocity = Utils.uniformOnUnitBall(prng, dimensionality)
            else:
                #newVelocity = prng.normal(0, 1, dimensionality)
                newVelocity = np.random.normal(0, 1, dimensionality)
        else:
            if self.rfOptions.refreshmentMethod == OptionClasses.RefreshmentMethod.GLOBAL or self.rfOptions.refreshmentMethod == OptionClasses.RefreshmentMethod.LOCAL:
                #newVelocity = prng.normal(0, 1, dimensionality)
                newVelocity = np.random.normal(0, 1, dimensionality)
            elif self.rfOptions.refreshmentMethod == OptionClasses.RefreshmentMethod.RESTRICTED:
                newVelocity = Utils.uniformOnUnitBall(prng, dimensionality)
            elif self.rfOptions.refreshmentMethod == OptionClasses.RefreshmentMethod.PARTIAL:
                if initializing:
                    newVelocity = Utils.uniformOnUnitBall(prng, dimensionality)
                else:
                    newVelocity = Utils.partialRefreshmentBetaAngle(prng, self.currentVelocity(variables),
                                                                    self.rfOptions.alpha, self.rfOptions.beta)
            else:
                raise ValueError()

        self.nRefreshments += 1
        self.nRefreshedVariables += len(variables)

        for i in range(dimensionality):
            variable = variables[i]
            currentVelocity = newVelocity[i]
            oldValue = variable
            if initializing:
                self.initTrajectory(refreshmentTime, variable, i, currentVelocity)
                self.model.variables[i] = self.updateVariable(i, refreshmentTime)
            else:
                self.model.variables[i] = self.updateVariable(i, refreshmentTime)
                newKey = self.model.variables[i]
                if newKey != oldValue:
                    self.trajectories[i].position_t = newKey
                    self.trajectories[i].t = refreshmentTime
                    self.trajectories[i].velocity_t = currentVelocity

        for indexOfFactor, factor in enumerate(self.allFactors):
            self.updateCandidateCollision(prng, indexOfFactor, refreshmentTime)

    def iterate(self, prng,  maxNumberOfIterations, maxTrajectoryLen, maxTimeMilli=sys.maxsize):
        # randomState is the state of the "random" pseudo number generator, it is obtained by first setting the
        # the seed of "random" module

        # sys.maxsize equivalent to Long.MAX_Value
        if self.currentTime > 0:
            raise ValueError("LocalRFSampler.iterate() currently does not support being called several "
                             + "times on the same instance. Create another object.")

        # try to figure out what values maxTimeMilli should take
        if maxTimeMilli == sys.maxsize:
            createWatch = False
        else:
            createWatch = True
            # create a stopwatch in python
            startTime = dt.datetime.now()

        self.globalVelocityRefreshment(prng, 0.0, True)

        # figure out what rayProcessor is doing and see if I need to implement that class
        if np.isclose(self.rfOptions.refreshRate, 0):
            nextRefreshmentTime = np.inf
        else:
            # nextRefreshmentTime = prng.exponential(scale=1 / self.rfOptions.refreshRate, size=1)
            nextRefreshmentTime = np.random.exponential(scale=1 / self.rfOptions.refreshRate, size=1)

        for i in range(maxNumberOfIterations):
            watchEndTime = dt.datetime.now()
            if createWatch:
                timeElapsed = watchEndTime - startTime
                if timeElapsed > maxTimeMilli:
                    break

            nextCollisionTime = self.collisionQueue.peekTime()
            nextEventTime = np.min((nextCollisionTime, nextRefreshmentTime))

            if nextEventTime > maxTrajectoryLen:
                self.currentTime = maxTrajectoryLen
                break

            if nextCollisionTime < nextRefreshmentTime:
                self.doCollision(prng)
                self.currentTime = nextCollisionTime
            else:
                nextRefreshmentTimeCopy = copy.deepcopy(nextRefreshmentTime)
                self.currentTime = nextRefreshmentTimeCopy
                if isinstance(self.currentTime, np.ndarray):
                    self.currentTime = np.asscalar(self.currentTime)
                if self.rfOptions.refreshmentMethod == OptionClasses.RefreshmentMethod.LOCAL:
                    self.localVelocityRefreshment(prng, nextRefreshmentTimeCopy)
                else:
                    self.globalVelocityRefreshment(prng, nextRefreshmentTimeCopy, False)
                nextRefreshmentTime += np.random.exponential(scale=1 / self.rfOptions.refreshRate, size=1)
                #nextRefreshmentTime += prng.exponential(scale=1 / self.rfOptions.refreshRate, size=1)
            #print(self.currentTime)
            print(i)

        # self.model.variables = self.updateAllVariables(self.currentTime)
        # change the keys of the trajectory to the new keys
        for index, var in enumerate(self.model.variables):
            #oldKey = self.model.variables[index]
            self.model.variables[index] = self.updateVariable(index, self.currentTime)

        # Ensure that the remaining rays are processed
        self.globalVelocityRefreshment(prng, self.currentTime, False)

        return self.model.variables

    def getTrajectoryLength(self):
        return self.currentTime

    def getNCollisions(self):
        return self.nCollisions

    def getNCollidedVariables(self):
        return self.nCollidedVariables

    def getNRefreshments(self):
        return self.nRefreshments

    def getNRefreshedVariables(self):
        return self.nRefreshedVariables
