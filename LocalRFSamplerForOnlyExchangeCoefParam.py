import os
import sys
import sys
sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
import datetime as dt
from collections import OrderedDict  ## OrderedDict is equivalent to LinkedHashMap in java
from main.EventQueue import EventQueue
from main.TrajectoryRay import TrajectoryRay
from main.CollisionContext import CollisionContext
from main import Utils
from main.OptionClasses import RefreshmentMethod
import copy

epsilon = 1


class LocalRFSamplerForOnlyExchangeCoefParam:
    """Make model a class of ExpectedCompleteReversibleModel"""

    def __init__(self, model, rfOptions, mcmcOptions, allFactors, nStates):
        self.model = model
        self.rfOptions = rfOptions
        self.mcmcOptions = mcmcOptions
        self.allFactors = allFactors
        self.collisionQueue = EventQueue()
        self.isCollisionMap = OrderedDict()
        self.trajectories = {}
        self.nCollisions = 0
        self.nCollidedVariables = 0
        self.nRefreshments = 0
        self.nRefreshedVariables = 0
        ## this defines the global time
        self.currentTime = 0
        self.nStates = nStates

        self.bivariateIndDict = Utils.getBivariateLocalFactorIndexDictForExchangeCoefUsingnStates(self.nStates)
        self.neighborFactorsDict = Utils.neighborFactorsCTMCDictForOnlyExchange(self.nStates, self.allFactors, self.bivariateIndDict)
        self.neighborFactorsVarIndAsKeys = self.neighborFactorsDict['FactorsDictWithVarIndAsKeys']

    def currentVelocity(self, variables):
        ## variables represent the parameters to be estimated
        dimensionality = len(variables)
        result = np.zeros(dimensionality)
        for d in range(dimensionality):
            result[d] = self.trajectories[variables[d]].velocity_t
        return result

    def initTrajectory(self, refreshmentTime, variable, currentVelocity):
        ## variable provides the value of one element of the parameter
        if variable in self.trajectories.keys():
            raise ValueError("The current trajectory contains the variable as one of the key values")
        self.trajectories[variable] = TrajectoryRay(refreshmentTime, variable, currentVelocity)

    def updateTrajectory(self, time, variable, newVelocity):
        oldRay = self.trajectories[variable]
        newPosition = oldRay.position(time)
        newRay = TrajectoryRay(time, newPosition, newVelocity)
        self.trajectories[variable] = newRay
        # processRay(variable, oldRay, time)

    def getVelocityMatrix(self, collisionFactor):
        length = collisionFactor.nVariables()
        result = np.zeros(length)
        for i in range(length):
            result[i] = self.trajectories[collisionFactor.getVariable(i)].velocity_t
        return result

    def updateVariable(self, variable, currentTime):
        variable = variable
        ray = self.trajectories[variable]
        variable = ray.position(currentTime)
        return variable

    def updateAllVariables(self, currentTime):
        for index, var in enumerate(self.model.variables):
            self.model.variables[index] = self.updateVariable(var, currentTime)
        return self.model.variables

    def collideTrajectories(self, collisionFactor, collisionTime):
        gradient = collisionFactor.gradient()
        oldVelocity = self.getVelocityMatrix(collisionFactor)
        newVelocity = Utils.bounce(oldVelocity, gradient)

        length = len(newVelocity)
        for i in range(length):
            variable = collisionFactor.getVariable(i)
            newVelocityCoordinate = newVelocity[i]
            self.updateTrajectory(collisionTime, variable, newVelocityCoordinate)

    def updateCandidateCollision(self, collisionFactor, currentTime):
        self.collisionQueue.remove(collisionFactor)
        context = CollisionContext(self.getVelocityMatrix(collisionFactor))

        collisionInfo = collisionFactor.getLowerBoundForCollisionDeltaTime(context)
        candidateCollisionTime = currentTime + collisionInfo['deltaTime']
        if candidateCollisionTime < 0:
            raise ValueError("The collision time can't be smaller than zero.")

        self.isCollisionMap[collisionFactor] = collisionInfo['collision']

        if self.collisionQueue.containsTime(candidateCollisionTime):
            print('The sampler has hit an event of probability zero: two collisions scheduled exactly at the same time.')
            print('Because of numerical precision, this could possibly happen, but very rarely.')
            print('For internal implementation reasons, one of the collisions at time' + ' ' + str(
                candidateCollisionTime) + ' was moved to' + ' ' + str(candidateCollisionTime + epsilon))
            candidateCollisionTime += epsilon

        self.collisionQueue.add(collisionFactor, candidateCollisionTime)

    def doCollision(self):

        collision = self.collisionQueue.pollEvent()
        ## get the key values of the popitem of collisionQueue where collision is a pair with keys and values
        collisionTime = collision[0]
        collisionFactor = collision[1]
        ## print("The collision factor is" + collisionFactor)

        isActualCollision = self.isCollisionMap[collisionFactor]

        immediateNeighborVariablesIndex = Utils.neighborVariablesCTMCWithOnlyExchangeCoef(self.nStates, collisionFactor)
        # this is only true for the CTMC model with the exchangeable coefficients as the variables
        extendedNeighborVariablesIndex = immediateNeighborVariablesIndex

        currentNeighborFactors = Utils.neighborFactors(self.neighborFactorsVarIndAsKeys,
                                                       immediateNeighborVariablesIndex)

        #extendedNeighborVariablesIndex = Utils.neighborVariablesWithOnlyExchangeCoef(self.nStates,
        #                                                         currentNeighborFactors)

        self.nCollisions += 1
        self.nCollidedVariables += len(immediateNeighborVariablesIndex)

        ## oldKeys and extendedNeighborVariablesIndex only contains one variable, so that we can use the following code
        oldKeys = self.model.variables[extendedNeighborVariablesIndex]

        for index, variable in enumerate(self.model.variables[extendedNeighborVariablesIndex]):
            self.model.variables[extendedNeighborVariablesIndex[index]] = self.updateVariable(variable, collisionTime)

            ## TODO: think more about how to deal with the trajectories and how to update the trajectories
            self.trajectories[self.model.variables[extendedNeighborVariablesIndex[index]]] = self.trajectories[np.asscalar(oldKeys)]
            del self.trajectories[np.asscalar(oldKeys)]

            ## TODO: need to double check if this update is necessary
            # ## self.trajectories[self.model.variables[extendedNeighborVariablesIndex[index]]].position_t = self.trajectories[np.asscalar(oldKeys)].position(collisionTime)
            ## self.trajectories[self.model.variables[extendedNeighborVariablesIndex[index]]].t = collisionTime

            ## check if I need to update the value of the variables in all the corresponding factors
            ## we should update all the variables with index extendedNeighborVariablesIndex in all factors

            for factor in self.allFactors:
                factor.variables[extendedNeighborVariablesIndex] = self.model.variables[extendedNeighborVariablesIndex]

        if isActualCollision:
            self.collideTrajectories(collisionFactor, collisionTime)

        if isActualCollision:
            for factor in currentNeighborFactors:
                self.updateCandidateCollision(factor, collisionTime)
        else:
            self.updateCandidateCollision(collisionFactor, collisionTime)


    def localVelocityRefreshment(self, refreshmentTime):
        ## sample a factor
        allFactors = self.allFactors
        ## sample the index of the factor
        ind = int(np.asscalar(np.random.randint(0, len(allFactors), 1)))
        f = allFactors[ind]  ## f is a collision factor
        immediateNeighborVariablesIndex = Utils.neighborVariablesCTMCWithOnlyExchangeCoef(self.nStates, f)

        if len(immediateNeighborVariablesIndex) == 1:
            ## ensure irreducibility for cases where some factor is connected to only one variable
            increasedNeighborhood = set()
            increasedNeighborhood.update(immediateNeighborVariablesIndex)

            f2 = allFactors[int(np.asscalar(np.random.randint(0, len(allFactors), 1)))]
            immediateNeighborVariablesIndex2 = Utils.neighborVariablesCTMCWithOnlyExchangeCoef(self.nStates, f2)
            increasedNeighborhood.update(immediateNeighborVariablesIndex2)

            immediateNeighborVariablesIndex = list(increasedNeighborhood)

        neighborFactors = Utils.neighborFactors(self.neighborFactorsVarIndAsKeys, immediateNeighborVariablesIndex)
        extendedNeighborVariablesIndex = Utils.neighborVariablesWithOnlyExchangeCoef(self.nStates, neighborFactors)

        self.nRefreshments += 1
        self.nRefreshedVariables += len(immediateNeighborVariablesIndex)

        ## sample new velocity vector
        newVelocity = np.random.normal(size=len(immediateNeighborVariablesIndex))
        if len(immediateNeighborVariablesIndex)==1:
            newVelocity = np.array([newVelocity])

        for index, variable in enumerate(self.model.variables[extendedNeighborVariablesIndex]):
            oldValue = variable
            self.model.variables[extendedNeighborVariablesIndex[index]] = self.updateVariable(variable, refreshmentTime)
            newKey = self.model.variables[extendedNeighborVariablesIndex[index]]
            if newKey != oldValue:
                self.trajectories[newKey] = self.trajectories[oldValue]
                del self.trajectories[oldValue]



            ## check if I need to update the value of the variables in all the corresponding factors
            ## we should update all the variables with index extendedNeighborVariablesIndex in all factors
            for factor in self.allFactors:
                factor.variables[extendedNeighborVariablesIndex] = self.model.variables[extendedNeighborVariablesIndex]

                ## 2-update rays for variables in immediate neighborhood (and process)
        d = int(0)
        immediateNeighborVariables = self.model.variables[immediateNeighborVariablesIndex]

        for variable in immediateNeighborVariables:
            self.updateTrajectory(refreshmentTime, variable, newVelocity[int(d)])
            d += 1


        ## 3-recompute the collisions for the other factors touching the variables (including the one we just popped)
        for factor in neighborFactors:
            self.updateCandidateCollision(factor, np.asscalar(refreshmentTime))

    def globalVelocityRefreshment(self, refreshmentTime, initializing):
        ## initializing should be a boolean variable
        variables = self.model.variables
        dimensionality = len(variables)
        newVelocity = np.zeros(dimensionality)

        if initializing:
            if self.rfOptions.refreshmentMethod == RefreshmentMethod.RESTRICTED:
                newVelocity = Utils.uniformOnUnitBall(dimensionality)
            else:
                newVelocity = np.random.normal(0, 1, dimensionality)
        else:
            if self.rfOptions.refreshmentMethod == RefreshmentMethod.GLOBAL or self.rfOptions.refreshmentMethod == RefreshmentMethod.LOCAL:
                newVelocity = np.random.normal(0, 1, dimensionality)
            elif self.rfOptions.refreshmentMethod == RefreshmentMethod.RESTRICTED:
                newVelocity = Utils.uniformOnUnitBall(dimensionality)
            elif self.rfOptions.refreshmentMethod == RefreshmentMethod.PARTIAL:
                if initializing:
                    newVelocity = Utils.uniformOnUnitBall(dimensionality)
                else:
                    newVelocity = Utils.partialRefreshmentBetaAngle(self.currentVelocity(variables),
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
                self.initTrajectory(refreshmentTime, variable, currentVelocity)
                self.model.variables[i] = self.updateVariable(variable, refreshmentTime)
                for factor in self.allFactors:
                    factor.variables[i] = self.model.variables[i]
            else:
                self.model.variables[i] = self.updateVariable(variable, refreshmentTime)
                newKey = self.model.variables[i]
                if newKey != oldValue:
                    self.trajectories[newKey] = self.trajectories[oldValue]
                    del self.trajectories[oldValue]
                variable = self.model.variables[i]
                for factor in self.allFactors:
                    factor.variables[i] = self.model.variables[i]
                self.updateTrajectory(refreshmentTime, variable, currentVelocity)

        for factor in self.allFactors:
            self.updateCandidateCollision(factor, refreshmentTime)

    def iterate(self, maxNumberOfIterations, maxTrajectoryLen, maxTimeMilli=sys.maxsize):
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

        self.globalVelocityRefreshment(0.0, True)

        # figure out what rayProcessor is doing and see if I need to implement that class
        if np.isclose(self.rfOptions.refreshRate, 0):
            nextRefreshmentTime = np.inf
        else:
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
                self.doCollision()
                self.currentTime = nextCollisionTime
            else:
                nextRefreshmentTimeCopy = copy.deepcopy(nextRefreshmentTime)
                self.currentTime = nextRefreshmentTimeCopy
                self.currentTime = np.asscalar(self.currentTime)
                if self.rfOptions.refreshmentMethod == RefreshmentMethod.LOCAL:
                    self.localVelocityRefreshment(nextRefreshmentTimeCopy)
                else:
                    self.globalVelocityRefreshment(nextRefreshmentTimeCopy, False)
                nextRefreshmentTime += np.random.exponential(scale=1 / self.rfOptions.refreshRate, size=1)
            print(i)


        # self.model.variables = self.updateAllVariables(self.currentTime)
        # change the keys of the trajectory to the new keys
        for index, var in enumerate(self.model.variables):
            oldKey = self.model.variables[index]
            self.model.variables[index] = self.updateVariable(var, self.currentTime)
            newKey = self.model.variables[index]
            if newKey != oldKey:
                ## TODO: figure out when we need to update the time and the positon of the trajectory ray
                self.trajectories[self.model.variables[index]] = self.trajectories[oldKey]
                del self.trajectories[oldKey]

        for i in range(len(self.model.variables)):
            for factor in self.allFactors:
                factor.variables[i] = self.model.variables[i]


        # Ensure that the remaining rays are processed
        self.globalVelocityRefreshment(self.currentTime, False)

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
