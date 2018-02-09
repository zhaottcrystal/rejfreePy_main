import sys

import numpy as np

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
## need to comment this when submitting assignment
import os

#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")

import random
import CollisionFactor
import Utils

#from main.CollisionFactor import CollisionFactor
#from main.Utils import getBivariateFeatGradientIndexWithoutPiWithBivariateFeat
from scipy.optimize import fsolve

class TransitionCountFactorWithoutPiEstWithBinaryFactors(CollisionFactor.CollisionFactor):

    def __init__(self, expectedCompleteReversibleObjectiveWithBivariateFeat, state0, state1, nStates, bivariateFeatWeights, stationaryDist, bivariateFeatIndexDictionary):
        """objective: an instance of ExpectedCompleteReversibleOjective class,
                      which gives gives the sufficient statistics such as initial
                      count for the current state: state0
           state0: the current state, state0 is the starting point
           state1: the ending state, the chain makes transitions from state0 to state1
           nStates: the total number of states in the state space
           exchangeCoef: the exchangeable coefficients
           stationaryDist: the provided stationary distribution estimates
        """
        self.objective = expectedCompleteReversibleObjectiveWithBivariateFeat
        self.state0 = state0
        self.state1 = state1
        self.nStates = nStates

        ## bivariateFeatIndexDictionary is used to save a key-value pair
        ## key is a pair of states (state0, state1)
        ## the value is a vector, which is the index of the position, for example, if it returns 0, 1, 3, that means
        ## the exchangeable parameter of state 0 and state1 needs the 0th, 1st and 3rd bivariateWeights
        self.bivariateGradInd = Utils.getBivariateFeatGradientIndexWithoutPiWithBivariateFeat(state0, state1, bivariateFeatIndexDictionary)
        self.bivariateFeatWeights= bivariateFeatWeights
        self.nVariables = len(bivariateFeatWeights)
        self.stationaryDist = stationaryDist
        self.phi = np.zeros(self.nVariables)
        self.phi[self.bivariateGradInd] = 1
        self.transitCount = self.objective.nTrans[state0][state1]
        self.variables = self.bivariateFeatWeights


    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        # this gives the index of the corresponding exchangeable element for
        # state0, state1 in the gradient or variables vector

        v = collisionContext.velocity

        if np.dot(v, self.phi) > 0:
            t = np.inf

        else:
            ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
            c = -np.log(collisionContext.prng.uniform(0, 1, 1)[0])

            t = -c / (self.transitCount * np.dot(v, self.phi))

            ## using numerical solver to check if the solution from a numerical solver is close to the theoretical form
            ## The numerical solution proves the correctness of the theoretical form
            # functionValue1 = self.transitCount * (np.dot(v, self.phi)*t) + c
            # func = lambda tau: self.transitCount * (np.dot(v, self.phi)*tau) + c
            # tau_initial_guess = 0.001
            # tau_solution = fsolve(func, tau_initial_guess)
            # functionValue2 = self.transitCount * (np.dot(v, self.phi)*tau_solution) + c

        result = {'deltaTime': t, 'collision': True}

        return result

    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        gradient = np.zeros(self.nVariables)
        gradient[self.bivariateGradInd] = self.transitCount
        return gradient

    def getVariable(self, gradientCoordinate):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return self.variables[gradientCoordinate]

    def nVariables(self):
        """Get the dimension of the parameters"""
        return len(self.variables)

    def setPosision(self, position):
        """Set the position of the variables"""
        self.variables = position

    def logDensity(self):
        ## need to find out when logDensity() is used and whether the variable should be the values before the collision or not
        result = self.transitCount * (np.log(self.stationaryDist[self.state1]) + np.dot(self.variables, self.phi))
        return result