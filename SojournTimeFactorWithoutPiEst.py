import sys

import numpy as np

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
## need to comment this when submitting assignment
import os

#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import random

import CollisionFactor
import getBivariateFeatGradientIndex

#from main.CollisionFactor import CollisionFactor
#from main.getBivariateFeatGradientIndex import getBivariateFeatGradientIndexWithoutPi


class SojournTimeFactorWithoutPiEst(CollisionFactor.CollisionFactor):
    def __init__(self, expectedCompleteReversibleObjective, state0, state1, nStates, exchangeCoef, stationaryDist):
        """objective: an instance of ExpectedCompleteReversibleOjective class, 
                      which gives gives the sufficient statistics such as initial
                      count for the current state: state0
           state0: the current state, state0 is the starting point
           state1: the ending state, the chain makes transitions from state0 to state1
           nStates: the total number of states in the state space
           exchangeCoef: the exchangeable coefficients of the rate matrix
           stationaryDist: the stationary distribution of the rate matrix
        """
        self.objective = expectedCompleteReversibleObjective
        self.state0 = state0
        self.state1 = state1
        self.nStates = nStates
        self.bivariateGradInd = getBivariateFeatGradientIndex.getBivariateFeatGradientIndexWithoutPi(state0, state1, nStates)
        self.exchangeCoef = exchangeCoef
        # obtain pix' which is pi[state1]
        self.pi1 = stationaryDist[self.state1]
        # obtain theta[x, x'], which is theta[state0, state1]
        self.theta = self.exchangeCoef[self.bivariateGradInd]
        self.holdTime = self.objective.holdTimes[state0]
        self.variables = exchangeCoef


    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        # this gives the index of the corresponding exchangeable element for
        # state0, state1 in the gradient or variables vector
        biFeatInd = self.bivariateGradInd

        v = collisionContext.velocity

        ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
        c = -np.log(collisionContext.prng.uniform(0, 1, 1)[0])

        v01 = v[biFeatInd]

        t = c/(self.holdTime * self.pi1 * v01)

        if t < 0:
            t = np.inf

        result = {'deltaTime': t, 'collision': True}

        return result

    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        gradient = np.zeros(len(self.exchangeCoef))
        gradient[self.bivariateGradInd] = -self.holdTime * self.pi1
        return gradient

    def getVariable(self, gradientCoordinate):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return self.variables[gradientCoordinate]

    def nVariables(self):
        """Get the dimension of the parameters"""
        return len(self.exchangeCoef)

    def setPosition(self, position):
        """Set the position of the variables"""
        self.variables = position
        return self.variables

    def logDensity(self):

        result = -self.holdTime * (self.pi1 * self.theta)
        return result

