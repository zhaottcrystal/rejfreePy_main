import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
## need to comment this when submitting assignment
import os

#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import random

import CollisionFactor
import Utils


#from main.CollisionFactor import CollisionFactor
#from main.Utils import getBivariateFeatGradientIndexWithoutPiWithBivariateFeat
import numpy as np
from scipy.optimize import fsolve

class SojournTimeFactorWithoutPiEstWithBinaryFactors(CollisionFactor.CollisionFactor):
    def __init__(self, expectedCompleteReversibleObjectiveWithBivariateFeat, state0, state1, nStates, bivariateFeatWeights, stationaryDist, bivariateFeatIndexDictionary):
        """objective: an instance of ExpectedCompleteReversibleOjective class,
                      which gives gives the sufficient statistics such as initial
                      count for the current state: state0
           state0: the current state, state0 is the starting point
           state1: the ending state, the chain makes transitions from state0 to state1
           nStates: the total number of states in the state space
           exchangeCoef: the exchangeable coefficients of the rate matrix
           stationaryDist: the stationary distribution of the rate matrix
        """
        self.objective = expectedCompleteReversibleObjectiveWithBivariateFeat
        self.state0 = state0
        self.state1 = state1
        self.nStates = nStates
        self.bivariateGradInd = Utils.getBivariateFeatGradientIndexWithoutPiWithBivariateFeat(state0, state1,
                                                                                        bivariateFeatIndexDictionary)
        self.nVariables = len(bivariateFeatWeights)
        self.pi1 = stationaryDist[state1]
        self.phi = np.zeros(self.nVariables)
        self.phi[self.bivariateGradInd] = 1
        self.holdTime = self.objective.holdTimes[state0]
        self.variables = bivariateFeatWeights


    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        # this gives the index of the corresponding exchangeable element for
        # state0, state1 in the gradient or variables vector

        v = collisionContext.velocity

        if np.dot(v, self.phi) < 0:
            t = np.inf

        else:
            ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
            # c = -np.log(collisionContext.prng.uniform(0, 1, 1)[0])
            c = - np.log(np.random.uniform(0, 1, 1)[0])

            term1 = 1 / np.dot(v, self.phi)

            #debug1 = self.holdTime * self.pi1
            #debug2 = np.exp(np.dot(self.variables, self.phi))


            term2 = np.log(c / (self.holdTime * self.pi1 * np.exp(np.dot(self.variables, self.phi))) + 1.0)

            t = term1 * term2

            ## using numerical solver to check if the solution from a numerical solver is close to the theoretical form
            ## The numerical solution proves the correctness of the theoretical form
            # functionValue1 = self.holdTime * self.pi1 * (np.exp(np.dot(self.variables, self.phi)))*(np.exp(np.dot(v, self.phi) * t)-1.0)-c
            # func = lambda tau: self.holdTime * self.pi1 * (np.exp(np.dot(self.variables, self.phi))) * (np.exp(np.dot(v, self.phi) * tau)-1.0)-c
            # tau_initial_guess = 0.001
            # tau_solution = fsolve(func, tau_initial_guess)
            # functionValue2 = self.holdTime * self.pi1 * (np.exp(np.dot(self.variables, self.phi)))*(np.exp(np.dot(v, self.phi) * tau_solution) - 1.0)-c


        result = {'deltaTime': t, 'collision': True}

        return result


    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        gradient = -self.holdTime * self.pi1 * np.exp(np.dot(self.variables, self.phi)) * np.array(self.phi)
        return gradient

    def dimension(self):
        return 1

    def valueAt(self, x):
        # this returns the energy value or the negative log density value of the function
        result = -self.logDensity()
        return result

    def derivativeAt(self, x):
        # this returns the gradient of the energy value with respect to the time t
        result = 0
        return result


    def getVariable(self, gradientCoordinate):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return self.variables[gradientCoordinate]

    def nVariables(self):
        """Get the dimension of the parameters"""
        return len(self.variables)

    def setPosition(self, position):
        """Set the position of the variables"""
        self.variables = position
        return self.variables

    def logDensity(self):

        result = -self.holdTime * self.pi1 * np.exp(np.dot(self.variables, self.phi))
        return result

