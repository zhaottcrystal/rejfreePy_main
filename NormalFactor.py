import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
## need to comment this when submitting assignment
import os

#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")

import random
import CollisionFactor
#from main.CollisionFactor import CollisionFactor
import numpy as np
import math


class NormalFactor(CollisionFactor.CollisionFactor):

    def __init__(self,  variables, nVariables, gradientIndex, sd = 1.0):
        ## mean is the mean of the normal distribution
        ## standardError represents the standard deviaiton of the normal distribution
        ## variable represents the current value of the realization of the Normal distributed variable
        ## We assume the normal factor is a standard Normal distribution

        self.variables = variables
        self.gradientIndex = gradientIndex
        self.sd = sd



    def normalCollisionTime(self, exponential, xv, vv):
        if xv > 0:
            result = -xv/vv + np.sqrt(xv * xv+ vv*exponential)/vv
        else:
            result = -xv/vv + np.sqrt(vv*exponential)/vv
        return result



    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        # this gives the index of the corresponding exchangeable element for
        # state0, state1 in the gradient or variables vector

        v = collisionContext.velocity[self.gradientIndex]

        x = self.variables[self.gradientIndex]

        vv = v * v
        xv = x * v

        ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
        c = -np.log(collisionContext.prng.uniform(0, 1, 1)[0])

        t = self.normalCollisionTime(c, xv, vv)

        if t < 0:
            t = np.inf

        result = {'deltaTime': t, 'collision': True}

        ## using numerical solver to check if the solution from a numerical solver is close to the theoretical form
        ## The numerical solution proves the correctness of the theoretical form
        # functionValue1 = self.holdTime * self.pi1 * (np.exp(np.dot(self.variables, self.phi)))*(np.exp(np.dot(v, self.phi) * t)-1.0)-c
        # func = lambda tau: self.holdTime * self.pi1 * (np.exp(np.dot(self.variables, self.phi))) * (np.exp(np.dot(v, self.phi) * tau)-1.0)-c
        # tau_initial_guess = 0.001
        # tau_solution = fsolve(func, tau_initial_guess)
        # functionValue2 = self.holdTime * self.pi1 * (np.exp(np.dot(self.variables, self.phi)))*(np.exp(np.dot(v, self.phi) * tau_solution) - 1.0)-c

        return result

    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        gradient = np.zeros(len(self.variables))
        gradient[self.gradientIndex] = self.variables[self.gradientIndex]/(self.sd*self.sd)*(-1.0)
        return gradient

    def getVariable(self, gradientIndex):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return self.variables[gradientIndex]

    def nVariables(self):
        """Get the dimension of the parameters"""
        return len(self.variables)

    def setPosision(self, position):
        """Set the position of the variables"""
        self.variables[self.gradientIndex] = position
        return self.variables[self.gradientIndex]

    def logDensity(self):
        ## need to find out when logDensity() is used and whether the variable should be the values before the collision or not
        result = -0.5 * np.dot(self.variables[self.gradientIndex], self.variables[self.gradientIndex]) - np.log(math.sqrt(2* math.pi))-np.log(self.sd)
        return result