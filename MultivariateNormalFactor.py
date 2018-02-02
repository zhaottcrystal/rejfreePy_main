import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
## need to comment this when submitting assignment
import os

#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")

import random

from CollisionFactor import CollisionFactor
import numpy as np
import math


class MultivariateNormalFactor(CollisionFactor):

    def __init__(self, precision, variables, logConstant = None):
        ## mean is the mean of the normal distribution
        ## standardError represents the standard deviaiton of the normal distribution
        ## variable represents the current value of the realization of the Normal distributed variable
        self.precision = precision
        self.variables = variables
        if logConstant is not None:
            self.logConstant = logConstant
        else:
            self.logConstant = - len(variables)/ 2.0 * np.log( 2.0 * math.pi) +  0.5 * np.log(np.abs(np.linalg.det(precision)))



    def normalCollisionTime(self, exponential, xv, vv):
        s1 = 0
        if xv < 0:
            s1 = -xv / vv
        C = -exponential - s1 * ( xv + vv * s1 / 2.0)
        result = (-xv + np.sqrt(xv * xv-2.0*vv*C))/vv
        return result



    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        # this gives the index of the corresponding exchangeable element for
        # state0, state1 in the gradient or variables vector

        v = collisionContext.velocity

        x = self.variables

        vv = np.dot(v, v)

        xv = np.dot(x, v)

        ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
        c = -np.log(random.random())

        t = self.normalCollisionTime(c, xv, vv)

        if t < 0:
            t = np.inf

        result = {'deltaTime': t, 'collision': True}

        return result

    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        x = self.variables
        gradient = np.matmul(self.precision, x) * (-1.0)
        return gradient

    def getVariable(self, gradientIndex):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return self.variables[gradientIndex]

    def nVariables(self):
        """Get the dimension of the parameters"""
        return len(self.variables)

    def setPosision(self, position):
        """Set the position of the variables"""
        self.variables = position
        return self.variables

    def logDensity(self):
        ## need to find out when logDensity() is used and whether the variable should be the values before the collision or not
        result = -0.5 * np.dot(self.variables, self.variables) + self.logConstant
        return result