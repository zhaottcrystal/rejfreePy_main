"""
Created on Fri Mar  3 15:33:54 2017

@author: crystal
"""

import sys

import numpy as np

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os

os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import random
import CollisionFactor
import getBivariateFeatGradientIndex
import DifferentiableFunction
import MultipleConvexCollisionSolver

#from main.CollisionFactor import CollisionFactor
#from main.getBivariateFeatGradientIndex import getBivariateFeatGradientIndex
#from main.DifferentiableFunction import DifferentiableFunction
from scipy import special
#from main.MultipleConvexCollisionSolver import MultipleConvexCollisionSolver

class GammaDistFactor(CollisionFactor.CollisionFactor, DifferentiableFunction.DifferentiableFunction):
    def __init__(self, shape, rate, state0, state1, variables, nStates):
        """
        Here we define the factor with gamma distribution factor parameterized by 
        its shape and rate parameter
        """
        self.shape = shape
        self.rate = rate
        self.state0 = state0
        self.state1 = state1
        self.variables = variables
        self.bivariateGradInd = getBivariateFeatGradientIndex.getBivariateFeatGradientIndex(state0, state1, nStates)
        self.nStates = nStates
        self.variable = self.variables[self.bivariateGradInd]

    def dimension(self):
        return 1

    def valueAt(self, x):
        alpha = self.shape
        beta = self.rate
        result = - alpha * np.log(beta) + np.log(special.gamma(alpha))-(alpha-1) * np.log(x) + beta * x
        return result

    def derivativeAt(self, x):
        result = -(self.shape-1)/x + self.rate
        return result

    def functionToGetCollisionTime(self, c, v01, position, t):
        alpha = self.shape
        beta = self.rate
        term1 = np.exp((beta * v01 * t-c)/(alpha-1)) - (v01/position)*t -1



    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        v = collisionContext.velocity
        ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
        c = -np.log(random.random())

        ## get the velocity for this variable
        biFeatInd = self.bivariateGradInd

        v = collisionContext.velocity
        v01 = v[biFeatInd]

        alpha = self.shape
        beta = self.rate

        collisionSolver = MultipleConvexCollisionSolver.MultipleConvexCollisionSolver()
        x = self.variable

        # sample the exponential number from an exponential distribution
        exponential = -np.log(np.random.uniform(0, 1, 1))

        if alpha == 1.0:
            collisionTime = c/(beta*v01)
        else:
            collisionTime = collisionSolver.collisionTime(initialPoint=x, velocity=v01, energy=self, exponential=exponential)


        #position = self.variable
        #tmpF = lambda x: np.exp((beta * v01 * x-c)/(alpha-1))-v01*x/position-1.0
        #tmpF = lambda x: [np.exp((0.5*x-0.23)/0.5)-0.5*x/0.4-1]
        #this line of code will cause error
        #collisionTime = mpmath.findroot(tmpF, 0.5, solver='pegasus', multidimensional=False)

        if collisionTime > 0:
            return collisionTime
        else:
            return np.inf

        result = {'deltaTime': t, 'collision': True}
        return result

    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        result = np.zeros(len(self.variables))
        result[self.bivariateGradInd] = (self.shape-1)/self.variable - self.rate
        return result

    def getVariable(self, gradientCoordinate):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return self.variables[gradientCoordinate]

    def nVariables(self):
        """Get the dimension of the parameters"""
        return len(self.variables)

    def setPosision(self, position):
        """Set the position of the variables"""
        self.variables = position
        return self.variables

    def logDensity(self):
        result = (self.shape-1)/self.var - self.rate
        return result

