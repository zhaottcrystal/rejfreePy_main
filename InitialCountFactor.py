# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:33:54 2017

@author: crystal
"""

import sys

import numpy as np

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
## need to comment this when submitting assignment
import os
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import random

import CollisionFactor
#from main.CollisionFactor import CollisionFactor


class InitialCountFactor(CollisionFactor.CollisionFactor):

    def __init__(self,expectedCompleteReversibleObjective, state0, variables, nStates):
        """objective: an instance of ExpectedCompleteReversibleOjective class, 
                      which gives gives the sufficient statistics such as initial
                      count for the current state: state0
                      expectedCompleteReversibleObjective should come from the 
                      ExpectedCompleteReversibleObjective class
           state0: the current state, for example, this gives the input of the 
                   initial count
           variables: this gives the values of all the parameters including the
                      the stationary distribution elements (if there are 4 states,
                      we will have three stationary distribution elements and the
                      exchangeable coefficients)
        """
        self.objective = expectedCompleteReversibleObjective
        self.state0 = state0
        self.variables = variables
        self.nStates = nStates
        # obtain pix' which is pi[state1] 
        # we use restrict bouncy particle sampler, with nstates
        # pix if state is from 0, 1, 2, ..., n-2, they take the same values as 
        # the first n-2 elements in "variables", for the last stationary 
        # distribution element, since we have the constraint that all of them
        # sum to 1, so the last element = 1- variables[0]-...- variables[n-2]
        if state0 < (nStates-1):
            self.pi0 = variables[self.state0]
        else:
            self.pi0 = 1- sum(variables[0:(nStates-1)])
    

    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        v = collisionContext.velocity
        
        ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
        c = -np.log(random.random())
        
        if v[self.state0] >= 0:
            t = np.inf
        else:
            n = self.objective.nInit[self.state0]
            t = self.variables[self.state0]/v[self.state0]*(np.exp(-c/n)-1)
        if t<0 :
           t= np.inf
            
        result = {'deltaTime': t, 'collision':True}    
        return result
    
    
    
    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        result = np.zeros(len(self.variables))
        n = self.objective.nInit[self.state0]
        if self.state1 < self.nStates-1:
            result[self.state0] = n/self.pi0
        else:
            for state in range((self.nStates-1)):
                result[state] = -n/self.pi0
        
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
        n = self.objective.nInit[self.state0]
        result = n * np.log(self.pi0)
        return result