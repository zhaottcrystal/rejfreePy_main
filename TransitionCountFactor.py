#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 21:12:04 2017

@author: crystal
"""

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
#from main.getBivariateFeatGradientIndex import getBivariateFeatGradientIndex

class TransitionCountFactor(CollisionFactor.CollisionFactor):
    def __init__(self,expectedCompleteReversibleObjective, state0, state1, nStates, variables):
        """objective: an instance of ExpectedCompleteReversibleOjective class, 
                      which gives gives the sufficient statistics such as initial
                      count for the current state: state0
           state0: the current state, state0 is the starting point
           state1: the ending state, the chain makes transitions from state0 to state1
           nStates: the total number of states in the state space
           variables: this gives the values of all the parameters including the
                      the stationary distribution elements (if there are 4 states,
                      we will have three stationary distribution elements and the
                      exchangeable coefficients)
        """
        self.objective = expectedCompleteReversibleObjective
        self.state0 = state0
        self.state1 = state1
        self.nStates = nStates
        self.bivariateGradInd = getBivariateFeatGradientIndex.getBivariateFeatGradientIndex(state0, state1, nStates)
        self.variables = variables
        
        # obtain pix' which is pi[state1] 
        # we use restrict bouncy particle sampler, with nstates
        # pix if state is from 0, 1, 2, ..., n-2, they take the same values as 
        # the first n-2 elements in "variables", for the last stationary 
        # distribution element, since we have the constraint that all of them
        # sum to 1, so the last element = 1- variables[0]-...- variables[n-2]
        
        if state1 < (nStates-1):
            self.pi1 = variables[self.state1]
        else:
            self.pi1 = 1 - sum(variables[0:(nStates-1)])
            
        # obtain theta[x, x'], which is theta[state0, state1]
        self.theta = variables[self.bivariateGradInd]
        self.transitCount = self.objective.nTrans[state0][state1] 
        
        
    
    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        # this gives the index of the corresponding stationary distribution element
        # for state1 in the gradient or variables vector
        # we use state1 instead of state0 since 
        uniFeatInd = self.state1
        # this gives the index of the corresponding exchangeable element for 
        # state0, state1 in the gradient or variables vector
        biFeatInd = self.bivariateGradInd
        
        v = collisionContext.velocity
        
        ## generate random number c, where c = -Math.log(V ~ unif(0, 1))
        c = -np.log(collisionContext.prng.uniform(0, 1, 1)[0])
        
        v1 = v[uniFeatInd]
        v01 = v[biFeatInd]
        
        part1 = -(v1*self.theta+self.pi1*v01)
        part2 = np.sqrt(part1 * part1 - 4*v1*v01*self.pi1*self.theta*(1-np.exp(-c/self.transitCount)))
        
        t1 = (part1 - part2)/(2*v1*v01)
        t2 = (part1 + part2)/(2*v1*v01)
        
        if t1 < 0 and t2 < 0:
            t = np.inf
        if t1 > 0 or t2 > 0:
            t = np.min((np.max((t1, 0)),  np.max((t2, 0)) ))
        
        result = {'deltaTime': t, 'collision':True}  
        return result
    

    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        gradient = np.zeros(len(self.variables))
        
        if self.state1 < self.nStates-1:
            gradient[self.state1] = self.transitCount / self.pi1
        else:
            for state in range((self.nStates-1)):
                gradient[state] = -self.transitCount/self.pi1
                
        gradient[self.bivariateGradInd] = self.transitCount/self.theta
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
        return self.variables
    
    def logDensity(self):
        
        result = self.transitCount * (np.log(self.pi1)+ np.log(self.theta))
        return result
        
        