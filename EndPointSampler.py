#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:53:00 2017

@author: crystal
"""

import sys
sys.path
sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
import numpy as np
from scipy import linalg # use this module to calculate matrix exponential
import math
from numpy.random import RandomState

## define a class variable
MAX_N_TRANSITION = 1000000
cached = True
epsilon = 0.01
class EndPointSampler:
    
    
    def __init__(self, rateMtx, totalTime):
        self.totalTime = totalTime
        self.rateMtx = rateMtx
        self.cache = self.initCache()
        self.uniformizedTransition = self.uniformizedTransitionMtx(rateMtx, self.maxDepartureRate(rateMtx))
        self.sojournWorkArray = np.zeros(10)
        self.transitionWorkArray = np.zeros(rateMtx.shape[0])
    
    def sample(self, prng, startState, endState, T, statistics, path):
        """
        statistics is of class PathStatistics
        path is of class path
        """
        if path is not None and not path.isEmpty() and path.lastState() != startState:
            raise ValueError("Incompatible extension of the provided path")
        nTransitions = self.sampleNTransitions(prng, startState, endState, T)
        self.generatePath(prng, startState, endState, T, nTransitions, path, statistics)
        
    
    
    def initCache(self):
        """
        Output: return a list of length one and the first element is an identity
        matrix as the same dimension of the rate matrix
        """
        result = []
        nStates = self.rateMtx.shape[0]
        result.append(np.identity(nStates))
        return result
        
    def cacheSize(self):
        return len(self.cache)
    
    
    def ensureCache(self, power):
        """
        Input: power is an integer, which is the power of the transition probability matrix B
        Output: cache the B^0, B^1, ..., B^power in a list in self.cache
        """
        maxPowerInCache = len(self.cache)-1
        for curPower in np.arange((maxPowerInCache+1), (power+1)):
            tmpInd = (curPower-1)
            tmpMatrix = self.cache[tmpInd]
            self.cache.append(np.matmul(tmpMatrix, self.uniformizedTransition))
        return self.cache                   
    
   
    def maxDepartureRate(self, rateMtx):
        
        """
        correctness of this function has been checked
        Input: rate matrix
        Output: return the biggest diagonal elements of the rate matrix in 
                terms of its absolute value   
        """
        maxRate = np.inf*(-1)
        for i in range(rateMtx.shape[0]):
            current = np.abs(rateMtx[i][i])
            if current > maxRate:
                maxRate = current

        maxRate = maxRate + epsilon
            
        return maxRate
    
            
    def uniformizedTransitionMtx(self, rateMtx, mu):
        ## correctness of this function has been checked
        """
        Input: rate matrix and the uniformization rate mu
               mu can be the maximum departure rate (the biggest diagonal elements
               of the rate matrix in terms of its absolute value or it can be
               anything that is bigger than that)
        Output: returns a transition probability matrix which is the sum of
                an identity matrix I + rateMtx/mu
        """
        nStates = rateMtx.shape[0]
        result = np.zeros((nStates, nStates))
        for i in range(nStates):
            for j in range(nStates):
                if i==j:
                    result[i, j] = 1.0 + rateMtx[i,j]/mu
                else:
                    result[i, j] = rateMtx[i,j]/mu
        return result
    
    
    def getUniformizedTransitionPower(self, power):
        self.ensureCache(power)
        return self.cache[power]
        
        
    
    
    def sampleNTransitions(self, prng, initialState, endState, totalTime):
        """
        Input: 
            rand: the random seed
            initialState: the initial state of the path
            endState: the ending state of the path
            totalTime: the time elapsed between the starting state and the ending state
        Output:
            return the total number of transitions sampled from a truncated 
            Poisson Processes
        """
        transitionMarginal = linalg.expm(self.rateMtx*totalTime)
        if np.isclose(transitionMarginal[np.int(initialState), np.int(endState)], 0.0, 1e-08, 1e-10):
            raise ValueError('The transition probability between the initial and ending state is zero')
        if np.isclose(transitionMarginal[np.int(initialState), np.int(endState)], 1.0):
            return 0 ## since the probability is 1.0 so that no transitions needed
        
        ## generate a number form Unif[0, 1]
        uniform = prng.uniform(low=0.0, high=1.0, size=1)
        sumResult = 0
        mu = self.maxDepartureRate(self.rateMtx)
        logConstant = -mu*totalTime- np.log(transitionMarginal[np.int(initialState), np.int(endState)])
        logMuT = np.log(mu*totalTime)
        for i in range(MAX_N_TRANSITION):
            logNum = logConstant + i*logMuT + np.log(self.getUniformizedTransitionPower(i)[np.int(initialState), np.int(endState)])
            logDenom = math.log(math.factorial(i))
            current = np.exp(logNum - logDenom)
            sumResult += current
            #print(i)
            #print(sumResult)
            #print(uniform)
            if sumResult >= uniform:
                return i
        raise ValueError("Max number of transitions exceeded")
        
  
    def marginalTransitionProbability(rateMtx, totalTime):
        """
        Input:
            rateMtx: rate matrix
            totalTime: the time elapsed for a continuous time Markov chains
        
        Output:
            return the marginal transition probability which is matrix exponential 
            of (rateMtx*totalTime)
        """
        result = linalg.expm(rateMtx*totalTime)
        return result
    
    

    def getWorkArray(self, minLen):
        if(self.sojournWorkArray.shape[0]< minLen):
            self.sojournWorkArray = np.zeros(minLen * 2)
            
        return self.sojournWorkArray
        
  
    
    def generateSojournTimes(self, prng, nTransitions, T):
        # prng is something like prng = RandomState(1234567890)
        nTimes = nTransitions+1
        result = self.getWorkArray(nTimes)
        sumResult = 0

        for i in range(nTimes):
            cur = prng.exponential(1.0, 1)
            sumResult = sumResult + cur
            result[i] = cur
        for i in range(nTimes):
            result[i] = T * result[i]/sumResult
        return result          
            
    def generatePath(self, prng,  initialState, endState, T, nTransitions, resultPath, stat):
           sojournTimes = self.generateSojournTimes(prng, nTransitions, T)

           currentPoint = initialState
           if resultPath != None:
               resultPath.addSegment(currentPoint, sojournTimes[0])
           if stat != None:
               stat.addSojournTime(currentPoint, sojournTimes[0])
           for transitionIndex in range(nTransitions):
               for candidateState in range(len(self.transitionWorkArray)):
                   self.transitionWorkArray[np.int(candidateState)] = self.uniformizedTransition[np.int(currentPoint), np.int(candidateState)] \
                                      * self.getUniformizedTransitionPower(nTransitions-transitionIndex-1)[np.int(candidateState), np.int(endState)]
               ## Multinomial normalize transitionWorkArray
               self.transitionWorkArray = self.transitionWorkArray/np.sum(self.transitionWorkArray)
               ## sample a state according to the multinomial distribution
               nextState = np.argmax(prng.multinomial(1, self.transitionWorkArray, 1))
               if resultPath != None:
                   resultPath.addSegment(nextState, sojournTimes[transitionIndex+1])
               if stat != None:
                   stat.addSojournTime(nextState, sojournTimes[transitionIndex+1])
                   if currentPoint != nextState:
                       stat.addTransition(currentPoint, nextState)
               currentPoint = nextState