#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:03:55 2017

@author: crystal
"""
import numpy as np
import sys
import os
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

class PathStatistics:
    
    def __init__(self, nStates):
        self.initialCounts = np.zeros(nStates)
        self.counts = np.zeros((nStates, nStates))
       
    def getCountsAsSimpleMatrix(self):
        
        return self.counts
        
    
    def getSojournTime(self, state):
        return self.counts[state, state]


    def addSojournTime(self, currentPoint, time):
        self.counts[np.int(currentPoint), np.int(currentPoint)] = self.counts[np.int(currentPoint), np.int(currentPoint)] + time
        return self.counts[np.int(currentPoint), np.int(currentPoint)]
    
    def getTransitionCount(self,currentState, nextState):
        if currentState == nextState:
            raise ValueError("The currentState and nextState should be different")
        return self.counts[currentState, nextState]
    
    def addTransition(self, currentState, nextState):
        if currentState!=nextState:
            self.counts[np.int(currentState), np.int(nextState)] =  self.counts[np.int(currentState), np.int(nextState)] + 1
        return self.counts[np.int(currentState), np.int(nextState)]

    def getInitialCount(self, state):
        return self.initialCounts[state]
    
    def addInitial(self, state):
        self.initialCounts[state] = self.initialCounts[state]+1
        return self.initialCounts[state]
    

## check the correctness of the code

#stat = PathStatistics(4) 
#stat.addSojournTime(1, 0.5) ## the state should start from 0
#stat.counts
#stat.addTransition(0, 1)
#stat.getTransitionCount(0, 1)
#stat.counts
#stat.addInitial(0)
#stat.initialCounts
                       
    
    
                   
        
