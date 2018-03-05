#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:26:09 2017

@author: crystal
"""
import sys
import numpy as np
from enum import Enum

class RefreshmentMethod(Enum):
     GLOBAL = 1
     LOCAL = 2
     RESTRICTED = 3
     PARTIAL = 4

     def __str__(self):
         return self.name

     @staticmethod
     def from_string(s):
         try:
             return RefreshmentMethod[s]
         except KeyError:
             raise ValueError()



class RFSamplerOptions:
    def __init__(self, refreshRate=1, refreshmentMethod=RefreshmentMethod.LOCAL, alpha=1.0, beta=4.0, trajectoryLength=1.0, maxSteps= 1000000 ):
        ## refreshmentMethod can be RefreshmentMethod.LOCAL,
        self.refreshRate = refreshRate
        self.refreshmentMethod = refreshmentMethod
        self.alpha= alpha
        self.beta = beta
        self.trajectoryLength = trajectoryLength
        self.maxSteps = maxSteps


class LocalRFRunnerOptions:
    def __init__(self, maxRunningTimeMilli=sys.maxsize, rfOptions= RFSamplerOptions(), silent=False):
        self.maxRunningTimeMilli = maxRunningTimeMilli
        self.maxSteps = rfOptions.maxSteps
        self.trajectoryLength = rfOptions.trajectoryLength
        self.rfOptions = rfOptions
        self.silent = silent



class MCMCOptions:
    def __init__(self, nMCMCSweeps=10000, thinningPeriod=10, burnIn = 0):
        self.nMCMCSweeps = nMCMCSweeps
        self.thinningPeriod = thinningPeriod
        self.burnIn = burnIn
        
        
        
    
    
    
        