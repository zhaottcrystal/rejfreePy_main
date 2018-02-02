#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:18:18 2017

@author: crystal
"""

#import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
#import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import Utils
#from main.Utils import addLocalFactorsRestrictedCTMCNoPriorFactor
#from main.Utils import addLocalFactorsNonRestrictedCTMCForExchangeParamWithGammaPrior


class ExpectedCompleteReversibleModel:
    def __init__(self,  expectedCompleteReversibleObjective, nStates, useExchangeParam, variables=None, exchangeCoef=None, stationaryDist=None):
        """
        nStates: the number of states in the state space of a CTMC
        useExchangeParam should be a boolean variable, if true, exchangeCoef and stationaryDist shouldn't be None and variables should be None
        """
        self.auxObjective = expectedCompleteReversibleObjective
        self.nStates = nStates

        if useExchangeParam:
            if exchangeCoef is None or stationaryDist is None:
                raise ValueError("When we use exchange coefficients and stationary dist parameterization, the input shouldn't be None")
            self.exchangeCoef = exchangeCoef
            self.stationaryDist = stationaryDist
            self.variables = exchangeCoef
            self.localFactors = self.localFactorsForExchangeCoef()

        else:
            self.variables = variables
            self.localFactors = self.localFactors()

        
    
    def nVariables(self):
        return len(self.variables)
    
    def localFactors(self):
        """later we should consider the case when adding the prior distributions as factors"""
        self.localFactors = Utils.addLocalFactorsRestrictedCTMCNoPriorFactor(self.nStates, self.auxObjective, self.variables)
        return self.localFactors

    def localFactorsForExchangeCoef(self):
        self.localFactors = Utils.addLocalFactorsNonRestrictedCTMCForExchangeParamWithGammaPrior(self.nStates, self.auxObjective, self.exchangeCoef, self.stationaryDist)
        return self.localFactors