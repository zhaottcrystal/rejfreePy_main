#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:18:18 2017

@author: crystal
"""

import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os

#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import numpy as np
from SojournTimeFactorWithoutPiEstWithBinaryFactors import SojournTimeFactorWithoutPiEstWithBinaryFactors
from TransitionCountWithoutPiWithBinaryFactors import TransitionCountFactorWithoutPiEstWithBinaryFactors
import NormalFactor



def addLocalFactors(nStates, expectedCompleteReversibleObjective, bivariateWeights, stationaryDist, bivariateFeatIndexDictionary):
    """
    :param nStates: the number of states in the CTMC
    :param expectedCompleteReversibleObjective: the class which contains the sufficient statistics
    :param bivariateFeatures: the bivariateWeights used to calculate the exchangeable parameters
    :return: a list where each element of the list is a factor. We add sojourn time factors and transition count factors for
             all pairs of states, and we also add the normal prior factor for each bivariateWeights of the bivariate features
    """
    localFactors = []
    wholeStates = np.arange(0, int(nStates))

    ## add all prior normal distribution factors for each bivariateWeights
    dim = len(bivariateWeights)
    for i in range(dim):
        normalFactor = NormalFactor.NormalFactor(bivariateWeights, dim, i)
        localFactors.append(normalFactor)

    ## add all sojourn time factors
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            collisionFactor = SojournTimeFactorWithoutPiEstWithBinaryFactors(expectedCompleteReversibleObjective, state0, state1, nStates, bivariateWeights, stationaryDist, bivariateFeatIndexDictionary)
            localFactors.append(collisionFactor)

    ## add all transition count factors
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            collisionFactor = TransitionCountFactorWithoutPiEstWithBinaryFactors(expectedCompleteReversibleObjective, state0, state1, nStates, bivariateWeights, stationaryDist, bivariateFeatIndexDictionary)
            localFactors.append(collisionFactor)
    return localFactors


class ExpectedCompleteReversibleModelWithBinaryFactors:

    def __init__(self, expectedCompleteReversibleObjective, nStates, bivariateWeights, stationaryDist, bivariateFeatIndexDictionary):
        """
        nStates: the number of states in the state space of a CTMC
        useExchangeParam should be a boolean variable, if true, exchangeCoef and stationaryDist shouldn't be None and variables should be None
        """
        self.auxObjective = expectedCompleteReversibleObjective
        self.nStates = nStates
        self.stationaryDist = stationaryDist
        self.variables = bivariateWeights
        self.bivariateFeatIndexDictionary = bivariateFeatIndexDictionary
        self.localFactors = self.localFactors()

    def nVariables(self):
        return len(self.variables)

    def localFactors(self):
        self.localFactors = addLocalFactors(self.nStates, self.auxObjective,  self.variables, self.stationaryDist, self.bivariateFeatIndexDictionary)
        return self.localFactors