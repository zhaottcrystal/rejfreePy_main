#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:49:35 2017

@author: crystal
"""

import sys

import numpy as np
from numpy import linalg as LA

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
## need to comment this when submitting assignment
import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import math
import numpy as np
import InitialCountFactor
import SojournTimeFactor
import TransitionCountFactor
import getBivariateFeatGradientIndex
import SojournTimeFactorWithoutPiEst
import TransitionCountFactorWithoutPiEst
import GammaDistFactorForExchangeCoef
import PathStatistics
import Path
import EndPointSampler
import FullTrajectorGeneration


# from main.InitialCountFactor import InitialCountFactor
# from main.SojournTimeFactor import SojournTimeFactor
# from main.TransitionCountFactor import TransitionCountFactor
# from main.getBivariateFeatGradientIndex import getBivariateFeatGradientIndex
# from main.SojournTimeFactorWithoutPiEst import SojournTimeFactorWithoutPiEst
# from main.TransitionCountFactorWithoutPiEst import TransitionCountFactorWithoutPiEst
# from main.getBivariateFeatGradientIndex import getBivariateFeatGradientIndexWithoutPi
# from main.GammaDistFactorForExchangeCoef import GammaDistFactorForExchangeCoef
# from main.PathStatistics import PathStatistics
# from main.Path import Path
# from main.EndPointSampler import EndPointSampler
# from main.FullTrajectorGeneration import getFirstAndLastStateOfListOfSeq

from collections import OrderedDict


def bounce(oldVelocity, gradient):
    """bounce the old velocities of the bouncy particle sampler"""
    scale = 2 * np.dot(gradient, oldVelocity)/np.dot(gradient, gradient)
    result = oldVelocity - gradient * scale
    return result

## design a test case for validate the correctness of bounce(oldVelocity, gradient)
# gradient = np.zeros(4)
# gradient[0] = 5
# gradient[1] = 4
# gradient[2] = 3
# gradient[3] = 2
# oldVelocity = np.array((0.1, 0.2, 0.3,0.4))
# newVelocity = bounce(oldVelocity, gradient)
# print(newVelocity)
## the correct newVelocity calculated by hand is the same as newVelocity 



def addLocalFactorsNonRestrictedCTMCForExchangeParamWithGammaPrior(nStates, expectedCompleteReversibleObjective, exchangeCoef, stationaryDist):
    """
    :param nStates: the number of states in the CTMC
    :param expectedCompleteReversibleObjective: the class which contains the sufficient statistics 
    :param exchangeCoef: the exchangeable coefficients
    :return: a list where each element of the list is a factor. We add sojourn time factors and transition count factors for 
             all pairs of states, and we also add the gamma prior factor for each pair of states
    """
    localFactors = []
    wholeStates = np.arange(0, nStates)

    ## add all prior gamma distribution factors for each element of exchangeCoef
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            if state1 > state0:
                collisionFactor = GammaDistFactorForExchangeCoef(shape=1.0, rate=1.0, state0=state0, state1=state1,
                                                             exchangeCoef=exchangeCoef, nStates=nStates)
                localFactors.append(collisionFactor)

    ## add all sojourn time factors
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            collisionFactor = SojournTimeFactorWithoutPiEst(expectedCompleteReversibleObjective, state0, state1, nStates, exchangeCoef, stationaryDist)
            localFactors.append(collisionFactor)

    ## add all transition count factors
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            collisionFactor = TransitionCountFactorWithoutPiEst(expectedCompleteReversibleObjective, state0, state1, nStates, exchangeCoef, stationaryDist)
            localFactors.append(collisionFactor)
    return localFactors




def addLocalFactorsRestrictedCTMCNoPriorFactor(nStates, expectedCompleteReversibleObjective, variables):
    """
    This function returns a list, each element of the list is a factor
    We add all initial count factors, sojourn time factors and transition count factors
    """
    localFactors = []
    wholeStates = np.arange(0, nStates)
        
    ## add all initial count factors
    for state0 in range(nStates):
        collisionFactor = InitialCountFactor.InitialCountFactor(expectedCompleteReversibleObjective, state0, variables, nStates)
        localFactors.append(collisionFactor)  
        
    
    ## add all sojourn time factors
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            collisionFactor = SojournTimeFactor(expectedCompleteReversibleObjective, state0, state1, nStates, variables)
            localFactors.append(collisionFactor)
            
            
    ## add all transition count factors
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            collisionFactor = TransitionCountFactor(expectedCompleteReversibleObjective, state0, state1, nStates, variables)
            localFactors.append(collisionFactor)
            
    ## Todo: add gamma prior distribution factors and dirichlet distribution factor        
                   
    return localFactors        
        

def neighborVariablesCTMC(nStates, isRestricted, collisionFactor):
    
    """
    Given a collisionFactor, we check if the collisionFactor is a univariate factor 
    like initial count factor, return the variables connected with this factor.
    For the current model in consideration, the candidate factors can be:
    initial count factor, sojourn time factor, transition count factor, 
    prior distribution factors: dirichlet distribution and gamma distribution
    for each bivariate exchangeable parameters.
    
    isRestricted: true means we have the constraint that all elements of the
    stationary distribution sum to one and we only estimate the first nStates-1 
    of them and the last one is obtained by 1- pi_1-pi_2-...- pi_{nStates-1}
    
    This function returns the index of the variables that are neighbors of the
    current collision factor of interest.
    """
    
    if isinstance(collisionFactor, InitialCountFactor):
        if isRestricted:
            state0 = collisionFactor.state0
            if state0 < (nStates-1):
                return state0
            if state0 == (nStates-1):
                return np.arange(0, (nStates-1))
            if state0 > (nStates-1):
                raise ValueError("The state index cannot be bigger than the number of states minus 1")
        else:
            # if not restricted, each stationary distribution elements are all 
            # connected to all univariate variables, then we return all indices
            # of univariate variables
            return np.arange(0, nStates)
    
    if isinstance(collisionFactor, SojournTimeFactor) or isinstance(collisionFactor, TransitionCountFactor):
        
        state0 = collisionFactor.state0
        state1 = collisionFactor.state1
        ind2 = getBivariateFeatGradientIndex(state0, state1, nStates)
        
        if isRestricted:
            if state1 < (nStates-1):
                ## add the univariate state index in the gradient vector
                ind1 = state1
                return np.array((ind1, ind2))
            if state1 == (nStates-1):
                result = np.zeros(nStates)
                result[0:(nStates-1)] = np.arange(0, (nStates-1))
                result[(nStates-1)] = ind2
                return result
            if state1 > (nStates-1):
                raise ValueError("The state index cannot be bigger than the number of states minus 1")
        else:
            result = np.zeros((nStates+1))
            result[0:nStates] = np.arange(0, nStates)
            result[nStates] = ind2
            return result
                
    ## Todo: isinstance gamma factor and dirichlet factor


def neighborVariablesCTMCWithOnlyExchangeCoef(nStates, collisionFactor):
    """
    Given a collisionFactor, we check the class instance of the collision
    factor and return the variables connected with this factor.
    For the current model in consideration, the candidate factors can be:
    gamma prior distribution factor, sojourn time factor, transition count factor, 
    for each bivariate exchangeable parameters.
    
    This function returns the index of the variables that are neighbors of the
    current collision factor of interest.
    
    Given the current model setup, each sojourn time, transition count and
    gamma prior distribution factor is connected with only one exchange coefficient
    """

    if isinstance(collisionFactor, SojournTimeFactorWithoutPiEst) or isinstance(collisionFactor, TransitionCountFactorWithoutPiEst) or isinstance(collisionFactor, GammaDistFactorForExchangeCoef):

        state0 = collisionFactor.state0
        state1 = collisionFactor.state1
        ind2 = getBivariateFeatGradientIndexWithoutPi(state0, state1, nStates)

        result = np.array([ind2])
        return result


def neighborVariablesWithOnlyExchangeCoef(nStates, collisionFactors):
    result = set()

    for collisionFactor in collisionFactors:
        variables = neighborVariablesCTMCWithOnlyExchangeCoef(nStates, collisionFactor)
        for var in variables:
            result.add(var)
    return list(result)


def getNumOFFactorsCTMC(nStates):
    nInit = nStates
    nSojourn = nStates * (nStates-1)
    nTransit = nStates * (nStates-1)
    result = {"nInit": nInit, "nSojourn": nSojourn, "nTransit": nTransit}
    return result


    
def getBivariateLocalFactorIndexDict(nStates, nInit, nSojourn, nTransit):
    ## the nth row represents state0
    ## the nth column represents state 1
    ## the first sojourn matrix stores the index of the sojourn factors connected
    ## for state0 and state1
    ## the diagonal elements are set to zero since state0 and state1 should
    ## not be equal so that we don't need the diagonal elements values
    
    result1 = np.zeros((nStates, nStates))
    wholeStates = np.arange(0, nStates)
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for index, state1 in enumerate(support):
            result1[state0, state1] = state0*(nStates-1) + nInit + index
    if nTransit == nSojourn:
        result2 = result1 + nSojourn
        ## set the diagonal element values to zero
        np.fill_diagonal(result2, 0)
        
    else:
        raise ValueError("The number of sojourn time factors is not equal to the number of transition count factors")
    
    result1 = result1.astype(np.int64)
    result2 = result2.astype(np.int64)
    
    result = {"Sojourn": result1, "Transit": result2}
    return result       
        

# test the correctness of the code
# result = getBivariateLocalFactorIndexDict(4, 4, 12, 12)
# print(result["Sojourn"])
# print(result["Transit"])
    

def getBivariateLocalFactorIndexDictUsingDict(nStates, numOFFactorsDict):  
    nInit = numOFFactorsDict["nInit"]
    nSojourn = numOFFactorsDict["nSojourn"]
    nTransit = numOFFactorsDict["nTransit"]
    result = getBivariateLocalFactorIndexDict(nStates, nInit, nSojourn, nTransit)
    return result

def getNumOFFactorsCTMCOnlyExchangeCoef(nStates):
    nSojourn = nStates * (nStates - 1)
    nTransit = nStates * (nStates - 1)
    nPrior = nStates * (nStates-1)/2
    result = {'nSojourn': nSojourn, 'nTransit': nTransit, 'nPrior': nPrior}
    return result


def getBivariateLocalFactorIndexDictOnlyExchangeCoef(nStates, nPrior, nSojourn, nTransit):
    ## the nth row represents state0
    ## the nth column represents state 1
    ## the first sojourn matrix stores the index of the sojourn factors connected
    ## for state0 and state1
    ## the diagonal elements are set to zero since state0 and state1 should
    ## not be equal so that we don't need the diagonal elements values

    result1 = np.zeros((nStates, nStates))
    wholeStates = np.arange(0, nStates)
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for index, state1 in enumerate(support):
            result1[state0, state1] = state0 * (nStates - 1) + nPrior + index
    if nTransit == nSojourn:
        result2 = result1 + nSojourn
        ## set the diagonal element values to zero
        np.fill_diagonal(result2, 0)

    else:
        raise ValueError("The number of sojourn time factors is not equal to the number of transition count factors")

    result1 = result1.astype(np.int64)
    result2 = result2.astype(np.int64)
    result3 = {}

    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for state1 in support:
            if state1 > state0:
                result3[(state0, state1)] = getBivariateFeatGradientIndex.getBivariateFeatGradientIndexWithoutPi(state0, state1, nStates)


    result = {"Sojourn": result1, "Transit": result2, 'Prior': result3}
    return result


# test the correctness of the code
# result = getBivariateLocalFactorIndexDictOnlyExchangeCoef(4, 6, 12, 12)
# print(result["Sojourn"])
# print(result["Transit"])
# print(result['Prior'])

def getBivariateLocalFactorIndexDictForExchangeCoefUsingnStates(nStates):
    numOFFactorsDict = getNumOFFactorsCTMCOnlyExchangeCoef(nStates)
    nPrior = numOFFactorsDict["nPrior"]
    nSojourn = numOFFactorsDict["nSojourn"]
    nTransit = numOFFactorsDict["nTransit"]
    result = getBivariateLocalFactorIndexDictOnlyExchangeCoef(nStates, nPrior, nSojourn, nTransit)
    return result
## test the correctness of the function
## result = getBivariateLocalFactorIndexDictForExchangeCoefUsingnStates(nStates)
## print(result["Sojourn"])
## print(result["Transit"])
## print(result['Prior'])

def getBivariateLocalFactorIndexDictUsingnStates(nStates):  
    numOFFactorsDict = getNumOFFactorsCTMC(nStates)
    nInit = numOFFactorsDict["nInit"]
    nSojourn = numOFFactorsDict["nSojourn"]
    nTransit = numOFFactorsDict["nTransit"]
    result = getBivariateLocalFactorIndexDict(nStates, nInit, nSojourn, nTransit)
    return result    

## test the correctness of the function
## result = getBivariateLocalFactorIndexDictUsingnStates(nStates)
## print(result["Sojourn"])
## print(result["Transit"])  
    

# bivariateIndDict = getBivariateLocalFactorIndexDictUsingnStates(nStates) 
def neighborFactorsCTMCDict(nStates, isRestricted, localFactors, bivariateIndDict):
    """
    In the formulation of continuous time Markov chains:
    if we use the restricted version of the stationary distribution
    we have (nStates-1) univariate variables of interest to represent
    the stationary distribution. If we use the unrestricted formulation
    of the stationary distribution, we have nStates univariate variables
    to estimate. 
    
    Under the restricted case: 
    1. For each univariate state whose state is not the last state
    represented by nStates under the restricted case, it is connected 
    to one unary factor, which is the initial count factor depending 
    on the current univariate variable. It is also connected to 2(n-1) 
    bivariate factors including the sojourn time factor and transition 
    count factor whose ending state, ie. state1 is the current 
    univariate state variable. It is also connected to 2(n-1) bivariate
    factors including sojourn time factor and transition count factors
    where the ending state is the last state denoted as nStates since its
    stationary distribution elements depends on all the other univariate 
    variables. 
    
    It will also involve one prior factor such as the Dirichlet prior 
    distribution.
    
    2. For each bivaraite variables, it is connected to four factors including
    the sojourn time factor and transition count factor: h_state0,state1,
    h_state1,state0, c_state0,state1 and c_state1,state0
    
    Todo: for each bivariate variable, it should be connected to a one 
    dimension gamma prior distribution for this bivariate variable
    
    This function returns the index of the connected factors in the factor
    list which stores all the local factors of the model.
    """
    
    resultDict = {}
    resultDictIndex = {}
    resultVarIndDict = {}
    resultVarIndDictIndex = {}
    
    wholeStates = np.arange(0, nStates)
    
    sojournInd = bivariateIndDict["Sojourn"]
    transitInd = bivariateIndDict["Transit"]
    
    if isRestricted:
        for state0 in range(nStates-1):
            support = np.setdiff1d(wholeStates, state0)
            result = []
            indList = []
            ## add univariate factors
            result.append(localFactors[state0])
            indList.append(state0)
            result.append(localFactors[nStates-1])
            indList.append((nStates-1))
            ## add bivariate factors
            ## 1. add the index of the bivariate factors that ends in state0
            for elem in support:
                sojournFactorInd = sojournInd[elem, state0]
                result.append(localFactors[sojournFactorInd])
                indList.append(sojournFactorInd)
                transitFactorInd = transitInd[elem, state0]
                result.append(localFactors[transitFactorInd])
                indList.append(transitFactorInd)
            ## 2. add the index of the bivariate factors that ends in the last state "nStates-1"
            support = np.setdiff1d(wholeStates, (nStates-1))
            for elem in support:
                sojournFactorInd = sojournInd[elem, (nStates-1)]
                result.append(localFactors[sojournFactorInd])
                indList.append(sojournFactorInd)
                transitFactorInd = transitInd[elem, (nStates-1)]
                result.append(localFactors[transitFactorInd])
                indList.append(transitFactorInd)
            resultDict[state0] = result
            resultDictIndex[state0] = indList
            resultVarIndDict[state0] = result
            resultVarIndDictIndex[state0] = indList
        # add the factor index for bivariate features
        for state0 in range(nStates-1):
            support = np.setdiff1d(wholeStates, state0)
            for elem in support:
                if elem > state0:
                    result = []
                    indList = []
                    FeatInd = getBivariateFeatGradientIndex(state0, elem, nStates)
                    ## add sojourn factors for bivariate variables 
                    sojournFactorInd1 = sojournInd[elem, state0]
                    result.append(localFactors[sojournFactorInd1])
                    indList.append(sojournFactorInd1)
                    sojournFactorInd2 = sojournInd[state0, elem]
                    result.append(localFactors[sojournFactorInd2])
                    indList.append(sojournFactorInd2)
                    ## add transit count factors for bivariate variables
                    transitFactorInd1 = transitInd[elem, state0]
                    result.append(localFactors[transitFactorInd1])
                    indList.append(transitFactorInd1)
                    transitFactorInd2 = transitInd[state0, elem]
                    result.append(localFactors[transitFactorInd2])
                    indList.append(transitFactorInd2)
                    resultDict[(state0, elem)] = result
                    resultDictIndex[(state0, elem)] = indList
                    resultVarIndDict[FeatInd] = result
                    resultVarIndDictIndex[FeatInd] = indList
        totalResult = {"FactorsDict": resultDict, "FactorsDictIndex": resultDictIndex,
                       "FactorsDictWithVarIndAsKeys": resultVarIndDict, 
                       "FactorsDictIndexWithVarIndAsKeys": resultVarIndDictIndex}                            
        return totalResult
    
# test the correctness of neighborFactorCTMCDict
# expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(5*np.ones(4), 5*np.ones(4), np.zeros((4,4))) 
# localFactors = addLocalFactorsRestrictedCTMCNoPriorFactor(4, expectedCompleteReversibleObjective, np.arange(0.1, 1,0.1))  
# bivariateIndDict = getBivariateLocalFactorIndexDictUsingnStates(nStates=4)  
# factorDict = neighborFactorsCTMCDict(4, True, localFactors, bivariateIndDict)  
# check the correctness of the FactorsDictIndex: correct           


# bivariateIndDictForExchange = getBivariateLocalFactorIndexDictForExchangeCoefUsingnStates(nStates)
def neighborFactorsCTMCDictForOnlyExchange(nStates,localFactors, bivariateIndDictForExchange):
    """
    For each bivaraite variables, it is connected to four factors including
    the sojourn time factor and transition count factor: h_state0,state1,
    h_state1,state0, c_state0,state1 and c_state1,state0, and a prior distribution
    factor such as a gamma distribution factor

    Todo: for each bivariate variable, it should be connected to a one 
    dimension gamma prior distribution for this bivariate variable

    This function returns the index of the connected factors in the factor
    list which stores all the local factors of the model.
    """

    resultDict = {}
    resultDictIndex = {}
    resultVarIndDict = {}
    resultVarIndDictIndex = {}

    wholeStates = np.arange(0, nStates)

    sojournInd = bivariateIndDictForExchange["Sojourn"]
    transitInd = bivariateIndDictForExchange["Transit"]
    priorInd = bivariateIndDictForExchange["Prior"]

    # add the factor index for bivariate features
    for state0 in range(nStates - 1):
        support = np.setdiff1d(wholeStates, state0)
        for elem in support:
            if elem > state0:
                result = []
                indList = []
                FeatInd = priorInd[(state0, elem)]

                ## add sojourn factors for bivariate variables
                sojournFactorInd1 = sojournInd[elem, state0]
                result.append(localFactors[sojournFactorInd1])
                indList.append(sojournFactorInd1)
                sojournFactorInd2 = sojournInd[state0, elem]
                result.append(localFactors[sojournFactorInd2])
                indList.append(sojournFactorInd2)

                ## add transit count factors for bivariate variables
                transitFactorInd1 = transitInd[elem, state0]
                result.append(localFactors[transitFactorInd1])
                indList.append(transitFactorInd1)
                transitFactorInd2 = transitInd[state0, elem]
                result.append(localFactors[transitFactorInd2])
                indList.append(transitFactorInd2)

                ## add gamma prior distribution factors for bivariate variables
                gammaFactorInd = priorInd[(state0, elem)]
                result.append(localFactors[gammaFactorInd])
                indList.append(gammaFactorInd)

                resultDict[(state0, elem)] = result
                resultDictIndex[(state0, elem)] = indList
                resultVarIndDict[FeatInd] = result
                resultVarIndDictIndex[FeatInd] = indList
    totalResult = {"FactorsDict": resultDict, "FactorsDictIndex": resultDictIndex,
                   "FactorsDictWithVarIndAsKeys": resultVarIndDict,
                   "FactorsDictIndexWithVarIndAsKeys": resultVarIndDictIndex}
    return totalResult


# test the correctness of neighborFactorCTMCDict
#expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(4*np.ones(4), 4*np.ones(4), np.zeros((4,4)), np.array((1,2,3,4,5,6)))
#localFactors = addLocalFactorsNonRestrictedCTMCForExchangeParamWithGammaPrior(4, expectedCompleteReversibleObjective, np.array((1,2,3,4,5,6)), 0.25* np.ones(4))
#bivariateIndDict = getBivariateLocalFactorIndexDictForExchangeCoefUsingnStates(nStates=4)
#factorDict = neighborFactorsCTMCDictForOnlyExchange(4,localFactors, bivariateIndDict)
# check the correctness of the FactorsDictIndex: correct

def neighborFactors(neighborFactorsCTMCDictResult, immediateVariablesIndex):
    # neighborFactorsCTMCDictResult should be the result from 
    # neighborFactorsCTMCDict(nStates, isRestricted, localFactors, bivariateIndDict)
    neighborFactorsDictWithVarIndAsKeys = neighborFactorsCTMCDictResult
    result = set()
    
    ## result should be a data structure like hashset in java to store unique elements
    for ind in immediateVariablesIndex:
        factors = neighborFactorsDictWithVarIndAsKeys[ind]
        for factor in factors:
            result.add(factor)
    return result

def neighborVariables(nStates, isRestricted, collisionFactors):
    
    result = set()
    
    for collisionFactor in collisionFactors:
        variables = neighborVariablesCTMC(nStates, isRestricted, collisionFactor)
        for var in variables:
            result.add(var)
    return list(result)
        
    
                 
## provide several functions to simulate the velocities from different refreshement methods
def uniformOnUnitBall(prng,dimension):
    random = prng.normal(0, 1, dimension)
    norm = LA.norm(random)
    random = random/norm
    return random

## test the correctness of this code
## a = uniformOnUnitBall(5)
## LA.norm(a)

def project(toProject, unitTangentialVector):
    length = np.dot(toProject, unitTangentialVector)
    newTangentialVector = unitTangentialVector * length
    toProject = toProject - newTangentialVector
    return toProject

def partialRefreshment(currentVelocity, angleRad):
    delta = uniformOnUnitBall(len(currentVelocity))
    p = project(delta, currentVelocity)
    p = p/LA.norm(p)
    L = math.tan(angleRad)
    p = p * L
    result = currentVelocity + p
    result = result/LA.norm(result)
    return result
    
## the correctness of both functions project() and partialRefreshment() has been 
## checked and compared with its java implementation    

def partialRefreshmentBetaAngle(prng, currentVelocity, alpha=1, beta=4):
    angle = prng.beta(alpha, beta, 1) *2 * math.pi
    return partialRefreshment(currentVelocity, angle)   

   
def timeElapsed(startTime, endTime, timeFormat):
    result = endTime - startTime
    if timeFormat=="microseconds":
        return result
    if timeFormat=="seconds":
        return result/1e6
    if timeFormat == "minutes":
        return (result/1e6)/60

    
def position(initialPos, velocity, time):
    result = initialPos + velocity * time
    return result

def SquareDistance(a, b):
    """
    :param a: D by n array
    :param b: D by m array
    :return: an array that is n by m of all pairwise square distances
    """
    # the correctness of this function has been tested on various settings
    if len(b.shape)<2 or len(a.shape)<2:
        result = np.sum((a-b)**2)
        return result
    D = a.shape[0]
    m = b.shape[1]
    n = a.shape[1]
    part1 = np.sum(a, axis=1) / (n + m)
    part2 = np.sum(b, axis=1) / (n + m)
    mu = part1 + part2
    
    a = a - np.repeat(mu, n).reshape(mu.shape[0], n)
    b = b - np.repeat(mu, m).reshape(mu.shape[0], m)
    
    c = np.repeat(np.transpose(np.sum(a * a, axis=0)), m).reshape(len(np.transpose(np.sum(a * a, axis=0))) , m)
    c = c + np.tile(np.sum(b * b, axis=0), (n, 1)).reshape( n, len(np.sum(b * b, axis=0)))

    c = c - np.matmul(np.transpose(a), b) * 2
    c = np.maximum(c, 0)
    return c

# summarize the sufficient statistics using EndPointsampler
def summarizeSuffStatUsingEndPoint(seqList, bt, currRateMtx):

    nStates = currRateMtx.shape[0]
    ## get the initial and last state of eqch sequence
    firstAndlastStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState']
    firstStates = firstAndlastStates[:, 0]
    lastStates = firstAndlastStates[:, 1]

    ## get sufficient statistics
    ## get nInit
    unique, counts = np.unique(firstStates, return_counts=True)
    nInitCount = np.asarray((unique, counts)).T
    nInit = nInitCount[:, 1]

    if len(nInit) != nStates:
        raise ValueError("The length of the sequence is not long enough, some of the states don't show up as the first states of the sequences")

    ## get transition count and sojourn time
    pathStat2 = PathStatistics(nStates)
    postSampler = EndPointSampler(currRateMtx, bt)

    for i in range(len(seqList)):
        p2 = Path()
        postSampler.sample(firstStates[i], lastStates[i], bt, pathStat2, p2)

    m2 = pathStat2.getCountsAsSimpleMatrix()
    transition = np.copy(m2)
    np.fill_diagonal(transition, 0)
    sojournTime = np.diag(m2)

    result = {'nInit': nInit, 'nTrans': transition, 'holdTimes': sojournTime}
    return result



def getBivariateFeatGradientIndexWithoutPiWithBivariateFeat(state0, state1, bivariateFeatIndexDictionary):

    ## bivariateFeatIndexDictionary is used to save a key-value pair
    ## key is a pair of states (state0, state1)
    ## the value is a vector, which is the index of the position, for example, if it returns 0, 1, 3, that means
    ## the exchangeable parameter of state 0 and state1 needs the 0th, 1st and 3rd bivariateWeights

    keyPair = (state0, state1)
    return bivariateFeatIndexDictionary[keyPair]

def generateBivariateFeatGradientIndexWithoutPiWithBivariateFeatPairStates(state0, state1, nBivariateFeat, nChoosenFeatureRatio):
    ## In contrast with the function above getBivariateFeatGradientIndexWithoutPiWithBivariateFeat(state0, state1, bivariateFeatIndexDictionary),
    ## this function generates the index of the positions randomly to create the factor graph structure
    ## nBivariateFeat: total number of Bivariate features
    ## nChoosenFeatureRatio: this defines the ratio of the features that are going to be selected in the graph

    nChoosenFeature = int(np.ceil(nBivariateFeat * nChoosenFeatureRatio))
    keyPair = (state0, state1)
    index = np.random.choice(nBivariateFeat, nChoosenFeature, replace=False)
    ## create a dictionary to save the keys and values
    result = OrderedDict()
    result[keyPair] = index

    return result

def generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat(nStates, nBivariateFeat, nChoosenFeatureRatio):

    ## loop over state0 and state1 and use generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat(state0, state1, nBivariateFeat, nChoosenFeatureRatio)

    wholeStates = np.arange(0, nStates)
    result = OrderedDict()
    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for elem in support:
            if elem > state0:
                result.update(generateBivariateFeatGradientIndexWithoutPiWithBivariateFeatPairStates(state0, elem, nBivariateFeat, nChoosenFeatureRatio))


    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates,state0)
        for elem in support:
            if elem < state0:
                result[(state0, elem)] = result[(elem, state0)]

    ## using this method generateBivariateFeatGradientIndexWithoutPiWithBivariateFeatPairStates(),
    ## when we use np.random.choice, certain dimension of the feature weights may not get selected,
    ## if for certain dimension of an element, it does not selected, we add dimension to a particular
    ## pair of states, and we also need to add this feature to its symmetric pair

    ## create a dictionary with keys as dimension index from 0 to nBivariateFeat with stepsize 1

    counterForEachDim = OrderedDict()

    ## loop over all pairs of states as keys, and then loop over all elements of each value for each key
    ## count the dimeension that shows up in the values of result
    for i in range(nBivariateFeat):
        counterForEachDim[i] = 0

    for state0 in range(nStates):
        support = np.setdiff1d(wholeStates, state0)
        for elem in support:
            values = result[(state0, elem)]
            for value in values:
                counterForEachDim[value] = counterForEachDim[value] + 1

    ## find out key of counterForEachDim has value == 0
    dimHasZeroCount = list()
    for name, count in counterForEachDim.items():
        if count == 0:
            dimHasZeroCount.append(name)

    ## put all elements in dimHasZeroCount to a random pair of states eg (0, 1), then we also need to
    ## add this element to pair (1, 0)
    halfSize = int(len(result.keys())/2.0)
    dictKeys = result.keys()
    dictKeys = list(dictKeys)

    if len(dimHasZeroCount) > 0:
        print(dimHasZeroCount)
        ## add this element for the first half of the keys in result
        ## and then we add it to the symmetric key pair of states
        for i in range(len(dimHasZeroCount)):
            keyIndex = np.int(np.random.randint(low=0, high=halfSize, size=1))
            result[dictKeys[keyIndex]] = np.append( result[dictKeys[keyIndex]], dimHasZeroCount[i])
            state0 = dictKeys[keyIndex][0]
            state1 = dictKeys[keyIndex][1]
            result[(state1, state0)] = np.append(result[(state1, state0)], (dimHasZeroCount[i]))

    return result

# test the correctness of this
# nStates = 4
# nBivariateFeat = 10
# seed = np.arange(nBivariateFeat)
# nChoosenFeatureRatio = 0.3
# result = generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat(nStates, nBivariateFeat, nChoosenFeatureRatio)
# ## check if for each pair of states is in the dictionary
# ## check if the values of a symmetric key pair has the same value
# containsEachPair = True
# isSymmetric = True
# print(len(result.keys()))
# for elem0 in range(nStates):
#     for elem1 in range(nStates):
#         if elem1 != elem0:
#             if not result.keys().__contains__((elem0, elem1)):
#                 containsEachPair = False
#                 break
#             if not np.array_equal(result[(elem0, elem1)], result[(elem1, elem0)]):
#                 isSymmetric = False
#                 break
# print(containsEachPair)
# print(isSymmetric)
# print(result)
#json.dump(result, open("/Users/crystal/Dropbox/2017PythonForResearch/bivariateGraphStructure.txt", 'w'))
#data = json.load(open("/Users/crystal/Dropbox/2017PythonForResearch/bivariateGraphStructure.txt"))