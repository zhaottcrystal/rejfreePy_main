#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:28:36 2017

@author: crystal
"""
import sys

import numpy as np

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
## need to comment this when submitting assignment
import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")

from EndPointSampler import EndPointSampler
from PathStatistics import PathStatistics
from Path import Path
from SimuSeq import ForwardSimulation
from ReversibleRateMtx import ReversibleRateMtx
import numpy as np
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from HardCodedDictionaryUtils import getHardCodedDictChainGraph
from numpy.random import RandomState
class TestForwardAndEndPointSamplers:
    def __init__(self):
        pass

    @staticmethod
    def testOneSampleMatrix():
        EndPointSampler.cached=True
        nStates=2
        stationaryWeights = np.array((0.619, 0.592))
        bivariateWeights = 1.499
        bivariateFeatIndexDictionary = getHardCodedDictChainGraph(nStates=nStates)
        rateMtxObj = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights,
                                                                            bivariateWeights,
                                                                                bivariateFeatIndexDictionary)
        stationaryDist= rateMtxObj.getStationaryDist()
        rateMatrix = rateMtxObj.getRateMtx()
        T = 5.0

        pathStat2 = PathStatistics(nStates)

        postSampler = EndPointSampler(rateMatrix, T)
        fwdSampler = ForwardSimulation(T, rateMatrix)

        transition = np.zeros((nStates, nStates))
        sojournTime = np.zeros(nStates)

        nIters = 1000000
        for i in range(nIters):
            current = Path()
            prng = np.random.RandomState(i)
            startState = prng.choice(nStates, 1, replace=True, p=stationaryDist)
            curResult = fwdSampler.sampleStateTimeSeq(prng, startState)
            print(curResult['transitCount'])
            transition = transition + curResult["transitCount"]
            sojournTime = sojournTime + curResult["sojourn"]
            current.states = curResult["states"]
            current.times = curResult["time"]
            p2 = Path()
            postSampler.sample(np.random.RandomState(i + nIters), current.firstState(), current.lastState(), T, pathStat2, p2)
            print(i)

        m2 = pathStat2.getCountsAsSimpleMatrix() / nIters
        m1 = transition
        np.fill_diagonal(m1, sojournTime)
        m1 = m1 / nIters
        print(np.round(m1, 3))
        print(np.round(m2, 3))

    @staticmethod
    def test():
        EndPointSampler.cached = True
        ## define a known rate matrix K80
        nStates = 4
        weights = np.array((0, 0, 0, 0, 0, np.log(3), 0, 0, np.log(3), 0))
        rateMtxObj = ReversibleRateMtx(nStates, weights)
        stationaryDist = rateMtxObj.getStationaryDist()
        
        k80 = ReversibleRateMtx(nStates, weights).getNormalizedRateMtx()
        T = 3.0
        
        pathStat2 = PathStatistics(nStates)
        
        postSampler = EndPointSampler(k80, T)
        fwdSampler = ForwardSimulation(T, k80)
        
        transition = np.zeros((nStates, nStates))
        sojournTime = np.zeros(nStates)
        
        
        nIters = 1000000
        for i in range(nIters):
            current = Path()
            startState = np.argmax(np.random.multinomial(1, stationaryDist, 1))
            curResult = fwdSampler.sampleStateTimeSeq(startState)
            transition = transition + curResult["transitCount"]
            sojournTime = sojournTime + curResult["sojourn"]
            current.states = curResult["states"]
            current.times = curResult["time"]
            p2 = Path()
            postSampler.sample(np.random.RandomState(i + nIters), current.firstState(), current.lastState(), T, pathStat2, p2)
            print(i)
        
        m2 = pathStat2.getCountsAsSimpleMatrix()/nIters
        m1 = transition
        np.fill_diagonal(m1, sojournTime)
        m1 = m1/ nIters
        print(np.round(m1,3))
        print(np.round(m2,3))

    @staticmethod
    def testNonNormalizedGTRRateMtx():
        EndPointSampler.cached = True
        ## define a known rate matrix K80
        nStates = 4
        ## set a seed for the random generator
        np.random.seed(1)
        ## randomly generate a GTR rate matrix from the weights
        weights = np.random.normal(0, 1, 10)
        rateMtxObj = ReversibleRateMtx(nStates, weights)
        stationaryDist = rateMtxObj.getStationaryDist()

        k80 = ReversibleRateMtx(nStates, weights).getRateMtx()
        T = 3.0

        pathStat2 = PathStatistics(nStates)

        postSampler = EndPointSampler(k80, T)
        fwdSampler = ForwardSimulation(T, k80)

        transition = np.zeros((nStates, nStates))
        sojournTime = np.zeros(nStates)

        nIters = 1000000
        for i in range(nIters):
            current = Path()
            startState = np.argmax(np.random.multinomial(1, stationaryDist, 1))
            curResult = fwdSampler.sampleStateTimeSeq(startState)
            transition = transition + curResult["transitCount"]
            sojournTime = sojournTime + curResult["sojourn"]
            current.states = curResult["states"]
            current.times = curResult["time"]
            p2 = Path()
            postSampler.sample(np.random.RandomState(i + nIters), current.firstState(), current.lastState(), T, pathStat2, p2)
            print(i)

        m2 = pathStat2.getCountsAsSimpleMatrix() / nIters
        m1 = transition
        np.fill_diagonal(m1, sojournTime)
        m1 = m1 / nIters
        print(np.round(m1, 3))
        print(np.round(m2, 3))



def main():
    # TestForwardAndEndPointSamplers.test()
    TestForwardAndEndPointSamplers.testOneSampleMatrix()

if __name__ == "__main__": main()