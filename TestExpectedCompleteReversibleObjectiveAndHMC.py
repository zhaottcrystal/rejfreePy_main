import sys
sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

from main.ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from main.HMC import HMC
import numpy as np
from main.ReversibleRateMtxPiAndExchangeGTR import ReversibleRateMtxPiAndExchangeGTR
from main.FullTrajectorGeneration import getFirstAndLastStateOfListOfSeq
from main.FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist


def test():
    ## Provided the weights for the stationary distribution and the exchangeable coefficients
    ## we use them to generate the true rate matrix and the sequences to get the averaged
    ## sufficient statistics for the sequences throughout a larger number of replications
    ## Based on the sufficient statistics and we fix the exchangeable coefficients to
    ## its true values, we use HMC to estimate the weights for the stationary distribution
    ## The rate matrix is an un-normalized version
    
    ## The correctness of HMC and ExpectedCompleteReversibleObjective has been tested
    ## The estimated stationary distribution 'stationaryDistEst' is very close to stationaryDist

    nStates = 4
    nRep = 1000
    seedNum = np.arange(0, nRep)
    np.random.seed(123)
    weights = np.random.uniform(0, 1, nStates)
    print(weights)
    exchangeCoef = np.array((1, 2, 3, 4, 5, 6))

    ## get the rate matrix
    testRateMtx = ReversibleRateMtxPiAndExchangeGTR(nStates, weights, exchangeCoef)
    stationaryDist = testRateMtx.getStationaryDist()
    rateMtx = testRateMtx.getRateMtx()
    bt = 5.0
    nSeq = 100

    nInit = np.zeros(nStates)
    holdTimes = np.zeros(nStates)
    nTrans = np.zeros((nStates, nStates))

    for j in range(nRep):
        ## do forward sampling
        seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seedNum[j], rateMtx, stationaryDist, bt)
        ## summarize the sufficient statistics
        ## extract first state from sequences
        firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
        unique, counts = np.unique(firstStates, return_counts=True)
        nInitCount = np.asarray((unique, counts)).T
        nInit = nInit + nInitCount[:, 1]

        for i in range(nSeq):
            sequences = seqList[i]
            holdTimes = holdTimes + sequences['sojourn']
            nTrans = nTrans + sequences['transitCount']

    avgNTrans = nTrans / nRep
    avgHoldTimes = holdTimes / nRep
    avgNInit = nInit / nRep

    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTimes=avgHoldTimes, nInit=avgNInit, nTrans=avgNTrans, kappa=1, exchangeCoef=exchangeCoef)
    hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = np.random.uniform(0, 1, nStates)
    samples = hmc.run(0, 5000, sample)
    avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    stationaryDistEst = np.exp(avgWeights)/np.sum(np.exp(avgWeights))
    print(weights)
    print(avgWeights)
    print(stationaryDist)
    print(stationaryDistEst)
