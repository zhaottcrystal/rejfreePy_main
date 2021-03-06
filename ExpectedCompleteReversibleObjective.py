# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:32:39 2017

@author: crystal zhaott0416@gmail.com
"""

import sys

import numpy as np

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/main/")
from ReversibleRateMtxPiAndExchangeGTR import ReversibleRateMtxPiAndExchangeGTR
from FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from FullTrajectorGeneration import getFirstAndLastStateOfListOfSeq
from ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from HardCodedDictionaryUtils import getHardCodedDict
from scipy.linalg import expm
from RateMtxExpectations import RateMtxExpectations
from copy import deepcopy
from numpy.random import RandomState
from DataGenerationRegime import DataGenerationRegime
from Path import Path
from EndPointSampler import EndPointSampler
from PathStatistics import PathStatistics

class ExpectedCompleteReversibleObjective:

    def __init__(self, holdTimes, nInit, nTrans, kappa=1, exchangeCoef=None, nBivariateFeatWeightsDictionary=None):
        """ This class has one attributes:
            expectedStatistics should come from ExpectedStatistics class, 
            (refer to the java implementation to see ExpectedStatistics class)
            
            Later, we may need to implement other methods of this class
        """
        
        self.holdTimes = holdTimes
        self.nInit = nInit
        self.nStates = len(holdTimes)
        self.nTrans = nTrans
        self.kappa = kappa
        self.nStar = np.sum(self.nInit)
        self.TStar = np.sum(self.nTrans, axis=0)
        self.TStarStar = np.sum(self.TStar)
        self.holdTimesDiag = np.diag(self.holdTimes)
        ## the instance attributes of mTrans, mStar and mStarStar will be initialized in
        ## calculateForPiUnnormalized() function
        self.mTrans = None
        self.mStar = None
        self.mStarStar = None
        self.lastX = None

        self.exchangeCoef = exchangeCoef
        self.nBivariateFeatWeightsDictionary = nBivariateFeatWeightsDictionary

        self.lastValue = None
        self.lastDerivative = None

        self.fixDerivative = self.nInit + self.TStar


    def calculate(self, weights, nBivariateFeatWeightsDictionary):
        ## this function calculates the weights for both the stationary distribution features
        ## and the weights for the bivariate features

        ## First get the weights for the univariate features
        nStates = self.nStates
        dim = len(weights)
        weightsForPi = weights[0:nStates]
        weightsForBivariate = weights[nStates:dim]

        ## With the weights for the univariate and bivariate features to obtain the rate matrix,
        ## and stationary distribution

        rateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, weightsForPi, weightsForBivariate, nBivariateFeatWeightsDictionary)
        stationaryDist = rateMtx.getStationaryDist()
        unnormalizedRateMtx = rateMtx.getRateMtx()


        rateMtxCopy = np.copy(unnormalizedRateMtx)
        np.fill_diagonal(rateMtxCopy, 0)
        self.mTrans = np.matmul(self.holdTimesDiag, rateMtxCopy)
        self.mStar = np.sum(self.mTrans, axis=0)
        self.mStarStar = np.sum(self.mStar)

        gradient = np.zeros(dim)
        ## get the gradient for the univariate weights (for stationary distribution)
        # term1 = self.nInit
        term2 = -self.nStar * stationaryDist
        # term3 = self.TStar
        term4 = -self.TStarStar * stationaryDist
        term5 = -self.mStar
        term6 = self.mStarStar * stationaryDist
        gradient[0:nStates] = self.fixDerivative + term2 + term4 + term5 + term6

        ## combined with the Normal prior information for the weight
        gradient[0:nStates] = gradient[0:nStates] - self.kappa * weightsForPi
        ## get the gradient for the potential energy instead of the log-density
        gradient[0:nStates] = -gradient[0:nStates]

        ## get the gradient for the bivariate features
        ## loop over all pairs of states where the two states are not equal

        wholeStates = np.arange(0, nStates)
        for state0 in wholeStates:
            support = np.setdiff1d(wholeStates, state0)
            for state1 in support:
                bivariateWeightsDim = nBivariateFeatWeightsDictionary[(state0, state1)] + nStates
                tState0State1 = self.nTrans[state0][state1]
                mState0State1 = self.mTrans[state0][state1]
                ## assigning values to the gradient for the corresponding dimension
                gradient[bivariateWeightsDim] = gradient[bivariateWeightsDim] + tState0State1-mState0State1

        gradient[nStates:dim] = gradient[nStates:dim] - self.kappa * weightsForBivariate
        gradient[nStates:dim] = - gradient[nStates:dim]

        ## calculate the value of the objective function
        ## get the value of the objective function
        vterm1 = np.dot(self.nInit, np.log(stationaryDist))
        logRateMtxCopy = np.log(unnormalizedRateMtx)
        np.fill_diagonal(logRateMtxCopy, 0)
        vterm2 = np.sum(np.multiply(self.nTrans, logRateMtxCopy))
        vterm3 = -self.mStarStar
        value = vterm1 + vterm2 + vterm3
        value = value - self.kappa * np.dot(weights, weights) / 2
        value = -value

        result = {'gradient': gradient, 'value': value}
        return result



    def calculateForPiUnnormalized(self, weightsforPi, exchangeCoef):
        nStates = len(weightsforPi)
        rateMtxResult = ReversibleRateMtxPiAndExchangeGTR(nStates, weightsforPi, exchangeCoef)
        stationaryDist = rateMtxResult.getStationaryDist()
        unnormalizedRateMtx = rateMtxResult.getRateMtx()

        holdTimesDiag = np.diag(self.holdTimes)
        rateMtxCopy = np.copy(unnormalizedRateMtx)
        np.fill_diagonal(rateMtxCopy, 0)
        self.mTrans = np.matmul(holdTimesDiag, rateMtxCopy)
        self.mStar = np.sum(self.mTrans, axis=0)
        self.mStarStar = np.sum(self.mStar)

        gradient = np.zeros(nStates)
        term1 = self.nInit
        term2 = -self.nStar * stationaryDist
        term3 = self.TStar
        term4 = -self.TStarStar * stationaryDist
        term5 = -self.mStar
        term6 = self.mStarStar * stationaryDist
        gradient = term1 + term2 + term3 + term4 + term5 + term6

        ## combined with the Normal prior information for the weight
        gradient = gradient - self.kappa * weightsforPi
        ## get the gradient for the potential energy instead of the log-density
        gradient = -gradient

        ## get the value of the objective function
        vterm1 = np.dot(self.nInit, np.log(stationaryDist))
        logRateMtxCopy = np.log(unnormalizedRateMtx)
        np.fill_diagonal(logRateMtxCopy, 0)
        vterm2 = np.sum(np.multiply(self.nTrans, logRateMtxCopy))
        vterm3 = -self.mStarStar
        value = vterm1 + vterm2 + vterm3
        value = value - self.kappa * np.dot(weightsforPi, weightsforPi)/2
        value = -value

        result = {'gradient': gradient, 'value': value}
        return result



    def calculateForPiUnnormalizedFixedExchangeCoef(self, weightsforPi):

        if self.exchangeCoef is None:
            raise ValueError("When using this method, the exchangeable coefficients should be provided!")

        nStates = len(weightsforPi)
        rateMtxResult = ReversibleRateMtxPiAndExchangeGTR(nStates, weightsforPi, self.exchangeCoef)
        stationaryDist = rateMtxResult.getStationaryDist()
        unnormalizedRateMtx = rateMtxResult.getRateMtx()


        rateMtxCopy = np.copy(unnormalizedRateMtx)
        np.fill_diagonal(rateMtxCopy, 0)
        self.mTrans = np.matmul(self.holdTimesDiag, rateMtxCopy)
        self.mStar = np.sum(self.mTrans, axis=0)
        self.mStarStar = np.sum(self.mStar)

        term2 = -self.nStar * stationaryDist
        term4 = -self.TStarStar * stationaryDist
        term5 = -self.mStar
        term6 = self.mStarStar * stationaryDist
        gradient = self.fixDerivative + term2 + term4 + term5 + term6

        ## combined with the Normal prior information for the weight
        gradient = gradient - self.kappa * weightsforPi
        ## get the gradient for the potential energy instead of the log-density
        gradient = -gradient

        ## get the value of the objective function
        vterm1 = np.dot(self.nInit, np.log(stationaryDist))
        logRateMtxCopy = np.log(unnormalizedRateMtx)
        np.fill_diagonal(logRateMtxCopy, 0)
        vterm2 = np.sum(np.multiply(self.nTrans, logRateMtxCopy))
        vterm3 = -self.mStarStar
        value = vterm1 + vterm2 + vterm3
        value = value - self.kappa * np.dot(weightsforPi, weightsforPi)/2
        value = -value

        result = {'gradient': gradient, 'value': value}
        return result

    @staticmethod
    def requireUpdate(lastX, x):

        if lastX is None:
            return True

        if lastX is not None:
            if len(lastX) != len(x):
                raise ValueError("The length of lastX and x should be equal.")

        for i in range(len(x)):
            if lastX[i] != x[i]:
                return True
        return False

    def ensureCache(self, x):

        ## currently this implementation only works when x represents the weights for the stationary distribution
        if ExpectedCompleteReversibleObjective.requireUpdate(self.lastX, x):
            if not self.exchangeCoef is None:
                currentValueAndDerivative = self.calculateForPiUnnormalizedFixedExchangeCoef(x)
            else:
                currentValueAndDerivative = self.calculate(x, self.nBivariateFeatWeightsDictionary)

            self.lastValue = currentValueAndDerivative['value']
            self.lastDerivative = currentValueAndDerivative['gradient']
            if self.lastX is None:
                self.lastX = np.zeros(len(x))
            self.lastX = x



    def valueAt(self, x):
        self.ensureCache(x)
        return self.lastValue

    def derivativeAt(self, x):
        self.ensureCache(x)
        return self.lastDerivative


    def functionValue(self, vec):

        return self.valueAt(vec)

    def mFunctionValue(self, vec):

        return self.derivativeAt(vec)




## test the correctness of the gradient，test passed, using numerical gradient 


def testCalculate():

    ## test the correctness of the code using numeric gradient check
    nStates = 6
    nRep = 10000
    seedNum = np.arange(0, nRep)
    np.random.seed(123)
    weights = np.random.uniform(0, 1, 18)
    print(weights)
    # delta = 0.000001
    bivariateFeatDictionary = getHardCodedDict()
    weightsForPi = weights[0:nStates]
    weightsForBivariate = weights[nStates:len(weights)]

    ## get the rate matrix
    testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, weightsForPi, weightsForBivariate,
                                                                  bivariateFeatDictionary)
    stationaryDist = testRateMtx.getStationaryDist()
    rateMtx = testRateMtx.getRateMtx()

    bt = 3.0
    nSeq = 500
    ## simulate the observation data first
    seqList = generateFullPathUsingRateMtxAndStationaryDist(RandomState(seedNum[0]), nSeq, nStates, rateMtx,
                                                            stationaryDist,
                                                            bt)
    ## get observed sequences at a finite number of time points
    dataGenerationRegime = DataGenerationRegime(nStates=nStates,
                                                bivariateFeatIndexDictionary=bivariateFeatDictionary,
                                                btLength=bt, nSeq=nSeq, stationaryWeights=weightsForPi,
                                                bivariateWeights=weightsForBivariate, interLength=0.5)
    ## summarize the sufficient statistics
    obsData = dataGenerationRegime.generatingSeqGivenRateMtxAndBtInterval(seqList)
    marginalResult = dataGenerationRegime.summaryFirstLastStatesArrayIntoMatrix(obsData, nStates)
    obsNInit0 = marginalResult['nInit']
    ## get the marginal count from observations
    marginalCount = marginalResult['count']

    ## extract first state from sequences
    ## Below are our actual observation sequences
    firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
    unique, counts = np.unique(firstStates, return_counts=True)
    nInitCount = np.asarray((unique, counts)).T
    obsNInit = np.zeros(nStates)
    obsNInit = obsNInit + nInitCount[:, 1]
    print(obsNInit)  ## it should be equal to obsNinit0

    rateMtxExpectations = RateMtxExpectations(rateMtx, 0.5)
    marginalExpectations = rateMtxExpectations.expectationsWithMarginalCount(marginalCount)
    ## get expected holding time
    expectedHoldingTime = np.diag(marginalExpectations)
    ## get expected transition count
    expectedTransCount = deepcopy(marginalExpectations)
    np.fill_diagonal(expectedTransCount, 0)

    expectedCompleteObjectiveFromExptStat = ExpectedCompleteReversibleObjective(expectedHoldingTime, obsNInit,
                                                                                expectedTransCount,
                                                                                nBivariateFeatWeightsDictionary=bivariateFeatDictionary)
    expectedForwardResult = expectedCompleteObjectiveFromExptStat.calculate(weights, bivariateFeatDictionary)
    exptFuncValue = expectedForwardResult['value']
    exptGradient = expectedForwardResult['gradient']
    print(exptFuncValue)
    print(exptGradient)

    # nInit = np.zeros(nStates)
    # holdTimes = np.zeros(nStates)
    # nTrans = np.zeros((nStates, nStates))
    #
    # for j in range(nRep):
    # # ## do forward sampling
    #     seqList = generateFullPathUsingRateMtxAndStationaryDist(RandomState(seedNum[0]), nSeq, nStates, rateMtx, stationaryDist, bt)
    # #         ## summarize the sufficient statistics
    # #         ## extract first state from sequences
    #     firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
    #     unique, counts = np.unique(firstStates, return_counts=True)
    #     nInitCount = np.asarray((unique, counts)).T
    #     nInit = nInit + nInitCount[:, 1]
    #
    #     for i in range(nSeq):
    #         sequences = seqList[i]
    #         holdTimes = holdTimes + sequences['sojourn']
    #         nTrans = nTrans + sequences['transitCount']
    #
    # avgNTrans = nTrans/nRep
    # avgHoldTimes = holdTimes/nRep
    # avgNInit = nInit/nRep


    T = 0.5
    postSampler = EndPointSampler(rateMtx, T)
    pathStat2 = PathStatistics(nStates)
    nSegment = dataGenerationRegime.nPairSeq

    counter =0
    for j in range(nRep):
        ## do posterior path sampling
        ## for each segment of the observed path for the time series,
        ## loop over each segment of the sequence
        for i in range(nSegment):
            ## loop over each sequences
            for k in range(len(seqList)):
                p2 = Path()
                startState = obsData[i][int(k)][0]
                endState = obsData[i][k][1]
                postSampler.sample(np.random.RandomState(counter), int(startState), int(endState), T,
                                   pathStat2, p2)
                counter = counter + 1
        if j%10 == 0:
            print(j)

    m2 = pathStat2.getCountsAsSimpleMatrix() / nRep
    avgNInit = obsNInit
    avgNTrans = deepcopy(m2)
    np.fill_diagonal(avgNTrans, 0)
    avgHoldTimes = m2.diagonal()

    originalExpectedCompleteObjective = ExpectedCompleteReversibleObjective(avgHoldTimes, avgNInit, avgNTrans, nBivariateFeatWeightsDictionary=bivariateFeatDictionary)
    forwardResult = originalExpectedCompleteObjective.calculate(weights, bivariateFeatDictionary)
    funcValue = forwardResult['value']
    gradient = forwardResult['gradient']
    print(funcValue)
    print(gradient)



def main():
    testCalculate()

if __name__ == "__main__": main()


# ## check methods to test the correctness
# def test():
#     nStates = 4
#     nRep = 100
#     seedNum = np.arange(0, nRep)
#     np.random.seed(123)
#     weights = np.random.uniform(0, 1, nStates)
#     print(weights)
#     delta = 0.0001
#     exchangeCoef = np.array((1, 2, 3, 4, 5, 6))
#
#     ## get the rate matrix
#     testRateMtx = ReversibleRateMtxPiAndExchangeGTR(nStates, weights, exchangeCoef)
#     stationaryDist = testRateMtx.getStationaryDist()
#     rateMtx = testRateMtx.getRateMtx()
#     bt = 5.0
#     nSeq = 100
#
#     nInit = np.zeros(nStates)
#     holdTimes = np.zeros(nStates)
#     nTrans = np.zeros((nStates, nStates))
#
#     for j in range(nRep):
#         ## do forward sampling
#         seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seedNum[j], rateMtx, stationaryDist, bt)
#         ## summarize the sufficient statistics
#         ## extract first state from sequences
#         firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
#         unique, counts = np.unique(firstStates, return_counts=True)
#         nInitCount = np.asarray((unique, counts)).T
#         nInit = nInit + nInitCount[:, 1]
#
#         for i in range(nSeq):
#             sequences = seqList[i]
#             holdTimes = holdTimes + sequences['sojourn']
#             nTrans = nTrans + sequences['transitCount']
#
#     avgNTrans = nTrans/nRep
#     avgHoldTimes = holdTimes/nRep
#     avgNInit = nInit/nRep
#
#     originalExpectedCompleteObjective = ExpectedCompleteReversibleObjective(avgHoldTimes, avgNInit, avgNTrans)
#     forwardResult = originalExpectedCompleteObjective.calculateForPiUnnormalized(weights, exchangeCoef)
#     funcValue = forwardResult['value']
#     gradient = forwardResult['gradient']
#
#     ## another way of calculate the function value using for loop instead of vectorization
#     term1 = np.dot(avgNInit, np.log(stationaryDist))
#     term2 = 0
#     term3 = 0
#     for i in range(nStates):
#         term3 = term3 + avgHoldTimes[i] * rateMtx[i, i]
#         for j in range(nStates):
#             if j!=i:
#                 term2 = term2 + avgNTrans[i,j]* np.log(rateMtx[i,j])
#
#     funcValueAnotherApproach = -(term1 + term2 + term3 -0.5*1* np.dot(weights, weights))
#     print(funcValueAnotherApproach)
#     print(funcValue)
#
#     ## Here is the numerical method that we use to check the correctness of the gradient calculation
#     ## http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
#     ## Basically, we change each element of the weight one by one, denote funcDelta as the change in the function value
#     ## and the corresponding gradient for each element is funcDelta/delta, where delta is the change in each element of
#     ## the variables
#     newGradient = np.zeros(nStates)
#     for i in range(nStates):
#         newWeights = weights
#         nInitNew = np.zeros(nStates)
#         holdTimesNew = np.zeros(nStates)
#         nTransNew = np.zeros((nStates, nStates))
#         newWeights[i] = weights[i] + delta
#         testRateMtxNew = ReversibleRateMtxPiAndExchangeGTR(nStates, newWeights, exchangeCoef)
#         stationaryDistNew = testRateMtxNew.getStationaryDist()
#         rateMtxNew = testRateMtxNew.getRateMtx()
#         for j in range(nRep):
#             seqList =generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seedNum[j],rateMtxNew,stationaryDistNew, bt)
#             firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
#             unique, counts = np.unique(firstStates, return_counts=True)
#             nInitCount = np.asarray((unique, counts)).T
#             nInitNew = nInitNew + nInitCount[:, 1]
#
#             for k in range(nSeq):
#                 sequences = seqList[k]
#                 holdTimesNew = holdTimesNew + sequences['sojourn']
#                 nTransNew = nTransNew + sequences['transitCount']
#         avgNTransNew = nTransNew/nRep
#         avgHoldTimesNew = holdTimesNew/nRep
#         avgNInitNew = nInitNew/nRep
#         newExpectedCompleteObjective = ExpectedCompleteReversibleObjective(holdTimes=avgHoldTimesNew, nInit=avgNInitNew, nTrans=avgNTransNew)
#         newforwardResult = newExpectedCompleteObjective.calculateForPiUnnormalized(newWeights, exchangeCoef)
#         newFuncValue = newforwardResult['value']
#         newGradient[i] = (newFuncValue-funcValue)/delta
#
#     print(newGradient)
#    print(gradient)