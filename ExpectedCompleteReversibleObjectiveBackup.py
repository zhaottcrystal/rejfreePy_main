# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:32:39 2017

@author: crystal zhaott0416@gmail.com
"""

import sys

import numpy as np

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os

os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

from main.ReversibleRateMtxPiAndExchangeGTR import ReversibleRateMtxPiAndExchangeGTR
from main.FullTrajectorGeneration import getFirstAndLastStateOfListOfSeq
from main.FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
import copy
from scipy import optimize

class ExpectedCompleteReversibleObjective:
    def __init__(self, holdTimes, nInit, nTrans, kappa=1, exchangeCoef=None):
        """ This class has one attributes:
            expectedStatistics should come from ExpectedStatistics class,
            (refer to the java implementation to see ExpectedStatistics class)

            Later, we may need to implement other methods of this class
        """

        self.holdTimes = holdTimes
        self.nInit = nInit
        self.nTrans = nTrans
        self.kappa = kappa
        self.nStar = np.sum(self.nInit)
        self.TStar = np.sum(self.nTrans, axis=0)
        self.TStarStar = np.sum(self.TStar)
        ## the instance attributes of mTrans, mStar and mStarStar will be initialized in
        ## calculateForPiUnnormalized() function
        self.mTrans = None
        self.mStar = None
        self.mStarStar = None
        self.lastX = None
        if exchangeCoef is not None:
            self.exchangeCoef = exchangeCoef

        self.lastValue = None
        self.lastDerivative = None

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
        value = value - self.kappa * np.dot(weightsforPi, weightsforPi) / 2
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
        value = value - self.kappa * np.dot(weightsforPi, weightsforPi) / 2
        value = -value

        result = {'gradient': gradient, 'value': value}
        return result

    @staticmethod
    def requireUpdate(lastX, x):

        if lastX is not None:
            if len(lastX) != len(x):
                raise ValueError("The length of lastX and x should be equal.")

        if lastX is None:
            return True

        for i in range(len(x)):
            if lastX[i] != x[i]:
                return True
        return False

    def ensureCache(self, x):

        ## currently this implementation only works when x represents the weights for the stationary distribution
        if ExpectedCompleteReversibleObjective.requireUpdate(self.lastX, x):
            if not self.exchangeCoef is None:
                currentValueAndDerivative = self.calculateForPiUnnormalizedFixedExchangeCoef(x)
                self.lastValue = currentValueAndDerivative['value']
                self.lastDerivative = currentValueAndDerivative['gradient']
                if self.lastX is None:
                    self.lastX = np.zeros(len(x))
                self.lastX = x
                ## Todo: implementation when x includes both the weights for the stationary and the exchangeable coefficients

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


def gradientCheck(weights, nStates, nRep, nSeq, bt, rateMtxParameterizationName, useExchangeParam, exchangeParameters):

    ## rateMtxParameterizationName should be the string of the name of the rate matrix parameterization method

    seedNum = np.arange(0, nRep)
    print("The weights values are ")
    print(weights)
    print("The content of rateMtxParameterizationName is ")
    print(rateMtxParameterizationName)

    if useExchangeParam:
        rateMtxFunc = eval(rateMtxParameterizationName)(nStates, weights, exchangeParameters)

    stationaryDist = rateMtxFunc.getStationaryDist()
    rateMtx = rateMtxFunc.getRateMtx()

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

    originalExpectedCompleteObjective = ExpectedCompleteReversibleObjective(avgHoldTimes, avgNInit, avgNTrans)
    forwardResult = originalExpectedCompleteObjective.calculateForPiUnnormalized(weights, exchangeParameters)
    funcValue = forwardResult['value']

    print("The gradient is ")
    gradient = forwardResult['gradient']
    print(gradient)

    return funcValue



def testGradientCheck():

    nStates = 4
    np.random.seed(123)
    weights = np.random.uniform(0, 1, nStates)
    print(weights)
    nRep = 1000
    nSeq = 100
    bt = 5.0
    rateMtxParameterizationName = "ReversibleRateMtxPiAndExchangeGTR"
    useExchangeParam = True
    exchangeParameters = np.array((1, 2, 3, 4, 5, 6))

    numericGradient = optimize.approx_fprime(weights, gradientCheck, 0.000001, nStates, nRep, nSeq, bt,
                                             rateMtxParameterizationName, useExchangeParam, exchangeParameters)

    print("The numericGraident is ")
    print(numericGradient)








# test the correctness of the gradient，test passed, using numerical gradient
# check methods to test the correctness
# def test():
#     nStates = 4
#     nRep = 1000
#     seedNum = np.arange(0, nRep)
#     np.random.seed(123)
#     weights = np.random.uniform(0, 1, nStates)
#     print(weights)
#     delta = 0.000000000000001
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
#     avgNTrans = nTrans / nRep
#     avgHoldTimes = holdTimes / nRep
#     avgNInit = nInit / nRep
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
#             if j != i:
#                 term2 = term2 + avgNTrans[i, j] * np.log(rateMtx[i, j])
#
#     funcValueAnotherApproach = -(term1 + term2 + term3 - 0.5 * 1 * np.dot(weights, weights))
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
#         weightsCopy = copy.deepcopy(weights)
#         newWeights = copy.deepcopy(weights)
#         nInitNew = np.zeros(nStates)
#         holdTimesNew = np.zeros(nStates)
#         nTransNew = np.zeros((nStates, nStates))
#         print("The new weights are")
#         newWeights[i] = weightsCopy[i] + delta
#         print(newWeights)
#         print("The old weights are")
#         print(weights)
#         testRateMtxNew = ReversibleRateMtxPiAndExchangeGTR(nStates, newWeights, exchangeCoef)
#         stationaryDistNew = testRateMtxNew.getStationaryDist()
#         rateMtxNew = testRateMtxNew.getRateMtx()
#         for j in range(nRep):
#             seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, seedNum[j], rateMtxNew,
#                                                                     stationaryDistNew, bt)
#             firstStates = getFirstAndLastStateOfListOfSeq(seqList)['firstLastState'][:, 0]
#             unique, counts = np.unique(firstStates, return_counts=True)
#             nInitCount = np.asarray((unique, counts)).T
#             nInitNew = nInitNew + nInitCount[:, 1]
#
#             for k in range(nSeq):
#                 sequences = seqList[k]
#                 holdTimesNew = holdTimesNew + sequences['sojourn']
#                 nTransNew = nTransNew + sequences['transitCount']
#
#         avgNTransNew = nTransNew / nRep
#         avgHoldTimesNew = holdTimesNew / nRep
#         avgNInitNew = nInitNew / nRep
#         newExpectedCompleteObjective = ExpectedCompleteReversibleObjective(holdTimes=avgHoldTimesNew, nInit=avgNInitNew,
#                                                                            nTrans=avgNTransNew, exchangeCoef = exchangeCoef)
#         newforwardResult = newExpectedCompleteObjective.calculateForPiUnnormalized(newWeights, exchangeCoef)
#         newFuncValue = newforwardResult['value']
#         newGradient[i] = (newFuncValue - funcValue) / delta
#
#     print(newGradient)
#     print(gradient)



def main():
    testGradientCheck()

if __name__ == "__main__": main()