import numpy as np
from collections import OrderedDict
import numpy as np

class ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure:

    def __init__(self, nstates, stationaryWeights, binaryWeights, bivariateFeatIndexDictionary):
        self.nstates = nstates
        self.weights = stationaryWeights
        self.binaryWeights = binaryWeights
        self.graphDictionary = bivariateFeatIndexDictionary



    def getStationaryDist(self):
        unnormalized = np.exp(self.weights)
        totalSum = np.sum(unnormalized)
        result = unnormalized / totalSum
        return result

    def getRateMtx(self):
        ## define a nstates by nstates array
        result = np.zeros((self.nstates, self.nstates))
        stationary = self.getStationaryDist()


        wholeStates = np.arange(0, self.nstates)
        for state0 in range(self.nstates):
            support = np.setdiff1d(wholeStates, state0)
            for state1 in support:
                keyPair = (state0, state1)
                exchangeCoef = np.exp(np.sum(np.take(self.binaryWeights, self.graphDictionary[keyPair])))
                result[state0, state1] = exchangeCoef * stationary[state1]

        ## fill diagonal elements for each row
        for i in range(0, self.nstates):
            result[i, i] = -np.sum(result[i, :])

        return result

    def getExchangeCoef(self):
        exchangeList = list()
        wholeStates = np.arange(0, self.nstates)
        for state0 in range(self.nstates):
            support = np.setdiff1d(wholeStates, state0)
            for state1 in support:
                if state1 > state0:
                    keyPair = (state0, state1)
                    #print(keyPair)
                    exchangeList.append(np.exp(np.sum(np.take(self.binaryWeights, self.graphDictionary[keyPair]))))
        return exchangeList




    def getNormalization(self):
        ## get the normalization of the rate matrix to make sure then expected number of
        ## changes in one unit time is one.
        result = 0
        stationary = self.getStationaryDist()
        for i in range(0, self.nstates):
            result = result + stationary[i] * self.getRateMtx()[i, i]
        beta = -1 / result
        return beta

    def getNormalizedRateMtx(self):
        return self.getRateMtx() * self.getNormalization()



## test the correctness of the code
# nStates = 4
# ## generate the exchangeable coefficients
# ## set the seed so that we can reproduce generating the
# seed = 234
# np.random.seed(seed)
# nBivariateFeat = 10
# nChoosenFeatureRatio = 0.3
#
# bivariateWeights = np.random.normal(0, 1, nBivariateFeat)
# print(bivariateWeights)
# ##[ 0.81879162 -1.04355064  0.3509007   0.92157829 -0.08738186 -3.12888464
# ## -0.96973267  0.93466579  0.04386634  1.4252155 ]
#
#
#
# np.random.seed(seed)
# stationaryWeights = np.random.normal(0, 1, nStates)
# print(stationaryWeights)
# ## [ 0.81879162 -1.04355064  0.3509007   0.92157829]
#
#
# bivariateFeatIndexDictionary = generateBivariateFeatGradientIndexWithoutPiWithBivariateFeat(nStates, nBivariateFeat, nChoosenFeatureRatio)
# OrderedDict(bivariateFeatIndexDictionary)
#
# # OrderedDict([((0, 1), array([1, 2, 6])),
# #              ((1, 2), array([9, 4, 5])),
# #              ((3, 2), array([9, 1, 7])),
# #              ((1, 3), array([9, 7, 5])),
# #              ((3, 0), array([7, 1, 6])),
# #              ((3, 1), array([9, 7, 5])),
# #              ((2, 1), array([9, 4, 5])),
# #              ((0, 2), array([0, 2, 4])),
# #              ((2, 0), array([0, 2, 4])),
# #              ((2, 3), array([9, 1, 7])),
# #              ((1, 0), array([1, 2, 6])),
# #             ((0, 3), array([7, 1, 6]))])
#
#
#
#
# testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(4,  stationaryWeights, bivariateWeights, bivariateFeatIndexDictionary)
# print(testRateMtx.getStationaryDist())
# np.exp(stationaryWeights)/np.sum(np.exp(stationaryWeights)) ## this is correct
#
# ## check the correctness of the rate matrix
# print(testRateMtx.getRateMtx())  ## correct
# ## check the correctness of sevaral elements
# np.exp((bivariateWeights[1]+ bivariateWeights[2]+ bivariateWeights[6])) * testRateMtx.getStationaryDist()[1]
# np.exp((bivariateWeights[9]+ bivariateWeights[4]+ bivariateWeights[5])) * testRateMtx.getStationaryDist()[2]
# print(testRateMtx.getNormalizedRateMtx())
# print(testRateMtx.getExchangeCoef())