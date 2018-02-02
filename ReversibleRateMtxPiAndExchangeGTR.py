import numpy as np
class ReversibleRateMtxPiAndExchangeGTR:
    def __init__(self, nstates, weights, exchangeCoef):
        self.nstates = nstates
        self.weights = weights
        self.exchangeCoef = exchangeCoef

    def checkCorrectnessOfParameterization(self):
        if self.nstates != len(self.weights):
            raise ValueError(
                "The number of states is different from the dimension of the weights for the stationary dist.")
        if len(self.exchangeCoef) != int(self.nstates * (self.nstates - 1) / 2):
            raise ValueError(
                "The number of exchangeable parameters is not correct according to the GTR parameterization")

    def getStationaryDist(self):

        self.checkCorrectnessOfParameterization()
        unnormalized = np.exp(self.weights)
        totalSum = np.sum(unnormalized)
        result = unnormalized / totalSum
        return result

    def getRateMtx(self):
        ## define a nstates by nstates array
        result = np.zeros((self.nstates, self.nstates))
        ## get the upper triangle elements of the matrix, without diagonal elements
        upperIdx = np.triu_indices(self.nstates, 1)
        result[upperIdx] = self.exchangeCoef
        result = np.triu(result).T + np.triu(result)

        stationary = self.getStationaryDist()
        for i in range(0, self.nstates):
            result[:, i] = result[:, i] * stationary[i]

        ## fill diagonal elements for each row
        for i in range(0, self.nstates):
            result[i, i] = -np.sum(result[i, :])

        return result

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


## test the correctness of ReversibleRateMtxPiAndExchangeGTR
testRateMtx = ReversibleRateMtxPiAndExchangeGTR(4, np.array((1, 2, 3, 4)), np.array((1, 2, 3, 4, 5, 6)))
print(testRateMtx.getStationaryDist())
print(np.exp(np.array((1, 2, 3, 4))) / np.sum(np.exp(np.array((1, 2, 3, 4)))))
## check the correctness of the rate matrix
print(testRateMtx.getRateMtx())  ## correct
print(testRateMtx.getNormalizedRateMtx())