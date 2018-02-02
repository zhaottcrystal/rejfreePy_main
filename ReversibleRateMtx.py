## create the class where once weight is given, we can calculate stationary distribution and rate matrix
import numpy as np
from scipy import linalg
import math

class ReversibleRateMtx:
    def __init__(self, nstates, weights):
        self.nstates = nstates
        self.weights = weights
        
    def getStationaryDist(self):
        ## get the first nstate elements of weights, which are the weights for univariate features
        result = self.weights[0:self.nstates]
        candidate = np.exp(result)
        sum = np.sum(candidate)
        result = candidate/sum
        return result
    
    def getRateMtx(self):
        ## define a nstates by nstates array
        result = np.zeros((self.nstates, self.nstates))
        ## get the upper triangle elements of the matrix, without diagonal elements
        upperIdx = np.triu_indices(self.nstates, 1)
        result[upperIdx] = np.exp(self.weights[(self.nstates): self.weights.size])
        ##copy the upper triangle part to the lower triangle part
        #lowerIdx = np.tril_indices(nstates, -1)
        #result[lowerIdx] = np.exp(weights[(nstates): weights.size])
        result = np.triu(result).T + np.triu(result) 
        
        stationary = self.getStationaryDist()    
        for i in range(0, self.nstates):
            result[:, i] = result[:, i] * stationary[i]
        
        ## fill diagonal elements for each row
        for i in range(0, self.nstates):
            result[i, i] = -np.sum(result[i,:])
            
        return result
        
        
    def getNormalization(self):
        ## get the normalization of the rate matrix to make sure then expected number of
        ## changes in one unit time is one.
        result = 0
        stationary = self.getStationaryDist()
        for i in range(0, self.nstates):
            result = result + stationary[i]*self.getRateMtx()[i, i]
        beta = -1/result
        return beta
        
    def getNormalizedRateMtx(self):
        return self.getRateMtx()*self.getNormalization()





        
        
        
        
        
        

