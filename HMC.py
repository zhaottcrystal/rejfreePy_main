import math
import sys

import numpy as np

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import DataStruct

#from main.DataStruct import  DataStruct

class HMC:
    def __init__(self, l, epsilon, gradient, func):
        self.l = l
        self.epsilon = epsilon
        self.gradient = gradient
        self.func = func

    def run(self, burnIn, totalNumSample, sample):

        # sample.shape[0] provides the number of rows of "sample"

        samples = np.zeros(((totalNumSample-burnIn), sample.shape[0]))

        for i in range(totalNumSample):
            result = self.doIter(self.l, self.epsilon, sample, self.gradient, self.func, True)
            sample = result.next_q
            if (i+1) % 100 == 0:
                print("Iteration" + str(i+1)+".")
            if i >= burnIn:
                samples[(i-burnIn), :] = sample

        return samples

    @staticmethod
    def doIter(l, epsilon, lastSample, gradient, func, randomizedNumberOfSteps):

        D = lastSample.shape[0]

        if randomizedNumberOfSteps:
            randomStep = math.ceil(np.random.uniform(0, 1, 1)*l)
        else:
            randomStep = l

        proposal = lastSample

        ## generate Momentum Vector
        old_p = np.random.normal(0, 1, D)

        p = old_p - gradient.mFunctionValue(proposal) * 0.5 * epsilon

        for i in range(randomStep):
            proposal = proposal + p * epsilon
            p = p - gradient.mFunctionValue(proposal) * 0.5 * epsilon

        p = p + gradient.mFunctionValue(proposal) * 0.5 * epsilon

        proposed_E = func.functionValue(proposal)
        original_E = func.functionValue(lastSample)

        proposed_K = np.dot(p, p)/2
        original_K = np.dot(old_p, old_p)/2

        mr = -proposed_E + original_E + original_K-proposed_K

        if not np.isnan(mr):
            if mr > 1:
                mr =1
            else:
                mr = np.min((math.exp(mr), 1.0))
        else:
            mr = 0.0

        nextSample = proposal

        accept = True

        ## should I use this negative sign or not
        ## the function value returns the potential energy
        ## double check about this
        energy = - proposed_E
        if np.random.uniform(0, 1, 1) > mr:
            nextSample = lastSample
            accept = False
            energy = -original_E

        return DataStruct.DataStruct(nextSample, accept, proposal, lastSample, mr, randomStep, energy)

class GaussianExample:

    def __init__(self, targetMean, targetSigma):
        self.targetSigma = targetSigma
        self.targetMean = targetMean

    def functionValue(self, vec):
        # this returns the potential energy or in other words,
        # the negative of the log-density
        diff = vec - self.targetMean
        term1 = np.linalg.solve(self.targetSigma, diff)
        result = 0.5 * np.matmul(diff, term1)
        return result

    def mFunctionValue(self, vec):
        diff = vec - self.targetMean
        return np.linalg.solve(self.targetSigma, diff)

def main():
    # my code here
    targetSigma = np.array((1.0, 0.99, 0.99, 1.0))
    targetSigma = targetSigma.reshape(2,2)
    targetMean = np.array((3.0, 5.0))
    ge = GaussianExample(targetMean, targetSigma)
    hmc = HMC(40, 0.05, ge, ge)
    sample = np.array((1.0, 2.0))
    samples = hmc.run(0, 5000, sample)
    print(np.sum(samples, axis=0)/samples.shape[0])

if __name__ == '__main__':
    main()
























