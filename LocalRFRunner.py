import os
import sys

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import OptionClasses

#from main.OptionClasses import LocalRFRunnerOptions
from numpy.random import RandomState


class LocalRFRunner:
    def __init__(self, sampler, model, prng, rfOptions, maxTimeMilli = sys.maxsize):
        ## sampler should be an instance of LocalRFRSampler or LocalRFSamplerForOnlyExchangeCoefParam
        ## model should be an instance of expectedCompleteReversibleModel
        ## randomSeed is the seed of "random"

        self.sampler = sampler
        self.model = model
        self.options = rfOptions
        self.prng = prng
        self.maxTimeMilli = maxTimeMilli


    def isInit(self):
        return self.model is not None

    def checkInit(self):
        if not self.isInit():
            raise ValueError("Model should first be initialized")

    def init(self, model):
        if self.isInit():
            raise ValueError("The model has been already initialized")
        self.model = model
        self.sampler.model = model

    def run(self):
        maxNumberOfIterations = self.options.maxSteps
        maxTrajectoryLen = self.options.trajectoryLength
        maxTimeMilli = self.maxTimeMilli
        newPoints = self.sampler.iterate(self.prng, maxNumberOfIterations, maxTrajectoryLen, maxTimeMilli)
        return newPoints



