import os
import sys

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import OptionClasses
import LocalRFRunner

## from main.OptionClasses import LocalRFRunnerOptions
#from main.LocalRFRunner import LocalRFRunner
from numpy.random import RandomState

class PhyloLocalRFMove:

    def __init__(self, model, sampler, initialPoints, options, prng= None, randomSeed = None):
        self.model = model
        self.sampler = sampler
        self.options = options
        self.parameters = initialPoints

        if prng is not None:
            self. prng = prng
        else:
            if randomSeed is not None:
                self.prng = RandomState(randomSeed)
            else:
                raise Exception("prng and random seed can't be None at the same time")

    def execute(self):
        rfRunner = LocalRFRunner.LocalRFRunner(sampler=self.sampler, model=self.model, prng  = self.prng, rfOptions=self.options)
        newPoints = rfRunner.run()
        return newPoints
    


