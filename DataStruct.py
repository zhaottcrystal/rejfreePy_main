import sys

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")


class DataStruct:
    def __init__(self, next_q=None, accept=False, proposal=None, q=None, mr=0, randomStep=-1, energy=0):
        self.next_q = next_q
        self.accept = accept
        self.proposal = proposal
        self.q = q
        self.mr = mr
        self.randomStep = randomStep
        self.energy = energy

