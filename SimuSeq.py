#import os
#cwd = os.getcwd()
#cwd

import sys
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
# import ReversibleRateMtx
from numpy.random import RandomState

class ForwardSimulation:

    def __init__(self, totalTime, rateMtx):
        self.totalTime = totalTime
        self.rateMtx = rateMtx

    def nextStateProb(self):
        # this returns a 4 by 4 matrix where the diagonal elements are zero
        # change the diagonal elements of the rateMtx to zero
        QProb = np.copy(self.rateMtx)
        np.fill_diagonal(QProb, 0)
        rowSumsQ = np.sum(QProb, axis=1)
        for i in range(0, self.rateMtx.shape[0]):
            QProb[i,:] = QProb[i,:]/rowSumsQ[i]
        return QProb
        
    def sampleNextState(self, prng,  curState):
        samplingProb = self.nextStateProb()[curState,:]
        ## sample the next state given the current state curState
        newState = prng.choice(self.rateMtx.shape[0] , p=samplingProb)
        return newState
        
    def getExpIntensity(self,curState):
        intensity = - self.rateMtx[curState, curState]
        return intensity
        
    def sampleSojournTime(self, prng, curState):
        ## the first parameter of np.random.exponential is scale but not rate parameter
        time = prng.exponential(1/self.getExpIntensity(curState), 1)
        return time
        
    def sampleStateTimeSeq(self, prng, curState):
        trest = self.totalTime
        prevTime = 0
        ## use a data structure to save state sequence and time sequence
        tlist = []
        slist = []
        ## saves the interval time between each two times when the state changes, 
        ## its length should be one shorter than tlist and slist
        deltaTlist = [] 

        ## initial state: time=0, state=curState
        tlist.append(0)
        slist.append(curState)
        
        ## get sojourn time at each state
        ## get transition counts between pairs of states
        ## create an array with same number of the 
        nstates = self.rateMtx.shape[0]
        sojournTime = np.zeros(nstates) 
        
        ## set a nstate by nstate array to save the number of transition counts
        ## between pairs of states
        transitCount = np.zeros((nstates, nstates))
        while(trest>0):
            waitingTime = self.sampleSojournTime(prng, curState)

            if(waitingTime > trest):
                sojournTime[curState] = sojournTime[curState]+trest
                tlist.append(self.totalTime)
                slist.append(curState)
                deltaTlist.append(trest)
                trest = 0

            if(waitingTime <= trest):
                sojournTime[curState] = sojournTime[curState] + waitingTime
                prevTime = prevTime + waitingTime
                nextState = self.sampleNextState(prng, curState)
                transitCount[curState, nextState] = transitCount[curState,nextState]+1
                curState = nextState
                tlist.append(prevTime)
                slist.append(nextState)
                deltaTlist.append(waitingTime)
                trest = trest -waitingTime

        ## the result that we return is a dictionary with three keys and three values
        ## where the keys are states, time, deltatime and the values are the three 
        ## corresponding lists
        result = {"time": tlist, "states": slist, "deltatime": deltaTlist,
        "sojourn": sojournTime, "transitCount": transitCount}
        return result

#################################################################
#################################################################
## this part of code is used to test the correctness of this class
#np.random.seed(123)
#rateMtxQ = ReversibleRateMtx.ReversibleRateMtx(4, np.random.uniform(0,1,10))
#Q = rateMtxQ.getRateMtx()
#Q
#stationary = rateMtxQ.getStationaryDist()
#simulator = ForwardSimulation(5000, Q)
#simulator.nextStateProb()
#simulator.getExpIntensity(1)     
## see if the next state frequency is consistent with nextStateProb
#nextStateList = np.zeros(5000)
#for i in range(0, 5000):
#    nextStateList[i] = simulator.sampleNextState(1)
#import pandas as pd
#ps = pd.Series([i for i in nextStateList])
#counts = ps.value_counts()
#print(counts/nextStateList.size)
#simulator.nextStateProb()[1]  ## the counts and nextStateProb are close, should be correct

#a = simulator.sampleStateTimeSeq(0)
#print(a["sojourn"]/simulator.totalTime)
#[0.32070426  0.21112307  0.18764373  0.28052893]
## compare it with stationary distribution
#stationary = rateMtxQ.getStationaryDist()
#array([ 0.31710181,  0.21037529,  0.19826512,  0.27425778])

##calculate (N(a,b)+N(b,a))/(pi(a)*h(b)+ pi(b)*h(a)), this is used to estimate the 
##exchangeable coefficients, theta(a,b), which is q(a,b)/pi(b)
##N(a, b) is the transition count between a and b, h(b) is the sojourn time for b, 
##pi(a) is the stationary distribution for a
##this estimate method comes from the paper "Maximum likelihood estimation of phylogenetic tree and
##substitution rates via generalized neighbor-joining and the EM algorithm"

#nstates = Q.shape[0]
#estimateMtx = np.zeros((Q.shape[0], Q.shape[0]))
#for i in range(0, nstates):
#    for j in range(0, nstates):
#        if(i != j):
#            estimateMtx[i,j] =  (a["transitCount"][i,j] + a["transitCount"][j,i])/(rateMtxQ.getStationaryDist()[i]*(a["sojourn"][j])+rateMtxQ.getStationaryDist()[j]*(a["sojourn"][i]))
#print(estimateMtx)
#[[ 0.          2.08011322  1.49000741  2.70417128]
# [ 2.08011322  0.          2.01391924  1.63019538]
# [ 1.49000741  2.01391924  0.          1.43815213]
# [ 2.70417128  1.63019538  1.43815213  0.        ]]

#exchangeCoef = np.copy(Q)
#for i in range(0, nstates):
#    exchangeCoef[:,i]= exchangeCoef[:,i]/stationary[i]
#print(exchangeCoef)
#[[-4.62302558  2.05334253  1.52669682  2.66649319]
# [ 2.05334253 -7.0730676   1.9834341   1.61758113]
# [ 1.52669682  1.9834341  -6.59377679  1.48011164]
# [ 2.66649319  1.61758113  1.48011164 -5.39384309]]
# This confirms the correctness of our ForwardSimulation code           
            
            
        
        
        
        
    

