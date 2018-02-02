#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:05:11 2017

@author: crystal
"""

import sys
sys.path
#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")

class Path:
    
    def __init__(self):
        self.times = []
        self.states = []
        
    def nSojourns(self):
        return len(self.times)
    
    def isEmpty(self):
        return self.nSojourns() == 0
    
    def lastState(self):
        return self.states[(len(self.states)-1)]
      
    def firstState(self):
        return self.states[0]
    
    def totalLength(self):
        return sum(self.times)
    
    def addSegment(self, currentPoint, time):
        if len(self.times)!= len(self.states):
            raise ValueError("The length of the times and states should be equal")
        
        if not self.isEmpty() and self.lastState()==currentPoint:
            self.times[(len(self.times)-1)] = self.times[(len(self.states)-1)] + time
                      
        else:
            self.times.append(time)
            self.states.append(currentPoint)
            
        
## check the correctness of the code
## pathExam = Path()
## pathExam.isEmpy()
## pathExam.addSegment(0, 1.2) 
## pathExam.addSegment(1, 1.5)
## pathExam.addSegment(2, 0.8)  
## pathExam.isEmpy()
## pathExam.firstState()
## pathExam.lastState()
## pathExam.addSegment(2, 1.2)
## pathExam.states
## pathExam.times