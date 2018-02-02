#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:33:20 2017

@author: crystal
"""

class TrajectoryRay:
    """A ray is parameterized by a starting point (time, position at that time),
    and the velocity just after that time (i.e. that starting point is
    assumed to be a collision, so the we store the velocity just after the
    bounce)."""
    
    def __init__(self, t, position_t, velocity_t):
        self.t = t
        self.position_t = position_t
        self.velocity_t = velocity_t
        
    def position(self, time):
        if(time < self.t):
            raise ValueError("Current time cannot be smaller than start t")
        
        return self.position_t + (time - self.t) * self.velocity_t
    
    def toString(self):
        print("TrajectoryRay [t=" + str(self.t) + ", position_t=" + str(self.position_t)
                + ", velocity_t=" + str(self.velocity_t) + "]")
    
    
## test case to test the correctness of this class
# trajectory = TrajectoryRay(0, 1, 2)
# newLoc = trajectory.position(2)
# print(newLoc)
# trajectory.toString()