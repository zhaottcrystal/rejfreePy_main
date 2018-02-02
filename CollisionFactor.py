#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:01:29 2017

@author: crystal
"""

import abc

class CollisionFactor(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def getLowerBoundForCollisionDeltaTime(self, collisionContext):
        return
    
    @abc.abstractmethod
    def gradient(self):
        """Get the gradient of a factor in terms of the parameters"""
        return
    
    @abc.abstractmethod
    def getVariable(self, gradientCoordinate):
        """Get the value of the variables for the gradientCoordinate dimension"""
        return
    
    @abc.abstractmethod
    def nVariables(self):
        """Get the dimension of the parameters"""
        return
    
    @abc.abstractmethod
    def setPosision(self, position):
        """Set the position of the variables"""
        return 
        