#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:56:49 2017

@author: crystal
"""

from sortedcontainers import SortedDict
import numpy as np

class EventQueue:
    
    def __init__(self):
        self.sortedEvents = SortedDict()
        self.eventTimes = {}
    
    def pollEvent(self):
        return self.sortedEvents.popitem(last=False)
    
    def peekEvent(self):
        return self.sortedEvents.peekitem(index=0)
    
    def remove(self, event):
        time = self.eventTimes.get(event)
        if time != None:
            if time in self.sortedEvents:
                self.sortedEvents.pop(time)
            if event in self.eventTimes:
                self.eventTimes.pop(event)
            
    def clear(self):
        self.sortedEvents.clear()
        self.eventTimes.clear()
        
        
    def add(self, event, time):
        if np.isinf(time):
            return None
        if self.containsTime(time):
            raise ValueError("EventQueue does not support two events at the same time" + " " + str(time))

        if isinstance(time, np.ndarray):
            time = np.asarray(time, dtype=np.float)[0]
        self.sortedEvents[time] = event
        self.eventTimes[event] = time
        
        
    def containsTime(self, time):
        keys = np.copy(self.sortedEvents.keys())
        keys = np.array(keys)
        result = time in keys
        return result
        
        
    def peekTime(self):
        
        return self.sortedEvents.keys()[0]
        
        
## create test cases to validate the correctness of the code
#trajectories = EventQueue()
#trajectories.add("one", 1)
#trajectories.add("two", 2)
#trajectories.add("three", 3)
#trajectories.sortedEvents
#trajectories.eventTimes
#
#trajectories.peekEvent()
#trajectories.pollEvent()
#trajectories.sortedEvents
#
#trajectories.containsTime(3)
#trajectories.containsTime(4)
#trajectories.remove("two")
#trajectories.sortedEvents
#trajectories.eventTimes
#trajectories.peekTime()
#trajectories.add("four", 4)
#trajectories.sortedEvents
#trajectories.eventTimes
#trajectories.clear()
#trajectories.sortedEvents
#trajectories.eventTimes        
#        
        
    
    