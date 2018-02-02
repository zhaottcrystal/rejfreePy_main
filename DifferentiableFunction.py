# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:01:29 2017

@author: crystal
"""

import abc


class DifferentiableFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def dimension(self):
        return

    @abc.abstractmethod
    def valueAt(self, x):
        """Get the value of the function evaluated at x"""
        return

    @abc.abstractmethod
    def derivativeAt(self, x):
        """Get the value of the derivative of the function evaluated at x"""
        return
