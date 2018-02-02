#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:00:31 2017

@author: crystal
"""

def getBivariateFeatGradientIndex(state0, state1, nStates):
    if state0 > state1:
        tmpState = state1
        state1 = state0
        state0 = tmpState

    if state0 == 0:
        result = nStates-2 + state1 - state0
    if state0 >= 1:
        tmp = (2*nStates- state0-1)*state0
        if tmp % 2 !=0:
            raise ValueError("The number should be divided by 2")
        else:
            result = int((nStates -2) + tmp/2 + state1-state0)
                 
    return result

## Use the following input to test the correctness of the code
## this version corresponds to the restricted parameterization of the stationary distribution

## 0,1: 3    0,2:4, 0,3:5   1,2:6,   1,3:7,    2,3:8 when nStates = 4
## 1,0: 3    2,0:4   3,0:5   2,1:6   3,1:7     3:2:8 when nStates = 4
# print(getBivariateFeatGradientIndex(0, 1, 4))
# print(getBivariateFeatGradientIndex(0, 2, 4))
# print(getBivariateFeatGradientIndex(0, 3, 4))
# print(getBivariateFeatGradientIndex(1, 2, 4))
# print(getBivariateFeatGradientIndex(1, 3, 4))
# print(getBivariateFeatGradientIndex(2, 3, 4))
# print(getBivariateFeatGradientIndex(1, 0, 4))
# print(getBivariateFeatGradientIndex(2, 0, 4))
# print(getBivariateFeatGradientIndex(3, 0, 4))
# print(getBivariateFeatGradientIndex(2, 1, 4))
# print(getBivariateFeatGradientIndex(3, 1, 4))
# print(getBivariateFeatGradientIndex(3, 2, 4))



def getBivariateFeatGradientIndexWithoutPi(state0, state1, nStates, scalar=0):
    ## scalar represents the number of elements -1 used to represent the stationary distribution
    ## We use this function to get the index of the elements in the variables given the start
    ## and end state
    if state0 > state1:
        tmpState = state1
        state1 = state0
        state0 = tmpState

    if state0 == 0:
        result = state1-state0-1
    if state0 == 1:
        result = nStates-2 + state1-state0
    if state0 > 1:
        tmp = (2 * nStates - state0 - 1) * state0
        if tmp % 2 != 0:
            raise ValueError("The number should be divided by 2")
        else:
            result = int(tmp / 2 -1 + state1 - state0)

    return result

## test the correctness of the code when state0 is bigger than state1
# print(getBivariateFeatGradientIndexWithoutPi(1, 0, 4))
# print(getBivariateFeatGradientIndexWithoutPi(2, 0, 4))
# print(getBivariateFeatGradientIndexWithoutPi(3, 0, 4))
# print(getBivariateFeatGradientIndexWithoutPi(2, 1, 4))
# print(getBivariateFeatGradientIndexWithoutPi(3, 1, 4))
# print(getBivariateFeatGradientIndexWithoutPi(3, 2, 4))

## Use the following input to test the correctness of the code
## assuming we have 4 states and only the exchangeble coefficients are considered as our variables
## 0,1:0    0,2:1   0,3:2    1,2:3   1,3:4   2,3:5
## 1，0:0    2,0:1   3,0:2    2,1:3   3,1:4   3,2:5
## due to symmetry

##print(getBivariateFeatGradientIndexWithoutPi(0, 1, 4))
##print(getBivariateFeatGradientIndexWithoutPi(0, 2, 4))
##print(getBivariateFeatGradientIndexWithoutPi(0, 3, 4))
##print(getBivariateFeatGradientIndexWithoutPi(1, 2, 4))
##print(getBivariateFeatGradientIndexWithoutPi(1, 3, 4))
##print(getBivariateFeatGradientIndexWithoutPi(2, 3, 4))