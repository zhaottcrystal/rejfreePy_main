import sys

import numpy as np

#sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import os
#os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")

import scipy.optimize
DELTA = 1.0

# The correctness of this class has been tested

class MultipleConvexCollisionSolver:
    def __init__(self):
        pass

    def collisionTime(self, initialPoint, velocity, energy, exponential):

        directionMin = self.lineMinimize(initialPoint, velocity, energy)
        if directionMin == None:
            return np.inf

        time1 = self.time(initialPoint, directionMin, velocity)

        initialEnergy = energy.valueAt(directionMin)

        class LineSolvingFunction:

            def __init__(self):
                pass

            def value(self, time):
                candidatePosition = MultipleConvexCollisionSolver.position(directionMin, velocity, time)
                candidateEnergy = energy.valueAt(candidatePosition)
                delta = candidateEnergy - initialEnergy
                if delta < - 1e-6:
                    raise ValueError("Did not expect negative delta for convex objective Delta="+ str(delta) + ", time= " + str(time))
                return exponential - delta

        lineSolvingFunction = LineSolvingFunction()

        upperBound = self.findUpperBound(lineSolvingFunction)
        maxEval = 100

        time2 = scipy.optimize.brentq(lineSolvingFunction.value, a=0, b=upperBound)
        return time1 + time2


    def lineMinimize(self, initialPoint, velocity, energy):
        """
        :param initialPoint: initial position which is the current values of the parameters
        :param velocity: the velocity of the parameter
        :param energy: a differentiable function (interface/abstract class) which has two functions: one is valueAt() 
        the other is derivativeAt()
        :return: the position when the potential energy achieves its minimum
        """

        class LineRestricted:

            def __init__(self, initialPoint, velocity, energy):
                self.initialPoint = initialPoint
                self.velocity = velocity
                self.energy = energy

            def valueAt(self, _time):
                time = _time
                position = MultipleConvexCollisionSolver.position(initialPoint, velocity, time)
                return energy.valueAt(position)

            def dimension(self):
                return int(1)

            def derivativeAt(self, _time):
                time = _time
                position = MultipleConvexCollisionSolver.position(initialPoint, velocity, time)
                fullDerivative = energy.derivativeAt(position)
                directionalDeriv = np.dot(fullDerivative, velocity)
                return directionalDeriv

        lineRestricted = LineRestricted(initialPoint, velocity, energy)
        minResult = scipy.optimize.fmin_l_bfgs_b(lineRestricted.valueAt, x0=0.1, fprime=lineRestricted.derivativeAt, args=())
        if minResult[2]['warnflag']==0:
            minTime, f, d = minResult
        else:
            raise ValueError("Line search fails to find the local minimum")

        if minTime < 0.0:
            minTime = 0.0

        minValue = lineRestricted.valueAt(minTime)
        valuePlusDelta = lineRestricted.valueAt(minTime + DELTA)
        if valuePlusDelta < minValue:
            return None
        return MultipleConvexCollisionSolver.position(initialPoint, velocity, minTime)

    def time(self, initialPos, finalPosition, velocity):
        xInit = initialPos
        xFinal = finalPosition
        v = velocity
        return (xFinal-xInit)/v

    def findUpperBound(self, lineSolvingFunction):
        result = 1.0
        maxNIterations = 1022
        for i in range(maxNIterations):
            if lineSolvingFunction.value(result) < 0.0:
                return result
            else:
                result = result * 2.0

        raise ValueError("UpperBound exceeded")
    
    @staticmethod
    def position(initialPos, velocity, time):
        result = initialPos + velocity * time
        return result













