import numpy as np
from scipy.linalg import expm

class RateMtxExpectations:
    """
     Using and augmented matrix with a Pade+scaling approach (faster for big matrices and
     does not require diagonalizable matrices)

     You can consult the following paper  "Comparison of methods for calculating conditional
     expectations of sufficient statistics for continuous time Markov chains"
     by Tataru (2011) to understand how this implementation works in order to get those
     sufficient statistics

     We are implementing algorithm 3 "EXPM" of this paper
    """


    def __init__(self, rateMtx, T):
        self.rateMtx = rateMtx
        self.T = T


    def expectations(self):
        """
        for (a != b):
            A[i][j][a][b] = E[N(a->b)|X_0=i, X_T=j]
        for (a == b):
            A[i][j][a][a] = E[T(a)|X_0=i, X_T=j]
        where
        N(a->b) is the number of transitions from a to b in the interval [0,T] and
        T(a) is the time spent at time in state a in the interval [0,T]

        :return: sufficient statistics of the the holding time and transitions
        """
        rateMtx = self.rateMtx
        T = self.T

        n = int(rateMtx.shape[0])
        simpleExp = expm(rateMtx * T)

        ## create 4-dimensional array to save result
        result = np.full((n, n, n, n), 0)
        emptyMtx = np.zeros((2*n, 2*n))

        for state1 in range(n):
            for state2 in range(n):
                current = self._expectations(state1, state2, simpleExp,emptyMtx)
                for i in range(n):
                    for j in range(n):
                        result[i][j][state1][state2] = current[i][j]

        return result


    def expectationsWithMarginalCount(self, marginalCounts):
        rateMtx = self.rateMtx
        T = self.T
        n = int(rateMtx.shape[0])
        simpleExp = expm(rateMtx * T)
        auxMtx = np.zeros((2*n, 2*n))
        result = np.zeros((n, n))

        for state1 in range(n):
            for state2 in range(n):
                current = self._expectations(state1, state2, simpleExp, auxMtx)
                sum = 0.0
                for i in range(n):
                    for j in range(n):
                        sum = sum + current[i][j] * marginalCounts[i][j]
                result[state1][state2] = sum

        return result


    def _expectations(self, state1, state2, matrixExponential, emptyMtx):

        rateMtx = self.rateMtx
        T = self.T

        n = int(rateMtx.shape[0])

        if rateMtx.shape[0] != matrixExponential.shape[0]:
            raise ValueError("The dimension of the rate matrix and the exponential of rate matrix doesn't match")

        aux = emptyMtx
        for i in range(n):
            for j in range(n):
                aux[i][j] = aux[(i+n)][(j+n)] = rateMtx[i][j] * T

        aux[state1][(state2+n)] = 1.0 * T

        exponentiatedAux = expm(aux)

        result = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if state1 == state2:
                    result[i][j] = exponentiatedAux[i][(n+j)]/matrixExponential[i][j]
                else:
                    result[i][j] = exponentiatedAux[i][(n+j)]/matrixExponential[i][j] * rateMtx[state1][state2]

        aux[state1][(state2+n)] = 0.0

        return result