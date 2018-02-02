import numpy as np

import Utils


class CovSEARD:

    def __init__(self, hyp):
        # hyp is the hyper-parameters of the kernel
        # hyp should be an n-by-2 array
        self.hyp = hyp
        range = np.arange(0, (hyp.shape[0]-1))
        self.ell = self.hyp[range, 0]
        self.sf2 = hyp.flattern()[(hyp.shape[0]-1)]

    def cov(self, x, y):
        """
        :param x: n by D matrix
        :param y: m by D matrix
        :return: kernel matrix
        """
        di = np.fill_diagonal(1/self.ell)
        K = Utils.SquareDistance(np.matmul(di, x), np.matmul(di, x))
        result = np.exp(-K/2.0 * self.sf2)
        return result


def main():
    a = np.array((0.1, 0.1))
    b = np.array((0.2, 0.1))
    print(Utils.SquareDistance(a, b))

if __name__ == '__main__':
    main()


