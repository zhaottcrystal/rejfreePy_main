from enum import Enum

class CovModel(Enum):

    seIso = 1
    seArd = 2

    @staticmethod
    def cov(x, y):
        return None