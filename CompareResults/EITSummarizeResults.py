import plotnine
import pandas as pd
from plotnine import ggplot
from plotnine import aes
from plotnine import geom_histogram
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import numpy as np
import os
import argparse
import sys

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('-dir_name', action='store', dest='dir_name', type=str, default= "/Users/crystal/Dropbox/rejfreePy_main/EIT/", help='store the directory name to save the csv files')
results = parser.parse_args()

default_path = results.dir_name
os.chdir(default_path)


def TwoSamplesTestColwise(df1, df2, alpha=0.05, testName="KS"):
    ## alpha represents the significance level
    ## we conduct wilcox test for each same column of df1 and df2
    ## testName could be "KS" or "wilcox" indicating we are either using KS or wilcoxon to compare two samples

    if df1.shape[1] != df2.shape[1]:
        raise ValueError("The number of columns of the two data frames are different")

    ## create a pandas data frame to save the result
    result_empty = pd.DataFrame({'columnIndex':[],  'pvalue':[], 'stat': [], 'testName':[]})

    for i in range(df1.shape[1]):
        if testName == "KS":
            test = ks_2samp(df1[i], df2[i])
        elif testName == "Wilcox":
            test = wilcoxon(df1[i], df2[i])
        else:
            raise ValueError("We don't support other tests except KS test and wilcox test")

        
        stat = test[0]
        pvalue = test[1]
        result_empty.loc[i]  = [i,  pvalue, stat, testName]
    return result_empty

def TwoSamplesTestColwiseFromFilePath(filepath1, filepath2, header=None, alpha=0.05, testName="KS"):
    df1 = pd.read_csv(filepath1, header)
    df2 = pd.read_csv(filepath2, header)
    testResult = TwoSamplesTestColwise(df1, df2, alpha=0.05, testName=testName)
    if testResult.shape[0] >0 :
        print(testResult)
    return testResult


orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

## example use of this function
## test for exchangeable parameters
fHMC = pd.read_csv("fExchangehmc.csv", header=None)
mHMC = pd.read_csv("hmcExchangeCoef.csv", header=None)
exchangeHMC_KS = TwoSamplesTestColwise(fHMC, mHMC, alpha=0.05)
exchangeHMC_wilcox = TwoSamplesTestColwise(fHMC, mHMC, alpha=0.05, testName="Wilcox")
if exchangeHMC_KS.shape[0] > 0:
    print("The exchangeable parameters from HMC using KS doesn't follow the same dist as the prior")
    print(exchangeHMC_KS)
if exchangeHMC_wilcox.shape[0] > 0:
    print("The exchangeable parameters from HMC using Wilcox doesn't follow the same dist as the prior")
    print(exchangeHMC_wilcox)

fBPS = pd.read_csv("fExchange.csv", header=None)
mBPS = pd.read_csv("bpsExchangeCoef.csv", header=None)
exchangeBPS_KS = TwoSamplesTestColwise(fBPS, mBPS, alpha=0.05)
exchangeBPS_wilcox = TwoSamplesTestColwise(fBPS, mBPS, alpha=0.05,testName="Wilcox")
if exchangeBPS_KS.shape[0] > 0:
    print("The exchangeable parameters from BPS using KS doesn't follow the same dist as the prior")
    print(exchangeBPS_KS)
if exchangeBPS_wilcox.shape[0] > 0:
    print("The exchangeable parameters from BPS using Wilcox doesn't follow the same dist as the prior")
    print(exchangeBPS_wilcox)



## test for stationary distribution
fstationary = pd.read_csv("fStationary.csv", header=None)
mstationary = pd.read_csv("bpsStationaryDist.csv", header=None)
stationaryDistBPS_KS = TwoSamplesTestColwise(fstationary, mstationary, alpha=0.05)
stationaryDistBPS_wilcox = TwoSamplesTestColwise(fstationary, mstationary, alpha=0.05,testName="Wilcox")
if stationaryDistBPS_KS.shape[0] > 0:
    print("The stationary dist from BPS using KS doesn't follow the same dist as the prior")
    print(stationaryDistBPS_KS)
if stationaryDistBPS_wilcox.shape[0] >0:
    print("The stationary dist from BPS using Wilcox doesn't follow the same dist as the prior")
    print(stationaryDistBPS_wilcox)

fstationaryHMC = pd.read_csv("fStationaryhmc.csv", header=None)
mstationaryHMC = pd.read_csv("hmcStationaryDist.csv", header=None)
stationaryDistHMC_KS = TwoSamplesTestColwise(fstationaryHMC, mstationaryHMC, alpha=0.05)
stationaryDistHMC_wilcox = TwoSamplesTestColwise(fstationaryHMC, mstationaryHMC, alpha=0.05, testName="Wilcox")
if stationaryDistHMC_wilcox.shape[0] >0:
    print("The stationary dist from HMC using KS doesn't follow the same dist as the prior")
    print(stationaryDistHMC_KS)
if stationaryDistHMC_wilcox.shape[0] >0:
    print("The stationary dist from HMC using Wilcox doesn't follow the same dist as the prior")
    print(stationaryDistHMC_wilcox)


fweights = pd.read_csv("fWeights.csv", header=None)
nStates = fstationary.shape[1]
fstationaryWeights = fweights.iloc[:,0:nStates]
mstationaryWeights = pd.read_csv("bpsStationaryWeights.csv", header=None)
stationaryWeightsBPS_KS = TwoSamplesTestColwise(fstationaryWeights, mstationaryWeights, alpha=0.05)
stationaryWeightsBPS_wilcox = TwoSamplesTestColwise(fstationaryWeights, mstationaryWeights, alpha=0.05, testName="Wilcox")
if stationaryWeightsBPS_KS.shape[0] > 0:
    print("The stationary weights from BPS using KS doesn't follow the same dist as the prior")
    print(stationaryWeightsBPS_KS)
if stationaryWeightsBPS_wilcox.shape[0] >0:
    print("The stationary weights from BPS using Wilcox doesn't follow the same dist as the prior")
    print(stationaryWeightsBPS_wilcox)


##  test for binary weights
nExchangeCoef = int(nStates * (nStates-1)/2)
fBinaryWeights = fweights.iloc[:, nStates:int(nStates+nExchangeCoef)]
fBinaryWeights.columns = np.arange(0, nExchangeCoef, 1)
mBinaryWeights = pd.read_csv("bpsBinaryWeights.csv", header=None)
binaryWeightsBPS_KS = TwoSamplesTestColwise(fBinaryWeights, mBinaryWeights, alpha=0.05)
binaryWeightsBPS_wilcox = TwoSamplesTestColwise(fBinaryWeights, mBinaryWeights, alpha=0.05, testName="Wilcox")
if binaryWeightsBPS_KS.shape[0] > 0:
    print("The binary weights from BPS using KS doesn't follow the same dist as the prior")
    print(binaryWeightsBPS_KS)
if binaryWeightsBPS_wilcox.shape[0] >0:
    print("The binary weights from BPS using Wilcox doesn't follow the same dist as the prior")
    print(binaryWeightsBPS_wilcox)


fweightsHMC = pd.read_csv("fWeightshmc.csv", header=None)
fstationaryWeightsHMC = fweightsHMC.iloc[:, 0:nStates]  ## weights 4, reject the null hypothesis
mstationaryWeightsHMC = pd.read_csv("hmcStationaryWeights.csv", header=None)
stationaryWeightsHMC_KS = TwoSamplesTestColwise(fstationaryWeightsHMC, mstationaryWeightsHMC, alpha=0.05)
stationaryWeightsHMC_wilcox = TwoSamplesTestColwise(fstationaryWeightsHMC, mstationaryWeightsHMC, alpha=0.05,testName="Wilcox")
if stationaryWeightsHMC_KS.shape[0] > 0:
    print("The stationary weights from HMC using KS doesn't follow the same dist as the prior")
    print(stationaryWeightsHMC_KS)
if stationaryWeightsHMC_wilcox.shape[0] >0:
    print("The stationary weights from HMC using Wilcox doesn't follow the same dist as the prior")
    print(stationaryWeightsHMC_wilcox)


##  test for binary weights
fBinaryWeightsHMC = fweightsHMC.iloc[:, nStates:int(nStates+nExchangeCoef)]
fBinaryWeightsHMC.columns = np.arange(0, nExchangeCoef, 1)
mBinaryWeightsHMC = pd.read_csv("hmcBinaryWeights.csv", header=None)
binaryWeightsHMC_KS = TwoSamplesTestColwise(fBinaryWeightsHMC, mBinaryWeightsHMC, alpha=0.05)
binaryWeightsHMC_wilcox = TwoSamplesTestColwise(fBinaryWeightsHMC, mBinaryWeightsHMC, alpha=0.05, testName="Wilcox")
if binaryWeightsHMC_KS.shape[0] > 0:
    print("The binary weights from HMC using KS doesn't follow the same dist as the prior")
    print(binaryWeightsHMC_KS)
if binaryWeightsHMC_wilcox.shape[0] >0:
    print("The binary weights from HMC using Wilcox doesn't follow the same dist as the prior")
    print(binaryWeightsHMC_wilcox)


sys.stdout = orig_stdout
f.close()
