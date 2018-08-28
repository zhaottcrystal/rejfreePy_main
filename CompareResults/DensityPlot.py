import pandas as pd
from plotnine import ggplot
from plotnine import aes
from plotnine import geom_density
from plotnine import xlab
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
from seaborn import kdeplot
from plotnine import scale_x_continuous
from plotnine import geom_vline
import numpy as np
import pymc3 as pm

import os
os.chdir("/Users/crystal/Dropbox/rejfreePy_main/CompareResults")
from ESS import essCalculator

hmcSamples = pd.read_csv("/Users/crystal/Dropbox/rejfreePy_main/newResults/2States/exchangeableCoefHMC10000initialWeightDistFixednSeq500nStates2mcmcSamplingSeed3trajectoryLength0.1nLeapFrogSteps40stepSize0.001mcmcSeeed3.csv",  header=None)
hmcSamples.head(3)

bpsSamples = pd.read_csv("/Users/crystal/Dropbox/rejfreePy_main/newResults/2States/exchangeableCoefHMCPlusBPS20000initialWeightDistFixedLOCALnSeq500nStates2mcmcSamplingSeed3trajectoryLength0.1nLeapFrogSteps40stepSize0.001mcmcSeeed3.csv",   header=None)
bpsSamples = bpsSamples.rename(columns={0:'exchangeCoef1'})

hmcSamples.rename(columns={0:'exchangeCoef1'}, inplace=True)
bpsSamples.rename(columns={0:'exchangeCoef1'}, inplace=True)

#pm.autocorrplot(hmcSamples, varnames=['exchangeCoef1'])

ks_2samp(hmcSamples[1000:10000]['exchangeCoef1'], bpsSamples[1000:20000][ 'exchangeCoef1'])
## wilcox test
wilcoxon(hmcSamples[1000:10000]['exchangeCoef1'], bpsSamples[1000:10000][ 'exchangeCoef1'])

## throw the burnin period away
burninRatio =0.3
burninHMC = int(hmcSamples.shape[0] * burninRatio)
burninBPS = int(bpsSamples.shape[0] * burninRatio)
t1 = bpsSamples[burninBPS:bpsSamples.shape[0]]
t2 = hmcSamples[burninHMC:hmcSamples.shape[0]]

ess_t1 = essCalculator(t1['exchangeCoef1'])
ess_t2 = essCalculator(t2['exchangeCoef1'])

## read and get the running time of the two files in seconds 
timeHMC = pd.read_csv("/Users/crystal/Dropbox/rejfreePy_main/newResults/2States/wallTimeHMC100002stepSize0.001nLeapFrogSteps40.csv", header=None)[0][1]
timeBPS = pd.read_csv("/Users/crystal/Dropbox/rejfreePy_main/newResults/2States/wallTimeHMCPlusBPS200002trajectoryLength0.1refreshementMethodLOCAL.csv", header=None)[0][1]

ess_bps_per_sec = ess_t1/(float(timeBPS)*(1-burninRatio))
ess_hmc_per_sec = ess_t2/(float(timeHMC)**(1-burninRatio))

print(ess_bps_per_sec)
# 0.004061
print(ess_hmc_per_sec)
# 0.005769



frames = [bpsSamples[burninBPS:bpsSamples.shape[0]], hmcSamples[burninHMC:hmcSamples.shape[0]]]
allDF = pd.concat(frames)
## add a new column to DF 
list1 = ["lbps"]*(bpsSamples.shape[0]-burninBPS)
list1.extend(["hmc"]*(hmcSamples.shape[0]-burninHMC))
allDF['method'] = list1
allDF.head(3)

ggplot(allDF, aes('exchangeCoef1', fill='method')) + geom_density(alpha=0.3, position='identity')+ scale_x_continuous(breaks=np.arange(3.5, 5.5, 0.3)) 

#import matplotlib.pyplot as plt

#kdeplot(bpsSamples['exchangeCoef1'][1000:10000], shade=True)
#kdeplot(hmcSamples['exchangeCoef1'][1000:10000], shade=True,color='r')

#ggplot(df, aes( x=values, fill=method)) +
#  geom_density(alpha=.3, position="identity")+facet_wrap(~variable, ncol=3, scales="free")+
#  geom_vline(aes(xintercept=vl), data=vline.dat, color="red", linetype="dashed")
  #sc
  