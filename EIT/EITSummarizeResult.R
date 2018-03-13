fHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/fExchangehmc.csv", header=FALSE)
mHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/hmcExchangeCoef.csv", header=FALSE)
ks.test(fHMC[,4], mHMC[,4])  ## reject null hypothesis
wilcox.test(fHMC[,6], mHMC[,6])

fBPS = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/fExchange.csv", header=FALSE)
mBPS = mHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/bpsExchangeCoef.csv", header=FALSE)

print(ks.test(fBPS[, 4], mBPS[, 4])$p.value)
## significantly different, 2,3,4
wilcox.test(fBPS[, 6], mBPS[,6])

## test for stationary distribution
fstationary = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/fStationary.csv", header=FALSE)
mstationary = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/bpsStationaryDist.csv", header=FALSE)
print(ks.test(fstationary[,4], mstationary[,4])$p.value)
wilcox.test(fstationary[,2], mstationary[,2])

fstationaryHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/fStationaryhmc.csv", header=FALSE)
mstationaryHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/hmcStationaryDist.csv", header=FALSE)
ks.test(fstationaryHMC[,1], mstationaryHMC[,1])
wilcox.test(fstationaryHMC[,1], mstationaryHMC[,1])

wilcoxTestForEachColumn = function(df1, df2, significanceLevel){
  ## this function conduct wilcox test for each column of 
  ## the two data frames, and compare it to the significance level
  ## if the pvalue of the test is smaller than the significance level
  ## the corresponding pvalue and the column index are returned
  
  for(i in 1:dim(df1)[2]){
    
    
    
    
    
  }
  
  
  
  
  
  
  
}

## test for stationary weights
fweights = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/fWeights.csv", header=FALSE)
fstationaryWeights = fweights[, 1:4]
mstationaryWeights = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/bpsStationaryWeights.csv", header=FALSE)
print(ks.test(fstationaryWeights[,1], mstationaryWeights[,1])$p.value)  ## 1 significant different

wilcox.test(fstationaryWeights[,1], mstationaryWeights[,1])

##  test for binary weights  
fBinaryWeights = fweights[, 4:10]
mBinaryWeights = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/bpsBinaryWeights.csv", header=FALSE)
print(ks.test(fBinaryWeights[,5], mBinaryWeights[,5])$p.value)  ## 5, significant different
wilcox.test(fBinaryWeights[,1], mBinaryWeights[,1]) ##1, 5, 4 are all significantly different 

fweightsHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/fWeightshmc.csv", header=FALSE)
fstationaryWeightsHMC = fweightsHMC[, 1:4]  ## weights 4, reject the null hypothesis
mstationaryWeightsHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/hmcStationaryWeights.csv", header=FALSE)
ks.test(fstationaryWeightsHMC[,4], mstationaryWeightsHMC[,4])  ## weights 4, reject the null hypothesis
wilcox.test(fstationaryWeightsHMC[,4], mstationaryWeightsHMC[,4])

##  test for binary weights  
fBinaryWeightsHMC = fweightsHMC[, 4:10]
mBinaryWeightsHMC = read.csv("/Users/crystal/Dropbox/rejfreePy_main/EIT/hmcBinaryWeights.csv", header=FALSE)
ks.test(fBinaryWeightsHMC[,6], mBinaryWeightsHMC[,6])
wilcox.test(fBinaryWeightsHMC[,2], mBinaryWeightsHMC[,2])


