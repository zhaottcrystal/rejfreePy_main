import os
import sys

sys.path.append("/Users/crystal/Dropbox/rejfree/rejfreePy/")
os.chdir("/Users/crystal/Dropbox/rejfree/rejfreePy/")
import numpy as np
from main.FullTrajectorGeneration import generateFullPathUsingRateMtxAndStationaryDist
from main.OptionClasses import MCMCOptions
from main.OptionClasses import RFSamplerOptions
from main.Utils import summarizeSuffStatUsingEndPoint
from main.ExpectedCompleteReversibleObjective import ExpectedCompleteReversibleObjective
from main.ExpectedCompleteReversibleModelBinaryFactors import ExpectedCompleteReversibleModelWithBinaryFactors
from main.HMC import HMC
from main.LocalRFSamplerForBinaryWeights import LocalRFSamplerForBinaryWeights
from main.PhyloLocalRFMove import PhyloLocalRFMove
from main.ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure import \
    ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure
from main.HardCodedDictionaryUtils import getHardCodedDictChainGraph
from main.FullTrajectorGeneration import getObsArrayAtSameGivenTimes
from main.FullTrajectorGeneration import endPointSamplerSummarizeStatisticsOneBt
from collections import OrderedDict
from datetime import datetime
from numpy.random import RandomState


nStates = 5
## generate the exchangeable coefficients
## set the seed so that we can reproduce generating the
seed = 1234567890
prng = RandomState(seed)

nBivariateFeat = np.int(nStates * (nStates-1) / 2)
bivariateWeights = prng.uniform(0, 1, nBivariateFeat)

np.random.seed(2)
stationaryWeights = prng.uniform(0, 1, nStates)

print("The true stationary weights are")
# change the extreme stationary weights to have a balanced stationary distribution
#stationaryWeights[2] = -0.5
#stationaryWeights[3] = 0.5
#stationaryWeights[11] = 0.6

## The true stationary weights are
## [-0.41675785 -0.05626683 -1.35        0.05       -1.         -0.84174737]
print("The true stationary distribution is")
print(np.exp(stationaryWeights) / np.sum(np.exp(stationaryWeights)))

# The true stationary distribution is
#[ 0.04780083  0.06854799  0.0439829   0.11955791  0.01206568  0.03125107
#  0.11990291  0.02087418  0.02517493  0.02921824  0.12587067  0.13213193
#  0.07559123  0.02370949  0.12432004]


bivariateFeatIndexDictionary = getHardCodedDictChainGraph(nStates)
print(bivariateFeatIndexDictionary)

print("The true binary weights are:")
print(bivariateWeights)
# [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763 -2.3015387
#   1.74481176 -0.7612069   0.3190391  -0.24937038  1.46210794 -2.06014071
#  -0.3224172  -0.38405435  1.13376944 -1.09989127 -0.17242821 -0.87785842
#   0.04221375  0.58281521 -1.10061918  1.14472371  0.90159072  0.50249434
#   0.90085595 -0.68372786 -0.12289023 -0.93576943 -0.26788808  0.53035547
#  -0.69166075 -0.39675353 -0.6871727  -0.84520564 -0.67124613 -0.0126646
#  -1.11731035  0.2344157   1.65980218  0.74204416 -0.19183555 -0.88762896
#  -0.74715829  1.6924546   0.05080775 -0.63699565  0.19091548  2.10025514
#   0.12015895  0.61720311  0.30017032 -0.35224985 -1.1425182  -0.34934272
#  -0.20889423  0.58662319  0.83898341  0.93110208  0.28558733  0.88514116
#  -0.75439794  1.25286816  0.51292982 -0.29809284  0.48851815 -0.07557171
#   1.13162939  1.51981682  2.18557541 -1.39649634 -1.44411381 -0.50446586
#   0.16003707  0.87616892  0.31563495 -2.02220122 -0.30620401  0.82797464
#   0.23009474  0.76201118 -0.22232814 -0.20075807  0.18656139  0.41005165
#   0.19829972  0.11900865 -0.67066229  0.37756379  0.12182127  1.12948391
#   1.19891788  0.18515642 -0.37528495 -0.63873041  0.42349435  0.07734007
#  -0.34385368  0.04359686 -0.62000084  0.69803203 -0.44712856  1.2245077
#   0.40349164  0.59357852 -1.09491185]



rfOptions = RFSamplerOptions()
rfOptions.trajectoryLength = 0.125
mcmcOptions = MCMCOptions(10000, 1, 0)

## create the rate matrix based on the sparse graphical structure
testRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, stationaryWeights, bivariateWeights,
                                                                        bivariateFeatIndexDictionary)
stationaryDist = testRateMtx.getStationaryDist()
print("The true stationary distribution is")
print(stationaryDist)
#The true stationary distribution is
#[ 0.04780083  0.06854799  0.0439829   0.11955791  0.01206568  0.03125107
#  0.11990291  0.02087418  0.02517493  0.02921824  0.12587067  0.13213193
#  0.07559123  0.02370949  0.12432004]


rateMtx = testRateMtx.getRateMtx()
print("The true rate matrix is ")
print(rateMtx)


print("The true exchangeable parameters are ")
trueExchangeCoef = testRateMtx.getExchangeCoef()
print(trueExchangeCoef)

#The true exchangeable parameters are
#[5.075095607482667, 2.7527184481647362, 0.31984199654874601, 0.20166641159878343, 0.8125636797020549, 0.23784619170668767, 0.57308173259146478, 2.6740785756861607, 0.64264178800599125, 1.0721529402109236, 3.3626776006407701, 0.54989233610573274, 0.092314143655253963, 0.49338199922389642, 2.116396943114645, 1.0344585761575418, 0.28018099473874553, 0.34983746241176095, 0.43359486518784385, 1.8683000634246019, 0.5958275714055804, 1.0450915953483446, 7.7393246589747902, 4.0717995820958128, 4.0688088393985122, 1.2425032442977648, 0.44636508539579689, 0.34692049033296873, 0.30009460204503197, 1.3001340663506022, 0.85103222408390089, 0.3367500623359253, 0.33826481058549102, 0.21602128345931407, 0.2194893049324016, 0.5046396192249113, 0.32304134929678052, 0.41358399604511992, 6.6473473139428663, 11.043547686994144, 1.7336146264027894, 0.33977742216073065, 0.19499385030060301, 2.573575834869624, 5.715960535364097, 0.55644447323771729, 0.64013245576975453, 9.8865042607617113, 9.2111443094471692, 2.0904138530530836, 2.5027082102948239, 0.94925337300690082, 0.22430062645738763, 0.22495364495285303, 0.57221701895649402, 1.458967447544371, 4.1603807854860193, 5.8713553124930042, 3.3759926734941295, 3.2243406815737923, 1.1396751016957103, 1.6462010085945011, 5.8462356509426598, 1.2396597977428661, 1.209764014724616, 1.5112640706637746, 2.8750143755564799, 14.174523079656018, 40.665994325759968, 2.2013681887623715, 0.058390028947956966, 0.14247629152534882, 0.7086250090980708, 2.8185032751681334, 3.2930160203641141, 0.18148790464315412, 0.09745103537045012, 1.6850085365257392, 2.8808038735813111, 2.6969079570568968, 1.7154630388961551, 0.65502216020813386, 0.98590361969849505, 1.8159577930501578, 1.8373997032228031, 1.3734260244320573, 0.57599653122593708, 0.74594866115907799, 1.6477077132825499, 3.4949014536348049, 10.261528321023848, 3.9911295951375978, 0.82685284956133087, 0.36275944238695396, 0.80635107580507093, 1.6500975748102924, 0.76604557990261135, 0.74062798911062422, 0.56191539275901414, 1.0811563795458397, 1.285186018419509, 2.175762417763067, 5.0936738418208956, 2.7103293666828412, 0.60572249762476293]


## generate data sequences of a CTMC with an un-normalized rate matrix
bt = 5.0
nSeq = 5000
seqList = generateFullPathUsingRateMtxAndStationaryDist(nSeq, nStates, prng, rateMtx, stationaryDist, bt)
observedTimePoints = np.arange(0, (bt + 1))
observedSeqList = getObsArrayAtSameGivenTimes(seqList, observedTimePoints)
observedAllSequences = observedSeqList[1:observedSeqList.shape[0], :]

# input = summarizeSuffStatUsingEndPoint(seqList, bt, rateMtx)

## initial guess of the parameters
initialWeights = [1/nStates] * nStates
print("The weights for the initial stationary distirbution are")
print(initialWeights)


initialBinaryWeights = prng.uniform(0, 1.0, len(bivariateWeights))
print("The initial binary feature weights at 0th iteration are: ")


initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights,
                                                                           initialBinaryWeights,
                                                                           bivariateFeatIndexDictionary)
initialStationaryDist = initialRateMtx.getStationaryDist()
print("The initial stationary distribution is ")
print(initialStationaryDist)

initialRateMatrix = initialRateMtx.getRateMtx()
print("The initial exchangeable parameters at 0th iteration are")
initialExchangeCoef = initialRateMtx.getExchangeCoef()
print(initialExchangeCoef)

print("The initialRateMtx is")
print(initialRateMatrix)

## obtain the sufficient statistics based on the current values of the parameters and perform MCMC sampling scheme
nMCMCIters = mcmcOptions.nMCMCSweeps
thinningPeriod = MCMCOptions().thinningPeriod
burnIn = MCMCOptions().burnIn

stationarySamples = np.zeros((nMCMCIters, nStates))
stationaryWeightsSamples = np.zeros((nMCMCIters, nStates))
binaryWeightsSamples = np.zeros((nMCMCIters, nBivariateFeat))
exchangeableSamples = np.zeros((nMCMCIters, len(initialExchangeCoef)))

# to debug code, set nMCMCIters=1 temporarily
nMCMCIters = 1000

firstLastStatesArrayAll = list()
nPairSeq = int(len(observedTimePoints) - 1)

for i in range(nPairSeq):
    pairSeq = observedAllSequences[:, i:(i + 2)]
    firstLastStatesArrayAll.append(pairSeq)


## create a three dimensional array to save the rate matrix elements
rateMatrixSamples = np.zeros((nMCMCIters, nStates, nStates))

startTime = datetime.now()
print(startTime)

timeElapsedHMCArray = np.zeros(nMCMCIters)
timeElapsedlbpsArray = np.zeros(nMCMCIters)

for i in range(nMCMCIters):

    # save the samples of the parameters
    stationarySamples[i, :] = initialStationaryDist
    binaryWeightsSamples[i, :] = initialBinaryWeights
    exchangeableSamples[i, :] = initialExchangeCoef
    #rateMatrixSamples[i, :, :] = initialRateMatrix
    stationaryWeightsSamples[i, :] = initialWeights

    # use endpointSampler to collect sufficient statistics of the ctmc given the current values of the parameters
    # suffStat = summarizeSuffStatUsingEndPoint(seqList, bt, initialRateMatrix)
    #
    # # get each sufficient statistics element
    # nInit = suffStat['nInit']
    # holdTime = suffStat['holdTimes']
    # nTrans = suffStat['nTrans']

    nInit = np.zeros(nStates)
    holdTime = np.zeros(nStates)
    nTrans = np.zeros((nStates, nStates))

    for j in range(nPairSeq):
        suffStat = endPointSamplerSummarizeStatisticsOneBt(True, prng, initialRateMatrix, firstLastStatesArrayAll[j], 1.0)
        nInit = nInit + suffStat['nInit']
        holdTime = holdTime + suffStat['holdTimes']
        nTrans = nTrans + suffStat['nTrans']

    # construct expected complete reversible model objective
    expectedCompleteReversibleObjective = ExpectedCompleteReversibleObjective(holdTime, nInit, nTrans, 1.0,
                                                                              initialExchangeCoef)

    # sample stationary distribution elements using HMC
    startTimehmc = datetime.now()
    hmc = HMC(40, 0.02, expectedCompleteReversibleObjective, expectedCompleteReversibleObjective)
    sample = np.random.uniform(0, 1, nStates)
    samples = hmc.run(0, 1000, sample)
    avgWeights = np.sum(samples, axis=0) / samples.shape[0]
    endTimehmc = datetime.now()
    timeElapsedhmc = endTimehmc- startTimehmc
    timeElapsedHMCArray[i] = (timeElapsedhmc).total_seconds()
    
    initialWeights = avgWeights
    stationaryDistEst = np.exp(avgWeights) / np.sum(np.exp(avgWeights))
    # update stationary distribution elements to the latest value
    initialStationaryDist = stationaryDistEst

    # sample exchangeable coefficients using local bouncy particle sampler
    ## define the model
    model = ExpectedCompleteReversibleModelWithBinaryFactors(expectedCompleteReversibleObjective, nStates,
                                                             initialBinaryWeights, initialStationaryDist,
                                                             bivariateFeatIndexDictionary)

    ## define the sampler to use
    ## local sampler to use
    startTimelbps = datetime.now()
    localSampler = LocalRFSamplerForBinaryWeights(model, rfOptions, mcmcOptions, nStates, bivariateFeatIndexDictionary)
    phyloLocalRFMove = PhyloLocalRFMove( model, localSampler, initialBinaryWeights, randomSeed = seed)
    initialBinaryWeights = phyloLocalRFMove.execute()
    endTimelbps = datetime.now()
    timeElapsedlbps = (endTimelbps - startTimelbps).total_seconds()
    timeElapsedlbpsArray[i] = timeElapsedlbps
    print("The initial estimates of the binary weights are:")
    print(initialBinaryWeights)

    initialRateMtx = ReversibleRateMtxPiAndBinaryWeightsWithGraphicalStructure(nStates, initialWeights,
                                                                               initialBinaryWeights,
                                                                               bivariateFeatIndexDictionary)

    initialStationaryDist = np.round(initialRateMtx.getStationaryDist(), 3)
    initialRateMatrix = np.round(initialRateMtx.getRateMtx(), 3)
    initialExchangeCoef = np.round(initialRateMtx.getExchangeCoef(), 3)
    print(i)
    
endTime = datetime.now()
timeElapsed = 'Duration: {}'.format(endTime - startTime)
print("The elapsed time interval is ")
print(timeElapsed)

download_dir = "timeElapsed4States1000.csv"  # where you want the file to be downloaded to
csv = open(download_dir, "w")
# "w" indicates that you're writing strings to the file
columnTitleRow = "elapsedTime\n"
csv.write(columnTitleRow)
row = timeElapsed
csv.write(str(row))
csv.close()

np.savetxt('stationaryDistribution5States1000.csv', stationarySamples, fmt='%.3f', delimiter=',')
np.savetxt('stationaryWeight5States1000.csv', stationaryWeightsSamples, fmt='%.3f', delimiter=',')
np.savetxt('exchangeableParameters5States1000.csv', exchangeableSamples, fmt='%.3f', delimiter=',')
np.savetxt('binaryWeights5States1000.csv', binaryWeightsSamples, fmt='%.3f', delimiter=',')
np.savetxt('timeElapsedHMC5.csv', timeElapsedHMCArray, fmt='%.3f', delimiter=',')
np.savetxt("timeElapsedlbps5.csv", timeElapsedlbpsArray,fmt='%.3f', delimiter=',')
#np.save('3dsave4States1000.npy', rateMatrixSamples)
