import sys
import numpy as np
import pickle
import os
import cProfile
import argparse
from numpy.random import RandomState

#os.chdir("/Users/crystal/Dropbox/rejfreePy_main")
import MCMCForOnlyNormalFactors
import OptionClasses

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
## add the number of states as arguments
## add the trajectory length if we use local bps
parser.add_argument('-trajectoryLength', action="store", dest='trajectoryLength', default = 1, help='save the trajectory length of the local bps sampler', type=float)
## add the total number of mcmc iterations
parser.add_argument('-nMCMCIter', action="store", dest='nMCMCIter', default=50000, type=int, help='store the total number of posterior samples')
## add the burning period of the posterior samples
parser.add_argument('-burnIn', action='store', dest='burnIn', default=0, type=int, help='store the burnIn period of the posterior samples')
## add the boolean variable to indicate whether we store the result in the end or we write results to csv files
parser.add_argument('--dumpResultIteratively', action='store_true', help='flag indicating we write results to csv iteratively instead of in the end')
## the number of iterations we write results to disk
parser.add_argument('-dumpResultIterations', action='store', dest='dumpResultIterations', default=50, type=int, help='store the number of iteration interval that we write results to csv')
##store the directory that we would lie to save the result
parser.add_argument('-dir_name', action='store', dest='dir_name', type=str, help='store the directory name to save the csv files')
## store the seed to used to generate the data
parser.add_argument('-seedGenData', action='store', dest='seed', default = 1234567890, type=int, help='store the seed we use to generate the sequences')
## store the seed for Markov chain sampling
## store the total number of generated time series sequences
parser.add_argument('-nSeq', action='store', dest='nSeq', type=int, default= 500, help='store the number of sequences of the time series')
parser.add_argument('-dim', action='store', dest='dim', type=int, default= 400, help='store the dimension of the multivariate Normal factor')
## add the refreshment rate of lbps algorithm
parser.add_argument('-refreshmentRate', action='store', dest='interLength', type=float, default=1, help='store the refreshment rate for the lbps algorithm')
## add the method we use to generate the initial weights
parser.add_argument('-refreshmentMethod', action='store', dest='refreshmentMethod', default= "LOCAL", type=OptionClasses.RefreshmentMethod.from_string, choices=list(OptionClasses.RefreshmentMethod))
parser.add_argument('-batchSize', action='store', dest='batchSize', type=int, default=50, help='the batch size when updating the ergodic mean')
parser.add_argument('--unknownTrueRateMtx', action="store_true", help='unknown rate matrix flag, the existence of the argument indicates the rate matrix used to generate the data is unknown.')


results = parser.parse_args()
dir_name = results.dir_name
nSeq = results.nSeq
nMCMCIter = results.nMCMCIter

trajectoryLength = results.trajectoryLength
dumpResultIterations = results.dumpResultIterations
refreshmentMethod = results.refreshmentMethod
seedGenData = results.seed
batchSize = results.batchSize
dim = results.dim


prng = RandomState(seedGenData)
#mean = prng.normal(0, 1,dim)
#sd = prng.uniform(0, 1, dim)
    ## change sd to a diagonal matrix
#cov = np.diag(sd)
    ## generate observations for multivariate Normal
#data = prng.multivariate_normal(mean, cov, nSeq)

#trueMean = mean
#trueSD = sd
#trueMeanStr = "trueMean" + str(dim)
#trueSDStr = "trueSD" + str(dim)
#format = '.csv'
# if dir_name is None:
#     dir_name = "/Users/crystal/Dropbox/NormalTry"
# trueMeanFileName = os.path.join(dir_name, trueMeanStr + format)
# trueSDFileName = os.path.join(dir_name, trueSDStr + format)
# np.savetxt(trueMeanFileName, trueMean, fmt='%.4f', delimiter=',')
# np.savetxt(trueSDFileName, trueSD, fmt='%.4f', delimiter=',')
#
# dataFileName = "dim" + str(dim) + "seedGenData" + str(seedGenData)+ "nObs" + str(nSeq)
# directory = os.path.join(dir_name, dataFileName)
# if not os.path.exists(directory):
#     os.makedirs(directory)
#     os.chdir(directory)
# dataFileName = dataFileName + ".file"
#
# with open(dataFileName, "wb") as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
# with open(dataFileName, "rb") as f:
#     dataRegime = pickle.load(f)



mcmcRegime = MCMCForOnlyNormalFactors.MCMCForOnlyNormalFactors(nMCMCIter, thinning=1.0, burnIn=0, dim=dim, prng=prng,
                                                               trajectoryLength=trajectoryLength, dumpResultIteratively=True,
                                                               dumpResultIterations = 50,
                                                               rfOptions=OptionClasses.RFSamplerOptions(trajectoryLength=trajectoryLength, refreshmentMethod=refreshmentMethod),
                                                               dir_name=os.getcwd(), initialSampleDist="Normal",
                                                               refreshmentMethod= OptionClasses.RefreshmentMethod.LOCAL,
                                                               batchSize=50, unknownTrueRateMtx = True)


mcmcRegime.run()

#cProfile.run(mcmcRegimeIteratively.run())

