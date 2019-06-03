import numpy as np
import rnn_mcmc_plots as mcmcplt

mplt = mcmcplt.Mcmcplot()
mplt.initialiseweights(5000,5)
for i in range(5000):
    mplt.addweightdata(i,np.random.randn(1,5))
mplt.saveplots()

