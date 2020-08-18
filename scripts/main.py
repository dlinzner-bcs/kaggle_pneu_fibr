from AutoCTMC.ctmc.ctmc import ctmc
import numpy as np
import pandas as pd
from scipy.stats import norm
import copy



if __name__ == '__main__':

    #import data



    #params of ctmc
    T = 21 #time window end
    dt = 0.0005 # timestep for simulation
    D = 2 # number of states of ctmc
    alpha = 0.1 #prior over num of transitions
    beta  = 0.1 # prior dwelling time

    #generate random rate matrix
    Q = np.random.gamma(shape =2.0,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    #generate random initial state
    p0 = np.ones((1,D)).flatten()
    p0[0] = 0
    p0 = p0/sum(p0)
    #prior assumption on observation model
    mu = np.random.uniform(low= -2,high = 2,size =(D,1))
    sig= np.ones((D))*0.2
    params = (mu,sig)

    #init ctmc
    mc = ctmc(Q,p0,alpha,beta,T,dt,params)