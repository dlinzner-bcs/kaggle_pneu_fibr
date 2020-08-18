from AutoCTMC.ctmc.ctmc import ctmc
from patient.patient import patient
import numpy as np
import pandas as pd
from scipy.stats import norm
import copy



if __name__ == '__main__':

    #import data

    X = pd.read_csv("../data/train.csv")
    ids = set(X.Patient)
    T = max(X['Weeks'].to_numpy())  # time window end
    T_min = min(X['Weeks'].to_numpy()) # time window start


    patients=[]
    for id in ids:
        p = patient(id)
        p.load_patient(X)
        patients.append(p)

    #params of ctmc

    dt = 0.005 # timestep for simulation
    D = 100 # number of states of ctmc
    alpha = 0.1 #prior over num of transitions
    beta  = 0.1 # prior dwelling time

    #generate random rate matrix
    Q = np.random.gamma(shape =2.0,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    #generate random initial state
    p0 = np.ones((1,D)).flatten()/D
    print(sum(p0))
    #prior assumption on observation model
    mu = np.arange(0,D,100)
    sig= np.ones((D))*0.2
    params = (mu,sig)

    #init ctmc
    mc = ctmc(Q,p0,alpha,beta,T,dt,params)