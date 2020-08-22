from AutoCTMC.ctmc.ctmc import ctmc
from patient.patient import patient
import numpy as np
import pandas as pd
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':

    #import data

    X = pd.read_csv("../data/train.csv")
    ids = list(set(X.Patient))
    T_max = max(X['Weeks'].to_numpy())  # time window end
    T_min = min(X['Weeks'].to_numpy()) # time window start
    T =T_max - T_min+1

    patients=[]
    for id in ids:
        p = patient(id)
        p.load_patient(X)
        patients.append(p)

    #params of ctmc

    dt = 0.001# timestep for simulation
    D = 5 # number of states of ctmc
    alpha = 0.0001 #prior over num of transitions
    beta  = 0.01 # prior dwelling time

    #generate random rate matrix
    Q = np.random.gamma(shape =2.0,scale=1.0,size=(D,D))
    for i in range(0,D):
        Q[i,i] = 0
        Q[i, i] = -sum(Q[i, :])
    #generate random initial state
    p0 = np.zeros((1,D)).flatten()/D
    p0[D-1]=1
    #prior assumption on observation model
    mu0 = np.arange(0,100,np.floor(100/D))
    sig= np.ones((D))*10
    params = (mu0,sig)

    #init ctmc
    mc = ctmc(Q,p0,alpha,beta,T,dt,params)

    print(T_min)
    dat = []
    for j in range(0,len(patients)):
        if patients[j].smoker.pop() == 'Ex-smoker':
            if patients[j].sex.pop() == 'Female':
                dat.append((patients[j].traj_p[0]-T_min+0.1,patients[j].traj_p[1]))

    M = 100  # number of EM iterations
    mc.reset_stats()
    mc.estimate_Q()

    for m in range(0, M):
        llh, sols = mc.process_emissions(dat)
        mc.update_obs_model(sols, dat)
        mc.estimate_Q()

        # log-likelihood
        print("log-likelihood:\n %s" % llh)
        # current obs params estimate
        print("mu_estimate:\n %s" % mc.params[0])

        mc.reset_stats()

        a = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        a0 = copy.deepcopy(mc.Q_estimate)
        np.fill_diagonal(a0, 0)
        np.fill_diagonal(b, 0)
        # mse of rate matrix estimate
        print("Q_estimate:\n %s" % a0)

        mc.T = 133
        llh, sols = mc.process_emissions(dat)
        plt.figure(1)
        for k in range(0,10):

            l = random.randint(0,len(dat)-1)

            x = sols[l][1]+T_min-0.1
            y = sols[l][0]
            mu0 = mc.params[0]
            y = np.multiply(y,mu0[:,None])*patients[l].base_fvc/100
            y = np.sum(y,axis=0)

            plt.plot(x,y)


            x = dat[l][0]+T_min-0.1
            y = dat[l][1]*patients[l].base_fvc/100
            plt.plot(x, y,'r+')

        plt.ylim((0, 6000))
        plt.show()
        mc.T = T_max - T_min+1