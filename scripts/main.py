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
    #prior assumption on observation model
    mu = np.arange(0,D)
    sig= np.ones((D))*1
    params = (mu,sig)

    #init ctmc
    mc = ctmc(Q,p0,alpha,beta,T,dt,params)

    dat = []
    for j in range(0,len(patients)):
        dat.append((patients[j].traj_p[0]-T_min,patients[j].traj_p)[1])

    M = 100  # number of EM iterations
    mc.reset_stats()
    mc.estimate_Q()
    print(2)
    for m in range(0, M):
        llh, sols = mc.process_emissions(dat)
        mc.update_obs_model(sols, dat)
        mc.estimate_Q()

        # log-likelihood
        print("log-likelihood:\n %s" % llh)
        # current obs params estimate
        print("mu_estimate:\n %s" % mc.params[0])
        print("mu_truth:\n %s" % np.array([0, 1]))

        mc.reset_stats()

        a = copy.deepcopy(mc.Q_estimate)
        b = copy.copy(mc.Q)
        a0 = copy.deepcopy(mc.Q_estimate)
        np.fill_diagonal(a0, 0)
        np.fill_diagonal(b, 0)
        # mse of rate matrix estimate
        print("Q_estimate:\n %s" % a0)
        print("Q_truth:\n %s" % b)
