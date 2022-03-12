# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:44:30 2017

@author: yexlwh
"""

import numpy as np
import scipy.special as ssp
from normwish1 import *

def vdpmm_expectationCNNKnearset(data,params,labdaPos,posGammas):
    K = params['a'].shape[0];
    N,D = data.shape
    eq_log_Vs = np.zeros((K,1));
    eq_log_1_Vs = np.zeros((K,1));
    log_V_prob= np.zeros((K,1))
    pob=np.zeros((N,K))
    log_gamma_tilde = np.zeros((data.shape[0],K));

    for i in range(K):
        eq_log_Vs[i] = ssp.psi(params['g'][i,0]) - ssp.psi(params['g'][i,0]+params['g'][i,1]);
        eq_log_1_Vs[i] = ssp.psi(params['g'][i,1]) - ssp.psi(params['g'][i,0]+params['g'][i,1]);
        log_V_prob[i] = eq_log_Vs[i] + np.sum(eq_log_1_Vs[np.arange(i)])
        pob[:,i] = normwish(data,params['mean'][i,:],params['beta'][i],params['a'][i],params['B'][:,:,i]);

        log_gamma_tilde[:,i] = log_V_prob[i] + (1-labdaPos)*pob[:,i]+labdaPos*posGammas[:,i];

    gammas = np.exp(log_gamma_tilde);
    gammas = gammas / repmat(np.sum(gammas,axis=1),K,1).T;


    return gammas
