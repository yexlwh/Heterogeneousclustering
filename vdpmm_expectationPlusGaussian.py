import numpy as np
import scipy.special as ssp
from normwish1 import *

def vdpmm_expectationPlusGaussian(data,params):
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

        log_gamma_tilde[:,i] =   log_V_prob[i] + pob[:,i];

    gammas = log_gamma_tilde;
    gammas = gammas / repmat(np.sum(gammas, axis=1), K, 1).T;

    # ############################################################################################################
    # if (T)%7==0:
        # print(gammas)
    u_SVD, s_SVD, v_SVD = np.linalg.svd(gammas, full_matrices=True)
        #
        # s_SVD = np.diag(s_SVD)
    D_u = (u_SVD.shape)[0]
    D_s = (s_SVD.shape)[0]
    S = s_SVD+0.0001
    s_SVD = np.zeros((D_u, D_s))
    s_SVD[:D_s, :D_s] = np.diag(S)
        # print(s_SVD)
    gammas = np.dot(u_SVD, np.dot(s_SVD, v_SVD))
        # print(gammas)
    gammas = (gammas + abs(gammas)) / 2;
    # ######################################################################
    gammas = gammas / repmat(np.sum(gammas,axis=1),K,1).T;
    return gammas
