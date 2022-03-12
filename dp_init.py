__author__ = 'ye'
import numpy as np
from numpy.matlib import repmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

def vdpmm_init(testData,K):
    params={}
    num,dim=testData.shape
    gammas = np.random.rand(num,K)
    temp=repmat(np.sum(gammas,axis=1),K,1)
    temp1=temp.transpose()
    gammas = gammas /temp1
    params['eq_alpha'] = 30
    params['beta'] = np.zeros((K,1))
    params['a'] = np.zeros((K,1))
    params['meanN'] = np.zeros((dim,K))
    params['B'] = np.ones((dim,dim,K))
    params['sigma'] = np.ones((dim,dim,K))
    params['mean'] = np.zeros((K,dim))
    params['g'] = np.zeros((K,2))
    params['ll'] = -np.inf
    kmeans = KMeans(n_clusters=K, n_init=20)
    y_pred = kmeans.fit_predict(testData)
    params['meanN'] = (kmeans.cluster_centers_).T
    params['mean'] = (kmeans.cluster_centers_)
    of = OneHotEncoder(sparse=False).fit(y_pred.reshape(-1, 1))
    data_ohe1 = of.transform(y_pred.reshape(-1, 1))
    gammas=data_ohe1
    print(y_pred)
    return params,gammas
