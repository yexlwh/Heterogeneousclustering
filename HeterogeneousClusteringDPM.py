from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from sklearn.manifold import spectral_embedding
from sklearn.decomposition import PCA
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from vdpmm_expectationCNNKnearset import *
from vdpmm_maximizeCNN import *
from gan_7 import GAN

import scipy.io as sio
from numpy import random as nr
from sklearn.manifold import TSNE


from dp_init import *
from vdpmm_maximizePlusGaussian import *
from vdpmm_expectationPlusGaussian import *
from sklearn import metrics

dataInput=sio.loadmat('D:\ye\heterUpload\coilFull2.mat')
batch=10;
batchSize=100;
X=dataInput['features']
X=X[0:batchSize*batch,:]
dataInput=sio.loadmat('D:\ye\heterUpload\Wcoil20.mat')

W=dataInput['W']


y=dataInput['label'][0]

data=dataInput['data']
start = time.clock()
Knum=7
W2 = kneighbors_graph(data,Knum, mode='distance', include_self=True)

print(y)

maps = spectral_embedding(W, n_components=100)

maps2 = spectral_embedding(W2, n_components=100)

newData=maps


numits=1;
maxits=50;
K=40;
maps=maps
paramsGaussian,posGaussian = vdpmm_init(maps,K)
paramsGaussian2,posGaussian2 = vdpmm_init(maps2,K)
model = GAN(100, batchSize, 1e-1,maps)
training_loss=0
training_loss1=0
params,pos = vdpmm_init(data,K)
for i in range(maxits):
    paramsGaussian = vdpmm_maximizePlusGaussian(maps, paramsGaussian, posGaussian)
    posGaussian = vdpmm_expectationPlusGaussian(maps, paramsGaussian)

    paramsGaussian2 = vdpmm_maximizePlusGaussian(maps2, paramsGaussian2, posGaussian2)
    posGaussian2 = vdpmm_expectationPlusGaussian(maps2, paramsGaussian2)
    for i in range(batch):
        images=X[batchSize*i:batchSize*(i+1),:]/255
        maps1=maps[batchSize*i:batchSize*(i+1),:]

        R_loss, loss_value, loss_value1 = model.update_params1(images, images, maps1)
        loss_value, loss_value1, dtemp1, dtemp2 = model.update_params(images, images, maps1)
        model.update_params2(images, images, maps1)
        training_loss += loss_value
        training_loss1 += loss_value1
    training_loss = abs(training_loss) / batch
    training_loss1 = abs(training_loss1) / batch
    model.generate_and_save_images(batchSize, "", maps1)
    print(training_loss, training_loss1)
    training_loss = 0

print(posGaussian)
[Nz,Dz]=maps.shape
temp=np.max(posGaussian,axis=1)
temp.shape=(Nz,1)
index1=np.where(temp==posGaussian)
preLabel=list(index1[1])


print(posGaussian2)
[Nz,Dz]=maps.shape
temp=np.max(posGaussian2,axis=1)
temp.shape=(Nz,1)
index2=np.where(temp==posGaussian2)
preLabel2=list(index2[1])

srcLabel=(y.T)

print(np.unique(preLabel).shape)
elapsed = (time.clock() - start)
print("Time used:",elapsed)



print('Our method:',metrics.adjusted_mutual_info_score(preLabel, srcLabel))

