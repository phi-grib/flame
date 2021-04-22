#! -*- coding: utf-8 -*-

# Description    Flame graphic functions
#
# Authors: Manuel Pastor (manuel.pastor@upf.edu),
#
# Copyright 2020 Manuel Pastor
#
# This file is part of Flame
#
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
#
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import copy
import pickle
# import umap
# from flame.stats.pca import pca    
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
# from flame.chem.compute_md import _RDKit_rdkFPS,_RDKit_morganFPS
# from sklearn.manifold import TSNE
from flame.graph.predtsne import PredictableTSNE
# import time 
# import joblib

from flame.util import utils, get_logger
LOG = get_logger(__name__)

def generateManifoldSpace(X,param,conveyor):
    
    LOG.info('Generating projected X space using t-SNE...')

    iY = np.empty((X.shape[0], 2))
    emb=PredictableTSNE().fit(X,iY)

    options = {"model_reduc":emb, "method": 't-SNE'}

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "wb") as f:
        pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')

    X_train=emb.transform(X)

    conveyor.addVal(X_train[:,0],'PC1',
                        'PCA PC1', 'method', 'objs',
                        'PCA PC1 score for graphic representation')
    
    conveyor.addVal(X_train[:,1],'PC2',
                        'PCA PC2', 'method', 'objs',
                        'PCA PC2 score for graphic representation')

def generatePCASpace(X, param, conveyor):
    ''' This function uses the scaled X matrix of the model to build a 2 PCs PCA model
        
        This model is saved and the scores are dumped to the conveyor, allowing to show
        the training series in Flame GUI. We also save the %SSX explained by each variable 

        Also, the model is saved as pca.npy and can be used to project predictions on top
    '''
    LOG.info('Generating projected X space using PCA...')

    emb=PCA(n_components=2,random_state=46).fit(X)

    options = {"model_reduc":emb, "method": "PCA"}

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "wb") as f:
        pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)

    X_train=emb.transform(X)

    conveyor.addVal(X_train[:,0],'PC1',
                        'PCA PC1', 'method', 'objs',
                        'PCA PC1 score for graphic representation')
    
    conveyor.addVal(X_train[:,1],'PC2',
                        'PCA PC2', 'method', 'objs',
                        'PCA PC2 score for graphic representation')

    # modelSSX = mpca.SSXex/mpca.SSX
    # conveyor.addVal(modelSSX, 'SSX',
    #                 'X Sum of Squares explained', 'method', 'single',
    #                 'X Sum of Squares explained by each PC dimension')

    return

def projectReduced(X, param, conveyor):
    '''
        This method projects X vectors into the existing PCA space generated for the
        current model (from param.getVal('model_path'))

        We assume a two dimension model

        The method returs scores for dimensions 1 and 2, as well as the distance to model (DModX)
        for a model of dimensionality 2

        The values of the Distance to Model (DModX in SIMCA) provided in the vector dmod 
        is the  normalized value (si/s0), where s0 was estimated directly using all the compounds in 
        the training set. It was suggested that s0 computed this way leads to too narrow CI.
        A much better estimation would be obtained using jackknifing (see Flaten et al. Chem 
        Intell Lab Sys 2004: 72, 101-9) and this method must be considered in future versions

    '''
    
    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "rb") as f:
        options = pickle.load(f)
    emb = options["model_reduc"]

    X_test = emb.transform(X)

    conveyor.addVal(X_test[:,0], 'PC1proj',
                       'Model projected D1', 'method', 'objs',
                       'Model projected scores D1 for graphic representation')

            
    conveyor.addVal(X_test[:,1], 'PC2proj',
                       'Model projected D2', 'method', 'objs',
                       'Model projected scores D2 for graphic representation')

    if options['method'] == 'PCA':
        nobj, nvarx = np.shape(X)
        X_pred = emb.inverse_transform(X_test)
        dmodx = []
        for iobj in range (nobj):
            mse = mean_squared_error (X_pred[iobj], X[iobj])
            dmodx.append(np.sqrt(mse))

        conveyor.addVal(dmodx, 'PCDMODX',
                        'DModX', 'method', 'objs',
                        'Distance of object to a 2PC PCA model')

    return
