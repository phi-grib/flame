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
import time
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from flame.graph.predtsne import PredictableTSNE

from flame.util import utils, get_logger
LOG = get_logger(__name__)

def generateManifoldSpace(X,param,conveyor):

    nobj, nvarx = np.shape(X)
    iY = np.empty((nobj, 2))

    t1= time.time()

    # for large matrices, simplify the t-SNE by running PCA and perform
    # the algorithm on the scores
    if nvarx > 1000 and nobj > 1000:

        # number of PCs is the min of nobj-1 and 1/20 nvarx, but never more than 500!
        A = int(min(nobj-1, nvarx/20, 500))
        LOG.info(f'Simplifying the matrix using {A} PCs...')
        pre = PCA(n_components=A ,random_state=46).fit(X)
        X_red = pre.transform(X)

        LOG.info('Generating projected X space using t-SNE...')
        emb = PredictableTSNE().fit(X_red,iY)
        X_train=emb.transform(X_red)

        options = {"model_pre": pre, 
                   "model_reduc": emb,           
                   "method": 't-SNE'}

    else:
        LOG.info('Generating projected X space using t-SNE...')
        emb = PredictableTSNE().fit(X,iY)
        X_train=emb.transform(X)

        options = {"model_reduc": emb,           
                   "method": 't-SNE'}

    LOG.info (f'...completed in {time.time()-t1 :.1f} seconds')

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "wb") as f:
        pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')

    conveyor.addVal(X_train[:,0],'PC1',
                        't-SNE X', 'method', 'objs',
                        'X value for a t-SNE representation')
    
    conveyor.addVal(X_train[:,1],'PC2',
                        't-SNE Y', 'method', 'objs',
                        'X value for a t-SNE representation')

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

    conveyor.addVal(emb.explained_variance_ratio_, 'SSX',
                    'Explained variance ratio', 'method', 'single',
                    'Ratio of variance explainded by each PC dimension')

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
    
    label = {'PCA':['PCA PC1', 'PCA PC2'], 't-SNE':['t-SNE X', 't-SNE Y']}

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "rb") as f:
        options = pickle.load(f)

    emb = options["model_reduc"]
    method = options['method']

    if "model_pre" in options:
        pre = options["model_pre"]
        X_red = pre.transform(X)
        X_test = emb.transform(X_red)
    else:
        X_test = emb.transform(X)

    conveyor.addVal(X_test[:,0], 'PC1proj',
                       label[method][0], 'method', 'objs',
                       'Model projected scores D1 for graphic representation')

            
    conveyor.addVal(X_test[:,1], 'PC2proj',
                       label[method][1], 'method', 'objs',
                       'Model projected scores D2 for graphic representation')

    if method == 'PCA':
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
