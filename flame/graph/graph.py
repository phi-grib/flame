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
import umap
from flame.stats.pca import pca    
from sklearn.decomposition import PCA
from flame.chem.compute_md import _RDKit_rdkFPS,_RDKit_morganFPS
from sklearn.manifold import TSNE
from flame.graph.predtsne import PredictableTSNE
import time 
import joblib

from flame.util import utils, get_logger
LOG = get_logger(__name__)

def generateManifoldSpace(X,Y,param,conveyor):
    LOG.info('Generating projected X space...')

    if X.shape[1]==2048:
        emb=PredictableTSNE().fit(X,Y)
        X_train=emb.transform(X)
        # print('Uso TSNE')
        return X_train
    else:
        emb=PCA(n_components=2,random_state=46).fit(X)
        X_train=emb.transform(X)
        # print('USO PCA')
        return X_train

    options = {"model_reduc":emb}

    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "wb") as f:
        pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)

    conveyor.addVal(X_train[:,0],'PC1',
                        'Model D1','method','objs',
                        'Model D1 score for graphic representation')
    
    conveyor.addVal(X_train[:,1],'PC2',
                        'Model D2','method','objs',
                        'Model D2 score for graphic representation')

def projectManifoldPredictions(X, param, conveyor):
    '''
        This method projects X umap Vector, using X_pca

        We assume a two dimension model

        The method returs scores for dimensions 1 and 2

    '''
    models_path = os.path.join(param.getVal('model_path'),'models.pkl')
    with open(models_path, "rb") as f:
        options = pickle.load(f)

    emb  = options["model_reduc"]

    X=copy.copy(X)
    X_test = emb.transform(X)

    conveyor.addVal(X_test[:,0], 'PC1proj',
                       'Model projected D1', 'method', 'objs',
                       'Model projected scores D1 for graphic representation')

            
    conveyor.addVal(X_test[:,1], 'PC2proj',
                       'Model projected D2', 'method', 'objs',
                       'Model projected scores D2 for graphic representation')


# def generateManifoldSpace(X,self,param,conveyor):
#     ''' This function uses the scaled X matrix of the model to build 2D UMAP model
#         This model is saved and the two transformed vars are dumped to the conveyor 
#     '''
#     LOG.info('Generating UMAP transformed X space...')

#     # reduce original dimensionality
#     # pca = PCA(n_components=round(X.shape[0]/4),random_state=46)
#     # X_pca = pca.fit_transform(X)

#     # # print ('PCA generated in: ', a-time.time())
#     # a = time.time()

#     umap = umap.UMAP(n_components=2,random_state=46).fit(X)

#     options = {"model_umap":umap}

#     models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     with open(models_path, "wb") as f:
#         pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)


#     conveyor.addVal(umap.embedding_[:,0],'PC1',
#                         'UMAP D1','method','objs',
#                         'UMAP D1 score for graphic representation')
    
#     conveyor.addVal(umap.embedding_[:,1],'PC2',
#                         'UMAP D2','method','objs',
#                         'UMAP D2 score for graphic representation')

#     # print ('UMAP generated in: ', a-time.time())

#     # options = {"model_pca": pca, "model_umap":umap}

#     # # save models
#     # models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     # # options = {"model_pca": pca, "model_umap":umap}
#     # # with open(models_path, "wb") as f:
#     # #     pickle.dump( options, f,protocol=pickle.HIGHEST_PROTOCOL)
#     # joblib.dump(umap, models_path)

#     # # dumpy the two variables to Conveyor
#     # conveyor.addVal(new_x[:,0],'PC1',
#     #                     'UMAP D1','method','objs',
#     #                     'UMAP D1 score for graphic representation')
    
#     # conveyor.addVal(new_x[:,1],'PC2',
#     #                     'UMAP D2','method','objs',
#     #                     'UMAP D2 score for graphic representation')


# def projectManifoldPredictions(X, param, conveyor):
#     '''
#         This method projects X umap Vector, using X_pca
#         The method returs transformed X for dimensions 1 and 2
#     '''

#     models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     with open(models_path, "rb") as f:
#         options = pickle.load(f)

#     umap = options["model_umap"]

#     X=copy.copy(X)
#     # X_test = pca.transform(X)

#     umap_test = umap.transform(X)
    
    
#     conveyor.addVal(umap_test[:,0], 'PC1proj',
#                        'UMAP projected D1', 'method', 'objs',
#                        'UMAP projected scores D1 for graphic representation')

            
#     conveyor.addVal(umap_test[:,1], 'PC2proj',
#                        'UMAP projected D2', 'method', 'objs',
#                        'UMAP projected scores D2 for graphic representation')
#     # models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     # umap = joblib.load(models_path)


#     # with open(models_path, "rb") as f:
#     #     options = pickle.load(f)

#     # pca  = options["model_pca"]
#     # umap = options["model_umap"]

#     # X=copy.copy(X)
#     # X_test = pca.transform(X)

#     # test_embedding = umap.transform(X_test)
#     test_embedding = umap.transform(X)
#     print (test_embedding)    
#     conveyor.addVal(test_embedding[:,0], 'PC1proj',
#                        'UMAP projected D1', 'method', 'objs',
#                        'UMAP projected scores D1 for graphic representation')
            
#     conveyor.addVal(test_embedding[:,1], 'PC2proj',
#                        'UMAP projected D2', 'method', 'objs',
#                        'UMAP projected scores D2 for graphic representation')

# def generateIsomapSpace(X,param,conveyor):
#     ''' This function uses the scaled X matrix of the model to build a PCs PCA model
        
#         This model is saved and the scores are dumped to the conveyor, after a umap 
        
#         of 2 dimensions is obtained.
#     '''
#     LOG.info('Generating projected X space...')

#     # a = time.time()

#     pca = PCA(n_components=round(X.shape[0]/4),random_state=46)
#     X_pca = pca.fit_transform(X)

#     # print ('PCA generated in: ', a-time.time())
#     a = time.time()

#     isomap=Isomap(n_components=2).fit(X_pca)

#     print ('UMAP generated in: ', a-time.time())

#     options = {"model_pca": pca, "model_isomap":isomap}

#     models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     with open(models_path, "wb") as f:
#         pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)

#     #TODO: store both models

#     X_train=isomap.transform(X_pca)

#     conveyor.addVal(X_train[:,0],'PC1',
#                         'ISOMAP D1','method','objs',
#                         'ISOMAP D1 score for graphic representation')
    
#     conveyor.addVal(X_train[:,1],'PC2',
#                         'ISOMAP D2','method','objs',
#                         'ISOMAP D2 score for graphic representation')

# def projectIsomapPredictions(X, param, conveyor):
#     '''
#         This method projects X umap Vector, using X_pca

#         We assume a two dimension model

#         The method returs scores for dimensions 1 and 2

#     '''
#     models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     with open(models_path, "rb") as f:
#         options = pickle.load(f)

#     pca  = options["model_pca"]
#     isomap = options["model_isomap"]

#     X=copy.copy(X)
#     X_test = pca.transform(X)

#     test_isomap = isomap.transform(X_test)
    
    
#     conveyor.addVal(test_isomap[:,0], 'PC1proj',
#                        'ISOMAP projected D1', 'method', 'objs',
#                        'ISOMAP projected scores D1 for graphic representation')

            
#     conveyor.addVal(test_isomap[:,1], 'PC2proj',
#                        'ISOMAP projected D2', 'method', 'objs',
#                        'ISOMAP projected scores D2 for graphic representation')


# def generatetsneSpace(X,Y,param,conveyor):
    
#     ''' This function uses the scaled X matrix of the model to build a PCs PCA model
        
#         This model is saved and the scores are dumped to the conveyor, after a umap 
        
#         of 2 dimensions is obtained.
#     '''
#     LOG.info('Generating projected X space...')

#     a = time.time()

#     npc = round(min(X.shape[0]/2,200))
#     pca = PCA(n_components=npc,random_state=46)
#     # pca = PCA(n_components=round(X.shape[0]/4),random_state=46)
#     X_pca = pca.fit_transform(X)

#     print ('PCA generated in: ', a-time.time())

#     # DO NOT PASS the Y matrix. We need an empty y array of this shape but 
#     # I have no idea how it is using it
#     # tsne = PredictableTSNE().fit(X_pca,Y)
#     iY = np.empty((X.shape[0], 2))
#     tsne = PredictableTSNE().fit(X_pca, iY)


#     print ('TSNE generated in: ', a-time.time())
#     print ('with error:', tsne.loss_)

#     options = {"model_pca": pca, "model_tsne":tsne}

#     models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     with open(models_path, "wb") as f:
#         pickle.dump(options, f,protocol=pickle.HIGHEST_PROTOCOL)

#     train_tsne = tsne.transform(X_pca)

#     conveyor.addVal(train_tsne[:,0],'PC1',
#                         't-SNE D1','method','objs',
#                         't-SNE D1 score for graphic representation')
    
#     conveyor.addVal(train_tsne[:,1],'PC2',
#                         't-SNE D2','method','objs',
#                         't-SNE D2 score for graphic representation')

# def projecttsnePredictions(X, param, conveyor):
#     '''
#         This method projects X umap Vector, using X_pca

#         We assume a two dimension model

#         The method returs scores for dimensions 1 and 2

#     '''
#     models_path = os.path.join(param.getVal('model_path'),'models.pkl')
#     with open(models_path, "rb") as f:
#         options = pickle.load(f)

#     pca  = options["model_pca"]
#     tsne = options["model_tsne"]

#     X=copy.copy(X)
#     X_test = pca.transform(X)

#     tsne_test = tsne.transform(X_test)
    
    
#     conveyor.addVal(tsne_test[:,0], 'PC1proj',
#                        't-SNE projected D1', 'method', 'objs',
#                        't-SNE projected scores D1 for graphic representation')

            
#     conveyor.addVal(tsne_test[:,1], 'PC2proj',
#                        't-SNE projected D2', 'method', 'objs',
#                        't-SNE projected scores D2 for graphic representation')
 

# def generateProjectedSpace(X, param, conveyor):
#     ''' This function uses the scaled X matrix of the model to build a 2 PCs PCA model
        
#         This model is saved and the scores are dumped to the conveyor, allowing to show
#         the training series in Flame GUI. We also save the %SSX explained by each variable 

#         Also, the model is saved as pca.npy and can be used to project predictions on top
#     '''
#     LOG.info('Generating projected X space...')
#     mpca = pca()
#     mpca.build(X,targetA=2,autoscale=False)

#     pca_path = os.path.join(param.getVal('model_path'),'pca.npy')
#     mpca.saveModel(pca_path)

#     if np.isnan (np.sum(mpca.t[0])):
#         t = np.zeros(len(mpca.t[0]))
#     else:
#         t = mpca.t[0]

#     conveyor.addVal(t, 'PC1',
#                         'PCA PC1', 'method', 'objs',
#                         'PCA PC1 score for graphic representation')

#     if np.isnan (np.sum(mpca.t[1])):
#         t = np.zeros(len(mpca.t[1]))
#     else:
#         t = mpca.t[1]

#     conveyor.addVal(t, 'PC2',
#                         'PCA PC2', 'method', 'objs',
#                         'PCA PC2 score for graphic representation')

#     modelSSX = mpca.SSXex/mpca.SSX
#     conveyor.addVal(modelSSX, 'SSX',
#                     'X Sum of Squares explained', 'method', 'single',
#                     'X Sum of Squares explained by each PC dimension')

#     return

# def projectPredictions(X, param, conveyor):
#     '''
#         This method projects X vectors into the existing PCA space generated for the
#         current model (from param.getVal('model_path'))

#         We assume a two dimension model

#         The method returs scores for dimensions 1 and 2, as well as the distance to model (DModX)
#         for a model of dimensionality 2

#         The values of the Distance to Model (DModX in SIMCA) provided in the vector dmod 
#         is the  normalized value (si/s0), where s0 was estimated directly using all the compounds in 
#         the training set. It was suggested that s0 computed this way leads to too narrow CI.
#         A much better estimation would be obtained using jackknifing (see Flaten et al. Chem 
#         Intell Lab Sys 2004: 72, 101-9) and this method must be considered in future versions

#     '''
    
#     # PCA is destructive
#     X=copy.copy(X)
    
#     pca_path = os.path.join(param.getVal('model_path'),'pca.npy')

#     if not os.path.isfile(pca_path):
#         return

#     LOG.info('Projecting in X space...')

#     mpca = pca()
#     mpca.loadModel(pca_path)

#     if not 'numpy.float' in str(type (X[0,0])):
#         X = X.astype(np.float64)

#     success, result = mpca.projectPC(X,0)
#     if success:
#         X, t, dmodx = result
#         if np.isnan (np.sum(t)):
#             t = np.zeros(len(t))
#         conveyor.addVal(t, 'PC1proj',
#                        'PCA projected PC1', 'method', 'objs',
#                        'PCA projected scores PC1 for graphic representation')

#     success, result = mpca.projectPC(X,1)
#     if success:
#         X, t, dmodx = result
#         if np.isnan (np.sum(t)):
#             t = np.zeros(len(t))
            
#         conveyor.addVal(t, 'PC2proj',
#                        'PCA projected PC1', 'method', 'objs',
#                        'PCA projected scores PC1 for graphic representation')
        
#         conveyor.addVal(dmodx, 'PCDMODX',
#                        'DModX', 'method', 'objs',
#                        'Distance of object to a 2PC PCA model')

#     return