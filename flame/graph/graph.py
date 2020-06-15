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
from flame.stats.pca import pca    

from flame.util import utils, get_logger
LOG = get_logger(__name__)

    
def generateProjectedSpace(X, param, conveyor):
    # TODO: decide which is the best way to present the training space
    LOG.info('Generating projeced X space...')
    mpca = pca()
    mpca.build(X,targetA=2,autoscale=False)

    pca_path = os.path.join(param.getVal('model_path'),'pca.npy')
    mpca.saveModel(pca_path)

    if np.isnan (np.sum(mpca.t[0])):
        t = np.zeros(len(mpca.t[0]))
    else:
        t = mpca.t[0]

    conveyor.addVal(t, 'PC1',
                        'PCA PC1', 'method', 'objs',
                        'PCA PC1 score for graphic representation')

    if np.isnan (np.sum(mpca.t[1])):
        t = np.zeros(len(mpca.t[1]))
    else:
        t = mpca.t[1]

    conveyor.addVal(t, 'PC2',
                        'PCA PC2', 'method', 'objs',
                        'PCA PC2 score for graphic representation')


def projectPredictions(X, param, conveyor):
    
    # PCA is destructive
    X=copy.copy(X)
    
    pca_path = os.path.join(param.getVal('model_path'),'pca.npy')

    if not os.path.isfile(pca_path):
        return

    LOG.info('Projecting in X space...')

    mpca = pca()
    mpca.loadModel(pca_path)

    if not 'numpy.float' in str(type (X[0,0])):
        X = X.astype(np.float64)

    success, result = mpca.projectPC(X,0)
    if success:
        X, t, dmodx = result
        if np.isnan (np.sum(t)):
            t = np.zeros(len(t))
        conveyor.addVal(t, 'PC1proj',
                       'PCA projected PC1', 'method', 'objs',
                       'PCA projected scores PC1 for graphic representation')

    success, result = mpca.projectPC(X,1)
    if success:
        X, t, dmodx = result
        if np.isnan (np.sum(t)):
            t = np.zeros(len(t))
        conveyor.addVal(t, 'PC2proj',
                       'PCA projected PC1', 'method', 'objs',
                       'PCA projected scores PC1 for graphic representation')

    return