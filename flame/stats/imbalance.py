#! -*- coding: utf-8 -*-

# Description    imbalance data methods
##
# Authors:       Jose Carlos Gómez-Tamayo (josecarlos.gomez@upf.edu)
##
# Copyright 2018 Manuel Pastor
##
# This file is part of Flame
##
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from flame.util import utils, get_logger

LOG = get_logger(__name__)

def simple_subsampling(X, Y, random_seed):
    """
    Simple subsampling, adjusts the number of negative 
    samples to the positive one or viceversa.
    """

    positives = X[Y==1]
    negatives = X[Y==0]

    # Perform subsampling of negative instances
    if len(negatives) > len(positives):
        LOG.info('Subsampling of negative instances')
        size = positives.shape[0]
        negatives_sub = negatives[np.random.choice(
                        negatives.shape[0],
                        size=size,
                        replace=False)]
        Y = np.concatenate((np.ones(size), np.zeros(size)))
        X = np.concatenate((positives, negatives_sub), axis=0)
    # Perform subsampling of positive instances
    else:
        size = negatives.shape[0]
        positives_sub = positives[np.random.choice(
                        positives.shape[0],
                        size=size,
                        replace=False)]
        Y = np.concatenate((np.ones(size), np.zeros(size)))
        X = np.concatenate((positives_sub, negatives), axis=0)

    if Y.size == 0  or X.size == 0:
        raise ValueError("Error creating subsampled matrices")
    return X, Y

def run_imbalance(method, X, Y, random_seed=46):
    X_s = []
    Y_s = []
    if method == "simple_subsampling":
        X_s, Y_s = simple_subsampling(X, Y, random_seed)
    else:
        raise ValueError("Imbalance data method not recognized")
    return X_s, Y_s
            
