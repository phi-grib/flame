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

import pandas as pd
import numpy as np
from flame.util import utils, get_logger

LOG = get_logger(__name__)

def simple_subsampling(X, Y, random_seed):
    """
    Simple subsampling, adjusts the number of negative 
    samples to the positive one or viceversa.
    """
    # Create a Pandas DataFrame to facilitate data
    # handling
    frame = pd.DataFrame(X)
    frame["act"] = Y

    positives = frame[frame['act'] == 1]
    negatives = frame[frame['act'] == 0]
    X_s, Y_s = "",""

    # Perform subsampling of negative instances
    if len(negatives) > len(positives):
        LOG.info('Subsampling of negative instances')
        neg_sub = negatives.sample(frac=(float(len(positives))
                                /len(negatives)), random_state=46)
        new = pd.concat([positives, neg_sub], axis=0)
        Y_s = (new["act"].values)
        new = new.drop(["act"], axis=1)
        X_s = new.values

    # Perform subsampling of positive instances
    else:
        LOG.info('Subsampling of positive instances')
        pos_sub = positives.sample(frac=(float(len(negatives))
                                / len(positives)), random_state=46)
        new = pd.concat([negatives, pos_sub], axis=0)
        Y_s = (new["act"].values)
        new = new.drop(["act"], axis=1)
        X_s = new.values

    if Y_s.size == 0  or X_s.size == 0:
        raise ValueError("Error creating subsampled matrices")
    return X_s, Y_s

def run_imbalance(method, X, Y, random_seed=46):
    X_s = []
    Y_s = []
    if method == "simple_subsampling":
        X_s, Y_s = simple_subsampling(X, Y, random_seed)
    else:
        raise ValueError("Imbalance data method not recognized")
    return X_s, Y_s
            
