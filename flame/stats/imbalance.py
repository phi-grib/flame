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

def simple_subsampling (Y):
    """
    Simple subsampling, adjusts the number of negative 
    samples to the positive one or viceversa.
    """
    np.random.seed(46)
    nobj = len(Y)
    mask = np.zeros(nobj, dtype=int)

    positives = Y[Y==1]
    negatives = Y[Y==0]

    # Perform subsampling of negative instances
    if len(negatives) > len(positives):
        size = len(positives)
        negatives_sub = np.random.choice(len(negatives),size=size,replace=False)

        j = 0
        for i in range(nobj):
            if Y[i]==1:
                mask[i] = 1
            else:
                if j in negatives_sub:
                    mask[i] = 1
                j=j+1

    # Perform subsampling of positive instances
    else:
        size = len(negatives)
        positives_sub = np.random.choice(len(positives), size=size, replace=False)

        j = 0
        for i in range(nobj):
            if Y[i]==0:
                mask[i] = 1
            else:
                if j in positives_sub:
                    mask[i] = 1
                j=j+1

    return True, mask

def run_imbalance(method, Y):
    if method == "simple_subsampling":
        return simple_subsampling(Y)

    return False, "Imbalance data method not recognized"
            
