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

    num_positives = len(Y[Y==1])
    num_negatives = len(Y[Y==0])

    # Perform subsampling of negative instances
    if num_negatives > num_positives:
        negatives_sub = np.random.choice(num_negatives,size=num_positives,replace=False)

        j = 0
        for i in range(nobj):
            # select all positives
            if Y[i]==1:
                mask[i] = 1
            # select only the negatives in the negative subsample
            # use j as sequential counter of negatives
            else:
                if j in negatives_sub:
                    mask[i] = 1
                j=j+1

    # Perform subsampling of positive instances
    else:
        positives_sub = np.random.choice(num_positives, size=num_negatives, replace=False)

        j = 0
        for i in range(nobj):
            # select all negatives
            if Y[i]==0:
                mask[i] = 1
            # select only the positives in the positive subsample
            # use j as sequential counter of positives
            else:
                if j in positives_sub:
                    mask[i] = 1
                j=j+1

    return True, mask


def near_miss1(X, Y):
    from imblearn.under_sampling import NearMiss
    nm1 = NearMiss(version=1)
    nm1.fit_resample(X, Y)
    indexes = nm1.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)

def near_miss2(X, Y):
    from imblearn.under_sampling import NearMiss
    nm1 = NearMiss(version=2)
    nm1.fit_resample(X, Y)
    indexes = nm1.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)

def near_miss3(X, Y):
    from imblearn.under_sampling import NearMiss
    nm1 = NearMiss(version=3)
    nm1.fit_resample(X, Y)
    indexes = nm1.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)


def edited_KNN(X, Y):
    from imblearn.under_sampling import EditedNearestNeighbours
    enn = EditedNearestNeighbours()
    enn.fit_resample(X, Y)
    indexes = enn.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)

def rep_edited_KNN(X, Y):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    renn = RepeatedEditedNearestNeighbours()
    renn.fit_resample(X, Y)
    indexes = renn.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)


def all_KNN(X, Y):
    from imblearn.under_sampling import AllKNN
    allknn = AllKNN()
    allknn.fit_resample(X, Y)
    indexes = allknn.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)


def iht(X, Y):
    from sklearn.linear_model import LogisticRegression
    from imblearn.under_sampling import InstanceHardnessThreshold
    iht = InstanceHardnessThreshold(random_state=0,
                                estimator=LogisticRegression(
                                solver='lbfgs', multi_class='auto'))
    Y = np.array(Y, dtype=int)
    iht.fit_resample(X, Y)
    indexes = iht.sample_indices_
    mask = []
    for i in range(len(X)):
        if i in indexes:
            mask.append(1)
        else:
            mask.append(0)
    return True, np.asarray(mask)


def run_imbalance(method, X, Y):
    if method == "simple_subsampling":
        return simple_subsampling(Y)
    if method == "near_miss1":
        return near_miss1(X, Y)
    if method == "near_miss2":
        return near_miss2(X, Y)
    if method == "near_miss3":
        return near_miss3(X, Y)
    if method == "edited_KNN":
        return edited_KNN(X, Y)
    if method == "rep_edited_KNN":
        return rep_edited_KNN(X, Y)
    if method == "all_KNN":
        return all_KNN(X, Y)
    if method == "iht":
        return iht(X, Y)
    return False, "Imbalance data method not recognized"
            
