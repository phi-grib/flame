#! -*- coding: utf-8 -*-

# Description    Flame feature selection methods
##
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.


""" This file contains implemented methods to perform
    feature selection"""

from sklearn.preprocessing import MinMaxScaler 
from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from flame.util import utils, get_logger, supress_log
import numpy as np

LOG = get_logger(__name__)


def selectkBest(X, Y, n, quantitative):
    function = ""
    if quantitative:
        function = f_regression
    else:
        scaler = MinMaxScaler(copy=True, feature_range=(0,1))
        X = scaler.fit_transform(X)
        function = chi2
    kbest = SelectKBest(function, n)
    kbest.fit(X,Y)
    mask = kbest.get_support()
    return mask


def run_feature_selection(X, Y, scaler, param):
    """Compute the number of variables to be retained.
    """

    nobj, nvarx = np.shape(X)
    variable_mask = ''


    # When auto, the 10% top informative variables are retained.
    if param.getVal("feature_number") == "auto":
        # Use 10% of the total number of objects:
        # The number of variables is greater than the 10% of the objects
        # And the number of objects is greater than 100
        if nvarx > (nobj * 0.1) and not nobj < 100:
            n_features = int(nobj * 0.1)
        # If number of objects is smaller than 100 then n_features
        # is set to 10
        elif nobj < 100:
            n_features = 10
        # In any other circunstance set number of variables to 10 
        else:
            n_features = nvarx
    # Manual selection of number of variables
    else:
        n_features = int(param.getVal("feature_number"))

    # Apply variable selection.
    try:
        # Apply the variable selection algorithm obtaining
        # the variable mask.
        X_copy = X.copy()
        variable_mask = selectkBest(X_copy, Y, n_features, 
                                    param.getVal('quantitative'))
        
        # The scaler has to be fitted to the reduced matrix
        # in order to be applied in prediction.
        if param.getVal('modelAutoscaling') is not None\
                                 and scaler is not None:
            X = scaler.inverse_transform(X)
            X = X[:, variable_mask]
            scaler = scaler.fit(X)
            X = scaler.transform(X)
        else:
            X = X[:, variable_mask]

        LOG.info(f'Variable selection applied, number of final variables:'
                    f'{n_features}')
    except Exception as e:
        LOG.error(f'Error performing feature selection'
                    f' with exception: {e}')
        raise e 

    return variable_mask, scaler