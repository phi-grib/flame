
# -*- coding: utf-8 -*-

# Description    Flame Parent Model Class
##
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
#                Manuel Pastor (manuel.pastor@upf.edu)
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



from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from stats.base_model import BaseEstimator
from stats.base_model import getCrossVal
from stats.scale import scale, center
from stats.model_validation import CF_QuanVal

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc

import pandas as pd
import numpy as np

class PLS_da(PLSRegression):
    def __init__ (self, n_components=2, scale=False, max_iter=500,
                  tol=1e-6, copy=True, threshold=0.5 ):
            super(PLS_da, self).__init__(n_components=n_components, 
                  scale=scale, max_iter=max_iter, tol=tol, copy=copy)
            self.threshold = threshold

    def predict(self, X, copy=True):
        threshold = self.threshold
        if threshold is None:
            threshold = 0.5

        results =  super(PLS_da, self).predict(X, copy=True).ravel()
        results[results < threshold] = 0
        results[results > threshold] = 1
        results = results.astype(dtype=int)

        return results

    # def get_params(self):
    #     results =  super(PLS_da, self).get_params()
    #     return results



class PLSDA(BaseEstimator):

    def __init__ (self, X, Y, parameters):
            super(PLSDA, self).__init__(X, Y, parameters)

            self.estimator_parameters = parameters['PLSDA_parameters']
            self.tune = parameters['tune']
            self.tune_parameters = parameters['PLSDA_optimize']
            self.name = "PLSDA"
            

    def build (self):
        
        if self.failed:
            return False, "Error initiating model"
        if  self.quantitative:
            print("PLSDA only applies to qualitative data")
            return False, "PLSR only applies to qualitative data"
        if self.conformal:
            print("Conformal prediction no implemented in PLSDA yet")
            return False, "Conformal prediction no implemented in PLS-DA yet"

        X = self.X.copy()
        Y = self.Y.copy()


        if self.autoscale:
            X, self.mux = center(X)
            X, self.wgx = scale(X, self.autoscale)
            print ("autoscaling")

        results = []
        results.append (('nobj', 'number of objects', self.nobj))
        results.append (('nvarx', 'number of predictor variables', self.nvarx))

        if self.cv:
            self.cv = getCrossVal(self.cv, 46 , self.n, self.p)


        if self.tune :
            self.optimize(X, Y, PLS_da(n_components=2, scale=False, max_iter=500,
                                        tol=1e-6, copy=True, threshold=0.5), self.tune_parameters)
            results.append(('model','model type','PLSDA qualitative (optimized)'))


        else:
                print ("Building  Qualitative PLSDA")
                self.estimator =  PLS_da(**self.estimator_parameters)
                print (self.estimator.get_params())
                results.append(('model','model type','PLSDA qualitative'))


        self.estimator.fit(X, Y)

        return True, results



