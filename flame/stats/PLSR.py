
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

class PLS_r(PLSRegression):
    
    def predict(self, X, copy=True):
        results =  super(PLS_r, self).predict(X, copy=True).ravel()
        return results



class PLSR(BaseEstimator):

    def __init__ (self, X, Y, parameters):
            super(PLSR, self).__init__(X, Y, parameters)

            self.estimator_parameters = parameters['PLSR_parameters']
            self.tune = parameters['tune']
            self.tune_parameters = parameters['PLSR_optimize']
            self.name = "PLSR"
            

    def build (self):
        if not self.quantitative:
            print("PLSR only applies to quantitative data")
            return False, "PLSR only applies to quantitative data"
        if self.conformal:
            print("Conformal prediction no implemented in PLSR yet")
            return False, "Conformal prediction no implemented in PLSR yet"

        if self.failed:
            return False, "Error initiating model"

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
            self.optimize(X, Y, PLS_r(**self.estimator_parameters), self.tune_parameters)
            results.append(('model','model type','PLSR quantitative (optimized)'))


        else:
                print ("Building  Quantitative PLSR")
                self.estimator =  PLS_r(**self.estimator_parameters)
                results.append(('model','model type','PLSR quantitative'))


        self.estimator.fit(X, Y)

        return True, results



