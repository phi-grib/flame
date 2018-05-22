#! -*- coding: utf-8 -*-

# Description    Flame Parent Model Class
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from stats.base_model import BaseEstimator
from stats.base_model import getCrossVal
from stats.scale import scale, center
from stats.model_validation import CF_QuanVal

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc

class RF(BaseEstimator):

    def __init__(self, X, Y, parameters):
        super(RF, self).__init__(X, Y, parameters)

        self.estimator_parameters = parameters['RF_parameters']
        self.tune = parameters['tune']
        self.tune_parameters = parameters['RF_optimize']

        if self.quantitative:
            self.name = "RF-R"
            self.tune_parameters.pop("class_weight")
        else:
            self.name = "RF-C"


    def build(self):
        '''Build a new RF model with the X and Y numpy matrices '''

        if self.failed:
            return False

        X = self.X.copy()
        Y = self.Y.copy()

        if self.autoscale:
            X, self.mux = center(X)
            X, self.wgx = scale(X, self.autoscale)

        results = []

        results.append (('nobj', 'number of objects', self.nobj))
        results.append (('nvarx', 'number of predictor variables', self.nvarx))

        if self.cv:
            self.cv = getCrossVal(self.cv,
                                  self.estimator_parameters["random_state"],
                                  self.n, self.p)
        if self.tune:
            if self.quantitative:
                self.optimize(X, Y, RandomForestRegressor(),
                              self.tune_parameters)
                results.append(('model','model type','RF quantitative (optimized)'))
            else:
                self.optimize(X, Y, RandomForestClassifier(),
                              self.tune_parameters)
                results.append (('model','model type','RF cualitative (optimized)'))
        else:
            if self.quantitative:
                print("Building Quantitative RF model")
                self.estimator_parameters.pop('class_weight', None)

                self.estimator = RandomForestRegressor(
                    **self.estimator_parameters)
                results.append(('model','model type','RF quantitative'))

            else:
                print("Building Qualitative RF_model")
                self.estimator = RandomForestClassifier(
                    **self.estimator_parameters)
                results.append(('model','model type','RF cualitative'))

        if self.conformal:
            if self.quantitative:
                self.conformal_pred = AggregatedCp(IcpRegressor(RegressorNc(RegressorAdapter(self.estimator))),
                                                   BootstrapSampler())
                self.conformal_pred.fit(X, Y)
                results.append(('model','model type','conformal RF quantitative'))  #overrides non-conformal
                
            else:
                self.conformal_pred = AggregatedCp(IcpClassifier(ClassifierNc(ClassifierAdapter(self.estimator),
                                                                              MarginErrFunc())), BootstrapSampler())
                self.conformal_pred.fit(X, Y)
                results.append(('model','model type','conformal RF cualitative'))   #overrides non-conformal
                
        self.estimator.fit(X, Y)

        return True, results
