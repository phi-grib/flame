
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


from sklearn.naive_bayes import GaussianNB
from flame.stats.base_model import BaseEstimator
from flame.stats.base_model import getCrossVal
from flame.stats.scale import scale, center
from flame.stats.model_validation import CF_QuanVal

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc


class GNB(BaseEstimator):

    def __init__(self, X, Y, parameters):
        super(GNB, self).__init__(X, Y, parameters)

        self.estimator_parameters = parameters['GNB_parameters']

        if self.quantitative:
            return

        else:
            self.name = "GNB-Classifier"

    def build(self):
        '''Build a new qualitative GNB model with the X and Y numpy matrices'''
        if self.failed:
            return False, "Error initiating model"

        X = self.X.copy()
        Y = self.Y.copy()


        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        if self.cv:
            self.cv = getCrossVal(self.cv, 46, self.n, self.p)

        if self.quantitative:
            print("GNB only applies to qualitative data")
            return False, "GNB only applies to qualitative data"

        else:
            print("Building GaussianNB model")
            print(self.estimator_parameters)
            self.estimator = GaussianNB(**self.estimator_parameters)
            results.append(('model', 'model type', 'GNB qualitative'))

        if self.conformal:
            self.conformal_pred = AggregatedCp(IcpClassifier(ClassifierNc(ClassifierAdapter(
                self.estimator),                                                                MarginErrFunc())), BootstrapSampler())
            self.conformal_pred.fit(X, Y)
            # overrides non-conformal
            results.append(
                ('model', 'model type', 'conformal GNB qualitative'))

        self.estimator.fit(X, Y)
        return True, results
