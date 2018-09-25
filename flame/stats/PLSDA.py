
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


# To ignore warnings comming from data precision in Cross-validation
# Study more in deep

from flame.stats.base_model import BaseEstimator
from flame.stats.base_model import getCrossVal
from flame.stats.scale import scale, center
from flame.stats.model_validation import CF_QuanVal

import copy

from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc, f1_score as f1

import warnings
warnings.filterwarnings('ignore')


class PLS_da(PLSRegression):
    def __init__(self, n_components=2, scale=False, max_iter=500,
                 tol=1e-6, copy=True, threshold=None):
        super(PLS_da, self).__init__(n_components=n_components,
                                     scale=scale, max_iter=max_iter, tol=tol, copy=copy)
        self.threshold = threshold

    def predict(self, X, copy=True):
        threshold = self.threshold
        if threshold is None:
            return super(PLS_da, self).predict(X, copy=True).ravel()

        results = super(PLS_da, self).predict(X, copy=True).ravel()
        results[results < threshold] = 0
        results[results >= threshold] = 1
        results = results.astype(dtype=float)

        return results


class PLSDA(BaseEstimator):

    def __init__(self, X, Y, parameters):

        super(PLSDA, self).__init__(X, Y, parameters)

        self.estimator_parameters = parameters['PLSDA_parameters']
        self.tune = parameters['tune']
        self.tune_parameters = parameters['PLSDA_optimize']
        self.name = "PLSDA"
        self.optimiz = self.estimator_parameters["optimize"]
        self.estimator_parameters.pop("optimize")

    def build(self):

        if self.failed:
            return False, "Error initiating model"
        if self.quantitative:
            print("PLSDA only applies to qualitative data")
            return False, "PLSR only applies to qualitative data"
        if self.conformal:
            raise ValueError("Conformal prediction no implemented " 
                             "in PLSDA yet, please change "
                             "conformal option to false in the parameter file")

        X = self.X.copy()
        Y = self.Y.copy()


        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        if self.cv:
            self.cv = getCrossVal(self.cv, 46, self.n, self.p)

        if self.tune:
            if self.optimiz == 'auto':
                super(PLSDA, self).optimize(X, Y, PLS_da(n_components=2, scale=False, max_iter=500,
                                                         tol=1e-6, copy=True, threshold=0.5), self.tune_parameters)
            elif self.optimiz == 'manual':
                self.optimize(X, Y, PLS_da(n_components=2, scale=False, max_iter=500,
                                           tol=1e-6, copy=True, threshold=None), self.tune_parameters)
            results.append(
                ('model', 'model type', 'PLSDA qualitative (optimized)'))

        else:
            print("Building  Qualitative PLSDA")
            self.estimator = PLS_da(**self.estimator_parameters)
            print(self.estimator.get_params())
            results.append(('model', 'model type', 'PLSDA qualitative'))

        print(len(Y[Y == 1]))
        self.estimator.fit(X, Y)

        return True, results

    def optimize(self, X, Y, estimator, tune_parameters):
        ''' optimizes a model using a grid search over a range of values for diverse parameters'''

        print("Optimizing PLS-DA algorithm")
        latent_variables = tune_parameters["n_components"]
        mcc_final = 0
        estimator0 = ""
        list_latent = []
        for n_comp in latent_variables:
            mcc0 = 0
            estimator.set_params(**{"n_components": n_comp})
            y_pred = cross_val_predict(estimator, X, Y, cv=self.cv, n_jobs=1)
            estimator1 = ""
            threshold_1 = 0
            for threshold in range(0, 100, 5):
                threshold = threshold / 100
                y_pred2 = copy.copy(y_pred)
                y_pred2[y_pred2 < threshold] = 0
                y_pred2[y_pred2 >= threshold] = 1
                mcc1 = mcc(Y, y_pred2)

                if mcc1 >= mcc0:
                    mcc0 = mcc1
                    estimator1 = copy.copy(estimator)
                    estimator1.set_params(**{'threshold': threshold})
                    threshold_1 = (threshold)

            if mcc0 >= mcc_final:
                mcc_final = mcc0
                estimator0 = copy.copy(estimator1)
                self.estimator = estimator0

            list_latent.append([n_comp, threshold_1, mcc0])

        print("MCC per lantent variable at best cutoff")
        for el in list_latent:
            print("Number of latent variables: %s \nBest cutoff: %s \nMCC: %s\n" %
                  (el[0], el[1], el[2]))

        self.estimator = estimator0
        self.estimator.fit(X, Y)
        print(self.estimator.get_params())
