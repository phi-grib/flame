
# -*- coding: utf-8 -*-

# Description    RF model classifier and regressor
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2017 Manuel Pastor
##
# This file is part of eTOXlab.
##
# eTOXlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# eTOXlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.

from sklearn import svm

from stats.base_model import *


class SVM(BaseEstimator):

    def __init__(self, X=None,
                 Y=None,
                 quantitative=False,
                 autoscale=False,
                 tune=False,
                 cv='loo',
                 n=2,
                 p=1,
                 lc=False,
                 conformalSignificance=0.05,
                 vpath='',
                 estimator_parameters={},
                 tune_parameters={},
                 conformal=False):
        if X is not None:
            super(SVM, self).__init__(X, Y, quantitative, autoscale,
                                      cv, n, p, lc, conformalSignificance, vpath, estimator_parameters, conformal)

            self.tune = tune
            self.tune_parameters = tune_parameters
            if self.quantitative:
                self.name = "SVM-R"
                self.estimator_parameters.pop("class_weight", None)
                self.estimator_parameters.pop("probability", None)
                self.estimator_parameters.pop("decision_function_shape", None)
                self.estimator_parameters.pop("random_state", None)
                self.tune_parameters.pop("class_weight", None)
                self.tune_parameters.pop("random_state", None)

            else:
                self.estimator_parameters.pop("epsilon", None)
                self.name = "SVM-C"

            self.failed = False

        else:
            self.failed = True

    def build(self):

        if self.failed:
            return False

        X = self.X.copy()
        Y = self.Y.copy()

        if self.autoscale:
            X, self.mux = center(X)
            X, self.wgx = scale(X, self.autoscale)

        if self.cv:
            self.cv = getCrossVal(self.cv, 1226, self.n, self.p)

        if self.tune:
            if self.quantitative:
                self.optimize(X, Y, svm.SVR(), self.tune_parameters)
            else:
                self.optimize(X, Y, svm.SVC(probability=True),
                              self.tune_parameters)

        else:
            if self.quantitative:
                print("Building Quantitative SVM-R model")

                self.estimator = svm.SVR(**self.estimator_parameters)
            else:
                print("Building Qualitative SVM-C")
                self.estimator = svm.SVC(**self.estimator_parameters)
        if self.conformal:
            if self.quantitative:
                self.conformal_pred = AggregatedCp(IcpRegressor(RegressorNc(RegressorAdapter(self.estimator))),
                                                   BootstrapSampler())
                self.conformal_pred.fit(X, Y)
            else:
                self.conformal_pred = AggregatedCp(IcpClassifier(ClassifierNc(ClassifierAdapter(self.estimator),
                                                                              MarginErrFunc())), BootstrapSampler())
                self.conformal_pred.fit(X, Y)

        self.estimator.fit(X, Y)

        return True
