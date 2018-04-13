#! -*- coding: utf-8 -*-

##    Description    Flame Parent Model Class
##
##    Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from stats.base_model import BaseEstimator
from stats.base_model import getCrossVal
from stats.scale import scale, center
from stats.model_validation import CF_QuanVal

from stats.base_model import *



class RF(BaseEstimator):

    def __init__ (self, X=None,
                        Y=None, 
                        quantitative=False, 
                        autoscale=False, 
                        tune=False,  
                        cv='loo',
                        n=2,
                        p=1, 
                        lc=False,
                        conformalSignificance=0.05,
                        vpath = '',
                        estimator_parameters={},
                        tune_parameters={}, 
                        conformal=False):
        if X is not None:
            super(RF,self).__init__(X, Y, quantitative, autoscale,
                                cv, n, p, lc, conformalSignificance, vpath 
                                , estimator_parameters, conformal)
                  
            self.tune = tune
            self.tune_parameters = tune_parameters
            
            if self.quantitative:
                self.name = "RF-R"
                self.tune_parameters.pop("class_weight")
            else:
                self.name = "RF-C"

            self.failed = False
                
        else:
            self.failed = True

    
    """Build a new RF model with the X and Y numpy matrices """

    def build (self):

        if self.failed:
            return False

        X = self.X.copy()
        Y = self.Y.copy()
        
        if self.autoscale:
            X, self.mux = center(X)
            X, self.wgx = scale(X, self.autoscale)

        if self.cv:
            self.cv = getCrossVal(self.cv, 
                                  self.estimator_parameters["random_state"],
                                  self.n, self.p)
        if self.tune :
            if self.quantitative:
                self.optimize (X, Y, RandomForestRegressor(), self.tune_parameters)
            else:
                self.optimize (X, Y, RandomForestClassifier(), self.tune_parameters)

        else:
            if self.quantitative:
                print ("Building Quantitative RF model")
                self.estimator_parameters.pop('class_weight', None)
                
                self.estimator = RandomForestRegressor(**self.estimator_parameters)
            else:
                print ("Building Qualitative RF_model")
                self.estimator = RandomForestClassifier(**self.estimator_parameters)

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