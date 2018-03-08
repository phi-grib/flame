# -*- coding: utf-8 -*-

##    Description    RF model classifier and regressor
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2017 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.


from base_model import BaseEstimator
from base_model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor



class RF(BaseEstimator):

    def __init__ (self, X=None, Y=None, quantitative=False, autoscale=False, tune=False,  
                        cv='loo', n=2, p=1, lc=True, conformalSignificance=0.05, vpath = '',
                        estimator_parameters={}, tune_parameters={}, conformal=False):
        if X is not None:
            parent = super(RF,self).__init__(X, Y, quantitative, autoscale,
                                cv, n, p, lc, conformalSignificance, vpath 
                                , estimator_parameters, conformal)

                  
            self.tune = tune
            self.tune_parameters = tune_parameters
            if self.quantitative:
                self.name = "RF-R"
            else:
                self.name = "RF-C"
                
        else:
            pass

        


    def build (self):
        """Build a new RF model with the X and Y numpy matrices

        """

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

        self.estimator.fit(X, Y)




## Old optimization. Now runs through gridsearchcv

'''    def optimize (self, X, Y ):
    """ Optimizes the number of trees (estimators) and max features used (features)
        and returns the best values, acording to the OOB criteria

        The results are shown in a diagnostic plot

        To avoid including many trees to produce tiny improvements, increments of OOB error
        below 0.01 are considered irrelevant
    """

    RANDOM_STATE = 1226
    errors = {}
    features = ['sqrt','log2','none']

    if self.quantitative:
        tclf = {'sqrt': RandomForestRegressor(warm_start=False, oob_score=True,
                    max_features="sqrt",random_state=RANDOM_STATE),
                'log2': RandomForestRegressor(warm_start=False, oob_score=True,
                    max_features="log2",random_state=RANDOM_STATE),
                'none': RandomForestRegressor(warm_start=False, oob_score=True,
                    max_features=None  ,random_state=RANDOM_STATE) }
    else:
        tclf = {'sqrt': RandomForestClassifier(warm_start=False, oob_score=True,
                    max_features="sqrt",random_state=RANDOM_STATE,
                    class_weight=self.class_weight),
                'log2': RandomForestClassifier(warm_start=False, oob_score=True,
                    max_features="log2",random_state=RANDOM_STATE,
                    class_weight=self.class_weight),
                'none': RandomForestClassifier(warm_start=False, oob_score=True,
                    max_features=None  ,random_state=RANDOM_STATE,
                    class_weight=self.class_weight) }

    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 700
    stp_estimators = 100

    num_steps = int((max_estimators-min_estimators)/stp_estimators)

    print ('optimizing RF....')
    updateProgress (0.0)

    optValue = 1.0e10
    j = 0
    for fi in features:
        errors[fi] = []
        count = 0
        for i in range(min_estimators, max_estimators + 1,stp_estimators):
            clf = tclf[fi]
            clf.set_params(n_estimators=i)
            clf.fit(X,Y)
            oob_error = 1 - clf.oob_score_
            errors[fi].append((i,oob_error))
            if oob_error < optValue:
                if np.abs(oob_error - optValue) > 0.01:
                    optValue = oob_error
                    optEstimators = i
                    optFeatures = fi

            updateProgress (float(count+(j*num_steps))/float(len(features)*num_steps))
            count = count+1
        j=j+1

    for ie in errors:
        xs, ys = zip (*errors[ie])
        plt.plot(xs, ys, label=ie)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators (Trees)")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    # plt.show()

    plt.savefig(self.vpath+"/rf-OOB-parameter-tuning.png")
    plt.savefig("./rf-OOB-parameter-tuning.png")

    print ('optimum features:', optFeatures, 'optimum estimators:', optEstimators, 'best OOB:', optValue)

    return (optEstimators, optFeatures)

'''