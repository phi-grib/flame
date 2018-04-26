#! -*- coding: utf-8 -*-

##    Description    Flame Apply class
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu), Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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

import numpy as np
import pickle
import util.utils as utils

class Apply:

    def __init__ (self, parameters, results):

        self.parameters = parameters
        self.results = results

        self.results['origin'] = 'apply'


    def run (self):
       
        # assume X matrix is present in 'xmatrix0
        X = self.results["xmatrix"]

        # retrieve data and dimensions from results
        try:
            nobj, nvarx = np.shape(X)
        except:
            self.results['error'] = 'Failed to generate MD'
            return self.results

        if (nobj==0) or (nvarx==0) :
            self.results['error'] = 'Failed to extract activity or to generate MD'
            return self.results

        try:
            model_file = self.parameters['model_path'] + '/model.pkl'
            with open(model_file, "rb") as input_file:
                estimator = pickle.load(input_file)
        except:
            self.results['error'] = 'No valid model estimator found'
            return self.results

        estimator.project(X, self.results)

        if 'error' in self.results:
            return self.results

        ## TODO: implement this for every prediction
        zero_array = np.zeros(nobj, dtype=np.float64)

        # if not 'CI' in self.results:
        #     self.results['CI'] = zero_array
        # if not 'RI' in self.results:
        #     self.results['RI'] = zero_array   

        utils.add_result (self.results, zero_array, 'CI', 'CI (95%)', 'confidence', 'objs', 'Approximate 95% Confidence Interval')

        utils.add_result (self.results, zero_array, 'RI', 'RI (prob)', 'confidence', 'objs', 'Reliability Index, from 0 (good) to 6 (bad)')

        return self.results

