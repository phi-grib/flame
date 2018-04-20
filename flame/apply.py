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

class Apply:

    def __init__ (self, parameters, results):

        self.parameters = parameters
        self.results = results
        
    def getMatrices (self):
        return self.results["xmatrix"]

    def run (self):
       
        X = self.getMatrices()
        # retrieve data and dimensions from results
        nobj, nvarx = np.shape(X)

        if (nobj==0) or (nvarx==0) :
            return False, 'failed to extract activity or to generate MD'

        # try:
        #     model_file = self.parameters['model_path'] + '/model.pkl'

        #     # select prediction tool from control
        #     with open(model_file, "rb") as input_file:
        #         estimator = pickle.load(input_file)

        #     zero_array = np.zeros(nobj, dtype=np.float64)
        #     self.results['origin'] = 'apply'
        #     self.results['projection'] = estimator.project(X)
        #     self.results['CI'] = zero_array
        #     self.results['RI'] = zero_array
        # except:
        #     return False, 'projection error'

        model_file = self.parameters['model_path'] + '/model.pkl'

        # select prediction tool from control
        with open(model_file, "rb") as input_file:
            estimator = pickle.load(input_file)

        zero_array = np.zeros(nobj, dtype=np.float64)

        self.results['origin'] = 'apply'
        #self.results['projection'] = estimator.project(X)
        #self.results['CI'] = zero_array
        #self.results['RI'] = zero_array

        projection = estimator.project(X)

        for key in projection:
            self.results[key] = projection[key]
            
        if not 'CI' in self.results:
            self.results['CI'] = zero_array
        if not 'RI' in self.results:
            self.results['RI'] = zero_array   
        #print (self.results)


        return True, self.results

