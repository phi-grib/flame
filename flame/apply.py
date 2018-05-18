#! -*- coding: utf-8 -*-

# Description    Flame Apply class
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu), Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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

import numpy as np
import pickle
import util.utils as utils


class Apply:

    def __init__(self, parameters, results):

        self.parameters = parameters
        self.results = results

        self.results['origin'] = 'apply'

    def run_internal(self):
        ''' 

        Runs prediction tasks using internally defined methods

        Most of these methods can be found at the stats folder

        '''

        # assume X matrix is present in 'xmatrix0
        X = self.results["xmatrix"]

        # retrieve data and dimensions from results
        try:
            nobj, nvarx = np.shape(X)
        except:
            self.results['error'] = 'Failed to generate MD'
            return

        if (nobj == 0) or (nvarx == 0):
            self.results['error'] = 'Failed to extract activity or to generate MD'
            return

        try:
            model_file = self.parameters['model_path'] + '/model.pkl'
            with open(model_file, "rb") as input_file:
                estimator = pickle.load(input_file)
        except:
            self.results['error'] = 'No valid model estimator found'
            return

        estimator.project(X, self.results)

        # TODO: implement this for every prediction
        # zero_array = np.zeros(nobj, dtype=np.float64)

        # if not 'CI' in self.results:
        #     self.results['CI'] = zero_array
        # if not 'RI' in self.results:
        #     self.results['RI'] = zero_array

        # utils.add_result (self.results, zero_array, 'CI', 'CI (95%)',
        # 'confidence', 'objs', 'Approximate 95% Confidence Interval')

        # utils.add_result (self.results, zero_array, 'RI', 'RI (prob)',
        # 'confidence', 'objs', 'Reliability Index, from 0 (good) to 6 (bad)')

        return

    def run_R(self):
        ''' Runs prediction tasks using an importer KNIME workflow '''
        self.results['error'] = 'R toolkit is not supported in this version'
        return

    def run_KNIME(self):
        ''' Runs prediction tasks using R code '''
        self.results['error'] = 'KNIME toolkit is not supported in this version'
        return

    def run_custom(self):
        ''' Template to be overriden in apply_child.py

            Input: must be already present in self.results
            Output: add prediction results to self.results using the utils.add_result() method 

        '''

        self.results['error'] = 'custom prediction must be defined in the model apply_chlid class'
        return

    def run(self):
        ''' 

        Runs prediction tasks using the information present in self.results. 

        Depending on the modelingToolkit defined in self.parameters this task will use internal methods
        or make use if imported code in R/KNIME

        The custom option allows advanced uses to write their own function 'run_custom' method in 
        the model apply_child.py

        '''

        if self.parameters['modelingToolkit'] == 'internal':
            self.run_internal()
        elif self.parameters['modelingToolkit'] == 'R':
            self.run_R()
        elif self.parameters['modelingToolkit'] == 'KNIME':
            self.run_KNIME()
        elif self.parameters['modelingToolkit'] == 'custom':
            self.run_custom()
        else:
            self.results['error'] = 'Unknown prediction toolkit to run ', self.parameters['modelingToolkit']

        return self.results
