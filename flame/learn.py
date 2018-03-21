#! -*- coding: utf-8 -*-

##    Description    Flame Learn class
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu), Jose Carlos Gómez (josecarlos.gomez@upf.edu)
##
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

from stats.RF import RF
from stats.SVM import SVM
import numpy as np
import pickle 

class Learn:

    def __init__ (self, parameters, results):

        self.parameters = parameters 

        self.X = results[0]
        self.Y = results[1]
        # TODO: make use of other results items
        
        self.model_path = self.parameters['model_path']
    
    def run_custom (self):

        return False, 'not implemented'
    
    def run_internal (self):

        nobj, nvarx = np.shape(self.X)

        if (nobj==0) or (nvarx==0) :
            return False, 'failed to extract activity or to generate MD'

        if (np.shape(self.Y)==0) :
            return False, 'no activity found'
        
        # initilizate estimator
        model = self.parameters['model']
        if  model == 'RF':
            model = RF(self.X,self.Y, self.parameters['quantitative'], self.parameters['modelAutoscaling'], self.parameters['tune'],
                        self.parameters['ModelValidationCV'], self.parameters['ModelValidationN'], self.parameters['ModelValidationP'], 
                        self.parameters['ModelValidationLC'] ,self.parameters['conformalSignificance'], self.model_path,
                        self.parameters['RF_parameters'], self.parameters['RF_optimize'], self.parameters['conformal'])
        elif model == 'SVM':
            model = SVM(self.X,self.Y, self.parameters['quantitative'], self.parameters['modelAutoscaling'], self.parameters['tune'],
                        self.parameters['ModelValidationCV'], self.parameters['ModelValidationN'], self.parameters['ModelValidationP'], 
                        self.parameters['ModelValidationLC'] ,self.parameters['conformalSignificance'], self.model_path,
                        self.parameters['SVM_parameters'], self.parameters['SVM_optimize'], self.parameters['conformal'])

            
        else:
            return False, 'modeling method not recognised'
            
        # build model       
        success = model.build()
        if not success:
            return success, 'error building '+self.parameters['model']+' model'

        # validate model
        success, results = model.validate()
        if not success:
            return success, results
            
        # TODO: this must be a class method even if we can define the base path
        # save model
        with open(self.model_path +  '/model.pkl', 'wb') as handle:
            pickle.dump(model , handle, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO: compute AD (when applicable)
        
        # copy any relevant information from the model building into a dictionary
        # what is relevant? to be defined...
        results = {'origin':'learn'}
        results = model.getResults(results)

        return True, results

    def run (self):

        toolkit = self.parameters['modelingToolkit']
        if toolkit == 'internal':
            success, results = self.run_internal ()
        elif toolkit == 'custom':
            success, results = self.run_custom ()
        else:
            return False, 'modeling Toolkit '+toolkit+' is not supported yet'

        return success, results
