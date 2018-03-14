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

from RF import RF
from SVM import SVM
import numpy as np
import pickle 

class Learn:

    def __init__ (self, control, results):

        self.control = control # control object defining the processing

        self.X = results[0]
        self.Y = results[1]
        # TODO: make use of other results items
        
        self.model_path = self.control.model_path

    def run (self):

        nobj, nvarx = np.shape(self.X)

        if (nobj==0) or (nvarx==0) :
            return False, 'failed to extract activity or to generate MD'

        if (np.shape(self.Y)==0) :
            return False, 'no activity found'
        
        # initilizate estimator
        if self.control.model == 'RF':
            model = RF(self.X,self.Y, self.control.quantitative, self.control.modelAutoscaling, self.control.tune,
                        self.control.ModelValidationCV, self.control.ModelValidationN, self.control.ModelValidationP, 
                        self.control.ModelValidationLC ,self.control.conformalSignificance, self.model_path,
                        self.control.RF_parameters, self.control.RF_optimize, self.control.conformal)
        elif self.control.model == 'SVM':
            model = SVM(self.X,self.Y, self.control.quantitative, self.control.modelAutoscaling, self.control.tune,
                        self.control.ModelValidationCV, self.control.ModelValidationN, self.control.ModelValidationP, 
                        self.control.ModelValidationLC ,self.control.conformalSignificance, self.model_path,
                        self.control.SVM_parameters, self.control.SVM_optimize, self.control.conformal)
            
        else:
            return False, 'modeling method not recognised'
            
        # build model       
        success = model.build()
        if not success:
            return success, 'error building '+self.control.model+' model'

        # validate model
        success, results = model.validate()
        if not success:
            return success, results
            
        # TODO: this must be a class method even if we can define the base path
        # save model
        with open(self.model_path +  '/model.pkl', 'wb') as handle:
            pickle.dump(model , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # copy any relevant information from the model building into a dictionary
        # what is relevant? to be defined...
        results = {'origin':'learn'}
        results = model.getResults(results)

        # TODO: compute AD (when applicable)

        return True, results
