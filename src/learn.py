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
        self.results = results # results is a tuple with X, Y and ... (to be defined)
        self.vpath = self.control.vpath

    def getMatrices (self):
        """ 
        
        Returns NumPy X and Y matrices extracted from results
        
        """
        
        # ncol = 0
        # xx = []
        # yy = []

        # # obtain X and Y from tuple elements 0 (MD) and 1 (Activity)
        # for i in self.results:
        #     print (i)
        #     if len(i[0])>ncol: ncol = len(i[0])
        #     xx.append(i[0])
        #     yy.append(i[1])

        # nrow = len (xx)

        # Y = np.array (yy,dtype=np.float64)
        # X = np.empty ((nrow,ncol),dtype=np.float64)

        # i=0
        # for row in xx:
        #     X[i,:]=np.array(row,dtype=np.float64)
        #     i+=1
        print (self.results[0].shape)

        return self.results[0], self.results[1]

    def runRF (self):

        print ('hi! I am Random Forest')

        return True, 'debug RF results'

    def run (self):
        X,Y = self.getMatrices ()

        nobj, nvarx = np.shape(X)

        if (nobj==0) or (nvarx==0) :
            return False, 'failed to extract activity or to generate MD'

        nobj = np.shape(Y)
        if (nobj==0) :
            return False, 'no activity found'
        
        model = ''
        
        # initilizate estimator
        if self.control.model == 'RF':
            model = RF(X,Y, self.control.quantitative, self.control.modelAutoscaling, self.control.tune,
                        self.control.ModelValidationCV, self.control.ModelValidationN, self.control.ModelValidationP, 
                        self.control.ModelValidationLC ,self.control.conformalSignificance, self.vpath,
                        self.control.RF_parameters, self.control.RF_optimize, self.control.conformal)
        elif self.control.model == 'SVM':
            model = SVM(X,Y, self.control.quantitative, self.control.modelAutoscaling, self.control.tune,
                        self.control.ModelValidationCV, self.control.ModelValidationN, self.control.ModelValidationP, 
                        self.control.ModelValidationLC ,self.control.conformalSignificance, self.vpath,
                        self.control.SVM_parameters, self.control.SVM_optimize, self.control.conformal)
            
        else:
            return False, 'modeling method not recognised'
            
        # build model
        
        model.build()

        # validate model
        
        model.validate()
            
        # save model

        with open(self.vpath +  '/model.pickle', 'wb') as handle:
            pickle.dump(model , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # compute AD (when applicable)

        success = True
        results = 'debug learn.run results'

        return success, results

