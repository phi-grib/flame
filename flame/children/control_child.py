#! -*- coding: utf-8 -*-

##    Description    Flame Control internal class
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
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
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
from control import Control

class ControlChild (Control):

    def __init__ (self):

        Control.__init__ (self)
        
        # this is COMPULSORY and must by called by child class to setup
        self.model_path = os.path.dirname(os.path.abspath(__file__))

        ###
        ### system settings
        ###

        self.verbose_error = True
        self.numCPUs = 2                            # (int)
        
        ###
        ### input settings
        ###

        self.input_type = 'molecule'                # 'molecule' | 'data'
        self.normalize_method = 'standardize'       # None | 'standardize'
        self.ionize_method = None                   # None | 'moka'
        self.convert3D_method = None                  # 'ETKDG' 
        self.computeMD_method = ['RDKit_properties']        # 'RDKit_properties'|'RDKit_md'|'custom'
        
        self.SDFile_name = 'GENERIC_NAME'           # (str)
        self.SDFile_activity = 'activity'           # (str)
        self.SDFile_experimental = 'IC50'           # (str)

        ###
        ### learn/apply settings
        ###

        self.modelingToolkit = 'internal'           # 'internal' | 'R' | 'KNIME' | 'custom'
        self.model = 'RF'
        self.modelAutoscaling = None
        self.quantitative = True
        self.tune = False
        # self.modelLV = None
        # self.modelCutoff = None

        ## Model Validation Settings
        self.ModelValidationCV = 'loo'
        self.ModelValidationN = 2
        self.ModelValidationP = 1
        self.ModelValidationLC = False # Learning curve

        # self.selVar = None
        # #self.selVarMethod = None
        # self.selVarLV = None
        # #self.selVarCV = None
        # self.selVarRun = None
        # self.selVarMask = None

        ## Random Forest
        self.RF_parameters = {   "n_estimators" : 200, 
                            "max_features" : "sqrt",
                            "class_weight" : "balanced", 
                            "random_state" : 46,
                            "oob_score"    : True, 
                            "n_jobs"       : -1, 
                            "max_depth" : None }

        self.RF_optimize = {'n_estimators': range(50, 200, 50),
                    'max_features': ['sqrt','log2'],
                    'class_weight' : [None, 'balanced'],
                    'oob_score' : [True],
                    'random_state' : [46],
                    }  
        ## SVM

        self.SVM_parameters = {"kernel" : "rbf",
                        "degree" : 3, 
                        "gamma" : "auto",
                        "coef0" : 0.0, 
                        "probability" : False,
                        "decision_function_shape" : "ovr",
                        "class_weight" : "balanced",
                        "tol" : 1e-3,
                        "epsilon" : 0.1,
                        "C" : 1.0,
                        "shrinking" : True,
                        "random_state" : 46}


        self.SVM_optimize = {'kernel': ['rbf', ],
                     'gamma': ['auto'],
                     'coef0': [0.0, 0.8, 100.0],
                     'C': [1, 10, 100] ,
                     'degree': [1, 3, 5],
                     'class_weight' : [None, 'balanced'],
                     'random_state' : [46]
                    }  # kernels: poly, sigmoid

        ## conformal predictor  settings
        self.conformal = False
        self.conformalSignificance = 0.2

        return
