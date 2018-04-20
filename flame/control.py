#! -*- coding: utf-8 -*-

##    Description    Flame Control class
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
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import util.utils as util
import os
import yaml

class Control:

    def __init__ (self, model, version):

        self.yaml_file = util.model_path(model,version) + '/parameters.yaml'

        success, parameters = self.load_parameters(model)

        #TODO: remove this code
        if not success:
            print ('CRITICAL ERROR: unable to load parameter file. Running with fallback defaults')
            parameters = self.get_defaults()
            # self.save_parameters(parameters)

        self.parameters = parameters
        self.parameters['model_path'] = util.model_path(model,version)
        self.parameters['md5'] = util.md5sum(self.yaml_file)


    def load_parameters (self, model):
        ''' Loads parameters from a yaml file '''

        if not os.path.isfile (self.yaml_file):
            return False, None

        try:
            with open (self.yaml_file, 'r') as pfile:
                parameters = yaml.load(pfile)
        except:
            return False, None
        
        return True, parameters

    # def save_parameters (self, parameters):
    #     yaml.dump(open(self.yaml_file,'w'), parameters)


    def get_parameters (self):
        ''' Commodity function to access stored parameters '''
        
        return self.parameters


    def get_model_set (self):
        ''' Returns a Boolean indicating if the model uses external input sources and a list with these sources '''

        ext_input = False
        model_set = None

        if 'ext_input' in self.parameters:
            if self.parameters['ext_input']:
                if 'model_set' in self.parameters:
                    if len(self.parameters['model_set'])>1:
                        model_set = self.parameters['model_set']
                        ext_input = True

        return ext_input, model_set


    def get_defaults (self):
        ''' Fallback for setting parameters even when no "config.yaml" file is found '''

        parameters = {
            ## system settings
            'verbose_error' : True,
            'numCPUs' : 1,                                  # (int)
            
            ## input settings
            'input_type' : 'molecule',                     # 'molecule' | 'data'
            'normalize_method' : 'standardize',             # None | 'standardize'
            'ionize_method' : None,                         # None | 'moka'
            'convert3D_method' : None,                      # 'ETKDG' 
            'computeMD_method' : ['RDKit_properties'],      # 'RDKit_properties'|'RDKit_md'|'custom'
            
            'SDFile_name' : 'GENERIC_NAME',                 # (str)
            'SDFile_activity' : 'activity',                 # (str)
            'SDFile_experimental' : 'IC50',                 # (str)

            ## learn/apply settings
            'modelingToolkit' : 'internal',                 # 'internal' | 'R' | 'KNIME' | 'custom'
            'model' : 'RF',
            'modelAutoscaling' : None,
            'quantitative' : True,
            'tune' : False,
            # self.modelLV = None
            # self.modelCutoff = None

            ## Model Validation Settings
            'ModelValidationCV' : 'loo',
            'ModelValidationN' : 2,
            'ModelValidationP' : 1,
            'ModelValidationLC' : False,                    # Learning curve

            # self.selVar = None
            # #self.selVarMethod = None
            # self.selVarLV = None
            # #self.selVarCV = None
            # self.selVarRun = None
            # self.selVarMask = None

            ## Random Forest
            'RF_parameters' : { 
                "n_estimators" : 200, 
                "max_features" : "sqrt",
                "class_weight" : "balanced", 
                "random_state" : 46,
                "oob_score"    : True, 
                "n_jobs"       : -1, 
                "max_depth" : None 
                },

            'RF_optimize' : {
                'n_estimators': range(50, 200, 50),
                'max_features': ['sqrt','log2'],
                'class_weight' : [None, 'balanced'],
                'oob_score' : [True],
                'random_state' : [46],
                },  

            ## SVM
            'SVM_parameters' : {
                "kernel" : "rbf",
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
                "random_state" : 46
                },

            'SVM_optimize' : {
                'kernel': ['rbf', ],                          # kernels: poly, sigmoid
                'gamma': ['auto'],
                'coef0': [0.0, 0.8, 100.0],
                'C': [1, 10, 100] ,
                'degree': [1, 3, 5],
                'class_weight' : [None, 'balanced'],
                'random_state' : [46]
                },  

            ## conformal predictor  settings
            'conformal' : False,
            'conformalSignificance' : 0.2
        }

        return parameters

                    
