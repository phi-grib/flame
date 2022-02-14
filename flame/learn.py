#! -*- coding: utf-8 -*-

# Description    Flame Learn class
#
# Authors: Manuel Pastor (manuel.pastor@upf.edu),
#          Jose Carlos Gómez (josecarlos.gomez@upf.edu)
#
# Copyright 2018 Manuel Pastor
#
# This file is part of Flame
#
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
#
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import os
# import pickle
import numpy as np
import yaml

# from sklearn.preprocessing import MinMaxScaler 
# from sklearn.preprocessing import StandardScaler 
# from sklearn.preprocessing import RobustScaler
# from flame.stats import feature_selection
from flame.stats.imbalance import run_imbalance  

from flame.stats.RF import RF
from flame.stats.SVM import SVM
from flame.stats.GNB import GNB
from flame.stats.PLSR import PLSR
from flame.stats.PLSDA import PLSDA
from flame.stats.combo import median, mean, majority, logicalOR, matrix
from flame.graph.graph import generateManifoldSpace, generatePCASpace

from flame.util import utils, get_logger
LOG = get_logger(__name__)

class Learn:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor

        self.registered_methods = [('RF', RF),
                              ('SVM', SVM),
                              ('GNB', GNB),
                              ('PLSR', PLSR),
                              ('PLSDA', PLSDA), 
                              ('median', median),
                              ('mean', mean),
                              ('majority', majority),
                              ('logicalOR', logicalOR),
                              ('matrix', matrix)]

        self.X = self.conveyor.getVal('xmatrix')
        self.Y = self.conveyor.getVal('ymatrix')
        self.nobj, self.nvarx = np.shape(self.X)

        # Preprocessing variables
        self.scaler = None


    def run_custom(self):
        '''
        Build a model using custom code to be defined in the learn child
        classes.
        '''

        self.conveyor.setError ('Not implemented')

    # def cpreprocess (self):
    #     ''' preprocesing for confidential models'''

    #     cpre = {}
    #     xmean= np.mean(self.X, axis=0)
    #     self.X = self.X.astype(float)
    #     self.X -= np.array(xmean)

    #     if self.param.getVal('modelAutoscaling') == 'StandardScaler':
    #         st = np.std(self.X, axis=0, ddof=1)
    #         wg = [1.0/sti if sti > 1.0e-7 else 0.00 for sti in st]
    #         wg = np.array(wg)
    #         self.X *= wg 
    #         cpre['wg'] = wg.tolist()

    #     cpre['xmean'] = xmean.tolist()

    #     model_file_path = utils.model_path(self.param.getVal('endpoint'), 0)
    #     model_file_name = os.path.join (model_file_path, 'confidential_preprocess.yaml')
    #     with open(model_file_name, 'w') as f:
    #         yaml.dump (cpre, f)

    #     return True, 'OK'

    # def preprocess(self):
    #     ''' Preprocessing workflow. 
        
    #     It includes three steps:

    #     1. imbalance: selects objects
    #         only for qualitative endpoints
    #         returns an object mask
    #         calls conveyor.mask_objects
        
    #     2. feature selection: selects variables
    #         if there is a scaler, a copy of the X matrix must be pre-scaled
    #         returns a variable mask
    #         calls conveyor.mask_variables
        
    #     3. scaler
    #         called last

    #     the variable mask and the scaled are saved in a pickl
    #     '''
    #     ###################################################################################
    #     ## STEP 1. SUBSAMPLING
    #     ###################################################################################
    #     if self.param.getVal("imbalance") is not None and not self.param.getVal("quantitative"):
            
    #         success, objmask = run_imbalance(self.param.getVal('imbalance'), self.X, self.Y)
    #         if not success:
    #             return False, objmask

    #         # ammend object variables
    #         objnum = np.count_nonzero(objmask==1)
    #         self.X = self.X[objmask==1]
    #         self.Y = self.Y[objmask==1]
    #         self.nobj= objnum

    #         # ammend conveyor
    #         self.conveyor.setVal('obj_num', objnum)
    #         self.conveyor.mask_objects(objmask)
            
    #         LOG.info(f'{self.param.getVal("imbalance")} performed')
    #         LOG.info(f'Number of objects after sampling: {objnum}')

    #     ###################################################################################
    #     ## INITIALIZE SCALER
    #     ###################################################################################
    #     scale_method = self.param.getVal('modelAutoscaling')

    #     # prevent the scaling of input which must be binary or with preserved values
    #     if scale_method is not None:
    #         non_scale_list = ['majority','logicalOR','matrix']

    #         if self.param.getVal('model') in non_scale_list:
    #             scale_method = None
    #             LOG.info(f"Method '{self.param.getVal('model')}' is incompatible with '{scale_method}' scaler. Forced to 'None'")

    #         if scale_method is not None:
    #             if scale_method == 'StandardScaler':
    #                 self.scaler = StandardScaler()
    #             elif scale_method == 'MinMaxScaler':
    #                 self.scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    #             elif scale_method == 'RobustScaler':
    #                 self.scaler = RobustScaler()
    #             else:
    #                 return False, 'Scaler not recognized'

    #         LOG.debug(f'scaler :{scale_method} initialized')
          
    #     ###################################################################################
    #     ## STEP 2. FEATURE SELECTION
    #     ###################################################################################
    #     # Run feature selection. Move to a instance method.
    #     varmask = None
    #     feature_selection_method = self.param.getVal("feature_selection")

    #     if feature_selection_method is not None:
    #         num_features = self.param.getVal("feature_number")
    #         quantitative = self.param.getVal("quantitative")
    #         X_copy = self.X.copy()
    #         Y_copy = self.Y.copy()

    #         if self.scaler is not None:
    #             self.scaler = self.scaler.fit(X_copy)
    #             X_copy = self.scaler.transform(X_copy)

    #         success, varmask = feature_selection.run_feature_selection(X_copy, Y_copy, 
    #             feature_selection_method, num_features, quantitative)

    #         LOG.debug(f'Feature selection :{feature_selection_method} finished')

    #         if not success:
    #             return False, varmask

    #         # ammend local variables
    #         varnum = np.count_nonzero(varmask==1)
    #         self.X = self.X[:, varmask]
    #         self.nvarx = varnum
            
    #         # ammend conveyor
    #         self.conveyor.mask_variables(varmask)

    #         LOG.info(f'Feature selection method: {feature_selection_method} completed. Selected {varnum} features')

    #     # Check X and Y integrity.
    #     if (self.nobj == 0) or (self.nvarx == 0):
    #         return False, 'No objects/variables in the matrix'

    #     if len(self.Y) == 0:
    #         self.failed = True
    #         return False, 'No activity values'

    #     ###################################################################################
    #     ## STEP 3. APPLY SCALER
    #     ###################################################################################
    #     if self.scaler is not None:
    #         self.scaler = self.scaler.fit(self.X)
    #         self.X = self.scaler.transform(self.X)

    #         LOG.info(f'Data scaled with method: {scale_method}')

    #     ###################################################################################
    #     ## SAVE
    #     ###################################################################################
    #     self.conveyor.addVal(self.X, 'xmatrix', 'X matrix', 'method', 'vars', 'Molecular descriptors')

    #     prepro = {'scaler':self.scaler,\
    #               'variable_mask': varmask,\
    #               'version':1}

    #     prepro_pkl_path = os.path.join(self.param.getVal('model_path'),'preprocessing.pkl')
        
    #     with open(prepro_pkl_path, 'wb') as handle:
    #         pickle.dump(prepro, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     LOG.debug('Preprocesing saved as:{}'.format(prepro_pkl_path))

    #     return True, 'OK'


    def run_internal(self):
        '''
        Builds a model using the internally defined machine learning tools.

        All input parameters are extracted from self.param.

        The main output is an instance of basemodel saved in
        the model folder as a pickle (model.pkl) and used for prediction.

        The results of building and validation are added to results,
        but also saved to the model folder as a pickle (info.pkl)
        for being displayed in manage tools.
        '''

        # expand with new methods here:
        # registered_methods = [('RF', RF),
        #                       ('SVM', SVM),
        #                       ('GNB', GNB),
        #                       ('PLSR', PLSR),
        #                       ('PLSDA', PLSDA), 
        #                       ('median', median),
        #                       ('mean', mean),
        #                       ('majority', majority),
        #                       ('logicalOR', logicalOR),
        #                       ('matrix', matrix)]

        if self.param.getVal('model') == 'XGBOOST':
            from flame.stats.XGboost import XGBOOST
            self.registered_methods.append( ('XGBOOST', XGBOOST))

        # check suitability of Y matrix
        if not self.param.getVal('quantitative') :
            success, yresult  = utils.qualitative_Y(self.Y)
            if not success:
                self.conveyor.setError(yresult)
                return

        # print (np.shape(self.X))

        # collect model information from parameters
        model_type_info = []
        model_type_info.append(('quantitative', 'True if the endpoint is quantitative', self.param.getVal('quantitative')))
        model_type_info.append(('conformal', 'True if the endpoint is conformal', self.param.getVal('conformal')))
        model_type_info.append(('confidential', 'True if the model is confidential', self.param.getVal('confidential')))
        model_type_info.append(('secret', 'True for barebone models exported by a confidential models', False))
        model_type_info.append(('ensemble', 'True if the model is an ensemble of models', self.param.getVal('input_type') == 'model_ensemble'))
        model_type_info.append(('ensemble_names', 'List of ensemble models', self.param.getVal('ensemble_names')))
        model_type_info.append(('ensemble_versions', 'List of ensemble versions', self.param.getVal('ensemble_versions')))
        model_type_info.append(('conformal_confidence', 'Confidence of the conformal model', self.param.getVal('conformalConfidence')))

        self.conveyor.addVal(
            model_type_info,
            'model_type_info',
            'model type information',
            'method',
            'single',
            'Information about the type of model')

        # instantiate an appropriate child of base_model
        model = None
        for imethod in self.registered_methods:
            if imethod[0] == self.param.getVal('model'):

                # we instantiate the subtype of base_model, 
                # passing 
                # - preteated X and Y matrices for model building
                # - model parameters (param) 
                # - already obtained results (conveyor)

                model = imethod[1](self.X, self.Y, self.param, self.conveyor)
                LOG.debug('Recognized learner: '
                          f"{self.param.getVal('model')}")
                break

        if not model:
            self.conveyor.setError(f'Modeling method {self.param.getVal("model")}'
                                    'not recognized')
            LOG.error(f'Modeling method {self.param.getVal("model")}'
                       'not recognized')
            return
            
        if self.conveyor.getError():
            return

        # build model
        LOG.debug('Starting model building')
        success, model_building_results = model.build()
        if not success:
            self.conveyor.setError(model_building_results)
            return

        self.conveyor.addVal(
                    model_building_results,
                    'model_build_info',
                    'model building information',
                    'method',
                    'single',
                    'Information about the model building')

        if hasattr(model, 'feature_importances'):
            self.conveyor.addVal(
                model.feature_importances,
                'feature_importances',
                'feature importances',
                'method',
                'vars',
                'Information about the relative importance of the model variables')

        if hasattr(model, 'feature_importances_method'):
            self.conveyor.addVal(
                model.feature_importances_method,
                'feature_importances_method',
                'feature importances_method',
                'method',
                'single',
                'Method used to compute the relative importance of the model variables')

        # validate model
        if self.param.getVal('input_type') == 'model_ensemble':
            validation_method = 'ensemble validation'
        else:
            validation_method = self.param.getVal("ModelValidationCV")
        LOG.info(f'Validating the model using method: {validation_method}')
        success, model_validation_results = model.validate()
        if not success:
            self.conveyor.setError(model_validation_results)
            return

        # model_validation_results is a dictionary which contains model_validation_info and 
        # (optionally) Y_adj and Y_pred, depending on the model type    
        

        self.conveyor.addVal(
            model_validation_results['quality'],
            'model_valid_info',
            'model validation information',
            'method',
            'single',
            'Information about the model validation')

        # non-conformal qualitative and quantitative models
        if 'Y_adj' in model_validation_results:
            self.conveyor.addVal(
                model_validation_results['Y_adj'],
                'Y_adj',
                'Y fitted',
                'result',
                'objs',
                'Y values of the training series fitted by the model')
        
        if 'Y_pred' in model_validation_results:
            self.conveyor.addVal(
                model_validation_results['Y_pred'],
                'Y_pred',
                'Y predicted',
                'result',
                'objs',
                'Y values of the training series predicted by the model')

        if 'Conformal_prediction_ranges' in model_validation_results:
            self.conveyor.addVal(
                model_validation_results['Conformal_prediction_ranges'],
                'Conformal_prediction_ranges',
                'Conformal prediction ranges',
                'method',
                'objs',
                'Interval for the cross-validated predictions')

        if 'Conformal_prediction_ranges_fitting' in model_validation_results:
            self.conveyor.addVal(
                model_validation_results['Conformal_prediction_ranges_fitting'],
                'Conformal_prediction_ranges_fitting',
                'Conformal prediction ranges fitting',
                'method',
                'objs',
                'Interval for the predictions in fitting')             

        # conformal qualitative models produce a list of tuples, indicating
        # if the object is predicted to belong to class 0 and 1
        if 'classes' in model_validation_results:
            for i in range(len(model_validation_results['classes'][0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i)
                class_list = model_validation_results['classes'][:, i].tolist()
                self.conveyor.addVal( class_list, 
                                class_key, class_label,
                                'result', 'objs', 
                                'Conformal class assignment',
                                'main')

        # conformal quantitataive models produce a list of tuples, indicating
        # the minumum and maximum value

        dimRed = self.param.getVal("dimensionality_reduction")
        if dimRed is None:
            nobj, nvarx = np.shape(self.X)
            if nvarx > 300:
                dimRed = 't-SNE'
            else:
                dimRed = 'PCA'

        if dimRed == 'PCA':
            generatePCASpace(self.X, self.param, self.conveyor)
        elif dimRed == 't-SNE':
            generateManifoldSpace(self.X, self.param, self.conveyor)

        # TODO: compute AD (when applicable)

        if self.param.getVal('confidential'):
            confidential_model = os.path.join (self.param.getVal('model_path'), 'confidential_model.yaml')
            conf_validation = {}
            for item in model_validation_results['quality']:
                conf_validation[item[0]]=float(item[2])
            with open(confidential_model, 'a') as f:
                yaml.dump (conf_validation, f)

        LOG.info('Model finished successfully')

        # save model
        model.save_model()

        return

    def run(self):
        '''
        Builds the model using the appropriate toolkit (internal or custom).
        '''

        toolkit = self.param.getVal('modelingToolkit')

        if toolkit == 'internal':
            LOG.info('Using internal machine learning toolkit')
            self.run_internal()

        elif toolkit == 'custom':
            LOG.info('Unsing custom machine learning toolkit')
            self.run_custom()
        else:
            LOG.error("Modeling toolkit is not yet supported")
            self.conveyor.setError( 'modeling Toolkit ' + \
                toolkit+' is not supported yet')

        return 
