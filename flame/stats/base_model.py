#! -*- coding: utf-8 -*-

# Description    Flame Parent Model Class
##
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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

import pickle
import numpy as np
import os
import time
import copy
import gc
import warnings

from flame.stats.feature_selection import *
from flame.stats.imbalance import *  
from flame.stats.crossval import getCrossVal

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, matthews_corrcoef as mcc
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import MarginErrFunc
from nonconformist.nc import AbsErrorErrFunc, SignErrorErrFunc, RegressorNormalizer
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.nc import RegressorNc
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler

from flame.util import utils, get_logger

LOG = get_logger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

class BaseEstimator:
    """
    Estimator parent class, contains all attributes methods shared
     by different algorithms.Particular implementation of these 
     methods are overwritten by child classes      
        
        Attributes
        ----------

        parameters : dict
            parameter values
        X_original : numpy.darray
            original X matrix
        Y_original : numpy.darray
            original X matrix
        variable_mask: numpy.darray
            variable mask from feature selection
        X : numpy.darray
            current X matrix
        Y : numpy.darray
            current Y vector/matrix
        scaler: sklearn scaler
            scaler object to scale prediction 
            instances
        nobj : int
            number of objects
        nvarx : int
            number of variables
        
        
        Methods
        -------

        build(X)
            Instance the estimator optimizing it
            if tune=true.
        CF_quantitative_validation(self)
            Performs conformal quantitative validation
        CF_qualitative_validation(self)
            Performs conformal qualitative validation
        quantitativeValidation(self)
            Performs quantitative validation
        qualitativeValidation
            Performs qualitative validation
        validate(self)
            Checks type of validation and calls it
        optimize(self, X, Y, estimator, tune_parameters)
            Performs GridSearchCV to optimize estimator
            hyperparameters
        regularProject(self, Xb, results)
            Returns prediction/s for unknown instance/s
        conformalProject(self, Xb, results)
            Returns conformal prediction/s for unknown instance/s
        project(self, Xb, results)
            Checks type of projection and calls it

    """
    def __init__(self, X, Y, parameters, conveyor=None):
        """Initializes the estimator.
        Actions
        -------
            - Attribute assignment
        """

        self.param = parameters
        self.conveyor = None
        if conveyor != None:
            self.conveyor = conveyor

        if X is None:
            return

        self.X = X
        self.Y = Y
        self.nobj, self.nvarx = np.shape(X)
        self.feature_importances = None

        self.cross_jobs = -1
        if utils.isSingleThread():
            self.cross_jobs = 1
            
        LOG.info(f'Num jobs set to {self.cross_jobs}')

        # Get cross-validator
        # Consider to include a Random Seed for cross-validator
        if self.param.getVal('ModelValidationCV'):
            try:
                self.cv = getCrossVal(
                                self.param.getVal('ModelValidationCV'),
                                46,
                                self.param.getVal('ModelValidationN'),
                                self.param.getVal('ModelValidationP'))
                LOG.debug('Cross-validator retrieved')
                LOG.debug(f'cv is: {self.cv}')
            except Exception as e:
                LOG.error(f'Error retrieving cross-validator with exception: {e}')
                self.conveyor.setError(f'Error retrieving cross-validator with exception: {e}')

    def quality_cualitative (self, y, yp):
        ''' convenience function to compute quality indexes for 
        cualitative models '''
        TN, FP, FN, TP = confusion_matrix(y, yp, labels=[0, 1]).ravel()

        MCC = mcc(y, yp)

        if np.isnan(MCC):
            MCC = 0.000

        if (TP+FN) > 0:
            sensitivity = (TP / float(TP + FN))
        else:
            sensitivity = 0.0

        if (TN+FP) > 0:
            specificity = (TN / float(TN + FP))
        else:
            specificity = 0.0

        if (FP + FN + TN + TP) > 0:
            accuracy = (float(TN + TP) /  float(FP + FN + TN + TP))
        else:
            accuracy = 0.0

        total = TP + TN + FP + FN

        return {'TP': float(TP), 'TN': float(TN), 'FP': float(FP), 'FN': float(FN), 'accuracy': float(accuracy), 'total': float(total), 
            'MCC': float(MCC), 'sensitivity': float(sensitivity), 'specificity': float(specificity) }

    def quality_quantitative (self, y, yp):
        ''' convenience function to compute quality indexes for 
        quantitative models '''

        MSE = mean_squared_error(y, yp)

        R2 = r2_score (y, yp)

        return {'R2': R2, 'MSE': MSE, 'SDEP': np.sqrt(MSE)}


    def external_validation(self):
        ''' when experimental values are available for the predicted compounds,
        run external validation '''

        ext_val_results = []
        
        # Ye are the y values present in the input file
        Ye = np.asarray(self.conveyor.getVal("ymatrix"))

        # For qualitative models, make sure the Y is qualitative as well
        if not self.param.getVal("quantitative"):
            qy, message = utils.qualitative_Y(Ye)
            if not qy:
                self.conveyor.setWarning(f'No qualitative activity suitable for external validation "{message}". Skipping.')
                LOG.warning(f'No qualitative activity suitable for external validation "{message}". Skipping.')
                return

        # there are four variants of external validation, depending if the method
        # if conformal or non-conformal and the model is qualitative and quantitative

        if not self.param.getVal("conformal"):

            # non-conformal 
            if not self.param.getVal("quantitative"):
                
                # non-conformal & qualitative
                Yp = np.asarray(self.conveyor.getVal("values"))

                if Ye.size == 0:
                    LOG.error ("Experimental activity vector is empty")
                    return
                if Yp.size == 0:
                    LOG.error ("Predicted activity vector is empty")
                    return

                quality = self.quality_cualitative(Ye, Yp)

                ext_val_results.append(('TP','True positives in external-validation', quality['TP']))
                ext_val_results.append(('TN','True negatives in external-validation', quality['TN']))
                ext_val_results.append(('FP','False positives in external-validation', quality['FP']))
                ext_val_results.append(('FN','False negatives in external-validation', quality['FN']))
                ext_val_results.append(('Sensitivity', 'Sensitivity in external-validation', quality['sensitivity']))
                ext_val_results.append(('Specificity', 'Specificity in external-validation', quality['specificity']))
                ext_val_results.append(('MCC', 'Mattews Correlation Coefficient in external-validation', quality['MCC']))

            else:

                # non-conformal & quantitative
                Yp = np.asarray(self.conveyor.getVal("values"))

                if Ye.size == 0:
                    LOG.error ("Experimental activity vector is empty")
                    return

                if Yp.size == 0:
                    LOG.error ("Predicted activity vector is empty")
                    return

                quality = self.quality_quantitative(Ye, Yp)

                ext_val_results.append(('scoringP', 'Mean Squared Error of the Predictions', quality['MSE']))
                ext_val_results.append(('Q2', 'Determination coefficient in cross-validation', quality['R2']))
                ext_val_results.append(('SDEP', 'Standard Deviation Error of the Predictions', quality['SDEP']))

        else:
            # conformal external validation

            if not self.param.getVal("quantitative"):
                
                # conformal & qualitative
                Yp = np.concatenate(
                    (np.asarray(self.conveyor.getVal('c0')).reshape(-1, 1), 
                     np.asarray(self.conveyor.getVal('c1')).reshape(-1, 1)), 
                    axis=1)

                if Ye.size == 0:
                    LOG.error ("Experimental activity vector is empty")
                    return
                if Yp.size == 0:
                    LOG.error ("Predicted activity vector is empty")
                    return

                not_predicted = 0

                Ye1 = []
                Yp1 = []
                for i in range(len(Ye)):
                    real = float(Ye[i])
                    predicted = Yp[i]
                    if predicted[0] != predicted[1]:
                        Ye1.append(real)
                        if predicted[0]:
                            Yp1.append(0)
                        else:
                            Yp1.append(1)
                    else:
                        not_predicted += 1

                coverage = float((len(Yp) - not_predicted) / len(Yp))
                quality = self.quality_cualitative(Ye1, Yp1)

                ext_val_results.append(('TP','True positives in external-validation', quality['TP']))
                ext_val_results.append(('TN','True negatives in external-validation', quality['TN']))
                ext_val_results.append(('FP','False positives in external-validation', quality ['FP']))
                ext_val_results.append(('FN', 'False negatives in external-validation', quality['FN']))
                ext_val_results.append(('Sensitivity', 'Sensitivity in external-validation', quality['sensitivity']))
                ext_val_results.append(('Specificity', 'Specificity in external-validation', quality['specificity']))
                ext_val_results.append(('MCC', 'Mattews Correlation Coefficient in external-validation', quality['MCC']))
                ext_val_results.append(('Conformal_coverage', 'Conformal coverage in external-validation', coverage))
                ext_val_results.append(('Conformal_accuracy', 'Conformal accuracy in external-validation', quality['accuracy']))

            else:
                # conformal & quantitative

                Yp_lower = np.asarray(self.conveyor.getVal('lower_limit'))
                Yp_upper = np.asarray(self.conveyor.getVal('upper_limit'))

                mean_interval = np.mean(np.abs(Yp_lower) - np.abs(Yp_upper))
                interval_means = (Yp_lower + Yp_upper) / 2

                inside_interval = (Yp_lower.reshape(-1, 1) < Ye) & (Yp_upper.reshape(-1, 1) > Ye)
                accuracy = len(inside_interval)/len(Ye)

                ext_val_results.append(('Conformal_mean_interval', 'Conformal mean interval', mean_interval))
                ext_val_results.append(('Conformal_accuracy', 'Conformal accuracy',  accuracy))
               
                # Compute classic Cross-validation quality metrics using inteval mean
                try:
                    quality = self.quality_quantitative(Ye, interval_means)

                    ext_val_results.append(('scoringP', 'Mean Squared Error of the Predictions', quality['MSE']))
                    ext_val_results.append(('Q2', 'Determination coefficient in cross-validation', quality['R2']))
                    ext_val_results.append(('SDEP', 'Standard Deviation Error of the Predictions', quality['SDEP']))

                except Exception as e:
                    LOG.error(f'Error in external validation with exception {e}')
                    return

        self.conveyor.addVal( ext_val_results,
                                'external-validation',
                                'external validation',
                                'method',
                                'single',
                                'External validation results')


    # Validation methods section
    def CF_quantitative_validation(self):
        ''' Performs internal  validation for conformal quantitative models '''

        # Make a copy of original matrices.
        X = self.X.copy()
        Y = self.Y.copy()

        # generate recalculate interval, values and mean range 
        Y_rec_in = self.estimator.predict(X, significance = 1.0 - self.param.getVal('conformalConfidence'))
        if Y_rec_in is None:
            return False, 'prediction error'

        Y_rec = np.mean(Y_rec_in, axis=1)
        interval_mean_rec = np.mean(np.abs((Y_rec_in[:, 0]) - (Y_rec_in[:, 1])))

        # Copy Y vector to use it as template to assign predictions
        Y_pred_in = copy.copy(Y).tolist()
        try:
            prob = 1.0 - self.param.getVal('conformalConfidence')
            for train_index, test_index in self.cv.split(X):
                # Generate training and test sets
                X_train = X[train_index]
                X_test = X[test_index]
                Y_train = Y[train_index]
                
                # Create the aggregated conformal regressor.
                conformal_pred = copy.copy(self.estimator)

                # Fit conformal regressor to the data
                conformal_pred.fit(X_train, Y_train)

                # Perform prediction on test set
                prediction = conformal_pred.predict(X_test, prob)

                # Assign the prediction its original index
                for index, el in enumerate(test_index):
                    Y_pred_in[el] = prediction[index]

        except Exception as e:
            return False, f'Quantitative conformal validation failed with exception: {e}'
        
        # Convert Y_pred to a numpy array
        Y_pred_in = np.asarray(Y_pred_in)

        # Add the n validation interval means
        Y_pred = np.mean(Y_pred_in, axis=1)
        interval_mean_pred = np.mean(np.abs((Y_pred_in[:, 0]) - (Y_pred_in[:, 1])))

        # Get boolean mask of instances within the applicability domain.
        inside_interval_pred = ((Y_pred_in[:, 0].reshape(1, -1) < Y) & 
                                (Y_pred_in[:, 1].reshape(1, -1) > Y)).reshape(1, -1)
        inside_interval_rec =  ((Y_rec_in[:, 0].reshape(1, -1) < Y) & 
                                (Y_rec_in[:, 1].reshape(1, -1) > Y)).reshape(1, -1)
       
        # Compute the accuracy (number of instances within the AD).
        accuracy_pred = np.sum(inside_interval_pred/len(Y))
        accuracy_rec  = np.sum(inside_interval_rec /len(Y))

        #Add quality metrics to results.
        info = []
        info.append(('Conformal_mean_interval_f','Conformal mean interval in fitting', interval_mean_rec))
        info.append(('Conformal_mean_interval', 'Conformal mean interval', interval_mean_pred))
        info.append(('Conformal_accuracy_f', 'Conformal accuracy in fitting', accuracy_rec))
        info.append(('Conformal_accuracy', 'Conformal accuracy', accuracy_pred))

        
        # Compute goodness of the fit statistics using recalculated
        # predictions
        try:
            quality = self.quality_quantitative(Y, Y_rec)

            info.append(('scoringR', 'Mean Squared Error of the Calculations', quality['MSE']))
            info.append(('R2', 'Determination coefficient', quality['R2']))
            info.append(('SDEC', 'Standard Deviation Error of the Calculations', quality['SDEP']))

            LOG.debug(f"R2: {quality['R2']:.4f}")
        except Exception as e:
            return False, f'Error computing goodness of fit with exception: {e}'

        # Compute classic Cross-validation quality metrics using inteval mean
        try:
            quality = self.quality_quantitative(Y, Y_pred)

            info.append(('scoringP', 'Mean Squared Error of the Prediction', quality['MSE']))
            info.append(('Q2', 'Determination coefficient in cross-validation',quality['R2']))
            info.append(('SDEP', 'Standard Deviation Error of the Predictions',quality['SDEP']))

            LOG.debug(f"Q2: {quality['R2']:.4f}")

        except Exception as e:
            return False, f'Error cross-validating the estimator with exception {e}'
              
        results = {}
        results ['quality'] = info
        results ['Y_adj'] = Y_rec
        results ['Y_pred'] = Y_pred
        results ['Conformal_prediction_ranges'] = Y_pred_in
        results ['Conformal_prediction_ranges_fitting'] = Y_rec_in
        return True, results

    def CF_qualitative_validation(self):
        ''' performs validation for conformal qualitative models '''

        # Make a copy of original matrices.
        X = self.X.copy()
        Y = self.Y.copy()

        fit = self.estimator.predict(X, significance = 1.0 - self.param.getVal('conformalConfidence'))
        if fit is None:
            return False, 'prediction error'

        info = []

        # Copy Y vector to use it as template to assign predictions
        Y_pred = copy.copy(Y).tolist()
        
        # try:

        prob = 1.0 - self.param.getVal('conformalConfidence')
        for train_index, test_index in self.cv.split(X):
            # Generate training and test sets
            X_train = X[train_index]
            X_test  = X[test_index]
            Y_train = Y[train_index]

            conformal_pred = copy.copy(self.estimator)
            
            # Fit the conformal classifier to the data
            conformal_pred.fit(X_train, Y_train)
            
            # Perform prediction on test set
            prediction = conformal_pred.predict(X_test, prob)
            if prediction is None:
                return False, 'prediction error'

            # Assign the prediction the correct index. 
            for index, el in enumerate(test_index):
                Y_pred[el] = prediction[index]
        
        not_predicted = 0
        Ye1 = []
        Yp1 = []
        for i in range(len(Y_pred)):
            real = float(Y[i])
            predicted = Y_pred[i]
            if predicted[0] != predicted[1]:
                Ye1.append(real)
                if predicted[0]:
                    Yp1.append(0)
                else:
                    Yp1.append(1)
            else:
                not_predicted += 1

        quality = self.quality_cualitative(Ye1, Yp1)

        not_predicted_f = 0
        Ye1 = []
        Yp1 = []
        for i in range(len(fit)):
            real = float(Y[i])
            predicted = fit[i]
            if predicted[0] != predicted[1]:
                Ye1.append(real)
                if predicted[0]:
                    Yp1.append(0)
                else:
                    Yp1.append(1)
            else:
                not_predicted_f += 1

        quality_f = self.quality_cualitative(Ye1, Yp1)

        # except Exception as e:
        #     return False, f'Qualitative conformal validation failed with exception: {e}'

        info.append(('TP_f', 'True positives in fitting', quality_f['TP']))
        info.append(('TN_f', 'True negatives in fitting', quality_f['TN']))
        info.append(('FP_f', 'False positives in fitting', quality_f['FP']))
        info.append(('FN_f', 'False negatives in fitting', quality_f['FN']))

        info.append(('TP', 'True positives in cross-validation', quality['TP']))
        info.append(('TN', 'True negatives in cross-validation', quality['TN']))
        info.append(('FP', 'False positives in cross-validation', quality['FP']))
        info.append(('FN', 'False negatives in cross-validation', quality['FN']))

        info.append(('Sensitivity_f', 'Sensitivity in fitting', quality_f['sensitivity']))
        info.append(('Specificity_f', 'Specificity in fitting', quality_f['specificity']))
        info.append(('MCC_f', 'Matthews Correlation Coefficient in fitting', quality_f['MCC']))

        info.append(('Sensitivity', 'Sensitivity in cross-validation', quality['sensitivity']))
        info.append(('Specificity', 'Specificity in cross-validation', quality['specificity']))
        info.append(('MCC', 'Matthews Correlation Coefficient in cross-validation', quality['MCC']))

        coverage_f = quality_f['total'] / float (not_predicted_f + quality_f['total'])
        coverage = quality['total'] / float (not_predicted + quality['total'])

        info.append(('Conformal_coverage_f', 'Conformal coverage in fitting', coverage_f))
        info.append(('Conformal_accuracy_f', 'Conformal accuracy in fitting', quality_f['accuracy']))                                                    
        info.append(('Conformal_coverage', 'Conformal coverage in cross-validation', coverage))
        info.append(('Conformal_accuracy', 'Conformal accuracy in cross-validation', quality['accuracy']))


        # convert to plain list to avoid problems with JSON serialization of results
        adj_list = fit.tolist()
        pred_list = [x.tolist() for x in Y_pred]

        results = {}
        results ['quality'] = info
        results ['Y_adj'] = adj_list
        results ['Y_pred'] = pred_list
        results ['classes'] = prediction

        return True, results

    def quantitativeValidation(self):
        ''' performs validation for quantitative models '''

        # Make a copy of the original matrices
        X = self.X.copy()
        Y = self.Y.copy()

        # Get predicted Y
        Yp = self.estimator.predict(X)
        if Yp is None:
            return False, 'prediction error'

        info = []
        # Compute Goodness of the fit metric (adjusted Y)
        try:
            quality = self.quality_quantitative(Y, Yp)

            info.append(('scoringR', 'Mean Squared Error of the Calculations', quality['MSE']))
            info.append(('R2', 'Determination coefficient', quality['R2']))
            info.append(('SDEC', 'Standard Deviation Error of the Calculations', quality['SDEP']))

            LOG.debug(f"R2: {quality['R2']:.4f}")
        except Exception as e:
            return False, f'Error computing goodness of fit with exception: {e}'

        # Compute Cross-validation quality metrics
        try:
            # Get predicted Y
            y_pred = cross_val_predict(self.estimator, X, Y, cv=self.cv, n_jobs=self.cross_jobs)

            quality = self.quality_quantitative(Y, y_pred)

            info.append(('scoringP', 'Mean Squared Error of the Predictions', quality['MSE']))
            info.append(('Q2', 'Determination coefficient in cross-validation', quality['R2']))
            info.append(('SDEP', 'Standard Deviation Error of the Predictions', quality['SDEP']))

            LOG.debug(f"Q2: {quality['R2']:.4f}")
        except Exception as e:
            return False, f'Error cross-validating the estimator with exception: {e}'
              
        results = {}
        results ['quality'] = info
        results ['Y_adj'] = Yp
        results ['Y_pred'] = y_pred
        return True, results

    def qualitativeValidation(self):
        ''' performs validation for qualitative models '''

        # Make a copy of the original matrices
        X = self.X.copy()
        Y = self.Y.copy()

        # Get predicted classes.
        Yp = self.estimator.predict(X)
        if Yp is None:
            return False, 'error in qualitative validation'

        if len(Yp) != len(Y):
            return False, 'Lenght of experimental and predicted Y do not match'

        info = []

        # Get confusion matrix for predicted Y
        try:
            quality_f = self.quality_cualitative (Y, Yp)

            info.append(('TP_f', 'True positives in fitting', quality_f['TP']))
            info.append(('TN_f', 'True negatives in fitting', quality_f['TN']))
            info.append(('FP_f', 'False positives in fitting', quality_f['FP']))
            info.append(('FN_f', 'False negatives in fitting', quality_f['FN']))
            info.append(('Sensitivity_f', 'Sensitivity in fitting', quality_f['sensitivity']))
            info.append(('Specificity_f', 'Specificity in fitting', quality_f['specificity']))
            info.append(('MCC_f', 'Matthews Correlation Coefficient in fitting', quality_f['MCC']))
   
            LOG.debug('Computed quality indicators for fitting')
        except Exception as e:
            return False, f'Error computing model quality in fitting: {e}'

        # Get cross-validated Y 
        try:
            y_pred = cross_val_predict(self.estimator, X, Y, cv=self.cv, n_jobs=self.cross_jobs)
        except Exception as e:
            return False, f'Cross-validation failed with exception: {e}'

        # Get confusion matrix
        try:
            quality = self.quality_cualitative (Y, y_pred)

            info.append(('TP', 'True positives in cross-validation', quality['TP']))
            info.append(('TN', 'True negatives in cross-validation', quality['TN']))
            info.append(('FP', 'False positives in cross-validation', quality['FP']))
            info.append(('FN', 'False negatives in cross-validation', quality['FN']))
            info.append(('Sensitivity', 'Sensitivity in cross-validation', quality['sensitivity']))
            info.append(('Specificity', 'Specificity in cross-validation', quality['specificity']))
            info.append(('MCC', 'Matthews Correlation Coefficient in cross-validation', quality['MCC']))

            LOG.debug('Computed quality indicators for prediction')
        except Exception as e:
            return False, f'Error computing model quality in prediction with exception: {e}'

        LOG.debug(f'Qualitative crossvalidation performed')

        results = {}
        results ['quality'] = info
        results ['Y_adj'] = Yp
        results ['Y_pred'] = y_pred
        return True, results

    def validate(self):
        ''' Validates the model and computes suitable
         model quality scoring values'''
        # Check estimator integrity
        if self.X is None or self.estimator is None:
            return False, 'no estimator'

        if not self.param.getVal('conformal'):
            if self.param.getVal('quantitative'):
                success, results = self.quantitativeValidation()
            else:
                success, results = self.qualitativeValidation()
        else:
            if self.param.getVal('quantitative'):
                success, results = self.CF_quantitative_validation()
            else:
                success, results = self.CF_qualitative_validation()

        return success, results

    def optimize(self, X, Y, estimator, tune_parameters):
        ''' optimizes a model using a grid search over a 
        range of values for diverse parameters'''

        # the default value is represented as 'default' in the YAML parameter file to
        # avoid problems with empty values and must be replaced by None here       
        
        for key, value in tune_parameters.items():
            if 'default' in value:
                tune_parameters[key] = [None if i == 'default' else i for i in value ]

        LOG.info('Computing best hyperparameter values')
        
        metric = None
        # Select the metric according to the type of model
        if self.param.getVal('quantitative'):
            metric = self.param.getVal('tune_metric_quantitative')
            if metric is None:
                metric = 'r2' 
        else:
            metric = self.param.getVal('tune_metric_qualitative')
            if metric is None:
                metric = 'mcc'

            if metric == 'mcc':
                metric = make_scorer(mcc)

        cv_fold = self.param.getVal('tune_cv_fold')
        if cv_fold is None:
            cv_fold = 5

        LOG.info(f'Scorer: {metric} using kfold with: {cv_fold} folds...')

        # tune_parameters = [tune_parameters]
        # Count computation time
        LOG.debug("Hyperparameter optimization ")
        start = time.time()

        try:
            # print (tune_parameters)
            tclf = GridSearchCV(estimator, tune_parameters, 
                                scoring=metric, cv=KFold(cv_fold), n_jobs=self.cross_jobs)
            tclf.fit(X, Y)

            means = tclf.cv_results_["mean_test_score"]
            stds = tclf.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, tclf.cv_results_["params"]):
                LOG.info("       %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            self.estimator = copy.copy(tclf.best_estimator_)
            if hasattr(self.estimator, 'feature_importances_'):
                self.feature_importances = self.estimator.feature_importances_

        except Exception as e:
            LOG.error(f'Error optimizing hyperparameters with'
            f'exception {e}')
            raise e

        end = time.time()
        LOG.info(f'Best parameters: {tclf.best_params_}')
        # LOG.info(f'Best score: {tclf.best_score_}')

        # myscore = cross_val_score(self.estimator, X, Y, scoring=metric, cv=self.cv)
        # LOG.info (f'mean of scores: {np.mean(myscore)}')
        # y_pred = cross_val_predict(self.estimator, X, Y, cv=self.cv)
        # LOG.info (f'validation: {r2_score(Y, y_pred)}')
        
        LOG.debug (f'Best estimator found in {(end-start):.2f} seconds')

        # Remove garbage in memory
        del(tclf)
        gc.collect()

    # Projection section

    def regularProject(self, Xb):
        ''' projects a collection of query objects in a regular model,
         for obtaining predictions '''

        Yp = self.estimator.predict(Xb)
        if Yp is None:
            return False, 'prediction error'

        # if conveyor contains experimental values for any of the objects replace the
        # predictions with the experimental results
        exp = self.conveyor.getVal('experim')
        if exp is not None:
            if len(exp) == len(Yp):
                for i in range (len(Yp)):
                    if not np.isnan(exp[i]):
                        # print (exp[i], Yp[i])
                        Yp[i] = exp[i]
                    else:
                    # if exp is nan, substitute it with a number which can be recognized
                    # to facilitate handling and do not replace Yp
                        exp[i]= float ('-99999')

        self.conveyor.addVal(Yp, 'values', 'Prediction',
                        'result', 'objs',
                        'Results of the prediction', 'main')


    def conformalBuild (self, X, Y):
        ''' uses self.estimator_temp to build a conformal estimator with the parameters defined in
        conformal_settings, fits X and Y and copy the result to self.estimator'''

        # Read conformal settings, defining the 
        # - aggregated (True/False)
        # - sampler (Boostrap, Random, Cross) 
        # - aggregation function (mean, median, min, max)
        # - normalizer (KNN, Underlying) (only for regressor)
        # 
        # However, the scorers were hardcoded to AbsErr for regressors and MarginErr for classifiers

        conformal_settings = self.param.getDict('conformal_settings')

        samplers = {"BootstrapSampler" : BootstrapSampler(), 
                    "RandomSubSampler" : RandomSubSampler(),
                    "CrossSampler"     : CrossSampler()}

        if 'aggregated' in conformal_settings:
            isACP = conformal_settings['aggregated']
        else:
            isACP = True

        if 'ACP_sampler' in conformal_settings:
            sampler_id = conformal_settings['ACP_sampler']
        else:
            sampler_id = 'BootstrapSampler'

        sampler = samplers[sampler_id]

        if 'conformal_predictors' in conformal_settings:
            n_predictors = conformal_settings['conformal_predictors']
        else:
            n_predictors = 10

        if 'aggregation_function' in conformal_settings:
            aggregation_f = conformal_settings['aggregation_function']
        else:
            aggregation_f = "median"
            
        if 'KNN_NN' in conformal_settings:
            n_neighbors=conformal_settings['KNN_NN']
        else:
            n_neighbors=15

        if 'normalizing_model' in conformal_settings:
            normalizing_id = conformal_settings['normalizing_model']
        else:
            normalizing_id = None
            
            # some old versions of Flame used 'error_model' terminology 
            if 'error_model' in conformal_settings:
                normalizing_id = conformal_settings['error_model']

        normalizers = {'KNN' : RegressorAdapter(
                                KNeighborsRegressor(
                                    n_neighbors=n_neighbors)),
                       'Underlying' : RegressorAdapter(self.estimator_temp),
                       'None' : None}

        if isACP :
            #########################################################################
            ###   ACP
            #########################################################################
            try:
                # Conformal regressor
                if self.param.getVal('quantitative'):
                    LOG.info("Building Quantitative Aggregated Conformal model")
                    LOG.info(f"Using {sampler_id} sampler, " \
                            +f"{aggregation_f} aggregator " \
                            +f"and {normalizing_id} normalizer")
                    LOG.info(f"Aggregation of {n_predictors} models")

                    # Normalizing model (lambda)
                    normalizing_model = normalizers[normalizing_id]

                    if normalizing_model is not None:
                        normalizer = RegressorNormalizer( self.estimator_temp,
                                                        normalizing_model,
                                                        AbsErrorErrFunc())
                    else:
                        normalizer = None

                    self.estimator = AggregatedCp(
                                        IcpRegressor(
                                            RegressorNc(
                                                RegressorAdapter(self.estimator_temp), 
                                                    AbsErrorErrFunc(), 
                                                normalizer
                                            )
                                        ),
                                        sampler=sampler, 
                                        aggregation_func=aggregation_f,
                                        n_models=n_predictors)

                # Conformal classifier
                else:
                    LOG.info("Building Qualitative Aggregated Conformal model")
                    LOG.info(f"Using {sampler_id} sampler, " \
                            +f"{aggregation_f} aggregator ")
                    LOG.info(f"Aggregation of {n_predictors} models" )

                    self.estimator = AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(self.estimator_temp),
                                                    MarginErrFunc()
                                            )
                                        ),
                                        sampler=sampler, 
                                        aggregation_func=aggregation_f,
                                        n_models=n_predictors)

                # Fit estimator to the data
                self.estimator.fit(X, Y)

                first = True
                features = None
                for p in self.estimator.predictors:
                    inner_estimator = p.nc_function.model.model
                    if hasattr(inner_estimator, 'feature_importances_'):    
                        fi = p.nc_function.model.model.feature_importances_
                        if first:
                            features = np.array(fi)
                            first = False
                        else:
                            features = np.vstack((features,fi))
                
                if features is not None:
                    self.feature_importances = np.mean (features, 0)

            except Exception as e:
                return False, f'Exception building conformal estimator with exception {e}'

        else :
            #########################################################################
            ###   ICP
            #########################################################################
            try:
                # Conformal regressor
                if self.param.getVal('quantitative'):
                    LOG.info("Building Quantitative Inductive Conformal model")
                    LOG.info(f"Using {normalizing_id} normalizer")

                    # Normalizing model (lambda)
                    normalizing_model = normalizers[normalizing_id]

                    if normalizing_model is not None:
                        normalizer = RegressorNormalizer( self.estimator_temp,
                                                        normalizing_model,
                                                        AbsErrorErrFunc())
                    else:
                        normalizer = None

                    self.estimator = IcpRegressor(
                                        RegressorNc(
                                            RegressorAdapter(self.estimator_temp), 
                                                AbsErrorErrFunc(), 
                                            normalizer
                                        )
                                    )

                # Conformal classifier
                else:
                    LOG.info("Building Qualitative Inductive Conformal model")

                    self.estimator = IcpClassifier(
                                        ClassifierNc(
                                            ClassifierAdapter(self.estimator_temp),
                                                MarginErrFunc()
                                        )
                                     )


                # Divide the data into 80% training set and 20% calibration set 
                np.random.seed(46)
                nobj, nvarx = np.shape(X)
                idx = np.random.permutation(nobj)
                train_size = int(np.floor(nobj*0.8))
                idx_train, idx_cal = idx[:train_size], idx[train_size:nobj]

                # Fit estimator to the data
                self.estimator.fit(X[idx_train, :], Y[idx_train])

                # Calibrate the data
                self.estimator.calibrate (X[idx_cal, :], Y[idx_cal])

                inner_estimator = self.estimator.nc_function.model.model
                if hasattr(inner_estimator, 'feature_importances_'):    
                    self.feature_importances = inner_estimator.feature_importances_
                    
            except Exception as e:
                return False, f'Exception building conformal estimator with exception {e}'

        return True, 'OK'


    def conformalProject(self, Xb):
        ''' projects a collection of query objects in a conformal model,
         for obtaining predictions '''

        if not 'nonconformist' in str(type(self.estimator)):
            self.conveyor.setError('Inconsistence error: non-conformal classifier found. Rebuild the model')
            return

        prediction = self.estimator.predict(Xb, significance = 1.0 - self.param.getVal('conformalConfidence'))

        if prediction is None:
            return False, 'prediction error'

        if self.param.getVal('quantitative'):
            Yp = np.mean(prediction, axis=1)
            
            lower_limit = prediction[:, 0]
            upper_limit = prediction[:, 1]

            # if conveyor contains experimental values for any of the objects replace the
            # predictions with the experimental results
            exp = self.conveyor.getVal('experim')
            if exp is not None:
                if len(exp) == len(Yp):
                    for i in range (len(Yp)):
                        if not np.isnan(exp[i]):
                            # print (exp[i], Yp[i])
                            Yp[i] = exp[i]
                            lower_limit[i] = exp[i]
                            upper_limit[i] = exp[i]
                        # if exp is nan, substitute it with a number which can be recognized
                        # to facilitate handling and do not replace Yp
                        else:
                            exp[i]= float ('-99999')

            self.conveyor.addVal(Yp, 'values', 'Prediction',
                    'result', 'objs',
                    'Results of the prediction', 'main')

            self.conveyor.addVal(lower_limit, 'lower_limit',
                             'Lower limit', 'confidence', 'objs',
                              'Lower limit of the conformal prediction')

            self.conveyor.addVal(upper_limit, 'upper_limit',
                             'Upper limit', 'confidence', 'objs',
                              'Upper limit of the conformal prediction')
            
            self.conveyor.addVal(self.param.getVal('conformalConfidence'), 'confidence',
                             'Conformal confidence', 'confidence', 'single',
                              'Confidence level used in the conformal method')
        else:
            # Returns a dictionary with class predictions
            # c0 / c1 

            # Repeat prediction without confidence to get probabilities
            # of class0 and class1 for each compound
            pvalues = self.estimator.predict(Xb)

            for i in range(len(prediction[0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i)
                class_list = prediction[:, i].tolist()
                self.conveyor.addVal(class_list, 
                                class_key, 
                                class_label,
                                'confidence', 'objs', 
                                'Conformal class assignment')

                if pvalues is not None:
                    class_key_p = 'p' + str(i)
                    class_label_p = 'p-value ' + str(i)
                    class_list_p = pvalues[:, i].tolist()
                    self.conveyor.addVal(class_list_p, 
                                    class_key_p, 
                                    class_label_p,
                                    'confidence', 'objs', 
                                    'Conformal class assignment')

            # the use of np.zeros defaults to 0 (negative)

            nobj, nvary = np.shape(prediction)

            Yp = np.zeros(nobj, dtype = np.float64)
            for j in range (nobj):
                p0 = prediction[j,0]
                if p0==prediction[j,1]:  # if both are equal results are unconclusive
                    Yp[j]=-1
                elif p0 == 0: # if do not belong to class 0 is must be class 1 (positive)
                    Yp[j]=1

            # if conveyor contains experimental values for any of the objects replace the
            # predictions with the experimental results
            # TODO: this section is incomplete and experimental. The replacement of cualitative
            # variables without checking can produce wrong results
            exp = self.conveyor.getVal('experim')
            if exp is not None:
                if len(exp) == len(Yp):
                    for i in range (len(Yp)):
                        if not np.isnan(exp[i]):
                            # print (exp[i], Yp[i])
                            # Yp is copied without any checking, BEWARE!!! 
                            Yp[i] = exp[i]
                        # if exp is nan, substitute it with a number which can be recognized
                        # to facilitate handling and do not replace Yp
                        else:
                            exp[i]= float ('-99999')

                #TODO: moddify the classes, by getting the classes, overwritting the values and
                # saving again

            self.conveyor.addVal(Yp, 'values', 'Prediction',
                    'result', 'objs',
                    'Results of the prediction', 'main')

    def project(self, Xb):
        ''' Uses the X matrix provided as argument to predict Y'''

        if self.estimator == None:
            self.conveyor.setError('failed to load classifier')
            return
        
        # Apply variable mask to prediction vector/matrix
        # if self.param.getVal("feature_selection"):
        #     Xb = Xb[:, self.variable_mask]
        # Scale prediction vector/matrix
        # if self.param.getVal('modelAutoscaling'):
            # Xb = Xb-self.mux
            # Xb = Xb*self.wgx
            # Xb = self.scaler.transform(Xb)
        # Select the type of projection

        if not self.param.getVal('conformal'):
            self.regularProject(Xb)
        else:
            self.conformalProject(Xb)
    
    def save_model(self):
        ''' This function saves estimator and scaler in a pickle file '''

        # This dictionary contain all the objects which will be needed
        # for prediction

        # Uncoment to inspect estimator contents 
        # print (self.estimator.__dict__)
        # print (self.estimator.estimators_[0].__dict__)

        dict_estimator = {'estimator' : self.estimator,
             'version': 1,
             'libraries': utils.module_versions()}

        model_pkl = os.path.join(self.param.getVal('model_path'),'estimator.pkl')

        with open(model_pkl, 'wb') as handle:
            pickle.dump(dict_estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        LOG.debug('Model saved as:{}'.format(model_pkl))

        # Add estimator parameters to Conveyor
        params = dict()
        if  self.param.getVal('conformal'):
            params = self.estimator_temp.get_params()
        else:
            params = self.estimator.get_params()

        self.conveyor.addVal(params, 'estimator_parameters',
                            'estimator parameters', 'method', 'single',
                            'Hyperparameter values for the algorithm')
        return

    def load_model(self):
        ''' This function loads estimator and scaler in a pickle file '''

        model_pkl = os.path.join(self.param.getVal('model_path'),'estimator.pkl')
        LOG.debug(f'Loading model from pickle file, path: {model_pkl}')
        try:
            with open(model_pkl, "rb") as input_file:
                dict_estimator = pickle.load(input_file)

        except FileNotFoundError:
            return False, f'No valid model estimator found at: {model_pkl}'

        # check if the pickle was created with a compatible version (currently, 1)
        self.version = dict_estimator['version']
        if self.version is not 1:
            return False, 'Incompatible model version'

        # check if the libraries used to build this model are similar to current libraries
        if 'libraries' in dict_estimator:
            success, results = utils.compatible_modules(dict_estimator['libraries'])
            if not success:
                LOG.warning(f"incompatible libraries detected, {results}. Use at your own risk")

        # load the estimator
        self.estimator = dict_estimator['estimator']
        if self.estimator is None:
            return False, 'No valid model estimator found. Try to rebuild the model'
    
        return True, 'model loaded'