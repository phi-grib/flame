#! -*- coding: utf-8 -*-

# Description    Flame documentation class
#
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import yaml
import json
import pickle
import pandas as pd

from flame.util import utils
from flame.conveyor import Conveyor


class Documentation:
    ''' Class storing the information needed to documentate models
        Fields are loaded from a YAML file (documentation.yaml)

        ...

        Attributes
        ----------

        fields : dict
            fields in the documentation
        version : int
            documentation version


        Methods
        -------

        load_results(results_dictionary)
            Accesses to build results and param class to retrieve all
            information needed to document the model.
        save_document()
            Updates YAML file with assigned values.
        create_template()
            creates a excel template with the available information
        create_QMRF(Creates a QMRF document filled with available information)
    '''

    def __init__(self, model, version=0):
        ''' Load the fields from the documentation file'''

        self.model = model
        self.version = version
        self.fields = None
        self.parameters = None
        self.conveyor = None

        # obtain the path and the default name of the model documents
        documentation_file_path = utils.model_path(self.model, self.version)
        documentation_file_name = os.path.join(documentation_file_path,
                                               'documentation.yaml')

        # load the main class dictionary (p) from this yaml file
        if not os.path.isfile(documentation_file_name):
            raise Exception('Documentation file not found')

        try:
            with open(documentation_file_name, 'r') as documentation_file:
                self.fields = yaml.load(documentation_file)
        except Exception as e:
            # LOG.error(f'Error loading documentation file with exception: {e}')
            raise e
        self.load_parameters()
        self.load_results()
        self.assign_parameters()
        self.assign_results()

    def load_parameters(self):
        '''This function takes info from results and
        param file and assigns it to corresponding fields
        in documentation dictionary'''

        # obtain the path and the default name of the parameter_file
        parameters_file_path = utils.model_path(self.model, self.version)
        parameters_file_name = os.path.join(parameters_file_path,
                                            'parameters.yaml')

        if not os.path.isfile(parameters_file_name):
            raise Exception('Parameter file not found')
        try:
            with open(parameters_file_name, 'r') as parameter_file:
                self.parameters = yaml.load(parameter_file)
        except Exception as e:
            # LOG.error(f'Error loading parameter file with exception: {e}')
            raise e

    def load_results(self):
        '''
            Load results pickle with model information
        '''
        # obtain the path and the default name of the results file
        results_file_path = utils.model_path(self.model, self.version)
        results_file_name = os.path.join(results_file_path,
                                         'results.pkl')
        self.conveyor = Conveyor()
        # load the main class dictionary (p) from this yaml file
        if not os.path.isfile(results_file_name):
            raise Exception('Results file not found')

        try:
            with open(results_file_name, "rb") as input_file:
                self.conveyor.load(input_file)
        except Exception as e:
            # LOG.error(f'No valid results pickle found at: {results_file_name}')
            raise e

    def assign_parameters(self):
        '''
            Fill documentation values corresponding to model parameter values
        '''

        if not self.parameters:
            raise ('Parameters were not loaded')

        self.fields['Algorithm']['subfields']['algorithm']['value'] = \
            self.parameters['model']['value']
        self.fields['Algorithm']['subfields']['descriptors']['value'] = \
            self.parameters['computeMD_method']['value']
        if self.parameters['conformal']['value']:
            self.fields['AD_method']['subfields']['name']['value'] = \
                'conformal prediction'
        if self.parameters['conformal']:
            self.fields['AD_parameters']['value'] = \
                (f'Conformal Significance: '
                    f'{self.parameters["conformalSignificance"]["value"]}')
        self.fields['Algorithm_settings']['subfields']['name']['value'] = \
            self.parameters['model']['value']

    def assign_results(self):
        '''
            Assign result values to documentation fields
        '''
        # Accepted validation keys
        allowed = ['Conformal_accuracy', 'Conformal_mean_interval',
                   'Sensitivity', 'Specificity', 'MCC',
                   'Conformal_coverage', 'Conformal_accuracy',
                   'Q2', 'SDEP']
        model_info = self.conveyor.getVal('model_build_info')
        validation = self.conveyor.getVal('model_valid_info')
        self.fields['Data_info']\
            ['subfields']['training_set_size']['value'] = \
            model_info[0][2]
        self.fields['Descriptors']\
            ['subfields']['final_number']['value'] = \
            model_info[1][2]
        self.fields['Descriptors']\
            ['subfields']['ratio']['value'] = \
            '{:0.2f}'.format(model_info[1][2]/model_info[0][2])
        internal_val = ''
        for stat in validation:
            if stat[0] in allowed:
                internal_val += f'{stat[0]} : {stat[2]}\n'
        self.fields['Internal_validation_1']\
            ['value'] = internal_val

    def get_string(self, dictionary):
        '''
        Convert a dictionary to string format for the model
        template
        '''
        text = ''
        for key, val in dictionary.items():
            text += f'{key} : {val["value"]}\n'
        return text

    def get_upf_template(self):
        '''
            This function creates a tabular model template based
            on the QMRF document type
        '''

        template = pd.DataFrame()
        template['ID'] = ['']
        template['Version'] = ['']
        template['Description'] = ['']
        template['Contact'] = ['']
        template['Institution'] = ['']
        template['Date'] = ['']
        template['Endpoint'] = ['']
        template['Endpoint_units'] = ['']
        template['Dependent_variable'] = ['']
        template['Species'] = ['']
        template['Limits_applicability'] = ['']
        template['Experimental_protocol'] = ['']
        template['Data_info'] = [self.get_string(
            self.fields['Data_info']['subfields'])]
        template['Model_availability'] = [\
            self.get_string(self.fields['Model_availability']
                            ['subfields'])]
        template['Algorithm'] = [self.get_string(
                                self.fields['Algorithm']['subfields']
                                )]
        template['Software'] = [self.get_string(
                                self.fields['Software']['subfields']
                                )]
        template['Descriptors'] = [self.get_string(
                                self.fields['Descriptors']['subfields']
                                )]
        template['Algorithm_settings'] = [self.get_string(
                        self.fields['Algorithm_settings']['subfields']
                        )]
        template['AD_method'] = [self.get_string(
                        self.fields['AD_method']['subfields']
                        )]
        template['AD_parameters'] = [self.fields['AD_parameters']['value']]
                        
        template['Goodness_of_fit_statistics'] = [self.fields\
                                ['Goodness_of_fit_statistics']['value']]
        template['Internal_validation_1'] = [self.fields[
                        'Internal_validation_1']['value']]
        template.to_csv('QMRF_template.tsv', sep='\t')

