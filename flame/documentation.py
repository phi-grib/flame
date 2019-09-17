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
import numpy as np

from flame.util import utils
from flame.conveyor import Conveyor
from flame.parameters import Parameters
from rdkit.Chem import AllChem


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

        load_parameters()
            Accesses to param file to retrieve all
            information needed to document the model.
        load_results()
            Accesses to build results to retrieve all
            information needed to document the model.
        assign_parameters()
            Fill documentation values corresponding to
             model parameter values
        assign_results()
            Assign result values to documentation fields
        get_upf_template()
            creates a spreedsheet QMRF-like
        get_prediction_template()
            Creates a reporting document for predictions
    
        '''

    def __init__(self, model, version=0, context='model'):
        ''' Load the fields from the documentation file'''

        self.model = model
        self.version = version
        self.fields = None
        self.parameters = Parameters()
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
        
        success, message = self.parameters.loadYaml(model, 0)

        if not success:
            print('Parameters could not be loaded. Please assure endpoint is correct')
            return
        
        # Remove this after acc
        #self.load_parameters()
        if context == 'model':
            self.load_results()
            self.assign_parameters()
            self.assign_results()

    # def load_parameters(self):
    #     '''This function takes info from results and
    #     param file and assigns it to corresponding fields
    #     in documentation dictionary'''
        # obtain the path and the default name of the parameter_file
        # parameters_file_path = utils.model_path(self.model, self.version)
        # parameters_file_name = os.path.join(parameters_file_path,
        #                                     'parameters.yaml')
        
        # success, message = self.parameters.loadYaml(self.model, 0)
        # if not os.path.isfile(parameters_file_name):
        #     raise Exception('Parameter file not found')
        # try:
        #     with open(parameters_file_name, 'r') as parameter_file:
        #         self.parameters = yaml.load(parameter_file)
        # except Exception as e:
        #     # LOG.error(f'Error loading parameter file with exception: {e}')
        #     raise e

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
            # LOG.error(f'No valid results pickle found at: 
            # {results_file_name}')
            raise e

    def assign_parameters(self):
        '''
            Fill documentation values corresponding to model parameter values
        '''

        if not self.parameters:
            raise ('Parameters were not loaded')

        self.fields['Algorithm']['subfields']['algorithm']['value'] = \
            self.parameters.getVal('model')
        self.fields['Algorithm']['subfields']['descriptors']['value'] = \
            self.parameters.getVal('computeMD_method')
        if self.parameters.getVal('conformal'):
            self.fields['AD_method']['subfields']['name']['value'] = \
                'conformal prediction'
            self.fields['AD_parameters']['value'] = \
                (f'Conformal Significance: '
                    f'{self.parameters.getVal("conformalSignificance")}')
        self.fields['Algorithm_settings']['subfields']['name']['value'] = \
            self.parameters.getVal('model')

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

    def get_upf_template2(self):
        '''
            This function creates a tabular model template based
            on the QMRF document type
        '''
        fields = ['ID', 'Version', 'Contact', 'Institution',\
            'Date', 'Endpoint', 'Endpoint_units', 'Dependent_variable', 'Species',\
                'Limits_applicability', 'Experimental_protocol', 'Data_info',\
                    'Model_availability', 'Algorithm', 'Software', 'Descriptors',\
                        'Algorithm_settings', 'AD_method', 'AD_parameters',\
                            'Goodness_of_fit_statistics', 'Internal_validation_1' ]
        template = pd.DataFrame(columns=['Field', 'Parameter name', 'Parameter value'])
        for field in fields: 
            subfields = self.fields[field]['subfields']
            if subfields is not None:
                for index, subfield in enumerate(subfields):
                    field2 = ''
                    if index == 0:
                        field2 = field
                    else:
                        field2 = ""
                    value = str(subfields[subfield]['value'])
                    # None types are retrieved as str from yaml??
                    if value == "None":
                        value = ""
                    row = dict(zip(['Field', 'Parameter name', 'Parameter value'],\
                        [field2, subfield, value]))
                    template = template.append(row, ignore_index=True)
            else:
                value = str(self.fields[field]['value'])
                if value == 'None':
                    value = ""
                row = dict(zip(['Field', 'Parameter name', 'Parameter value'],\
                    [field, "", value]))
                template = template.append(row, ignore_index=True)
        template.to_csv('QMRF_template3.tsv', sep='\t', index=False)



    def get_prediction_template(self):
        '''
            This function creates a tabular model template based
            on the QMRF document type
        '''
        # obtain the path and the default name of the results file
        results_file_path = utils.model_path(self.model, self.version)
        results_file_name = os.path.join(results_file_path,
                                         'prediction-results.pkl')
        conveyor = Conveyor()
        # load the main class dictionary (p) from this yaml file
        if not os.path.isfile(results_file_name):
            raise Exception('Results file not found')
        try:
            with open(results_file_name, "rb") as input_file:
                conveyor.load(input_file)
        except Exception as e:
            # LOG.error(f'No valid results pickle found at: {results_file_name}')
            raise e        

        # First get Name, Inchi and InChIkey

        names = conveyor.getVal('obj_nam')
        smiles = conveyor.getVal('SMILES')
        inchi = [AllChem.MolToInchi(
                      AllChem.MolFromSmiles(m)) for m in smiles]
        inchikeys = [AllChem.InchiToInchiKey(
                     AllChem.MolToInchi(
                      AllChem.MolFromSmiles(m))) for m in smiles]
        predictions = []
        applicability = []
        if self.parameters['quantitative']['value']:
            raise('Prediction template for quantitative endpoints'
                  ' not implemented yet')
        if not self.parameters['conformal']['value']:
            predictions = conveyor.getVal('values')
        else:
            c0 = np.asarray(conveyor.getVal('c0'))
            c1 = np.asarray(conveyor.getVal('c1'))

            predictions = []
            for i, j in zip(c0, c1):
                prediction = ''
                if i == j:
                    prediction = 'out of AD'
                    applicability.append('out')
                if i != j:
                    if i == True:
                        prediction = 'Inactive'
                    else:
                        prediction = 'Active'
                    applicability.append('in')

                predictions.append(prediction)

        # Now create the spreedsheats for prediction

        # First write summary
        summary = ("Study name\n" +
                "Endpoint\n" +
                "QMRF-ID\n" +
                "(Target)Compounds\n" +
                "Compounds[compounds]\tName\tInChiKey\n")
        
        for name, inch in zip(names, inchikeys):
            summary += f'\t{name}\t{inch}\n'

        summary += ("\nFile\n" + 
                    "Author name\n" +
                    "E-mail\n" +
                    "Role\n" +
                    "Affiliation\n" +
                    "Date\n")
                
        with open('summary_document.tsv', 'w') as out:
            out.write(summary)

        # Now prediction details
        # Pandas is used to ease the table creation.

        reporting = pd.DataFrame()

        reporting['InChI'] = inchi
        reporting['CAS-RN'] = '-'
        reporting['SMILES'] = smiles
        reporting['prediction'] = predictions
        reporting['Applicability_domain'] = applicability
        reporting['reliability'] = '-'
        reporting['Structural_analogue_1_CAS'] = '-'
        reporting['Structural_analogue_1_smiles'] = '-'
        reporting['Structural_analogue_1_source'] = '-'
        reporting['Structural_analogue_1_experimental_value'] = '-'
        reporting['Structural_analogue_2_CAS'] = '-'
        reporting['Structural_analogue_2_smiles'] = '-'
        reporting['Structural_analogue_2_source'] = '-'
        reporting['Structural_analogue_2_experimental_value'] = '-'
        reporting['Structural_analogue_3_CAS'] = '-'
        reporting['Structural_analogue_3_smiles'] = '-'
        reporting['Structural_analogue_3_source'] = '-'
        reporting['Structural_analogue_3_experimental_value'] = '-'

        reporting.to_csv('prediction_report.tsv', sep='\t',index=False)

        



        


        
        

        
        

            




        
