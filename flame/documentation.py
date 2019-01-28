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

from flame.util import utils
from flame.util import get_logger
LOG = get_logger(__name__)


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

        # obtain the path and the default name of the model documents
        documentation_file_path = utils.model_path(model, version=0)
        documentation_file_name = os.path.join(documentation_file_path,
                                              'documentation.yaml')

        # load the main class dictionary (p) from this yaml file
        if not os.path.isfile(documentation_file_name):
            raise Exception('Documentation file not found')

        try:
            with open(documentation_file_name, 'r') as documentation_file:
                self.fields = yaml.load(documentation_file)
        except Exception as e:
            LOG.error(f'Error loading documentation file with exception: {e}')
            raise e

        # MD5 hash, not necessary yet.
        # self.fields['md5'] = utils.md5sum(documentation_file_name)
        #self.fields['md5']

    def load_results(self, results, params):
        