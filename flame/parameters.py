#! -*- coding: utf-8 -*-

# Description    Flame Build class
#
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
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
import sys

from flame.util import utils, get_logger

LOG = get_logger(__name__)

class parameters:

        def __init__(self, model, version):
     
            parameters_file_path = utils.model_path(model, version)
            parameters_file_name = os.path.join (parameters_file_path,'parameters.yaml')

            if not os.path.isfile(parameters_file_name):
                LOG.critical('Unable to load model parameters. Aborting...')
                sys.exit()

            try:
                with open(parameters_file_name, 'r') as pfile:
                    self.p = yaml.load(pfile)
            except Exception as e:
                LOG.critical('Unable to load model parameters. Aborting...')
                sys.exit()

            self.setVal('endpoint',model)
            self.setVal('version',version)
            self.setVal('model_path',parameters_file_path)
            self.setVal('md5',utils.md5sum(parameters_file_name))

            return

        def getVal(self, key):
            if key in self.p:
                return self.p[key]
            return None

        def setVal(self, key, value):
            self.p[key] = value