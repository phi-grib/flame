#! -*- coding: utf-8 -*-

# Description    Flame Parent Space Class
##
# Authors:       Manuel Pastor (manuel.pastor@ufp.edu)
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

from flame.util import utils
import pickle
import numpy as np
import os
from flame.util import utils, get_logger, supress_log

LOG = get_logger(__name__)


class Space:
    def __init__(self, X, parameters):
        """Initializes the estimator.
        Actions
        -------
            - Attribute assignment
        """

    def build(self):
        ''' This function saves estimator and scaler in a pickle file '''

        return True, 'success'

    def save_space(self):
        ''' This function saves estimator and scaler in a pickle file '''


        return

    def load_space(self):
        ''' This function loads estimator and scaler in a pickle file '''
    
        return