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

from flame.stats.space import Space
from flame.util import utils, get_logger

LOG = get_logger(__name__)

class Slearn:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor
        self.conveyor.setOrigin('slearn')

        self.X = self.conveyor.getVal('xmatrix')


    def run (self):
        '''
        Builds a chemical space 

        '''

        # instances space object
        space = Space(self.param)

        # builds space from idata results
        LOG.info('Starting space building')
        success, space_building_results = space.build(self.X, self.conveyor.getVal('obj_nam'), self.conveyor.getVal('SMILES'))
        if not success:
            LOG.error('space_building_results')
            self.conveyor.setError(space_building_results)
            return

        self.conveyor.addVal(
                    space_building_results,
                    'space_build_info',
                    'space build info',
                    'method',
                    'single',
                    'Information about the building of the chemical space')

        # save model
        try:
            space.save_space()
        except Exception as e:
            LOG.error(f'Error saving space with exception {e}')
            self.conveyor.setError(f'Error saving space with exception {e}')
            return

        LOG.info('Space building finished successfully')

        return

