#! -*- coding: utf-8 -*-

# Description    Flame compute molecular 3D coordinates functions
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
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
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from flame.util import get_logger
import flame.chem.sdfileutils as sdfu

LOG = get_logger(__name__)


def _ETKDG(ifile) -> (bool, str):
    """ Assigns 3D structures to the molecular structures provided as input.
    """

    success_list = [True for i in range(sdfu.count_mols(ifile))]

    LOG.info('Converting to ETKDG 3D structures')
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.critical('Unable to create supplier')
        raise e
        # not true, UNABLE TO CREATE SUPPLIER
        # return False, 'unable to compute 3D structures'

    filename, fileext = os.path.splitext(ifile)
    ofile = filename + '_3d' + fileext
    LOG.debug(f'3D stucture ouput file is: {ofile}')

    with open(ofile, 'w') as fo:

        mcount = 0
        for mol in suppl:
            if mol is None:
                LOG.debug('Supplier failed to read'
                            f' molecule #{mcount+1} in {ifile}')
                continue
            try:
                mol3 = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol3, AllChem.ETKDG())
            except:
                LOG.error('Failed to generate 3D structures using'
                            f'ETKDG method for molecule #{mcount+1} in {ifile}')
                success_list[mcount]=False
                mcount += 1
                continue

            ## debug for testing error handling. This code simulates a 3D conversion error
            # if mcount == 3 :
            #     print ('@ETKDG debug, skipping mol 3')
            #     LOG.error('Failed to generate 3D structures using'
            #                 f'ETKDG method for molecule #{mcount+1} in {ifile}')
            #     success_list[mcount]=False
            #     mcount += 1
            #     continue

            fo.write(Chem.MolToMolBlock(mol3))
            fo.write('\n$$$$\n')  # end of mol
            mcount += 1

    return success_list, ofile
