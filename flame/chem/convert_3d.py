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


def _ETKDG(ifile):
    """ Assigns 3D structures to the molecular structures provided as input.
    """
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except:
        return False, 'unable to compute 3D structures'

    filename, fileext = os.path.splitext(ifile)
    ofile = filename + '_3d' + fileext

    num_obj = 0
    with open(ofile, 'w') as fo:
        for mol in suppl:

            if mol is None:
                print('ERROR: (@_ETKDG) Unable to obtain 3D structure for molecule #', str(
                    num_obj+1), 'in file ' + ifile)
                continue

            mol3 = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3, AllChem.ETKDG())

            fo.write(Chem.MolToMolBlock(mol3))
            fo.write('\n$$$$\n')

            num_obj += 1

    return True, ofile
