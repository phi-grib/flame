#! -*- coding: utf-8 -*-

# Description    Flame graphic functions
#
# Authors: Manuel Pastor (manuel.pastor@upf.edu),
#
# Copyright 2020 Manuel Pastor
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
import numpy as np
from flame.stats.pca import pca    
import matplotlib.pyplot as plt

from flame.util import utils, get_logger
LOG = get_logger(__name__)

    
def generateProjectedSpace(X, param, conveyor):
    # TODO: decide which is the best way to present the training space
    LOG.info('Generating projeced X space...')
    mpca = pca()
    mpca.build(X,targetA=2,autoscale=False)

    pca_path = os.path.join(param.getVal('model_path'),'pca.npy')
    mpca.saveModel(pca_path)

    obj_nam = conveyor.getVal('obj_nam')

    # generate TSV file with PCA scores
    with open('scores.tsv','w') as handler:
        for i in range(mpca.nobj):
            handler.write (f'{obj_nam[i]}\t{mpca.t[0][i]}\t{mpca.t[1][i]}\n')

    # dump to conveyor?

    # generate png with PCA scores
    scores=plt.figure(figsize=(9,6))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.scatter(mpca.t[0],mpca.t[1], c='red', marker='D', s=40, linewidths=0)
    scores.savefig("pca-scores12.png", format='png')

def projectPredictions(X, conveyor):
    return