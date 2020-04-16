#! -*- coding: utf-8 -*-

# Description    Flame Apply internal class
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
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from flame.stats.Keras import Keras_nn

class my_keras(Keras_nn):
    def __init__(self, X, Y, parameters, conveyor):
        Keras_nn.__init__(self, X, Y, parameters, conveyor)

# Function to create model, required for KerasClassifier
    def create_model(self, dim=20):
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=dim, activation='relu'))
        model.add(Dense(20, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        
        if self.param.getVal('quantitative'):
            loss = 'mean_squared_error'
        else:
            loss = 'binary_crossentropy'
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        return model