#! -*- coding: utf-8 -*-

##    Description    Flame predict command
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

from flpredict import flPredict

# hardcoded stuff for testing

inputFileName = 'test.sdf'
modelName = 'CACO2'

predict = flPredict (inputFileName, modelName)
success, results = predict.run()

print (success, results)
