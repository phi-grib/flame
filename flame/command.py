#! -*- coding: utf-8 -*-

##    Description    Command wrappers allowing to call predict and build
##                   for models making use of extenal input sources 
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.     

import multiprocessing as mp

from predict import Predict
from build import Build

def getExternalInput (task, model_set, infile):
    # parallel is approppriate for many external sources
    parallel = (len(model_set)>3)
    if parallel:
        task.setSingleCPU()

    # add input molecule to the model input definition of every internal model
    for mi in model_set:
        mi['infile']=infile

    model_suc = []
    model_res = []

    ## TODO: if any of the models belongs to another module, send a POST for
    ## obtaining the results

    if parallel :
        pool = mp.Pool(len(model_set))
        model_temp = pool.map(predict_cmd, model_set)

        for x in model_temp:
            model_suc.append(x[0])
            model_res.append(x[1])
    else:
        for mi in model_set:
            success, results = predict_cmd (mi)
            model_suc.append(success)
            model_res.append(results)

    if False in model_suc:
        return False, 'Some external input sources failed: ', str(model_suc)

    return True, model_res


def predict_cmd(model):
    ''' 
    
    Instantiates a Predict object to run a prediction using the given input file and model 
    
    This method must be self-contained and suitable for being called in cascade, by models
    which use the output of other models as input
    
    '''

    predict = Predict(model['endpoint'], model['version'])

    ext_input, model_set = predict.getModelSet()

    if ext_input :

        success, model_res = getExternalInput (predict, model_set, model['infile'])

        if not success:
            return False, model_res

        # now run the model using the data from the external sources            
        success, results = predict.run(model_res)    

    else:

        # run the model with the input file
        success, results = predict.run(model['infile'])

    return success, results


def build_cmd(model):
    ''' 
    
    Instantiates a Build object to build a model using the given input file and model 
    
    This method must be self-contained and suitable for being called in cascade, by models
    which use the output of other models as input
    
    '''

    build = Build(model['endpoint'])

    ext_input, model_set = build.getModelSet()

    if ext_input :

        success, model_res = getExternalInput (build, model_set, model['infile'])

        if not success:
            return False, model_res

        # now run the model using the data from the external sources            
        success, results = build.run(model_res)    

    else:

        # run the model with the input file
        success, results = build.run(model['infile'])

    return success, results