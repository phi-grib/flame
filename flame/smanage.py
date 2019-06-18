#! -*- coding: utf-8 -*-

# Description    Flame Manage class
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
import shutil
import tarfile
import json
import pickle
import pathlib
import numpy as np
from flame.util import utils, get_logger 
# from flame.parameters import Parameters
# from flame.conveyor import Conveyor

LOG = get_logger(__name__)

def set_space_repository(path=None):
    """
    Set the space repository path.
    This is the dir where flame is going to create and load spaces
    """
    utils.set_space_repository(path)

    # this is a console oriented tool which prints messages. Avoid use of LOG.info
    LOG.info(f'space repository updated to {path}')
    #print(f'space repository updated to {path}')
    return True, 'space repository updated'


def action_new(space):
    '''
    Create a new space tree, using the given name.
    This creates the development version "dev",
    copying inside default child classes
    '''

    if not space:
        return False, 'empty space label'

    # importlib does not allow using 'test' and issues a misterious error when we
    # try to use this name. This is a simple workaround to prevent creating spaces 
    # with this name 
    if space == 'test':
        #LOG.warning(f'the name "test" is disallowed, please use any other name')
        return False, 'the name "test" is disallowed, please use any other name'

    # space directory with /dev (default) level
    ndir = pathlib.Path(utils.space_tree_path(space)) / 'dev'

    # check if there is already a tree for this endpoint
    if ndir.exists():
        #LOG.warning(f'Endpoint {space} already exists')
        return False, f'Endpoint {space_tree_path} already exists'

    try:
        ndir.mkdir(parents=True)
        LOG.debug(f'{ndir} created')
    except:
        return False, f'Unable to create path for {space} endpoint'

    # Copy classes skeletons to ndir
    wkd = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    children_names = ['search', 'idata', 'odata', 'slearn']

    for cname in children_names:
        filename = cname + '_child.py'
        src_path = wkd / 'children' / filename
        dst_path = ndir / filename
        try:
            shutil.copy(src_path, dst_path)
        except:
            return False, f'Unable to copy {cname} file'

    LOG.debug(f'copied class skeletons from {src_path} to {dst_path}')
    
    # copy parameter yml file
    params_path = wkd / 'children/parameters.yaml'
    shutil.copy(params_path, ndir)

    # copy documentation yml file
    documentation_path = wkd / 'children/documentation.yaml'
    shutil.copy(documentation_path, ndir)
  

    LOG.info(f'New space {space} created')
    #print(f'New endpoint {space} created')
    return True, 'New space '+space+' created'


def action_kill(space):
    '''
    removes the space tree described by the argument
    '''

    if not space:
        return False, 'Empty space name'

    ndir = utils.space_tree_path(space)

    if not os.path.isdir(ndir):
        #LOG.error(f'space {space} not found')
        return False, f'space {space} not found'

    try:
        shutil.rmtree(ndir, ignore_errors=True)
    except:
        return False, f'Failed to remove space {space}'

    LOG.info(f'space {space} removed')
    #print(f'space {space} removed')
    return True, f'space {space} removed'


def action_publish(space):
    '''
    clone the development "dev" version as a new space version,
     assigning a sequential version number
    '''

    if not space:
        return False, 'Empty space label'

    base_path = utils.space_tree_path(space)

    if not os.path.isdir(base_path):
        #LOG.error(f'space {space} not found')
        return False, f'space {space} not found'

    # gets version number
    v = [int(x[-6:]) for x in os.listdir(base_path) if x.startswith("ver")]

    if not v:
        max_version = 0
    else:
        max_version = max(v)

    new_path = os.path.join(base_path,f'ver{max_version+1:06}')

    if os.path.isdir(new_path):
        #LOG.error(f'Versin {v} of space {space} not found')
        return False, f'Version {max_version+1} of space {space} already exists'

    src_path = os.path.join (base_path,'dev')

    try:
        shutil.copytree(src_path, new_path)
    except:
        return False, f'Unable to copy contents of dev version for space {space}'

    LOG.info(f'New space version created from {src_path} to {new_path}')
    return True, f'New space version created from {src_path} to {new_path}'


def action_remove(space, version):
    '''
    Remove the version indicated as argument from the space tree indicated
    as argument
    '''

    if not space:
        return False, 'Empty space label'

    if version == 0:
        return False, 'Development version cannot be removed, provide a version number'

    rdir = utils.space_path(space, version)
    if not os.path.isdir(rdir):
        return False, f'Version {version} not found'

    shutil.rmtree(rdir, ignore_errors=True)
    LOG.info(f'Version {version} of space {space} has been removed')
    return True, f'Version {version} of space {space} has been removed'

