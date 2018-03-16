#! -*- coding: utf-8 -*-

##    Description    Flame Manage class
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

import os
import sys
import shutil

import utils

class Manage:

    def __init__ (self, model, action):

        self.model = model
        self.action = action

        return

    def action_new (self, model):

        ndir = utils.base_path(model)
        
        # check if there is already a tree for this endpoint
        if os.path.isdir (ndir):
            return False, 'This endpoint already exists'
        try:
            os.mkdir (ndir)
        except:
            return False,'unable to create directory : '+ndir

        ndir+='/dev'
        try:
            os.mkdir (ndir)
        except:
            return False,'unable to create directory '+ndir

        # TODO: create templates directory with empty childs
        try:
            wkd = os.path.dirname(os.path.abspath(__file__))
            children_names = ['apply','control','idata','odata','learn']
            for cname in children_names:
                shutil.copy(wkd+'/children/'+cname+'_child.py',ndir+'/'+cname+'_child.py')
        except:
            return (False,'unable to create imodel.py at '+ndir)

        return True,'version created OK'

    def action_kill (self):
        print ('manage kill')
        return True, 'manage OK'

    def action_publish (self):
        print ('manage publish')
        return True, 'manage OK'

    def action_remove (self):
        print ('manage remove')
        return True, 'manage OK'

    def action_list (self):
        print ('manage list')
        return True, 'manage OK'

    def run (self):
        ''' Executes a default predicton workflow '''

        if self.action == 'new':
            success, results = self.action_new (self.model)

        elif self.action == 'kill':
            success, results = self.action_kill ()

        elif self.action == 'remove':
            success, results = self.action_remove ()

        elif self.action == 'publish':
            success, results = self.action_publish ()

        elif self.action == 'list':
            success, results = self.action_list ()

        return success, results

