#! -*- coding: utf-8 -*-

##    Description    Flame Odata class
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
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

class Odata:

    def __init__ (self, control, results):

        self.results = results
        self.control = control

    def run_learn (self):

        print ('odata : ', self.results)
        print ('building OK!')
        return True, 'ok'


    def run_apply (self):

        print ('odata : ', self.results)
        print ('predicting OK!')        
        return True, 'ok'


    def run (self):

        if not 'origin' in self.results:
            return False, 'invalid result format'

        if self.results['origin'] == 'learn':
            success, results  = self.run_learn ()

        elif self.results['origin'] == 'apply':
            success, results  = self.run_apply ()

        else:
            return False, 'invalid result format'

        return success, results

