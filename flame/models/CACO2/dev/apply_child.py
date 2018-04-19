#! -*- coding: utf-8 -*-

##    Description    Flame Apply internal class
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

from apply import Apply
import numpy as np

class ApplyChild (Apply):

    def __init__ (self, control, results):

        Apply.__init__(self, control, results)

    def run (self):

        self.results['values']=self.results['xmatrix'].mean(1)
        self.results['origin'] = 'apply'
        self.results['meta'] = {'main':['values']}
 
        return True, self.results

