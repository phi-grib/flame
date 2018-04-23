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

    ## example of run for overriding
    # def run (self):

    #     self.results['origin'] = 'apply'

    #     ## prediction results must be an array, with as many values as object
    #     ## are in the results
    #     self.results['example'] = np.zeros(len(self.results['obj_nam']), dsize=np.float64))

    #     ## label prediction output, inserting the key name in ['meta']['main'] 
    #     self.results['meta'] = {'main':['example']}

    #     return True, self.results

