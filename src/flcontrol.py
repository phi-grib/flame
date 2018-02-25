#! -*- coding: utf-8 -*-

##    Description    Flame flcontrol class
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

import hashlib

class flControl:

    def __init__ (self):
        self.model_name = ""
        self.model_version = 0.0

        self.input_type = 'molecule'                # 'molecule' | 'data'
        self.normalize_method = 'standardize'       # None | 'standardize'
        self.ionize_method = None                   # None | 'moka'

        return

    def md5stamp (self):

        m = hashlib.md5()
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, (int, float, str)):
                m.update (str(val).encode('utf-8'))

        return (m.hexdigest())