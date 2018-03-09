#! -*- coding: utf-8 -*-

##    Description    Flame Control internal class
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

from control import Control

class ControlChild (Control):

    def __init__ (self):

        Control.__init__ (self)

        self.model_name = 'DIPL2'
        self.model_version = 0.1
        self.quantitative = False
        self.tune = False
        self.conformal = False

        return
