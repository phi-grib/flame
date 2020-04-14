#! -*- coding: utf-8 -*-

# Description    Flame Parent Model Class
##
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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

""" Module to add cross-validation methods
"""
def getCrossVal(cv, rs, n, p):

    cv = str(cv)

    if cv == 'loo':
        from sklearn.model_selection import LeaveOneOut
        return LeaveOneOut()                   

    if cv == 'kfold':
        from sklearn.model_selection import KFold
        return KFold(n_splits=n, random_state=rs, shuffle=True)

    if cv == 'lpo':
        from sklearn.model_selection import LeavePOut 
        return LeavePOut(int(p))

