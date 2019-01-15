#! -*- coding: utf-8 -*-

# Description    Flame feature selection methods
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


""" This file contains implemented methods to perform
    feature selection"""


from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from flame.util import utils, get_logger, supress_log
LOG = get_logger(__name__)


def selectkBest(X, Y, n, quantitative):
    function = ""
    if quantitative:
        function = f_regression
    else:
        function = chi2
    kbest = SelectKBest(function, n)
    kbest.fit(X,Y)
    mask = kbest.get_support()
    return mask