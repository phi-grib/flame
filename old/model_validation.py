# -*- coding: utf-8 -*-

##    Description    tools for qualitative endpoints
##                   
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu) 
##
##    Copyright 2013 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

import warnings
##warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import learning_curve  # JC
from sklearn.model_selection import * #KP
from sklearn.model_selection import LeavePOut #KP
from sklearn.model_selection import LeaveOneOut #KP


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def getCrossVal(cv, rs, n, p):

    ###Splitter Classes:

    kfold = KFold(n_splits=n, random_state=rs, shuffle=False)                             ### K-Folds cross-validator
    gkfold = GroupKFold(n_splits=n)                                                       ### K-fold iterator variant with non-overlapping groups.
    stkfold = StratifiedKFold(n_splits=n, random_state=rs, shuffle=False)                 ### Stratified K-Folds cross-validator
    logo = LeaveOneGroupOut()                                                             ### Leave One Group Out cross-validator
    lpgo = LeavePGroupsOut(n_groups=n)                                                    ### Leave P Group(s) Out cross-validator
    loo = LeaveOneOut()                                                                   ### Leave-One-Out cross-validator
    lpo = LeavePOut(int(p))                                                               ### Leave-P-Out cross-validator
    shufsplit = ShuffleSplit(n_splits=n, random_state=rs, test_size=0.25, train_size=None)### Random permutation cross-validator
    gshufplit = GroupShuffleSplit(test_size=10, n_splits=n)                               ### Shuffle-Group(s)-Out cross-validation iterator
    stshufsplit = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=0)       ### Stratified ShuffleSplit cross-validator
    psplit = PredefinedSplit(test_fold=[ 0,  1, -1,  1])                                  ### Predefined split cross-validator
    tssplit = TimeSeriesSplit(n_splits=n)
    
    splitClass = {'kfold': kfold, 'gkfold': gkfold, 'stkfold': stkfold, 'logo': logo,
                  'lpgo': lpgo, 'loo': loo, 'lpo': lpo, 'shufsplit': shufsplit,
                  'gshufplit': gshufplit, 'stshufsplit': stshufsplit,
                  'psplit': psplit, 'tssplit': tssplit}

##    splitClass = {'kfold': kfold, 'stkfold': stkfold,
##                  'loo': loo, 'lpo': lpo,
##                  'shufsplit': shufsplit}
    
    cv = splitClass.get(str(cv))

    return cv
