# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.model_selection import LeavePOut  
# from sklearn.model_selection import LeavePGroupsOut
# from sklearn.model_selection import PredefinedSplit
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
# from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import *  # KP



def getCrossVal(cv, rs, n, p):

    cv = str(cv)

    if cv == 'loo':
        from sklearn.model_selection import LeaveOneOut
        return LeaveOneOut()                   

    if cv == 'kfold':
        from sklearn.model_selection import KFold
        return KFold(n_splits=n, random_state=rs, shuffle=False)

    if cv == 'lpo':
        from sklearn.model_selection import LeavePOut 
        return LeavePOut(int(p))

    if cv == 'logo':
        from sklearn.model_selection import LeaveOneGroupOut
        return LeaveOneGroupOut()                   

    if cv == 'lpgo':
        from sklearn.model_selection import LeavePGroupsOut
        return LeavePGroupsOut(n_groups=n)

