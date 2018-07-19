from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression


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