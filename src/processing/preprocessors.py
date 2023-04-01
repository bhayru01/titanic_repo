import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropVars(BaseEstimator, TransformerMixin):
	# Temporal elapsed time transformer

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# so that we do not over-write the original dataframe
        X = X.copy()
        X.drop(labels=self.variables, axis=1, inplace=True)
        return X



class ReplaceQuestionMarks(BaseEstimator, TransformerMixin):
    	# Temporal elapsed time transformer

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# so that we do not over-write the original dataframe
        X = X.copy()
        X = X.replace('?', np.nan)
        return X



class CastVarsToFloat(BaseEstimator, TransformerMixin):
    	# Temporal elapsed time transformer

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):

    	# so that we do not over-write the original dataframe
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].astype('float')
        return X
