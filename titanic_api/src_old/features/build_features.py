# for the preprocessors
from sklearn.base import BaseEstimator, TransformerMixin
import re
from src.config.core import config


class ExtractLetter(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variable):
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        self.variable = variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()
        X[self.variable] = X[self.variable].str[0]

        return X


class ExtractTitle(BaseEstimator, TransformerMixin):
    # Extract title from fullname

    def __init__(self, variable):
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        self.variable = variable


    def get_title(self, passenger):
        line = passenger
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'


    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self


    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()
        X[config.model_config.title_var_name] = X[self.variable].apply(self.get_title)

        return X