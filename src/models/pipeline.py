# for encoding categorical variables
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# for imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

# for the preprocessors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.core import config
from src.features.build_features import ExtractLetter, ExtractTitle
from src.processing.preprocessors import DropVars

# set up the pipeline
titanic_pipe = Pipeline(
    [
        # ===== FEATURE EXTRACTION =====
        # Extract letter from cabin
        (
            "extract_letter",
            ExtractLetter(variable=config.model_config.var_to_extract_letter),
        ),
        # Extract title from fullname
        (
            "extract_title",
            ExtractTitle(variable=config.model_config.var_to_extract_title),
        ),
        # ===== FEATURE DROPING =====
        ("drop_features", DropVars(variables=config.model_config.vars_to_drop)),
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_config.numerical_vars
            ),
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05, n_categories=1, variables=config.model_config.categorical_vars
            ),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True, variables=config.model_config.categorical_vars
            ),
        ),
        # scale
        ("scaler", StandardScaler()),
        ("Logit", LogisticRegression(C=0.0005, random_state=0)),
    ]
)
