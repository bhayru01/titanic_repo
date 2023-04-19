import pandas as pd
from src.processing.data_manager import load_dataset, sequential_split
from src.config.core import config
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    titanic_dataset = load_dataset(config.app_config.data_url)
    df_train, df_test, y_test = sequential_split(data = titanic_dataset,
                                                 split_ratio = config.model_config.test_size,
                                                 target = config.model_config.target)
    return df_test


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}