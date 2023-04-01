import pandas as pd
from src.processing.data_manager import load_dataset
from src.config.core import config
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return load_dataset(filename=config.app_config.test_data_file)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}