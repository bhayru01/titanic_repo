import pytest
from src.config.core import config
from src.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    return load_dataset(config.app_config.test_data_file)
