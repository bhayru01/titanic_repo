import pytest
from src.config.core import config
from src.processing.data_manager import load_dataset, sequential_split


@pytest.fixture()
def sample_input_data():
    titanic_dataset = load_dataset(config.app_config.data_url)
    df_train, df_test, y_test = sequential_split(data = titanic_dataset,
                                                 split_ratio = config.model_config.test_size,
                                                 target = config.model_config.target)
    return df_test
