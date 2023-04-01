from pathlib import Path
from strictyaml import YAML, load
from typing import Dict, List, Sequence
from pydantic import BaseModel

#import src
# Project Directories
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = ROOT / 'data' / 'raw'
TRAINED_MODEL_DIR = ROOT / 'models'


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    data_url: str
    train_data_file: str
    test_data_file: str
    y_test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int
    var_to_extract_title: str
    title_var_name: str
    vars_to_drop: List[str]
    numerical_vars: List[str]
    categorical_vars: List[str]
    var_to_extract_letter: str
    vars_to_cast: List[str]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file_path() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file_path()

    if cfg_path:
        with open(cfg_path, 'r') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_config(parsed_config: YAML = None) -> Config:
    """Create config."""

    if not parsed_config:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
                     app_config = AppConfig(**parsed_config.data),
                     model_config = ModelConfig(**parsed_config.data)
                    )
    return _config


config = create_config()