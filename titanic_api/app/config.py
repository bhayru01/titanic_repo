from src import __version__ as model_version
from logging.handlers import TimedRotatingFileHandler
from logging import StreamHandler
from pydantic import BaseSettings
from pathlib import Path
import logging
import sys


FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
API_ROOT = Path(__file__).parent.parent.absolute()
LOG_FILE = API_ROOT / "api.log"

class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO

    def get_console_handler(self):
        console_handler = StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        return console_handler

    def get_file_handler(self):
        file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
        file_handler.setFormatter(FORMATTER)
        return file_handler

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.LOGGING_LEVEL)
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False
        return logger


with open(API_ROOT / "VERSION") as version_file:
    app_version = version_file.read().strip()


class Settings(BaseSettings):
    PROJECT_NAME = 'Titanic Survival Prediction API'
    API_VERSION = app_version
    API_V1_STR: str = "/api/v1"
    MODEL_VERSION = model_version

settings = Settings()
