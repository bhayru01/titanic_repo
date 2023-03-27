from app.config import API_ROOT

with open(API_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
