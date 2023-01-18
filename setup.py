from setuptools import find_packages, setup
from pathlib import Path
BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Task from "Deployment of machine learning" course',
    author='beray',
    python_requires="3.9.12",
    install_requires=[required_packages],
    license='',
)
