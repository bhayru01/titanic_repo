from setuptools import find_packages, setup
# from pathlib import Path
# BASE_DIR = Path(__file__).parent

# # Load packages from requirements.txt
# with open(Path(BASE_DIR, "requirements//requirements.txt")) as file:
#     required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name='titanic_core',
    packages=find_packages(),
    version='0.2.0',
    description='Task from "Deployment of machine learning" course',
    author='beray',
    python_requires='>=3.9',
    install_requires=['numpy>=1.20.0,<1.21.0',
                        'pandas>=1.3.5,<1.4.0',
                        'pydantic>=1.8.1,<1.9.0',
                        'scikit-learn>=1.0.2,<1.1.0',
                        'strictyaml',
                        'ruamel.yaml==0.17.10',
                        'feature-engine>=1.0.2,<1.1.0',
                        'joblib>=1.0.1,<1.1.0'],
    package_data={'': ['config.yml', 'VERSION']},
    include_package_data=True,
    license='MIT'
)
