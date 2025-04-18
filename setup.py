from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="experanto",
    version="0.1",
    description="Python package to interpolate recordings and stimuli of neuroscience experiments",
    url="https://github.com/sensorium-competition/experanto",
    packages=find_packages() + ['configs'],
    package_data={
        'configs': ['*.yaml']  # includes all yaml files in config_folder
    },
    install_requires=requirements,
)
