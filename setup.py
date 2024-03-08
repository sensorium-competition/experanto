from setuptools import setup, find_packages

setup(
    name="experanto",
    version="0.0",
    description="Python package to interpolate recordings and stimuli of neuroscience experiments",
    url="https://github.com/sensorium-competition/experanto",
    packages=find_packages(),
    # requried packages
    install_requires=["numpy"],
)
