FROM jupyter/datascience-notebook

WORKDIR /src
ADD . /src/experanto
USER root
RUN pip install -e /src/experanto
USER jovyan
WORKDIR /home/jovyan
