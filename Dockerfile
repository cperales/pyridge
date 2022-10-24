FROM python:3.9-slim-buster

# Environments
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install \
    git \
    wget \
    unzip

# Install requirements
RUN pip3 install --upgrade --no-cache-dir pip && pip install --no-cache-dir \
    numpy==1.20 \
    pandas==1.3.5 \
    sklearn \
    pymongo \
    cvxopt \
    scipy \
    pyscenarios \
    llvmlite \
    toolz \
    six \
    pyyaml \
    pytest \
    git+https://github.com/cperales/contGA.git

# Install PyRidge library
ADD . /repo/
WORKDIR /repo/
RUN pip3 install -e .

# Test PyRidge with examples
RUN wget https://www.dropbox.com/s/c5p3fg0x8selqqv/data.zip && unzip data
RUN wget https://www.dropbox.com/s/6en0h3pxv8pbaj5/data_regression.zip && unzip data_regression.zip 
RUN pytest
RUN rm -rf data && rm -rf data_regression

VOLUME /repo
WORKDIR /repo