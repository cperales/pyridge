FROM ubuntu:21.04

# Environments
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install \
    python3.9 \
    python3-pip \
    git \
    wget \
    unzip

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

# Install library
ADD . /repo/
WORKDIR /repo/
RUN pip3 install -e .
RUN wget https://www.dropbox.com/s/jj5zmr4hza3jd8d/data.zip && unzip data
RUN wget https://www.dropbox.com/s/xvan4n74w690rg9/data_regression.zip && unzip data_regression.zip 

VOLUME /repo
WORKDIR /repo