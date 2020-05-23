FROM ubuntu:latest

MAINTAINER Vaibhav Mehta <mehta8292@gmail.com>

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
      build-essentials \
      g++ \
      git \
      openssh-client \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-virtualenv \
      python3-wheel \
      pkg-config \
      libopenblas-base \
      python3-numpy \
      python3-scipy \
      python3-h5py \
      python3-yaml \
      python3-pydot && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install -U numpy==1.13.3 && \
    pip3 install --upgrade h5py && \
    pip3 install --upgrade pillow

ARG TENSORFLOW_VERSION=1.5.0
ARG TENSORFLOW_DEVICE=cpu
ARG TENSORFLOW_APPEND=
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=2.1.4
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

RUN python3 -c "import tensorflow; print(tensorflow.__version__)" \
  && dpkg-query -l > /dpkg-query-l.txt \
  && pip3 freeze > /pip3-freeze.txt

WORKDIR /data/

COPY . .
