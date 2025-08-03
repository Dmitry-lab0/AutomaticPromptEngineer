FROM python:3.11 AS cicd
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-distutils \
        python3-dev \
        build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH=${PYTHONPATH}:/app 

CMD ["bash"]

