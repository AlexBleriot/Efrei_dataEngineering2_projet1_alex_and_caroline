FROM jupyter/scipy-notebook

USER root

WORKDIR /home

RUN mkdir ./mlsample
RUN mkdir ./mlsample/model

ENV MODEL_DIR=/home/mlsample/model
ENV MODEL_FILE=/clf.joblib
ENV METADATA_FILE=metadata.json

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY docker-ml.py .
COPY docker-ml-inference.py .

RUN python ./docker-ml.py
RUN python ./docker-ml-inference.py