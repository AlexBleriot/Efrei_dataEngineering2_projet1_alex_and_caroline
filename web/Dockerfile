FROM python:3.6

WORKDIR /app

RUN mkdir ./mlsample

RUN mkdir ./mlsample/model

ENV MODEL_DIR=/app/mlsample/model

ENV MODEL_FILE=/clf.joblib

ENV METADATA_FILE=metadata.json

ENV FLASK_APP=app.py

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY reviewDataCleaned.csv .

COPY docker-ml.py .

RUN python ./docker-ml.py

COPY app.py .

COPY templates ./templates

EXPOSE 5000

CMD ["python", "app.py"]