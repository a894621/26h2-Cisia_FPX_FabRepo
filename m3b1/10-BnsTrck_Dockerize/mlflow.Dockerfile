FROM python:3.11-slim
RUN pip install mlflow
CMD ["mlflow","ui","--host","0.0.0.0"]
