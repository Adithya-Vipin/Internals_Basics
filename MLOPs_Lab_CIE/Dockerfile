FROM python:3.12-slim 
WORKDIR /app 
COPY . . 
RUN pip install mlflow scikit-learn pandas numpy 
CMD ["python", "src/predict_cli.py"]
