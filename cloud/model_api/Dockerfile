FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_api.py .

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "9002"]