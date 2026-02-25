FROM python:3.12-slim

WORKDIR /app

COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/polymarket_pb2.py .
COPY ingest/src/ ./src/

CMD ["python", "-m", "src.main"]