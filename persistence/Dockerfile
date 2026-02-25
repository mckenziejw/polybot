FROM python:3.12-slim

WORKDIR /app

COPY persistence/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/polymarket_pb2.py .
COPY persistence/src/ ./src/

CMD ["python", "-m", "src.main"]