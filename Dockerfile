FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .
RUN python -m tools.download_embedding_model
EXPOSE 10000
CMD ["sh", "-c", "uvicorn yiti_ai.api:app --host 0.0.0.0 --port ${PORT:-10000}"]