FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    MODEL_CACHE_DIR=/app/models

WORKDIR /app

# 1) dependencies first (this layer is cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) download the embedding model at build time, so the container works without waiting on first start
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBED_MODEL}', cache_dir='/app/models')"

# 3) application code + unprivileged user
COPY . .
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8000

# default: the Streamlit GUI. For the API run:  uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
