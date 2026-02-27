# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy lock + project metadata first for caching
COPY pyproject.toml uv.lock ./

# Create venv and install deps
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv sync --frozen --no-dev --active

# Copy application code
COPY config.toml ./
COPY src/ ./src/
COPY apps/ ./apps/
COPY data/ ./data/

EXPOSE 8501

CMD ["/opt/venv/bin/streamlit", "run", "apps/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

