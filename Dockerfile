# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8080 \
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

# Create venv + install deps from lock 
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN uv sync --frozen --no-dev

# Copy application code and data
COPY childlanguagenet ./childlanguagenet
COPY data ./data
COPY streamlit_app.py main.py ./

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
