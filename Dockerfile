
# Feedback AI/NLP Kit - Docker image
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# Build deps for hdbscan/umap/bertopic
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Separate layer for dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

# Copy project
COPY . /workspace

EXPOSE 8501

# Default command: run Streamlit dashboard
CMD ["bash","-lc","streamlit run app/streamlit_app.py --server.headless true --server.port 8501 --server.address 0.0.0.0"]
