FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for geopandas/shapely/rtree
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    g++ \
    gcc \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    geos-bin \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit port
EXPOSE 8080

# Streamlit config via flags to honor Cloud Run $PORT
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless true --server.enableCORS false --server.enableXsrfProtection false"]

