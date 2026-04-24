FROM python:3.11-slim

WORKDIR /app

# System deps for building wheels (chromadb, sentence-transformers, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Streamlit config: disable CORS/XSRF for Cloud Run, headless mode
RUN mkdir -p ~/.streamlit && \
    echo '[server]\n\
headless = true\n\
port = 8080\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base="light"\n' > ~/.streamlit/config.toml

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "main.py"]
