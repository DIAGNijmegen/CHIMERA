# ===== Base image with Python =====
FROM python:3.10-slim

# ===== Set working directory =====
WORKDIR /app

# ===== Install OS dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== Copy requirements and install Python packages =====
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ===== Copy your model code and scripts =====
COPY . .

# ===== Default command =====
CMD ["python", "scripts/test_FE.py"]

