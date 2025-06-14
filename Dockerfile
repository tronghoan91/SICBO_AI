FROM python:3.11-slim

WORKDIR /app

# Cài đầy đủ các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cài pip/setuptools/wheel/cython phiên bản mới nhất
RUN pip install --upgrade pip setuptools wheel cython

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
