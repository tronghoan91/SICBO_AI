FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc build-essential libpq-dev

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel cython
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000

CMD ["python", "main.py"]
