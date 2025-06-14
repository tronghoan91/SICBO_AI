FROM python:3.12-slim

WORKDIR /app

# Cài thêm các thư viện hệ thống cần thiết cho build wheel
RUN apt-get update && apt-get install -y gcc build-essential

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
