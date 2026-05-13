FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /aiproject

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .