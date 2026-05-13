FROM mcr.microsoft.com/devcontainers/python:3.8

WORKDIR /aiproject

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 
