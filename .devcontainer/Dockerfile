FROM mcr.microsoft.com/devcontainers/python:3.8
WORKDIR /aiproject
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY data_classes/ ./data_classes/
COPY system_classes/ ./system_classes/
COPY final_datasets/ ./final_datasets/
COPY presentation.ipynb .