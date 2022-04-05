FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

WORKDIR /app
RUN pip install --no-cache-dir torchvision Pillow numpy onnxruntime python-multipart
COPY service.py main.py
COPY best_model.onnx .
