FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt /service/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /service/app
COPY cloud-inference/inference_api.py /service/inference_api.py

ENV PYTHONPATH=/service

CMD exec gunicorn --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 --threads 1 --timeout 300 inference_api:app
