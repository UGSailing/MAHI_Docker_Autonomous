FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

CMD ["python", "src/main.py"]
