FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-opencv \
    ffmpeg libsm6 libxext6 libx264-dev libopenh264-dev wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . /app

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--log-level", "critical"] 