FROM python:3.11.9-slim-bullseye

ENV DEEPGRAM_API_KEY=""

# Install Deepgram STT
WORKDIR /usr/src
COPY requirements.txt .

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        -r requirements.txt \
    && apt-get purge -y --auto-remove \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY run.sh ./
COPY app ./app

EXPOSE 10000

ENTRYPOINT ["python", "-m", "app"]