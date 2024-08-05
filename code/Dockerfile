# Stage 1 - Install build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 - Copy only necessary files to the runner stage
FROM python:3.11-slim

ARG FILENAME
ARG PORT=8000

ENV FILENAME=${FILENAME}
ENV PORT=${PORT}

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY $FILENAME .
COPY chainlit.json .
COPY static ./static

EXPOSE ${PORT}

CMD ["sh", "-c", "python -m chainlit run ${FILENAME} --port=${PORT} -w"]