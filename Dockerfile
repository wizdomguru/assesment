
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY Requirements.txt ./

# Install CPU-only PyTorch first
RUN pip install --upgrade pip \
    && pip install torch==2.5.1+cpu \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r Requirements.txt

COPY app ./app
COPY Ui ./Ui

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]