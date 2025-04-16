FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Установим gcc и python3-dev
RUN apt-get update && \
    apt-get install -y gcc python3-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
RUN mkdir -p /app/code \
    mkdir -p /app/datasets

COPY /code/4.py /app/code
COPY /datasets/db_nl_preprocessed-edit.csv /app/datasets

EXPOSE 8061

WORKDIR /app/code

CMD ["python", "4.py"]