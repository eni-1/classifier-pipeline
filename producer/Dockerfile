FROM python:3.11-slim
RUN apt-get update && apt-get install -y netcat-traditional
WORKDIR /app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./wait-for-kafka.sh .
COPY ./producer.py .
RUN chmod +x wait-for-kafka.sh
CMD ["/bin/sh", "./wait-for-kafka.sh", "kafka", "9092", "python", "producer.py"]