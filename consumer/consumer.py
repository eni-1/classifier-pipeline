import os
import sys
import signal
import time
import json
import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from kafka import KafkaConsumer
from kafka.errors import KafkaError

class Config:
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'arxiv_papers')
    API_URL = os.getenv('API_URL', 'http://api:8000/v1/predict')
    GROUP_ID = os.getenv('GROUP_ID', 'arxiv-classifier-group')
    API_TIMEOUT_SECONDS = int(os.getenv('API_TIMEOUT_SECONDS', '30'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
shutdown_flag = False
def signal_handler(sig, frame):
    global shutdown_flag
    logger.info("stopping consumer...")
    shutdown_flag = True
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_kafka_consumer(broker_url, topic, group_id):
    for attempt in range(5):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=[broker_url],
                auto_offset_reset='earliest',
                group_id=group_id,
                enable_auto_commit=False, # We will commit manually
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info("Consumer connected successfully.")
            return consumer
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka (attempt {attempt+1}/5): {e}")
            time.sleep(5 * (attempt + 1))
    logger.critical("Could not connect to Kafka. Exiting.")
    sys.exit(1)

def create_api_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    return session

def consume_and_process(consumer: KafkaConsumer, api_session: requests.Session, config: Config):
    stats = {'total_consumed': 0, 'successful_predictions': 0, 'failed_predictions': 0}
    logger.info("Starting to consume messages from Kafka...")
    try:
        for message in consumer:
            if shutdown_flag:
                break
            stats['total_consumed'] += 1
            paper_data = message.value
            paper_id = paper_data.get("id", "N/A")
            try:
                response = api_session.post(
                    config.API_URL,
                    json={"abstract": paper_data.get("abstract", "")},
                    timeout=config.API_TIMEOUT_SECONDS
                )
                response.raise_for_status()
                prediction = response.json()
                category = prediction.get('category')
                confidence = prediction.get('confidence', 0)
                logger.info(
                    f"SUCCESS: Paper ID: {paper_id} -> "
                    f"Predicted Category: {category} (Confidence: {confidence:.4f})"
                )
                stats['successful_predictions'] += 1
                consumer.commit()
            except requests.exceptions.RequestException as e:
                stats['failed_predictions'] += 1
                logger.error(f"API request failed for paper ID {paper_id}: {e}")
            except Exception as e:
                stats['failed_predictions'] += 1
                logger.error(f"An unexpected error occurred for paper ID {paper_id}: {e}")
    finally:
        logger.info(f"Consumer shutting down. Final: {stats}")

if __name__ == "__main__":
    kafka_consumer = create_kafka_consumer(Config.KAFKA_BROKER, Config.KAFKA_TOPIC, Config.GROUP_ID)
    api_session = create_api_session()
    consume_and_process(kafka_consumer, api_session, Config)
    kafka_consumer.close()
    logger.info("Consumer done.")