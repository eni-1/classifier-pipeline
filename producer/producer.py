import os
import sys
import signal
import time
import json
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

class Config:
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'arxiv_papers')
    DATA_FILE = os.getenv('DATA_FILE', '/data/arxiv-metadata-oai-snapshot.json')
    STREAM_RATE_SECONDS = float(os.getenv('STREAM_RATE_SECONDS', '2.0'))
    MAX_PAPERS_TO_SEND = int(os.getenv('MAX_PAPERS_TO_SEND', '1000'))
    TARGET_CATEGORIES = {'cs.LG', 'cs.CV', 'cs.AI', 'cs.CL', 'cs.NE'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
shutdown_flag = False
def signal_handler(sig, frame):
    global shutdown_flag
    logger.info("stopping producer...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_kafka_producer(broker_url):
    for attempt in range(5):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker_url],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info("Kafka Producer connected.")
            return producer
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka (attempt {attempt+1}/5): {e}")
            time.sleep(5 * (attempt + 1))
    logger.critical("Could not connect to Kafka. Exiting.")
    sys.exit(1)

def on_send_success(record_metadata):
    logger.info(f"Message sent to topic '{record_metadata.topic}' partition {record_metadata.partition}")

def on_send_error(excp):
    logger.error('Error sending message', exc_info=excp)

def validate_paper(paper):
    return all(paper.get(field) for field in ['id', 'title', 'abstract'])


def stream_data(producer: KafkaProducer, config: Config):
    stats = {'total_read': 0, 'total_sent': 0, 'total_skipped': 0}
    logger.info(f"Starting, will stop after {config.MAX_PAPERS_TO_SEND} papers.")
    try:
        with open(config.DATA_FILE, 'r') as f:
            for line in f:
                if shutdown_flag or stats['total_sent'] >= config.MAX_PAPERS_TO_SEND:
                    break
                stats['total_read'] += 1
                try:
                    paper = json.loads(line)
                    if paper.get('categories', '').split(' ')[0] in config.TARGET_CATEGORIES:
                        if not validate_paper(paper):
                            stats['total_skipped'] += 1
                            continue
                        message = {
                            "id": paper.get("id"),
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract")
                        }
                        producer.send(config.KAFKA_TOPIC, value=message).add_callback(on_send_success).add_errback(on_send_error)
                        stats['total_sent'] += 1
                        time.sleep(config.STREAM_RATE_SECONDS)
                except (json.JSONDecodeError, AttributeError):
                    stats['total_skipped'] += 1
                    continue
    except FileNotFoundError:
        logger.critical(f"File not found at {config.DATA_FILE}. Exiting.")
        sys.exit(1)
    finally:
        producer.flush()
        logger.info(f"Producer shutting down. Final: {stats}")

if __name__ == "__main__":
    kafka_producer = create_kafka_producer(Config.KAFKA_BROKER)
    stream_data(kafka_producer, Config)
    kafka_producer.close()
    logger.info("Producer done.")