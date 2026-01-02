# Real-Time NLP Pipeline

- Data Streaming: Apache Kafka pipeline for high-throughput, real-time data ingestion.
- AI / Machine Learning: Fine-tuned Transformer (SciBERT) in PyTorch for multi-class NLP classification.
- API & MLOps: Containerized FastAPI REST API (Docker) for live, low-latency model inference.
- System Orchestration: Resilient, event-driven microservices architecture orchestrated with Docker Compose.


```
producer_service   | 2026-01-02 20:17:25,957 - INFO - Message sent to topic 'arxiv_papers' partition 0
arxiv_api_service  | 2026-01-02 20:17:25 - INFO - Received prediction request for abstract of length 537.
arxiv_api_service  | 2026-01-02 20:17:26 - INFO - Prediction successful: cs.CL (Confidence: 0.9846)
arxiv_api_service  | INFO:     172.18.0.6:48458 - "POST /v1/predict HTTP/1.1" 200 OK
consumer_service   | 2026-01-02 20:17:26,104 - INFO - SUCCESS: Paper ID: 0707.3972 -> Predicted Category: cs.CL (Confidence: 0.9846)
producer_service   | 2026-01-02 20:17:26,958 - INFO - Message sent to topic 'arxiv_papers' partition 0
arxiv_api_service  | 2026-01-02 20:17:26 - INFO - Received prediction request for abstract of length 961.
arxiv_api_service  | 2026-01-02 20:17:27 - INFO - Prediction successful: cs.LG (Confidence: 0.6208)
arxiv_api_service  | INFO:     172.18.0.6:48458 - "POST /v1/predict HTTP/1.1" 200 OK
consumer_service   | 2026-01-02 20:17:27,100 - INFO - SUCCESS: Paper ID: 0707.3979 -> Predicted Category: cs.LG (Confidence: 0.6208)
```
