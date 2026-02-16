# SENTIENTCITY AI Project Structure

## Directory Structure

```
sentient_city/
│
├── edge_ai/
│   ├── yolov26_research/          # YOLOv26 research variant
│   │   ├── __init__.py
│   │   └── yolov26.py
│   │
│   ├── pose_extraction/            # Pose detection and analysis
│   │   ├── __init__.py
│   │   ├── pose_detector.py
│   │   ├── fall_detector.py
│   │   └── panic_detector.py
│   │
│   ├── tracking_engine/            # Multi-object tracking
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   ├── ocsort.py
│   │   ├── deepsort.py
│   │   └── reid.py
│   │
│   ├── edge_inference_runner/      # Edge inference pipeline
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── video_processor.py
│   │
│   └── __init__.py
│
├── streaming/
│   ├── kafka_producer/             # Kafka producer and consumer
│   │   ├── __init__.py
│   │   ├── kafka_producer.py
│   │   └── kafka_consumer.py
│   │
│   ├── event_router/                # Event routing and caching
│   │   ├── __init__.py
│   │   ├── event_router.py
│   │   └── redis_cache.py
│   │
│   └── __init__.py
│
├── intelligence/
│   ├── behavior_models/            # Behavior analysis models
│   │   ├── __init__.py
│   │   ├── behavior_model.py
│   │   └── memory_model.py
│   │
│   ├── crowd_prediction/            # Crowd crush prediction
│   │   ├── __init__.py
│   │   └── crowd_crush_predictor.py
│   │
│   ├── risk_engine/                 # Risk assessment
│   │   ├── __init__.py
│   │   ├── risk_engine.py
│   │   └── accident_detector.py
│   │
│   ├── anomaly_detection/          # Anomaly detection
│   │   ├── __init__.py
│   │   └── anomaly_detector.py
│   │
│   └── __init__.py
│
├── memory_engine/
│   └── vector_store/                # Vector database and memory
│       ├── __init__.py
│       ├── vector_db.py
│       ├── behavioral_memory.py
│       └── identity_memory.py
│
├── explainability/
│   └── llm_reasoner/                # LLM-based explanations
│       ├── __init__.py
│       ├── llm_reasoner.py
│       ├── alert_generator.py
│       └── incident_summarizer.py
│
├── backend_api/
│   └── fastapi_server/             # FastAPI backend
│       ├── __init__.py
│       ├── main.py
│       ├── websocket_manager.py
│       └── routes/
│           ├── __init__.py
│           ├── health.py
│           ├── alerts.py
│           ├── cameras.py
│           └── analytics.py
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── config.py
│   └── logger.py
│
├── pipeline.py                     # Main processing pipeline
└── __init__.py

dashboard/
└── react_ui/                        # React dashboard
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── components/
    │       ├── Dashboard.jsx
    │       ├── AlertPanel.jsx
    │       ├── CameraView.jsx
    │       └── Analytics.jsx
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js

deployment/
├── docker/                          # Docker configurations
│   ├── Dockerfile.backend
│   └── Dockerfile.dashboard
├── edge_config/                     # Edge deployment configs
└── cloud_config/                    # Cloud deployment configs

research_experiments/                # Research and training
├── __init__.py
└── train_yolov26.py

configs/                             # Configuration files
├── config.yaml
└── yolov26_config.yaml

tests/                               # Test suite
├── __init__.py
├── test_edge_ai.py
└── test_tracking.py
```

## Module Organization

### Edge AI (`edge_ai/`)
- **yolov26_research/**: YOLOv26 model implementation
- **pose_extraction/**: MediaPipe-based pose detection
- **tracking_engine/**: OC-SORT and DeepSORT tracking
- **edge_inference_runner/**: Real-time inference pipeline

### Streaming (`streaming/`)
- **kafka_producer/**: Kafka producer and consumer
- **event_router/**: Event routing and Redis caching

### Intelligence (`intelligence/`)
- **behavior_models/**: Transformer and LSTM models
- **crowd_prediction/**: Crowd crush risk prediction
- **risk_engine/**: Unified risk assessment
- **anomaly_detection/**: Anomaly detection engine

### Memory Engine (`memory_engine/`)
- **vector_store/**: FAISS-based vector storage

### Explainability (`explainability/`)
- **llm_reasoner/**: LLM-based explanation generation

### Backend API (`backend_api/`)
- **fastapi_server/**: FastAPI application and routes

### Dashboard (`dashboard/`)
- **react_ui/**: React frontend application

### Deployment (`deployment/`)
- **docker/**: Docker configurations
- **edge_config/**: Edge deployment configs
- **cloud_config/**: Cloud deployment configs

## Import Examples

```python
# Edge AI
from sentient_city.edge_ai.yolov26_research import YOLOv26Detector
from sentient_city.edge_ai.edge_inference_runner import EdgeDetector
from sentient_city.edge_ai.pose_extraction import PoseDetector
from sentient_city.edge_ai.tracking_engine import OCSortTracker

# Streaming
from sentient_city.streaming.kafka_producer import KafkaProducer
from sentient_city.streaming.event_router import RedisCache

# Intelligence
from sentient_city.intelligence.behavior_models import BehaviorTransformer
from sentient_city.intelligence.crowd_prediction import CrowdCrushPredictor
from sentient_city.intelligence.risk_engine import RiskEngine

# Memory
from sentient_city.memory_engine.vector_store import VectorDatabase

# Explainability
from sentient_city.explainability.llm_reasoner import LLMReasoner

# Backend
from sentient_city.backend_api.fastapi_server import app
```

## Benefits of This Structure

1. **Clear Separation**: Each major component has its own directory
2. **Modularity**: Easy to extend or replace individual components
3. **Scalability**: Structure supports multiple sub-modules per component
4. **Maintainability**: Clear organization makes code easier to navigate
5. **Deployment Flexibility**: Separate deployment configs for different environments
