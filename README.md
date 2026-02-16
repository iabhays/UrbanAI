# SENTIENTCITY AI — Multi-Agent Smart City Intelligence Platform

## 🏗️ Architecture Overview

SENTIENTCITY AI is a production-grade, research-level multi-agent intelligence platform designed for smart city operations. The system provides real-time monitoring, analysis, and prediction capabilities across multiple domains including crowd management, surveillance, defense, disaster detection, and traffic safety.

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD LAYER                          │
│              React + Tailwind CSS Frontend                   │
│         Real-time feeds, alerts, analytics, playback         │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket / REST API
┌──────────────────────▼──────────────────────────────────────┐
│                    BACKEND API LAYER                        │
│              FastAPI Microservices                          │
│    REST APIs | WebSocket Streaming | Authentication         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Event Streaming
┌──────────────────────▼──────────────────────────────────────┐
│                  STREAMING LAYER                            │
│         Kafka Event Bus | Redis Cache                       │
│         Async Event Routing | Pub/Sub                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌───▼──────────────┐
│ INTELLIGENCE │ │  EXPLAIN   │ │  MEMORY ENGINE   │
│    LAYER     │ │     AI     │ │   (FAISS/Vector) │
│              │ │   LAYER    │ │                  │
│ Transformers │ │ LLM Reason │ │ Behavioral Hist  │
│ LSTM Memory  │ │ Summarize  │ │ Identity Embed   │
│ Risk Engine  │ │ Explain    │ │                  │
└───────┬──────┘ └────────────┘ └──────────────────┘
        │
        │ Processed Events
┌───────▼────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                        │
│  Edge AI → Tracking → Pose → Behavior → Intelligence        │
└───────┬────────────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│                    EDGE AI LAYER                            │
│              YOLOv26 Research Variant                       │
│  Multi-task Detection | Crowd Density | Behavior Embedding  │
└───────┬────────────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│              VIDEO INPUT SOURCES                            │
│         RTSP | Webcam | File | Network Streams              │
└────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Edge AI Layer (`edge_ai/`)
- **YOLOv26 Research Variant**: Multi-head detection architecture
- Real-time object detection (persons, vehicles, anomalies)
- Crowd density estimation
- Behavioral embedding extraction
- Pose-aware detection fusion
- Temporal feature buffering

#### 2. Tracking Engine (`tracking_engine/`)
- OC-SORT / DeepSORT implementation
- Person Re-Identification (ReID)
- Persistent ID management
- Multi-camera tracking support
- Trajectory analysis

#### 3. Pose Extraction (`pose_extraction/`)
- MediaPipe / OpenPose integration
- Skeleton keypoint extraction
- Fall detection algorithms
- Panic movement detection
- Activity classification

#### 4. Streaming Layer (`streaming/`)
- Kafka-based event streaming
- Redis caching layer
- Async event routing
- Pub/Sub messaging
- Event serialization

#### 5. Intelligence Layer (`intelligence/`)
- Transformer-based temporal behavior models
- LSTM memory networks
- Crowd crush risk prediction
- Suspicious behavior detection
- Anomaly detection engine
- Accident probability estimation

#### 6. Memory Engine (`memory_engine/`)
- FAISS vector database abstraction
- Behavioral history storage
- Identity embedding management
- Similarity search
- Temporal pattern storage

#### 7. Explainable AI (`explainability/`)
- LLM-based reasoning module
- Natural language alert generation
- Incident summarization
- Risk explanation
- Decision transparency

#### 8. Backend API (`backend_api/`)
- FastAPI microservice architecture
- WebSocket real-time streaming
- REST API endpoints
- Authentication & authorization
- Health checks & monitoring

#### 9. Dashboard (`dashboard/`)
- React + Tailwind CSS
- Live camera feed viewer
- Risk heatmap visualization
- Alert timeline
- Incident playback
- Analytics charts

#### 10. Deployment (`deployment/`)
- Docker Compose configuration
- Edge deployment configs
- GPU inference support
- TensorRT optimization placeholders
- Kubernetes manifests (optional)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for inference)
- Docker & Docker Compose (optional)
- Node.js 18+ (for dashboard)
- Kafka & Redis (or use Docker Compose)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd SENTIENTCITY

# Install Python dependencies
pip install -r requirements.txt

# Install dashboard dependencies
cd dashboard/react_ui
npm install
cd ../..

# Start services with Docker Compose
docker-compose up -d

# Or run manually:
# Terminal 1: Backend API
python scripts/run_api.py

# Terminal 2: Processing Pipeline
python scripts/run_pipeline.py --camera <camera_source>

# Terminal 3: Dashboard
cd dashboard/react_ui && npm run dev
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Run tests
pytest tests/ -v
```

### Configuration

Edit `configs/config.yaml` to configure:
- Model paths
- Kafka/Redis endpoints
- Camera sources
- Detection thresholds
- Risk parameters

## 📊 Capabilities

### 1. Crowd Crush Prediction
- Real-time crowd density analysis
- Movement pattern recognition
- Risk scoring algorithm
- Early warning system

### 2. Surveillance Intelligence
- Multi-camera tracking
- Person re-identification
- Behavior pattern analysis
- Anomaly detection

### 3. Defense Perimeter Monitoring
- Intrusion detection
- Unauthorized access alerts
- Perimeter breach analysis
- Threat assessment

### 4. Disaster Detection
- Fire/smoke detection
- Structural anomaly detection
- Environmental hazard recognition
- Emergency response triggers

### 5. Smart Traffic Accident Detection
- Vehicle collision detection
- Traffic flow analysis
- Accident probability estimation
- Emergency response coordination

### 6. Behavior Understanding
- Activity recognition
- Suspicious behavior detection
- Crowd dynamics analysis
- Temporal pattern learning

### 7. Explainable AI Incident Reporting
- Natural language incident reports
- Risk explanation
- Decision transparency
- Audit trail generation

## 🧪 Research & Development

The `research_experiments/` directory contains:
- Model training scripts
- Experiment configurations
- Evaluation metrics
- Ablation studies
- Dataset loaders

## 📁 Project Structure

```
sentient_city/
├── edge_ai/              # YOLOv26 detection models
├── tracking_engine/      # OC-SORT/DeepSORT tracking
├── pose_extraction/      # MediaPipe pose detection
├── streaming/            # Kafka/Redis streaming
├── intelligence/         # AI reasoning engines
├── memory_engine/        # Vector database interface
├── explainability/       # LLM explanation module
├── backend_api/          # FastAPI backend
├── dashboard/            # React frontend
├── deployment/           # Docker/K8s configs
├── research_experiments/ # Training & experiments
├── configs/              # Configuration files
└── tests/                # Unit & integration tests
```

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black .
flake8 .
mypy .
```

### Training Models
```bash
python research_experiments/train_yolov26.py --config configs/yolov26_config.yaml
```

## 📝 License

[Specify License]

## 🤝 Contributing

[Contributing Guidelines]

## 📧 Contact

[Contact Information]
