# Quick Start Guide

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)
- CUDA-capable GPU (recommended for inference)

## Installation

### 1. Clone and Setup

```bash
cd SENTIENTCITY
pip install -r requirements.txt
```

### 2. Configure

Edit `configs/config.yaml` to configure:
- Camera sources
- Model paths
- Kafka/Redis endpoints
- Detection thresholds

### 3. Start Services (Docker)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Start Services (Local)

#### Terminal 1: Backend API
```bash
python -m sentient_city.backend_api.main
```

#### Terminal 2: Dashboard
```bash
cd dashboard
npm install
npm run dev
```

#### Terminal 3: Processing Pipeline
```bash
python -m sentient_city.pipeline
```

## Access

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=sentient_city
```

## Development

```bash
# Format code
black sentient_city/

# Lint code
flake8 sentient_city/

# Type check
mypy sentient_city/
```

## Project Structure

```
SENTIENTCITY/
├── sentient_city/          # Main Python package
│   ├── edge_ai/           # Detection models
│   ├── tracking_engine/   # Multi-object tracking
│   ├── pose_extraction/   # Pose analysis
│   ├── streaming/         # Kafka/Redis
│   ├── intelligence/      # AI reasoning
│   ├── memory_engine/     # Vector storage
│   ├── explainability/    # LLM explanations
│   ├── backend_api/       # FastAPI server
│   └── pipeline.py        # Main pipeline
├── dashboard/             # React frontend
├── configs/               # Configuration files
├── deployment/            # Docker configs
├── tests/                 # Test suite
└── research_experiments/ # Training scripts
```

## Key Components

### Edge AI
- YOLOv26 multi-head detection
- Crowd density estimation
- Behavior embedding extraction

### Tracking
- OC-SORT/DeepSORT tracking
- Person re-identification
- Multi-camera support

### Intelligence
- Transformer behavior models
- LSTM memory networks
- Crowd crush prediction
- Anomaly detection
- Accident detection

### Explainability
- LLM-based explanations
- Natural language alerts
- Incident summarization

## Next Steps

1. **Configure Cameras**: Add your camera sources to `configs/config.yaml`
2. **Load Models**: Place model weights in `models/` directory
3. **Customize**: Modify detection thresholds and risk parameters
4. **Extend**: Add custom intelligence modules
5. **Deploy**: Use Docker Compose for production deployment

## Troubleshooting

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- Set `device: "cpu"` in config if GPU unavailable

### Kafka Connection Errors
- Ensure Kafka is running: `docker-compose ps`
- Check `KAFKA_BOOTSTRAP_SERVERS` in config

### Redis Connection Errors
- Ensure Redis is running: `docker-compose ps`
- Check `REDIS_HOST` and `REDIS_PORT` in config

### Model Loading Errors
- Ensure model weights exist in `models/` directory
- Check model paths in `configs/config.yaml`

## Support

For issues and questions:
- Check `ARCHITECTURE.md` for system design
- Review `README.md` for detailed documentation
- Examine code comments for implementation details
