# SENTIENTCITY AI Deployment Guide

## Production Deployment Guide

This guide covers deployment of SENTIENTCITY AI in production environments.

---

## Prerequisites

### Hardware Requirements

**Edge Deployment:**
- GPU: NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- CPU: 8+ cores
- RAM: 16GB+ (32GB recommended)
- Storage: 100GB+ SSD

**Cloud Deployment:**
- Kubernetes cluster (optional)
- Load balancer
- Persistent storage

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.10+
- CUDA 11.8+ (for GPU)
- NVIDIA Container Toolkit (for GPU)

---

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd SENTIENTCITY
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure System

Edit `configs/config.yaml`:
- Set camera sources
- Configure Kafka/Redis endpoints
- Set model paths
- Configure detection thresholds

### 4. Start Services

**Using Docker Compose:**
```bash
docker-compose up -d
```

**Manual Start:**
```bash
# Terminal 1: Backend API
python scripts/run_api.py

# Terminal 2: Pipeline
python scripts/run_pipeline.py --camera <camera_source>

# Terminal 3: Dashboard
cd dashboard/react_ui && npm install && npm run dev
```

---

## Docker Deployment

### Build Images

```bash
# Build backend
docker build -f deployment/docker/Dockerfile.backend -t sentientcity-backend .

# Build dashboard
docker build -f deployment/docker/Dockerfile.dashboard -t sentientcity-dashboard ./dashboard
```

### Run with Docker Compose

```bash
docker-compose up -d
```

### Verify Deployment

```bash
# Check services
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f pipeline

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Edge Deployment

### Single Node Deployment

Deploy all services on a single edge device:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure for edge
cp deployment/edge_config/edge_config.yaml configs/

# Start services
systemctl start sentientcity-pipeline
systemctl start sentientcity-api
```

### Distributed Edge Deployment

Deploy services across multiple edge nodes:

1. **Edge Node 1** (Inference):
   - Edge AI layer
   - Tracking engine
   - Pose extraction

2. **Edge Node 2** (Intelligence):
   - Intelligence layer
   - Risk engine
   - Memory engine

3. **Central Node** (Coordination):
   - Backend API
   - Dashboard
   - Kafka/Redis

---

## Cloud Deployment

### Kubernetes Deployment

**1. Create Namespace:**
```bash
kubectl create namespace sentientcity
```

**2. Deploy ConfigMap:**
```bash
kubectl create configmap sentientcity-config \
  --from-file=configs/config.yaml \
  -n sentientcity
```

**3. Deploy Services:**
```bash
kubectl apply -f deployment/cloud_config/kubernetes/
```

### AWS Deployment

**Using ECS:**
```bash
# Build and push images
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag sentientcity-backend:latest <account>.dkr.ecr.<region>.amazonaws.com/sentientcity-backend:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/sentientcity-backend:latest

# Deploy using ECS task definitions
aws ecs create-service --cluster sentientcity --service-name backend --task-definition sentientcity-backend
```

---

## Configuration

### Environment Variables

Set environment variables for configuration:

```bash
export SENTIENT_EDGE_AI__MODEL__DEVICE=cuda:0
export SENTIENT_STREAMING__KAFKA__BOOTSTRAP_SERVERS=kafka:9092
export SENTIENT_BACKEND__AUTHENTICATION__JWT_SECRET=<your-secret>
```

### Configuration Files

- `configs/config.yaml`: Main configuration
- `configs/yolov26_config.yaml`: YOLOv26 model configuration
- `deployment/edge_config/`: Edge-specific configs
- `deployment/cloud_config/`: Cloud-specific configs

---

## Security Setup

### 1. Generate JWT Secret

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to `configs/config.yaml`:
```yaml
backend:
  authentication:
    jwt_secret: "<generated-secret>"
```

### 2. Set Encryption Key

```bash
export ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
```

### 3. Configure RBAC

Edit `configs/config.yaml` to configure roles and permissions.

### 4. Enable HTTPS

Use reverse proxy (nginx/traefik) with SSL certificates.

---

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Pipeline health (check logs)
docker-compose logs pipeline | grep -i health
```

### Metrics Collection

Metrics are collected via Prometheus client:
- System metrics: CPU, memory, GPU
- Application metrics: Request rate, latency
- Model metrics: Inference time, accuracy

### Logging

Logs are written to:
- `logs/sentient_city.log`: Application logs
- `logs/audit.log`: Security audit logs

---

## Scaling

### Horizontal Scaling

**Backend API:**
```bash
# Scale API instances
docker-compose up -d --scale backend=4
```

**Pipeline Workers:**
```bash
# Run multiple pipeline instances
python scripts/run_pipeline.py --camera <camera1> &
python scripts/run_pipeline.py --camera <camera2> &
```

### Vertical Scaling

- **GPU**: Use TensorRT optimization
- **Memory**: Increase Redis cache size
- **CPU**: Increase worker processes

---

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Kafka Connection Issues

```bash
# Check Kafka
docker-compose logs kafka

# Test connection
docker exec -it sentientcity-kafka-1 kafka-topics --list --bootstrap-server localhost:9092
```

### High Memory Usage

- Reduce batch size in config
- Enable model quantization
- Reduce Redis cache TTL

---

## Backup & Recovery

### Backup Configuration

```bash
# Backup configs
tar -czf configs_backup.tar.gz configs/

# Backup models
tar -czf models_backup.tar.gz models/
```

### Backup Data

```bash
# Backup Redis
docker exec sentientcity-redis-1 redis-cli SAVE
docker cp sentientcity-redis-1:/data/dump.rdb ./backup/

# Backup FAISS indices
cp -r data/faiss_index.bin ./backup/
```

### Recovery

```bash
# Restore configs
tar -xzf configs_backup.tar.gz

# Restore models
tar -xzf models_backup.tar.gz

# Restore Redis
docker cp ./backup/dump.rdb sentientcity-redis-1:/data/
docker restart sentientcity-redis-1
```

---

## Performance Tuning

### GPU Optimization

1. **Enable TensorRT:**
   ```yaml
   deployment:
     gpu:
       tensorrt:
         enabled: true
         precision: "FP16"
   ```

2. **Model Quantization:**
   - Use INT8 quantization for faster inference
   - Trade-off: Slight accuracy reduction

### Inference Optimization

1. **Batch Processing:**
   - Increase batch size for better GPU utilization
   - Balance latency vs throughput

2. **Frame Skipping:**
   - Process every Nth frame for lower latency
   - Configure in `configs/config.yaml`

---

## Maintenance

### Model Updates

```bash
# Register new model
python -m sentient_city.mlops.model_registry register \
  --model-id yolov26 \
  --model-path models/yolov26_v2.pt \
  --version v2

# Set as production
python -m sentient_city.mlops.model_registry set-production \
  --model-id yolov26 \
  --version v2
```

### Log Rotation

Configure log rotation:
```yaml
# In configs/config.yaml
system:
  log_rotation: "10 MB"
  log_retention: "7 days"
```

### Database Cleanup

```bash
# Clean expired data
python -m sentient_city.memory_engine.vector_store cleanup
```

---

## Support

For issues and questions:
- Check logs: `logs/sentient_city.log`
- Review architecture: `ARCHITECTURE.md`
- Check configuration: `configs/config.yaml`

---

## Production Checklist

- [ ] Change default JWT secret
- [ ] Set encryption key
- [ ] Configure RBAC users
- [ ] Enable HTTPS
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Test failover
- [ ] Review security settings
- [ ] Load test system
- [ ] Document runbooks
