# SENTIENTCITY AI - System Architecture

## Executive Overview

SENTIENTCITY AI is an autonomous multi-agent smart city intelligence system designed for real-time surveillance, threat detection, crowd management, and emergency response. The platform implements a distributed microservices architecture with edge-to-cloud intelligence orchestration.

## Architecture Principles

### Design Philosophy
- **Event-Driven Architecture**: All components communicate via asynchronous event streams
- **Edge-First Processing**: Minimize latency with edge inference capabilities
- **Plugin-Based Intelligence**: Modular, hot-swappable analysis modules
- **Explainable AI**: All decisions include human-readable explanations
- **Privacy-by-Design**: Built-in anonymization and audit capabilities

### Technical Standards
- SOLID Principles
- 12-Factor App Methodology
- API-First Design with OpenAPI 3.0
- Typed Python with runtime validation

## System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  React Dashboard │ Mobile Apps │ Alert Systems │ APIs       │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                        API GATEWAY LAYER                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Gateway │ Auth │ Rate Limiting │ Load Balancing    │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     INTELLIGENCE LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Risk Engine │ Explainability │ LLM Integration │ Plugins   │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     PROCESSING LAYER                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Tracking │ Pose │ Behavior │ Event Streaming │ Memory      │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     INFERENCE LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Edge Inference │ YOLOv26 │ Crowd Density │ Anomaly Models  │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     DATA LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL │ Redis │ Kafka │ S3/MinIO │ Vector DB          │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                     INFRASTRUCTURE LAYER                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Kubernetes │ Docker │ GPU Runtime │ Edge Devices │ Cloud   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Services

### 1. Edge Inference Service
- **Purpose**: Real-time video frame processing at edge devices
- **Technology**: Python, ONNX Runtime, TensorRT
- **Scaling**: Horizontal per camera cluster
- **Latency Target**: <50ms per frame

### 2. Tracking Service
- **Purpose**: Multi-object tracking and re-identification across cameras
- **Technology**: Python, DeepSORT/ByteTrack, Feature stores
- **Features**: Cross-camera ReID, trajectory prediction

### 3. Pose Service
- **Purpose**: Human pose estimation and skeleton tracking
- **Technology**: Python, MediaPipe/OpenPose integration

### 4. Event Streaming Service
- **Purpose**: Distributed event bus for all system events
- **Technology**: Apache Kafka, Python aiokafka
- **Topics**: detections, tracks, alerts, metrics, audit

### 5. Intelligence Engine
- **Purpose**: Orchestrates plugin-based analysis modules
- **Technology**: Python, Plugin architecture

### 6. Memory Service
- **Purpose**: Temporal context and state management
- **Technology**: Redis, PostgreSQL, Vector embeddings

### 7. Explainability Service
- **Purpose**: Generate human-readable incident explanations
- **Technology**: Python, LLM integration

### 8. Backend API Gateway
- **Purpose**: Unified API for all client applications
- **Technology**: FastAPI, OAuth2, Rate limiting

### 9. Dashboard Service
- **Purpose**: Serve frontend and handle UI-specific APIs
- **Technology**: React, Tailwind CSS

### 10. Experiment Tracking Service
- **Purpose**: ML experiment management
- **Technology**: MLflow-compatible

### 11. Model Registry Service
- **Purpose**: Model versioning and deployment management

## Data Stores

- **PostgreSQL**: Persistent data, incidents, configs
- **Redis**: Real-time state, caching
- **Kafka**: Event streaming
- **MinIO/S3**: Models, videos, artifacts
- **Vector DB**: Embeddings, similarity search

## Security Architecture

- OAuth2/OIDC authentication
- JWT tokens with short expiry
- Role-Based Access Control (RBAC)
- Service-to-service mTLS
- Face anonymization pipeline
- Audit logging for all data access

## Performance Targets

- Frame Inference Latency: <50ms
- End-to-End Alert Latency: <500ms
- Tracking Accuracy (MOTA): >75%
- System Availability: 99.9%
- Concurrent Cameras: 1000+ per cluster
