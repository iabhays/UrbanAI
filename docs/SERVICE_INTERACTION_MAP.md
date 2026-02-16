# SENTIENTCITY AI - Service Interaction Map

## Service Communication Patterns

### Synchronous Communication (REST/gRPC)
- Dashboard → API Gateway → Services
- Services → Model Registry (model loading)
- Services → Memory Service (state queries)

### Asynchronous Communication (Kafka)
- Edge Inference → Event Streaming → All consumers
- Tracking Service → Event Streaming → Intelligence Engine
- Intelligence Engine → Event Streaming → Explainability Service

## Service Topology

```
                                    ┌─────────────────┐
                                    │   Dashboard     │
                                    │   (React/TS)    │
                                    └────────┬────────┘
                                             │ HTTP/WS
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API GATEWAY (FastAPI)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │   Auth   │ │  Rate    │ │  Route   │ │  Cache   │ │  Audit   │     │
│  │ Handler  │ │ Limiter  │ │  Logic   │ │  Layer   │ │  Logger  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  Intelligence │       │   Tracking    │       │    Memory     │
│    Engine     │◄─────►│   Service     │◄─────►│   Service     │
└───────┬───────┘       └───────┬───────┘       └───────────────┘
        │                       │                       ▲
        │                       │                       │
        ▼                       ▼                       │
┌───────────────┐       ┌───────────────┐              │
│ Explainability│       │     Pose      │              │
│   Service     │       │   Service     │              │
└───────────────┘       └───────────────┘              │
                                                       │
┌─────────────────────────────────────────────────────────────────────────┐
│                       EVENT STREAMING (Kafka)                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │ detections │ │   tracks   │ │   alerts   │ │   audit    │          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Edge Inference│       │  Experiment   │       │    Model      │
│   Service     │       │   Tracking    │       │   Registry    │
└───────────────┘       └───────────────┘       └───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          VIDEO SOURCES                                 │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│    │ RTSP    │  │  RTMP   │  │  HTTP   │  │  File   │  │  IoT    │  │
│    │ Streams │  │ Streams │  │ Images  │  │ Batch   │  │ Sensors │  │
│    └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

## Data Flow Patterns

### Pattern 1: Real-time Detection Flow
```
Camera → Edge Inference → Kafka[detections] → Tracking → Kafka[tracks] → Intelligence
```

### Pattern 2: Alert Generation Flow
```
Intelligence → Risk Engine → Kafka[alerts] → Explainability → Dashboard
```

### Pattern 3: Query/Response Flow
```
Dashboard → API Gateway → Memory Service → Response
```

### Pattern 4: Model Update Flow
```
MLOps Pipeline → Model Registry → Edge Inference (hot reload)
```

## Kafka Topic Schema

### Topic: `sentient.detections`
```json
{
  "camera_id": "string",
  "frame_id": "int64",
  "timestamp": "iso8601",
  "detections": [
    {
      "class_id": "int",
      "confidence": "float",
      "bbox": [x1, y1, x2, y2],
      "embedding": "float[]"
    }
  ]
}
```

### Topic: `sentient.tracks`
```json
{
  "camera_id": "string",
  "track_id": "string",
  "global_id": "string",
  "trajectory": [[x, y, t], ...],
  "state": "active|lost|confirmed",
  "features": "float[]"
}
```

### Topic: `sentient.alerts`
```json
{
  "alert_id": "uuid",
  "severity": "low|medium|high|critical",
  "type": "string",
  "timestamp": "iso8601",
  "location": {"camera_id": "string", "zone": "string"},
  "evidence": {"frame_ids": [], "track_ids": []},
  "risk_score": "float",
  "explanation": "string"
}
```

### Topic: `sentient.metrics`
```json
{
  "service": "string",
  "timestamp": "iso8601",
  "metrics": {
    "latency_ms": "float",
    "throughput_fps": "float",
    "memory_mb": "float",
    "gpu_util": "float"
  }
}
```

## Service Dependencies

### Edge Inference Service
- **Requires**: Model Registry, Kafka
- **Produces**: detections topic
- **Consumes**: model updates

### Tracking Service
- **Requires**: Kafka, Redis, Memory Service
- **Produces**: tracks topic
- **Consumes**: detections topic

### Intelligence Engine
- **Requires**: Kafka, Memory Service, Plugin System
- **Produces**: alerts topic
- **Consumes**: tracks, detections topics

### Explainability Service
- **Requires**: Kafka, LLM API, Memory Service
- **Produces**: explained alerts
- **Consumes**: alerts topic

### API Gateway
- **Requires**: All services (routing)
- **Produces**: HTTP responses, WebSocket streams
- **Consumes**: Service APIs

## Plugin Integration Points

```
┌────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE ENGINE                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Plugin Orchestrator                      │  │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │  │
│  │   │ Crowd   │ │ Anomaly │ │ Defense │ │ Traffic │   │  │
│  │   │Predict  │ │ Detect  │ │ Monitor │ │ Analysis│   │  │
│  │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │  │
│  │        │           │           │           │         │  │
│  │        └───────────┴───────────┴───────────┘         │  │
│  │                        │                              │  │
│  │                        ▼                              │  │
│  │              ┌─────────────────┐                     │  │
│  │              │   Risk Engine   │                     │  │
│  │              └─────────────────┘                     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Health Check Endpoints

All services expose:
- `GET /health` - Basic health
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /metrics` - Prometheus metrics
