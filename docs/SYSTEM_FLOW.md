# UrbanAI AI - System Flow Documentation

## End-to-End Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           UrbanAI PROCESSING PIPELINE                    │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │ VIDEO INPUT │
    │ RTSP/File   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ EDGE INFERENCE  │───►│ MODEL REGISTRY  │
    │ ┌─────────────┐ │    │                 │
    │ │  YOLOv26    │ │    │ - Versioning    │
    │ │  Detection  │ │    │ - Hot reload    │
    │ │  Density    │ │    └─────────────────┘
    │ │  Embeddings │ │
    │ └─────────────┘ │
    └────────┬────────┘
             │ Kafka: detections
             ▼
    ┌─────────────────┐    ┌─────────────────┐
    │    TRACKING     │◄──►│  MEMORY SERVICE │
    │ ┌─────────────┐ │    │                 │
    │ │ ByteTrack   │ │    │ - Track history │
    │ │ DeepSORT    │ │    │ - ReID features │
    │ │ Cross-cam   │ │    │ - State cache   │
    │ └─────────────┘ │    └─────────────────┘
    └────────┬────────┘
             │ Kafka: tracks
             ▼
    ┌─────────────────┐
    │ POSE EXTRACTION │
    │ ┌─────────────┐ │
    │ │ 17-keypoint │ │
    │ │ Skeleton    │ │
    │ │ Action feat │ │
    │ └─────────────┘ │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    BEHAVIOR     │
    │   EMBEDDINGS    │
    │ ┌─────────────┐ │
    │ │ Temporal    │ │
    │ │ Transformer │ │
    │ │ Encoding    │ │
    │ └─────────────┘ │
    └────────┬────────┘
             │
             ▼
    ┌────────────────────────────────────────────────────────────┐
    │              PLUGIN INTELLIGENCE MODULES                    │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
    │  │  Crowd   │ │ Anomaly  │ │ Defense  │ │ Traffic  │      │
    │  │ Predict  │ │ Detect   │ │ Monitor  │ │ Analysis │      │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
    │       └────────────┴────────────┴────────────┘             │
    │                          │                                  │
    │                          ▼                                  │
    │                  ┌──────────────┐                          │
    │                  │ RISK ENGINE  │                          │
    │                  │ Score: 0-100 │                          │
    │                  └──────────────┘                          │
    └────────────────────────┬───────────────────────────────────┘
                             │ Kafka: alerts
                             ▼
    ┌─────────────────┐
    │ EVENT STREAMING │
    │ ┌─────────────┐ │
    │ │ Kafka       │ │
    │ │ Partitioned │ │
    │ │ Replicated  │ │
    │ └─────────────┘ │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ LLM EXPLANATION │◄──►│  CONTEXT STORE  │
    │ ┌─────────────┐ │    │                 │
    │ │ GPT/Claude  │ │    │ - Incident DB   │
    │ │ Chain-of-   │ │    │ - Evidence      │
    │ │ Thought     │ │    │ - History       │
    │ └─────────────┘ │    └─────────────────┘
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ DASHBOARD ALERT │
    │ ┌─────────────┐ │
    │ │ WebSocket   │ │
    │ │ Push notif  │ │
    │ │ Webhook     │ │
    │ └─────────────┘ │
    └─────────────────┘
```

## Detailed Flow Descriptions

### Flow 1: Frame Ingestion to Detection

```python
# Pseudo-code flow
async def frame_ingestion_flow(frame: Frame):
    # 1. Decode frame from video source
    decoded = await video_decoder.decode(frame)
    
    # 2. Preprocess for inference
    tensor = preprocessor.normalize(decoded)
    
    # 3. Run YOLOv26 multi-head inference
    results = await yolov26.infer(tensor)
    # Returns: detections, density_map, embeddings
    
    # 4. Publish to Kafka
    await kafka.publish("sentient.detections", {
        "camera_id": frame.camera_id,
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "detections": results.detections,
        "density": results.density_map,
        "embeddings": results.embeddings
    })
```

### Flow 2: Detection to Tracking

```python
async def tracking_flow(detection_event: DetectionEvent):
    # 1. Get existing tracks from memory
    active_tracks = await memory.get_active_tracks(
        detection_event.camera_id
    )
    
    # 2. Associate detections with tracks
    associations = tracker.associate(
        detections=detection_event.detections,
        tracks=active_tracks
    )
    
    # 3. Update track states
    updated_tracks = tracker.update(associations)
    
    # 4. Cross-camera re-identification
    for track in updated_tracks:
        global_id = await reid.match_global(track.embedding)
        track.global_id = global_id
    
    # 5. Store and publish
    await memory.update_tracks(updated_tracks)
    await kafka.publish("sentient.tracks", updated_tracks)
```

### Flow 3: Intelligence Analysis

```python
async def intelligence_flow(track_event: TrackEvent):
    # 1. Enrich with pose data
    pose_data = await pose_service.get_pose(track_event.track_id)
    
    # 2. Generate behavior embedding
    behavior = await behavior_encoder.encode(
        trajectory=track_event.trajectory,
        pose_sequence=pose_data.sequence
    )
    
    # 3. Run all active plugins
    plugin_results = []
    for plugin in plugin_manager.active_plugins:
        result = await plugin.analyze({
            "track": track_event,
            "pose": pose_data,
            "behavior": behavior,
            "context": await memory.get_context(track_event.camera_id)
        })
        plugin_results.append(result)
    
    # 4. Aggregate in risk engine
    risk_score = risk_engine.calculate(plugin_results)
    
    # 5. Generate alert if threshold exceeded
    if risk_score > config.alert_threshold:
        await generate_alert(track_event, plugin_results, risk_score)
```

### Flow 4: Alert Explanation

```python
async def explanation_flow(alert: Alert):
    # 1. Gather evidence
    evidence = await memory.get_evidence(
        frame_ids=alert.evidence.frame_ids,
        track_ids=alert.evidence.track_ids
    )
    
    # 2. Build context prompt
    context = explanation_builder.build_context(
        alert=alert,
        evidence=evidence,
        historical=await memory.get_similar_incidents(alert)
    )
    
    # 3. Generate explanation via LLM
    explanation = await llm.generate(
        prompt=context,
        max_tokens=500,
        temperature=0.3
    )
    
    # 4. Update alert with explanation
    alert.explanation = explanation
    
    # 5. Publish to dashboard
    await kafka.publish("sentient.explained_alerts", alert)
    await websocket.broadcast(alert)
```

## State Machine: Track Lifecycle

```
                    ┌─────────────┐
                    │   CREATED   │
                    └──────┬──────┘
                           │ first detection
                           ▼
                    ┌─────────────┐
              ┌────►│  TENTATIVE  │◄────┐
              │     └──────┬──────┘     │
              │            │ N matches  │
              │ lost       ▼            │ lost
              │     ┌─────────────┐     │
              └─────│  CONFIRMED  │─────┘
                    └──────┬──────┘
                           │ M frames lost
                           ▼
                    ┌─────────────┐
                    │    LOST     │
                    └──────┬──────┘
                           │ timeout
                           ▼
                    ┌─────────────┐
                    │   DELETED   │
                    └─────────────┘
```

## Plugin Execution Order

1. **Crowd Prediction** (Priority: 1)
   - Input: Density maps, trajectories
   - Output: Crowd flow vectors, crush risk

2. **Anomaly Detection** (Priority: 2)
   - Input: Behavior embeddings, historical patterns
   - Output: Anomaly scores, type classification

3. **Defense Monitoring** (Priority: 3)
   - Input: Track positions, zone definitions
   - Output: Perimeter breaches, loitering alerts

4. **Traffic Analysis** (Priority: 4)
   - Input: Vehicle tracks, intersection data
   - Output: Congestion levels, accident detection

5. **Disaster Detection** (Priority: 5)
   - Input: Visual features, sensor data
   - Output: Fire/flood/accident alerts

## Error Handling & Recovery

### Circuit Breaker Pattern
```
Service Health: CLOSED → OPEN → HALF_OPEN → CLOSED
                  │         │         │
                  │ failures │ timeout │ success
                  ▼         ▼         ▼
             reject    test request  restore
```

### Dead Letter Queue
- Failed messages → `sentient.dlq.<topic>`
- Retry policy: 3 attempts, exponential backoff
- Alert on DLQ depth threshold

### Graceful Degradation
- Edge inference continues if cloud unavailable
- Local caching for 24-hour offline operation
- Quality reduction under load (skip frames)
