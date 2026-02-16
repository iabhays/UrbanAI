# Advanced Crowd Analysis System

A comprehensive computer vision system for detecting risk in crowded environments with person detection, tracking, motion analysis, and dynamic alert management.

## 🚀 Features

### Core Capabilities
- **Person Detection**: YOLOv8-based detection optimized for crowded scenes
- **Multi-Person Tracking**: ByteTrack-inspired tracking with occlusion handling
- **Motion Analysis**: Optical flow analysis with chaos detection
- **Risk Evaluation**: Crowd-aware risk assessment with per-person scoring
- **Alert Management**: Dynamic alert triggering with cooldown and confidence scoring
- **Visualization**: Comprehensive real-time visualization with debugging capabilities

### Advanced Features
- **Adaptive Thresholds**: Dynamic confidence adjustment based on crowd density
- **False Positive Filtering**: Intelligent filtering to reduce false alerts
- **Performance Optimization**: GPU acceleration and efficient processing
- **Debug Mode**: Detailed metrics and analysis information
- **Multi-Scenario Support**: Different risk logic for small vs large crowds

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU**: NVIDIA GPU with CUDA support (recommended for real-time performance)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for models and logs

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- OpenCV 4.8+
- PyTorch 2.0+

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SENTIENTCITY
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model
```bash
# The system will automatically download yolov8n.pt on first run
# Or manually download from ultralytics repository
```

## 🎯 Quick Start

### Basic Usage
```python
from sentient_city.perception.crowd_analysis import CrowdAnalysisSystem

# Initialize system
system = CrowdAnalysisSystem()

# Process single frame
results = system.process_frame(frame)

# Process video stream
stats = system.process_video_stream("input.mp4", "output.mp4")
```

### Advanced Configuration
```python
from sentient_city.perception.crowd_analysis import (
    CrowdAnalysisSystem, SystemConfig, AlertConfig
)

# Custom configuration
config = SystemConfig(
    confidence_threshold=0.6,
    enable_alerts=True,
    debug_mode=True,
    target_fps=30
)

# Initialize with custom config
system = CrowdAnalysisSystem(config=config)
```

## 📊 System Architecture

### Component Overview

```
CrowdAnalysisSystem (Main Orchestrator)
├── PersonDetector (YOLO-based detection)
├── PersonTracker (ByteTrack-inspired tracking)
├── MotionAnalyzer (Optical flow & chaos detection)
├── RiskEvaluator (Crowd-aware risk assessment)
├── AlertManager (Dynamic alert management)
└── Visualizer (Comprehensive visualization)
```

### Data Flow
1. **Input Frame** → PersonDetector → Detections
2. **Detections** → PersonTracker → TrackedPersons
3. **Frame + TrackedPersons** → MotionAnalyzer → MotionFeatures
4. **TrackedPersons + MotionFeatures** → RiskEvaluator → RiskAssessment
5. **RiskAssessment** → AlertManager → Alerts (if needed)
6. **All Data** → Visualizer → VisualizedFrame

## 🔧 Configuration

### System Configuration
```python
config = SystemConfig(
    # Detection settings
    model_path="yolov8n.pt",
    confidence_threshold=0.5,
    max_detections=500,
    
    # Tracking settings
    track_buffer=30,
    confirmation_threshold=3,
    
    # Risk evaluation
    enable_adaptive_thresholds=True,
    risk_history_length=30,
    
    # Alert settings
    enable_alerts=True,
    alert_cooldown_seconds=30,
    
    # Visualization
    enable_visualization=True,
    debug_mode=False,
    
    # Performance
    target_fps=30,
    enable_gpu=True
)
```

### Alert Configuration
```python
alert_config = AlertConfig(
    min_confidence_threshold=0.6,
    high_risk_threshold=0.7,
    critical_risk_threshold=0.9,
    cooldown_seconds=30,
    sustained_risk_frames=5,
    enable_risk_alerts=True,
    enable_anomaly_alerts=True
)
```

## 🎮 Usage Examples

### 1. Single Frame Processing
```python
import cv2
from sentient_city.perception.crowd_analysis import CrowdAnalysisSystem

# Initialize system
system = CrowdAnalysisSystem(debug_mode=True)

# Load frame
frame = cv2.imread("crowd_scene.jpg")

# Process frame
results = system.process_frame(frame)

# Access results
print(f"Detected {len(results['detections'])} persons")
print(f"Crowd risk level: {results['crowd_risk'].risk_level}")
print(f"Active alerts: {len(results['new_alerts'])}")

# Save result
if results['visualized_frame'] is not None:
    cv2.imwrite("result.jpg", results['visualized_frame'])
```

### 2. Video Stream Processing
```python
def alert_callback(alert):
    print(f"ALERT: {alert.severity} - {alert.message}")

# Initialize with alert callback
system = CrowdAnalysisSystem(alert_callbacks=[alert_callback])

# Process video
stats = system.process_video_stream(
    video_source="input.mp4",
    output_path="output.mp4",
    show_live=True
)

print(f"Processed {stats['total_frames']} frames")
print(f"Average FPS: {stats['avg_fps']:.1f}")
```

### 3. Real-time Camera Processing
```python
# Process from camera (index 0)
stats = system.process_video_stream(
    video_source=0,
    show_live=True
)
```

### 4. Custom Risk Evaluation
```python
from sentient_city.perception.crowd_analysis import RiskEvaluator, RiskThresholds

# Custom risk thresholds
thresholds = RiskThresholds(
    large_crowd_min_people=25,
    large_crowd_speed_spike_threshold=30.0,
    small_crowd_max_people=8,
    global_risk_threshold=0.7
)

# Initialize with custom thresholds
risk_evaluator = RiskEvaluator(thresholds=thresholds)
```

## 📈 Risk Detection Logic

### Large Crowd Risk (>20 people)
- **High Risk**: Triggers when:
  - Sudden chaotic direction variance
  - Rapid speed spikes across crowd
  - Compression density exceeds threshold
  - Multiple chaotic regions detected

### Small Crowd Risk (<10 people)
- **Low Risk**: Normal behavior
- **Medium/High Risk**: Only triggers for:
  - Abnormal running behavior
  - Fighting motion patterns
  - Collision risks
  - Panic indicators

### Risk Factors
- **Speed Variance**: Changes in movement speed
- **Direction Variance**: Chaotic movement patterns
- **Compression Density**: People per unit area
- **Chaos Index**: Motion pattern irregularity
- **Collision Risk**: Proximity and relative velocity
- **Anomaly Detection**: Unusual behavior patterns

## 🎨 Visualization Features

### Display Options
- **Bounding Boxes**: Color-coded by risk level
- **Track IDs**: Unique identifiers for each person
- **Risk Levels**: Visual indicators (Green/Yellow/Red)
- **Motion Trajectories**: Historical movement paths
- **Alert Banners**: Real-time alert notifications
- **Debug Panel**: Detailed metrics and analysis

### Color Coding
- 🟢 **Green**: Low risk
- 🟡 **Yellow**: Medium risk
- 🔴 **Red**: High risk
- 🟣 **Purple**: Critical risk

## 🔍 Debug Mode

Enable debug mode for detailed analysis:
```python
system = CrowdAnalysisSystem(debug_mode=True)
```

Debug information includes:
- Crowd density metrics
- Direction variance calculations
- Per-person velocity vectors
- Risk score breakdown
- Performance metrics
- Alert decision logic

## ⚡ Performance Optimization

### GPU Acceleration
```python
config = SystemConfig(enable_gpu=True)
system = CrowdAnalysisSystem(config=config)
```

### Adaptive Processing
- Dynamic confidence thresholds based on crowd density
- Batch processing for multiple frames
- Efficient memory management
- Optimized optical flow calculations

### Performance Monitoring
```python
# Get performance metrics
performance = system.get_system_performance()
print(f"Current FPS: {performance['system']['current_fps']:.1f}")
print(f"Processing time: {performance['system']['avg_processing_time']:.3f}s")
```

## 🚨 Alert Management

### Alert Types
1. **Risk Alerts**: Based on crowd risk assessment
2. **Anomaly Alerts**: Unusual behavior detection
3. **System Alerts**: Technical issues and errors

### Alert Levels
- **Low**: Minor concerns, monitoring advised
- **Medium**: Moderate risk, attention required
- **High**: Significant risk, immediate attention
- **Critical**: Emergency situation, immediate action

### Alert Callbacks
```python
def custom_alert_handler(alert):
    # Send to monitoring system
    send_to_monitoring(alert)
    
    # Log to database
    log_alert(alert)
    
    # Send notification
    send_notification(alert.message)

system = CrowdAnalysisSystem(alert_callbacks=[custom_alert_handler])
```

## 🧪 Testing

### Run Example Script
```bash
python example_crowd_analysis.py --mode all
```

### Test Different Scenarios
```bash
# Single frame test
python example_crowd_analysis.py --mode single

# Video processing test
python example_crowd_analysis.py --mode video --video test.mp4

# Risk scenario tests
python example_crowd_analysis.py --mode scenarios
```

## 📊 Monitoring & Logging

### Logging Configuration
```python
import loguru

# Setup custom logging
logger.add("crowd_analysis.log", rotation="10 MB", level="INFO")
```

### Performance Metrics
- Frame processing rate (FPS)
- Detection accuracy metrics
- Tracking stability
- Alert frequency and accuracy
- System resource usage

## 🔧 Troubleshooting

### Common Issues

1. **Low FPS Performance**
   - Enable GPU acceleration
   - Reduce confidence threshold
   - Lower maximum detections

2. **False Alerts**
   - Adjust risk thresholds
   - Enable false positive filtering
   - Increase sustained risk frames

3. **Tracking Issues**
   - Increase track buffer size
   - Adjust IoU threshold
   - Enable re-entry detection

4. **Memory Issues**
   - Reduce history lengths
   - Clear old data periodically
   - Use batch processing

### Debug Commands
```python
# Toggle debug mode
system.toggle_debug_mode()

# Update confidence threshold
system.update_confidence_threshold(0.6)

# Get active alerts
alerts = system.get_active_alerts()

# Acknowledge alert
system.acknowledge_alert(alert_id)

# Reset system
system.reset()
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ultralytics for YOLOv8 implementation
- ByteTrack authors for tracking inspiration
- OpenCV community for computer vision tools
- Contributors to the open-source computer vision ecosystem

---

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the example scripts
- Enable debug mode for detailed analysis

**System Status**: ✅ Production Ready
**Last Updated**: 2024
**Version**: 1.0.0
