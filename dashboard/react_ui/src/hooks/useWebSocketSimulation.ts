import { useState, useEffect, useCallback, useRef } from 'react'
import { WebSocketMessage, CommandCenterState, SystemState, Camera, Alert, SystemTelemetry, IntelligenceFeed } from '../types'

const generateMockSystemState = (): SystemState => ({
  operational: ['ONLINE', 'DEGRADED', 'OFFLINE', 'CRITICAL'][Math.floor(Math.random() * 4)] as SystemState['operational'],
  threatLevel: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][Math.floor(Math.random() * 4)] as SystemState['threatLevel'],
  activeIncidents: Math.floor(Math.random() * 10),
  aiBrainActivity: Math.random() * 100,
  gpuUtilization: Math.random() * 100,
  deploymentMode: ['EDGE', 'CLOUD', 'HYBRID'][Math.floor(Math.random() * 3)] as SystemState['deploymentMode'],
  systemTime: new Date(),
  emergencyOverride: Math.random() > 0.9
})

const generateMockCameras = (): Camera[] => {
  const locations = ['Main Entrance', 'Parking Lot A', 'Corridor B', 'Server Room', 'Emergency Exit', 'Roof Access']
  return Array.from({ length: 6 }, (_, i) => ({
    id: `CAM-${String(i + 1).padStart(3, '0')}`,
    name: locations[i],
    location: `Zone ${i + 1}`,
    status: 'ACTIVE',
    fps: Math.floor(Math.random() * 20) + 20,
    resolution: '1920x1080',
    coordinates: { x: Math.random() * 100, y: Math.random() * 100 },
    riskScore: Math.random() * 100,
    activeTracking: Math.random() > 0.3,
    videoUrl: `/videos/video${i + 1}.mp4`
  }))
}

const generateMockAlert = (): Alert => ({
  id: `ALERT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  timestamp: new Date(),
  severity: ['INFO', 'WARNING', 'CRITICAL', 'EMERGENCY'][Math.floor(Math.random() * 4)] as Alert['severity'],
  type: ['ANOMALY', 'THREAT', 'SYSTEM', 'PERIMETER', 'BEHAVIOR'][Math.floor(Math.random() * 5)] as Alert['type'],
  title: `Security Alert ${Math.floor(Math.random() * 1000)}`,
  description: 'Detected unusual activity pattern requiring immediate attention',
  cameraId: `CAM-${String(Math.floor(Math.random() * 6) + 1).padStart(3, '0')}`,
  objectId: `OBJ-${Math.random().toString(36).substr(2, 9)}`,
  confidence: Math.random() * 100,
  aiReasoning: 'AI model detected behavioral anomaly based on movement patterns and historical data',
  resolved: false
})

const generateMockTelemetry = (): SystemTelemetry => ({
  fps: Math.floor(Math.random() * 20) + 20,
  gpuUsage: Math.random() * 100,
  memoryUsage: Math.random() * 100,
  networkThroughput: Math.random() * 1000,
  eventProcessingRate: Math.random() * 500,
  trackingStability: Math.random() * 100,
  pluginLatency: Math.random() * 50,
  cpuUsage: Math.random() * 100,
  diskUsage: Math.random() * 100,
  temperature: Math.random() * 40 + 30
})

const generateMockIntelligence = (): IntelligenceFeed => ({
  id: `INT-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  timestamp: new Date(),
  type: ['BEHAVIOR', 'CROWD', 'THREAT', 'ANOMALY', 'PREDICTION'][Math.floor(Math.random() * 5)] as IntelligenceFeed['type'],
  classification: ['Normal', 'Suspicious', 'Aggressive', 'Anomalous', 'Unknown'][Math.floor(Math.random() * 5)],
  confidence: Math.random() * 100,
  cameraId: `CAM-${String(Math.floor(Math.random() * 6) + 1).padStart(3, '0')}`,
  region: `Region ${Math.floor(Math.random() * 10) + 1}`,
  metadata: {
    duration: Math.random() * 60,
    objects: Math.floor(Math.random() * 10),
    riskScore: Math.random() * 100
  }
})

export const useWebSocketSimulation = () => {
  const [state, setState] = useState<CommandCenterState>({
    system: generateMockSystemState(),
    cameras: generateMockCameras(),
    alerts: [],
    intelligence: [],
    telemetry: generateMockTelemetry(),
    riskHeatmap: {
      regions: [],
      timestamp: new Date()
    },
    aiBrain: {
      neurons: [],
      decisionFlow: [],
      processingLatency: 0,
      modelConfidence: 0
    },
    operatorActions: [],
    plugins: [],
    selectedCamera: null,
    autonomousMode: true,
    manualOverride: false
  })

  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const simulateMessage = useCallback(() => {
    const messageTypes = ['CAMERA_UPDATE', 'TRACKING', 'ALERT', 'TELEMETRY', 'INTELLIGENCE', 'RISK_UPDATE']
    const randomType = messageTypes[Math.floor(Math.random() * messageTypes.length)]

    setState(prevState => {
      const newState = { ...prevState }

      switch (randomType) {
        case 'CAMERA_UPDATE':
          newState.cameras = prevState.cameras.map(camera => ({
            ...camera,
            fps: Math.max(15, Math.min(45, camera.fps + (Math.random() - 0.5) * 5)),
            riskScore: Math.max(0, Math.min(100, camera.riskScore + (Math.random() - 0.5) * 10))
          }))
          break

        case 'ALERT':
          if (Math.random() > 0.7) {
            const newAlert = generateMockAlert()
            newState.alerts = [newAlert, ...prevState.alerts.slice(0, 49)]
          }
          break

        case 'TELEMETRY':
          newState.telemetry = generateMockTelemetry()
          break

        case 'INTELLIGENCE':
          if (Math.random() > 0.5) {
            const newIntelligence = generateMockIntelligence()
            newState.intelligence = [newIntelligence, ...prevState.intelligence.slice(0, 19)]
          }
          break

        case 'RISK_UPDATE':
          newState.system = {
            ...prevState.system,
            threatLevel: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][Math.floor(Math.random() * 4)] as SystemState['threatLevel'],
            activeIncidents: Math.max(0, prevState.system.activeIncidents + Math.floor((Math.random() - 0.5) * 3)),
            aiBrainActivity: Math.max(0, Math.min(100, prevState.system.aiBrainActivity + (Math.random() - 0.5) * 10))
          }
          break
      }

      return newState
    })
  }, [])

  useEffect(() => {
    intervalRef.current = setInterval(simulateMessage, 500)
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [simulateMessage])

  const sendMessage = useCallback((message: WebSocketMessage) => {
    console.log('WebSocket message sent:', message)
  }, [])

  return {
    state,
    sendMessage,
    connected: true
  }
}
