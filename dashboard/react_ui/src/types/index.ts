export interface SystemState {
  operational: 'ONLINE' | 'DEGRADED' | 'OFFLINE' | 'CRITICAL'
  threatLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  activeIncidents: number
  aiBrainActivity: number
  gpuUtilization: number
  deploymentMode: 'EDGE' | 'CLOUD' | 'HYBRID'
  systemTime: Date
  emergencyOverride: boolean
}

export interface Camera {
  id: string
  name: string
  location: string
  status: 'ACTIVE' | 'INACTIVE' | 'MAINTENANCE'
  fps: number
  resolution: string
  coordinates: { x: number; y: number }
  riskScore: number
  activeTracking: boolean
  videoUrl?: string
}

export interface TrackingObject {
  id: string
  bbox: { x: number; y: number; width: number; height: number }
  confidence: number
  class: string
  trackId: string
  poseKeypoints?: Array<{ x: number; y: number; confidence: number }>
  reid: string
  riskScore: number
  behavior: string
}

export interface Alert {
  id: string
  timestamp: Date
  severity: 'INFO' | 'WARNING' | 'CRITICAL' | 'EMERGENCY'
  type: 'ANOMALY' | 'THREAT' | 'SYSTEM' | 'PERIMETER' | 'BEHAVIOR'
  title: string
  description: string
  cameraId?: string
  objectId?: string
  confidence: number
  aiReasoning: string
  resolved: boolean
}

export interface IntelligenceFeed {
  id: string
  timestamp: Date
  type: 'BEHAVIOR' | 'CROWD' | 'THREAT' | 'ANOMALY' | 'PREDICTION'
  classification: string
  confidence: number
  cameraId: string
  region: string
  metadata: Record<string, any>
}

export interface SystemTelemetry {
  fps: number
  gpuUsage: number
  memoryUsage: number
  networkThroughput: number
  eventProcessingRate: number
  trackingStability: number
  pluginLatency: number
  cpuUsage: number
  diskUsage: number
  temperature: number
}

export interface RiskHeatmap {
  regions: Array<{
    id: string
    coordinates: { x: number; y: number; width: number; height: number }
    riskLevel: number
    trend: 'INCREASING' | 'DECREASING' | 'STABLE'
    factors: string[]
  }>
  timestamp: Date
}

export interface AIBrainActivity {
  neurons: Array<{
    id: string
    activation: number
    connections: string[]
    layer: string
  }>
  decisionFlow: Array<{
    id: string
    type: 'INPUT' | 'PROCESSING' | 'DECISION' | 'OUTPUT'
    data: any
    timestamp: Date
    confidence: number
  }>
  processingLatency: number
  modelConfidence: number
}

export interface OperatorAction {
  id: string
  timestamp: Date
  operator: string
  action: string
  target: string
  result: 'SUCCESS' | 'FAILED' | 'PENDING'
  metadata: Record<string, any>
}

export interface Plugin {
  id: string
  name: string
  version: string
  status: 'ACTIVE' | 'INACTIVE' | 'ERROR'
  enabled: boolean
  config: Record<string, any>
  lastExecution: Date
  latency: number
}

export interface WebSocketMessage {
  type: 'CAMERA_UPDATE' | 'TRACKING' | 'ALERT' | 'TELEMETRY' | 'INTELLIGENCE' | 'RISK_UPDATE' | 'SYSTEM'
  timestamp: Date
  data: any
}

export interface CommandCenterState {
  system: SystemState
  cameras: Camera[]
  alerts: Alert[]
  intelligence: IntelligenceFeed[]
  telemetry: SystemTelemetry
  riskHeatmap: RiskHeatmap
  aiBrain: AIBrainActivity
  operatorActions: OperatorAction[]
  plugins: Plugin[]
  selectedCamera: string | null
  autonomousMode: boolean
  manualOverride: boolean
}
