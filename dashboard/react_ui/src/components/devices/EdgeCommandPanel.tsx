import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Camera, 
  Cpu, 
  Wifi, 
  Battery, 
  Thermometer,
  Activity,
  Zap,
  Settings,
  Play,
  Pause,
  RotateCcw,
  Download,
  Upload,
  MapPin,
  Drone,
  Server,
  Monitor
} from 'lucide-react'
import { clsx } from '../../utils/clsx'

interface EdgeDevice {
  id: string
  name: string
  type: 'camera' | 'drone' | 'server' | 'gateway'
  location: string
  status: 'online' | 'offline' | 'maintenance' | 'error'
  batteryLevel?: number
  temperature: number
  cpuUsage: number
  memoryUsage: number
  networkLatency: number
  uptime: number
  lastHeartbeat: Date
  firmwareVersion: string
  modelInference: boolean
  gpuUtilization?: number
  storageUsage: number
}

interface DroneFleet {
  id: string
  name: string
  status: 'idle' | 'patrolling' | 'investigating' | 'returning' | 'maintenance'
  batteryLevel: number
  altitude: number
  speed: number
  location: { lat: number; lng: number }
  missionType?: string
  missionProgress: number
  estimatedReturnTime?: Date
  cameraStatus: 'active' | 'inactive'
  thermalImaging: boolean
}

interface ModelDeployment {
  id: string
  modelName: string
  version: string
  targetDevices: string[]
  status: 'pending' | 'deploying' | 'completed' | 'failed'
  progress: number
  deploymentTime: Date
  fileSize: number
  performanceMetrics: {
    accuracy: number
    latency: number
    throughput: number
  }
}

interface EdgeCommandPanelProps {
  className?: string
}

const EdgeCommandPanel: React.FC<EdgeCommandPanelProps> = ({ className }) => {
  const [devices, setDevices] = useState<EdgeDevice[]>([])
  const [droneFleet, setDroneFleet] = useState<DroneFleet[]>([])
  const [modelDeployments, setModelDeployments] = useState<ModelDeployment[]>([])
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'devices' | 'drones' | 'deployments'>('devices')

  useEffect(() => {
    // Generate mock edge devices
    const mockDevices: EdgeDevice[] = [
      {
        id: 'edge-cam-001',
        name: 'Edge Camera Alpha',
        type: 'camera',
        location: 'Building A - Floor 3',
        status: 'online',
        temperature: 42.5,
        cpuUsage: 67.8,
        memoryUsage: 54.2,
        networkLatency: 12,
        uptime: 86400000,
        lastHeartbeat: new Date(),
        firmwareVersion: 'v2.4.1',
        modelInference: true,
        gpuUtilization: 78.3,
        storageUsage: 73.1
      },
      {
        id: 'edge-cam-002',
        name: 'Edge Camera Beta',
        type: 'camera',
        location: 'Building B - Roof',
        status: 'online',
        temperature: 38.2,
        cpuUsage: 45.6,
        memoryUsage: 62.1,
        networkLatency: 8,
        uptime: 172800000,
        lastHeartbeat: new Date(),
        firmwareVersion: 'v2.4.1',
        modelInference: true,
        gpuUtilization: 56.7,
        storageUsage: 68.9
      },
      {
        id: 'edge-server-001',
        name: 'Edge Server Primary',
        type: 'server',
        location: 'Data Center - Rack A',
        status: 'online',
        temperature: 55.8,
        cpuUsage: 82.4,
        memoryUsage: 76.3,
        networkLatency: 3,
        uptime: 604800000,
        lastHeartbeat: new Date(),
        firmwareVersion: 'v3.1.2',
        modelInference: true,
        gpuUtilization: 91.2,
        storageUsage: 84.5
      },
      {
        id: 'edge-gateway-001',
        name: 'Network Gateway Alpha',
        type: 'gateway',
        location: 'Network Core',
        status: 'online',
        temperature: 41.3,
        cpuUsage: 34.7,
        memoryUsage: 28.9,
        networkLatency: 1,
        uptime: 2592000000,
        lastHeartbeat: new Date(),
        firmwareVersion: 'v1.8.3',
        modelInference: false,
        storageUsage: 45.2
      },
      {
        id: 'edge-cam-003',
        name: 'Edge Camera Gamma',
        type: 'camera',
        location: 'Parking Lot - Entrance',
        status: 'maintenance',
        temperature: 35.1,
        cpuUsage: 12.3,
        memoryUsage: 18.7,
        networkLatency: 45,
        uptime: 0,
        lastHeartbeat: new Date(Date.now() - 300000),
        firmwareVersion: 'v2.3.8',
        modelInference: false,
        storageUsage: 23.4
      }
    ]

    setDevices(mockDevices)

    // Generate mock drone fleet
    const mockDrones: DroneFleet[] = [
      {
        id: 'drone-001',
        name: 'Eagle Eye',
        status: 'patrolling',
        batteryLevel: 87,
        altitude: 120,
        speed: 25,
        location: { lat: 40.7128, lng: -74.0060 },
        missionType: 'Perimeter Patrol',
        missionProgress: 65,
        cameraStatus: 'active',
        thermalImaging: true
      },
      {
        id: 'drone-002',
        name: 'Hawk',
        status: 'investigating',
        batteryLevel: 62,
        altitude: 80,
        speed: 15,
        location: { lat: 40.7135, lng: -74.0055 },
        missionType: 'Incident Response',
        missionProgress: 40,
        cameraStatus: 'active',
        thermalImaging: true
      },
      {
        id: 'drone-003',
        name: 'Falcon',
        status: 'idle',
        batteryLevel: 95,
        altitude: 0,
        speed: 0,
        location: { lat: 40.7120, lng: -74.0065 },
        missionProgress: 0,
        cameraStatus: 'inactive',
        thermalImaging: false
      }
    ]

    setDroneFleet(mockDrones)

    // Generate mock model deployments
    const mockDeployments: ModelDeployment[] = [
      {
        id: 'deploy-001',
        modelName: 'YOLOv8-L',
        version: 'v8.0.0',
        targetDevices: ['edge-cam-001', 'edge-cam-002'],
        status: 'completed',
        progress: 100,
        deploymentTime: new Date(Date.now() - 3600000),
        fileSize: 245.7,
        performanceMetrics: {
          accuracy: 94.2,
          latency: 45.3,
          throughput: 28.7
        }
      },
      {
        id: 'deploy-002',
        modelName: 'DeepSORT',
        version: 'v2.1.0',
        targetDevices: ['edge-server-001'],
        status: 'deploying',
        progress: 73,
        deploymentTime: new Date(),
        fileSize: 89.3,
        performanceMetrics: {
          accuracy: 89.7,
          latency: 28.5,
          throughput: 45.2
        }
      },
      {
        id: 'deploy-003',
        modelName: 'PoseNet',
        version: 'v1.2.0',
        targetDevices: ['edge-cam-001', 'edge-cam-002', 'edge-cam-003'],
        status: 'pending',
        progress: 0,
        deploymentTime: new Date(),
        fileSize: 156.8,
        performanceMetrics: {
          accuracy: 87.3,
          latency: 35.7,
          throughput: 18.9
        }
      }
    ]

    setModelDeployments(mockDeployments)
  }, [])

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setDevices(prevDevices => 
        prevDevices.map(device => ({
          ...device,
          cpuUsage: Math.max(0, Math.min(100, device.cpuUsage + (Math.random() - 0.5) * 10)),
          memoryUsage: Math.max(0, Math.min(100, device.memoryUsage + (Math.random() - 0.5) * 8)),
          temperature: Math.max(20, Math.min(80, device.temperature + (Math.random() - 0.5) * 3)),
          networkLatency: Math.max(1, Math.min(100, device.networkLatency + (Math.random() - 0.5) * 5)),
          gpuUtilization: device.gpuUtilization ? Math.max(0, Math.min(100, device.gpuUtilization + (Math.random() - 0.5) * 8)) : undefined,
          lastHeartbeat: device.status === 'online' ? new Date() : device.lastHeartbeat
        }))
      )

      setDroneFleet(prevDrones =>
        prevDrones.map(drone => ({
          ...drone,
          batteryLevel: drone.status !== 'idle' ? Math.max(0, drone.batteryLevel - 0.1) : drone.batteryLevel,
          missionProgress: drone.status === 'patrolling' || drone.status === 'investigating' 
            ? Math.min(100, drone.missionProgress + Math.random() * 2) 
            : drone.missionProgress
        }))
      )

      setModelDeployments(prevDeployments =>
        prevDeployments.map(deployment => ({
          ...deployment,
          progress: deployment.status === 'deploying' ? Math.min(100, deployment.progress + Math.random() * 5) : deployment.progress,
          status: deployment.status === 'deploying' && deployment.progress >= 100 ? 'completed' : deployment.status
        }))
      )
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'camera': return <Camera className="w-4 h-4" />
      case 'drone': return <Drone className="w-4 h-4" />
      case 'server': return <Server className="w-4 h-4" />
      case 'gateway': return <Wifi className="w-4 h-4" />
      default: return <Monitor className="w-4 h-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-400 border-green-500/30 bg-green-900/20'
      case 'offline': return 'text-red-400 border-red-500/30 bg-red-900/20'
      case 'maintenance': return 'text-yellow-400 border-yellow-500/30 bg-yellow-900/20'
      case 'error': return 'text-red-400 border-red-500/30 bg-red-900/20'
      default: return 'text-gray-400 border-gray-500/30 bg-gray-900/20'
    }
  }

  const getMetricColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'text-green-400'
    if (value <= thresholds.warning) return 'text-yellow-400'
    return 'text-red-400'
  }

  const handleDeviceAction = (deviceId: string, action: 'restart' | 'toggle_inference') => {
    console.log(`Device ${deviceId} action: ${action}`)
    // In a real implementation, this would send commands to the edge devices
  }

  return (
    <div className={clsx('h-full flex flex-col bg-gray-900/50 rounded-lg border border-cyan-500/20', className)}>
      {/* Header */}
      <div className="bg-gray-800/90 backdrop-blur-sm border-b border-cyan-500/20 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Server className="text-cyan-400 w-5 h-5" />
            <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">
              Edge Device Operations
            </h3>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 text-xs font-mono">ONLINE</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <span className="text-gray-400">Devices:</span>
              <span className="text-cyan-400 font-mono">{devices.filter(d => d.status === 'online').length}/{devices.length}</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-gray-400">Drones:</span>
              <span className="text-cyan-400 font-mono">{droneFleet.filter(d => d.status !== 'maintenance').length}/{droneFleet.length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-cyan-500/20">
        {[
          { id: 'devices', label: 'Edge Devices', count: devices.length },
          { id: 'drones', label: 'Drone Fleet', count: droneFleet.length },
          { id: 'deployments', label: 'Model Deployments', count: modelDeployments.length }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={clsx(
              'flex-1 flex items-center justify-center space-x-2 py-3 text-xs font-mono uppercase tracking-wider transition-all',
              activeTab === tab.id
                ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-400/10'
                : 'text-gray-500 hover:text-gray-300'
            )}
          >
            {tab.id === 'devices' && <Server className="w-4 h-4" />}
            {tab.id === 'drones' && <Drone className="w-4 h-4" />}
            {tab.id === 'deployments' && <Upload className="w-4 h-4" />}
            <span>{tab.label}</span>
            <span className={clsx(
              'px-2 py-0.5 rounded text-xs',
              activeTab === tab.id ? 'bg-cyan-400/20 text-cyan-300' : 'bg-gray-800 text-gray-400'
            )}>
              {tab.count}
            </span>
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'devices' && (
          <div className="h-full overflow-y-auto p-4">
            <div className="grid grid-cols-1 gap-3">
              {devices.map((device) => (
                <motion.div
                  key={device.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={clsx(
                    'border rounded-lg p-3 cursor-pointer transition-all',
                    selectedDevice === device.id 
                      ? 'border-cyan-400 bg-cyan-400/10' 
                      : 'border-gray-700 hover:border-gray-600'
                  )}
                  onClick={() => setSelectedDevice(device.id === selectedDevice ? null : device.id)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gray-800 border border-gray-600">
                        {getDeviceIcon(device.type)}
                      </div>
                      <div>
                        <h5 className="text-gray-200 font-mono text-sm">{device.name}</h5>
                        <div className="flex items-center space-x-2 text-xs">
                          <MapPin className="text-gray-500 w-3 h-3" />
                          <span className="text-gray-400">{device.location}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={clsx(
                        'text-xs px-2 py-1 rounded border font-mono uppercase',
                        getStatusColor(device.status)
                      )}>
                        {device.status}
                      </span>
                      {device.modelInference && (
                        <div className="flex items-center space-x-1">
                          <Cpu className="text-cyan-400 w-3 h-3" />
                          <span className="text-cyan-400 text-xs">AI</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-4 gap-3 text-xs">
                    <div>
                      <div className="flex items-center space-x-1">
                        <Cpu className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">CPU</span>
                      </div>
                      <span className={clsx('font-mono', getMetricColor(device.cpuUsage, { good: 60, warning: 80 }))}>
                        {device.cpuUsage.toFixed(1)}%
                      </span>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-1">
                        <Activity className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">MEM</span>
                      </div>
                      <span className={clsx('font-mono', getMetricColor(device.memoryUsage, { good: 70, warning: 85 }))}>
                        {device.memoryUsage.toFixed(1)}%
                      </span>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-1">
                        <Thermometer className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">TEMP</span>
                      </div>
                      <span className={clsx('font-mono', getMetricColor(device.temperature, { good: 50, warning: 65 }))}>
                        {device.temperature.toFixed(1)}°C
                      </span>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-1">
                        <Wifi className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">LAT</span>
                      </div>
                      <span className={clsx('font-mono', getMetricColor(device.networkLatency, { good: 20, warning: 50 }))}>
                        {device.networkLatency.toFixed(0)}ms
                      </span>
                    </div>
                  </div>

                  {device.gpuUtilization && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Zap className="text-gray-500 w-3 h-3" />
                          <span className="text-gray-400 text-xs">GPU Utilization</span>
                        </div>
                        <span className={clsx('text-xs font-mono', getMetricColor(device.gpuUtilization, { good: 70, warning: 85 }))}>
                          {device.gpuUtilization.toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                        <motion.div
                          className={clsx(
                            'h-full',
                            device.gpuUtilization > 85 ? 'bg-red-400' :
                            device.gpuUtilization > 70 ? 'bg-yellow-400' :
                            'bg-green-400'
                          )}
                          initial={{ width: 0 }}
                          animate={{ width: `${device.gpuUtilization}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Device Actions */}
                  {selectedDevice === device.id && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-3 pt-3 border-t border-gray-700 flex space-x-2"
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDeviceAction(device.id, 'restart')
                        }}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600/20 border border-blue-500/30 text-blue-400 rounded text-xs font-mono hover:bg-blue-600/30 transition-colors"
                      >
                        <RotateCcw className="w-3 h-3" />
                        <span>Restart</span>
                      </button>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDeviceAction(device.id, 'toggle_inference')
                        }}
                        className={clsx(
                          'flex items-center space-x-1 px-3 py-1 border rounded text-xs font-mono transition-colors',
                          device.modelInference 
                            ? 'bg-green-600/20 border-green-500/30 text-green-400 hover:bg-green-600/30'
                            : 'bg-gray-600/20 border-gray-500/30 text-gray-400 hover:bg-gray-600/30'
                        )}
                      >
                        {device.modelInference ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                        <span>{device.modelInference ? 'Stop AI' : 'Start AI'}</span>
                      </button>
                      
                      <button className="flex items-center space-x-1 px-3 py-1 bg-gray-600/20 border border-gray-500/30 text-gray-400 rounded text-xs font-mono hover:bg-gray-600/30 transition-colors">
                        <Settings className="w-3 h-3" />
                        <span>Config</span>
                      </button>
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'drones' && (
          <div className="h-full overflow-y-auto p-4">
            <div className="grid grid-cols-1 gap-3">
              {droneFleet.map((drone) => (
                <motion.div
                  key={drone.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="border border-gray-700 rounded-lg p-3 hover:border-gray-600 transition-all"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gray-800 border border-gray-600">
                        <Drone className="text-cyan-400 w-4 h-4" />
                      </div>
                      <div>
                        <h5 className="text-gray-200 font-mono text-sm">{drone.name}</h5>
                        {drone.missionType && (
                          <span className="text-gray-400 text-xs">{drone.missionType}</span>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={clsx(
                        'text-xs px-2 py-1 rounded border font-mono uppercase',
                        drone.status === 'patrolling' ? 'text-blue-400 border-blue-500/30 bg-blue-900/20' :
                        drone.status === 'investigating' ? 'text-orange-400 border-orange-500/30 bg-orange-900/20' :
                        drone.status === 'returning' ? 'text-yellow-400 border-yellow-500/30 bg-yellow-900/20' :
                        drone.status === 'idle' ? 'text-green-400 border-green-500/30 bg-green-900/20' :
                        'text-gray-400 border-gray-500/30 bg-gray-900/20'
                      )}>
                        {drone.status}
                      </span>
                    </div>
                  </div>

                  {/* Drone Metrics */}
                  <div className="grid grid-cols-4 gap-3 text-xs">
                    <div>
                      <div className="flex items-center space-x-1">
                        <Battery className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">BAT</span>
                      </div>
                      <span className={clsx('font-mono', getMetricColor(drone.batteryLevel, { good: 50, warning: 25 }))}>
                        {drone.batteryLevel.toFixed(0)}%
                      </span>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-1">
                        <Activity className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">ALT</span>
                      </div>
                      <span className="text-gray-300 font-mono">{drone.altitude.toFixed(0)}m</span>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-1">
                        <Zap className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">SPD</span>
                      </div>
                      <span className="text-gray-300 font-mono">{drone.speed.toFixed(0)}km/h</span>
                    </div>
                    
                    <div>
                      <div className="flex items-center space-x-1">
                        <Camera className="text-gray-500 w-3 h-3" />
                        <span className="text-gray-400">CAM</span>
                      </div>
                      <span className={clsx(
                        'font-mono',
                        drone.cameraStatus === 'active' ? 'text-green-400' : 'text-gray-500'
                      )}>
                        {drone.cameraStatus === 'active' ? 'ON' : 'OFF'}
                      </span>
                    </div>
                  </div>

                  {/* Mission Progress */}
                  {drone.missionProgress > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-400 text-xs">Mission Progress</span>
                        <span className="text-cyan-400 text-xs font-mono">{drone.missionProgress.toFixed(0)}%</span>
                      </div>
                      <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                        <motion.div
                          className="h-full bg-cyan-400"
                          initial={{ width: 0 }}
                          animate={{ width: `${drone.missionProgress}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Drone Actions */}
                  <div className="mt-3 pt-3 border-t border-gray-700 flex space-x-2">
                    <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600/20 border border-blue-500/30 text-blue-400 rounded text-xs font-mono hover:bg-blue-600/30 transition-colors">
                      <MapPin className="w-3 h-3" />
                      <span>Track</span>
                    </button>
                    
                    <button className="flex items-center space-x-1 px-3 py-1 bg-orange-600/20 border border-orange-500/30 text-orange-400 rounded text-xs font-mono hover:bg-orange-600/30 transition-colors">
                      <Camera className="w-3 h-3" />
                      <span>View Feed</span>
                    </button>
                    
                    <button className="flex items-center space-x-1 px-3 py-1 bg-gray-600/20 border border-gray-500/30 text-gray-400 rounded text-xs font-mono hover:bg-gray-600/30 transition-colors">
                      <Settings className="w-3 h-3" />
                      <span>Control</span>
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'deployments' && (
          <div className="h-full overflow-y-auto p-4">
            <div className="grid grid-cols-1 gap-3">
              {modelDeployments.map((deployment) => (
                <motion.div
                  key={deployment.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="border border-gray-700 rounded-lg p-3 hover:border-gray-600 transition-all"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h5 className="text-gray-200 font-mono text-sm">{deployment.modelName}</h5>
                      <div className="flex items-center space-x-2 text-xs">
                        <span className="text-gray-400">v{deployment.version}</span>
                        <span className="text-gray-500">•</span>
                        <span className="text-gray-400">{deployment.fileSize.toFixed(1)}MB</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={clsx(
                        'text-xs px-2 py-1 rounded border font-mono uppercase',
                        deployment.status === 'completed' ? 'text-green-400 border-green-500/30 bg-green-900/20' :
                        deployment.status === 'deploying' ? 'text-blue-400 border-blue-500/30 bg-blue-900/20' :
                        deployment.status === 'failed' ? 'text-red-400 border-red-500/30 bg-red-900/20' :
                        'text-gray-400 border-gray-500/30 bg-gray-900/20'
                      )}>
                        {deployment.status}
                      </span>
                    </div>
                  </div>

                  {/* Deployment Progress */}
                  {deployment.status === 'deploying' && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-400 text-xs">Deployment Progress</span>
                        <span className="text-cyan-400 text-xs font-mono">{deployment.progress.toFixed(0)}%</span>
                      </div>
                      <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                        <motion.div
                          className="h-full bg-blue-400"
                          initial={{ width: 0 }}
                          animate={{ width: `${deployment.progress}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Target Devices */}
                  <div className="mb-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Server className="text-gray-500 w-3 h-3" />
                      <span className="text-gray-400 text-xs">Target Devices ({deployment.targetDevices.length})</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {deployment.targetDevices.map((deviceId, index) => (
                        <span key={index} className="text-xs bg-gray-700 px-2 py-1 rounded text-gray-300">
                          {deviceId}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Performance Metrics */}
                  {deployment.status === 'completed' && (
                    <div className="pt-3 border-t border-gray-700">
                      <div className="grid grid-cols-3 gap-3 text-xs">
                        <div>
                          <span className="text-gray-400">Accuracy</span>
                          <span className="ml-2 text-green-400 font-mono">{deployment.performanceMetrics.accuracy.toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Latency</span>
                          <span className="ml-2 text-yellow-400 font-mono">{deployment.performanceMetrics.latency.toFixed(1)}ms</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Throughput</span>
                          <span className="ml-2 text-cyan-400 font-mono">{deployment.performanceMetrics.throughput.toFixed(1)}fps</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Deployment Actions */}
                  <div className="mt-3 pt-3 border-t border-gray-700 flex space-x-2">
                    {deployment.status === 'pending' && (
                      <button className="flex items-center space-x-1 px-3 py-1 bg-blue-600/20 border border-blue-500/30 text-blue-400 rounded text-xs font-mono hover:bg-blue-600/30 transition-colors">
                        <Upload className="w-3 h-3" />
                        <span>Deploy</span>
                      </button>
                    )}
                    
                    {deployment.status === 'completed' && (
                      <button className="flex items-center space-x-1 px-3 py-1 bg-orange-600/20 border border-orange-500/30 text-orange-400 rounded text-xs font-mono hover:bg-orange-600/30 transition-colors">
                        <RotateCcw className="w-3 h-3" />
                        <span>Redeploy</span>
                      </button>
                    )}
                    
                    <button className="flex items-center space-x-1 px-3 py-1 bg-gray-600/20 border border-gray-500/30 text-gray-400 rounded text-xs font-mono hover:bg-gray-600/30 transition-colors">
                      <Download className="w-3 h-3" />
                      <span>Logs</span>
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default EdgeCommandPanel
