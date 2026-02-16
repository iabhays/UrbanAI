import React from 'react'
import { motion } from 'framer-motion'
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Wifi, 
  Zap,
  Monitor,
  Thermometer,
  Clock
} from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { SystemTelemetry } from '../../types'

interface SystemTelemetryStripProps {
  telemetry: SystemTelemetry
  connected: boolean
}

const SystemTelemetryStrip: React.FC<SystemTelemetryStripProps> = ({
  telemetry,
  connected
}) => {
  const getMetricColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'text-green-400'
    if (value <= thresholds.warning) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getProgressBarColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'bg-green-400'
    if (value <= thresholds.warning) return 'bg-yellow-400'
    return 'bg-red-400'
  }

  return (
    <div className="h-full flex items-center px-6 space-x-8">
      {/* Connection Status */}
      <div className="flex items-center space-x-2">
        <div className={clsx(
          'w-2 h-2 rounded-full',
          connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
        )} />
        <span className={clsx(
          'text-xs font-mono uppercase',
          connected ? 'text-green-400' : 'text-red-400'
        )}>
          {connected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {/* FPS */}
      <div className="flex items-center space-x-2">
        <Monitor className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">FPS</span>
        <span className={getMetricColor(telemetry.fps, { good: 25, warning: 15 })}>
          {telemetry.fps.toFixed(0)}
        </span>
      </div>

      {/* GPU Usage */}
      <div className="flex items-center space-x-2">
        <Cpu className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">GPU</span>
        <div className="w-16 bg-gray-700 rounded-full h-2 overflow-hidden">
          <motion.div
            className={clsx('h-full', getProgressBarColor(telemetry.gpuUsage, { good: 70, warning: 85 }))}
            initial={{ width: 0 }}
            animate={{ width: `${telemetry.gpuUsage}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <span className={clsx('text-xs w-10', getMetricColor(telemetry.gpuUsage, { good: 70, warning: 85 }))}>
          {telemetry.gpuUsage.toFixed(0)}%
        </span>
      </div>

      {/* Memory Usage */}
      <div className="flex items-center space-x-2">
        <HardDrive className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">MEM</span>
        <div className="w-16 bg-gray-700 rounded-full h-2 overflow-hidden">
          <motion.div
            className={clsx('h-full', getProgressBarColor(telemetry.memoryUsage, { good: 70, warning: 85 }))}
            initial={{ width: 0 }}
            animate={{ width: `${telemetry.memoryUsage}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <span className={clsx('text-xs w-10', getMetricColor(telemetry.memoryUsage, { good: 70, warning: 85 }))}>
          {telemetry.memoryUsage.toFixed(0)}%
        </span>
      </div>

      {/* Network Throughput */}
      <div className="flex items-center space-x-2">
        <Wifi className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">NET</span>
        <span className="text-cyan-400 text-xs">
          {(telemetry.networkThroughput / 1000).toFixed(1)}K
        </span>
      </div>

      {/* Event Processing Rate */}
      <div className="flex items-center space-x-2">
        <Activity className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">EVT</span>
        <span className="text-cyan-400 text-xs">
          {telemetry.eventProcessingRate.toFixed(0)}/s
        </span>
      </div>

      {/* Tracking Stability */}
      <div className="flex items-center space-x-2">
        <Zap className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">TRK</span>
        <div className="w-16 bg-gray-700 rounded-full h-2 overflow-hidden">
          <motion.div
            className={clsx('h-full', getProgressBarColor(telemetry.trackingStability, { good: 80, warning: 60 }))}
            initial={{ width: 0 }}
            animate={{ width: `${telemetry.trackingStability}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <span className={clsx('text-xs w-10', getMetricColor(telemetry.trackingStability, { good: 80, warning: 60 }))}>
          {telemetry.trackingStability.toFixed(0)}%
        </span>
      </div>

      {/* Plugin Latency */}
      <div className="flex items-center space-x-2">
        <Clock className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">LAT</span>
        <span className={getMetricColor(telemetry.pluginLatency, { good: 20, warning: 35 })}>
          {telemetry.pluginLatency.toFixed(0)}ms
        </span>
      </div>

      {/* CPU Usage */}
      <div className="flex items-center space-x-2">
        <Cpu className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">CPU</span>
        <div className="w-16 bg-gray-700 rounded-full h-2 overflow-hidden">
          <motion.div
            className={clsx('h-full', getProgressBarColor(telemetry.cpuUsage, { good: 70, warning: 85 }))}
            initial={{ width: 0 }}
            animate={{ width: `${telemetry.cpuUsage}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <span className={clsx('text-xs w-10', getMetricColor(telemetry.cpuUsage, { good: 70, warning: 85 }))}>
          {telemetry.cpuUsage.toFixed(0)}%
        </span>
      </div>

      {/* Temperature */}
      <div className="flex items-center space-x-2">
        <Thermometer className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono w-8">TEMP</span>
        <span className={getMetricColor(telemetry.temperature, { good: 60, warning: 75 })}>
          {telemetry.temperature.toFixed(0)}°C
        </span>
      </div>

      {/* System Time */}
      <div className="flex items-center space-x-2 ml-auto">
        <Clock className="text-gray-400 w-4 h-4" />
        <span className="text-gray-400 text-xs font-mono">
          {new Date().toLocaleTimeString()}
        </span>
      </div>
    </div>
  )
}

export default SystemTelemetryStrip
