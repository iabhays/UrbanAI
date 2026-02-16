import React from 'react'
import { motion } from 'framer-motion'
import { 
  Activity, 
  AlertTriangle, 
  Cpu, 
  Shield, 
  Zap, 
  Globe, 
  Clock,
  Power,
  Wifi,
  WifiOff
} from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { SystemState } from '../../types'

interface GlobalCommandBarProps {
  system: SystemState
  onEmergencyOverride: () => void
}

const GlobalCommandBar: React.FC<GlobalCommandBarProps> = ({ system, onEmergencyOverride }) => {

  const getOperationalColor = (state: SystemState['operational']) => {
    switch (state) {
      case 'ONLINE': return 'text-green-400'
      case 'DEGRADED': return 'text-yellow-400'
      case 'OFFLINE': return 'text-red-400'
      case 'CRITICAL': return 'text-red-600 animate-pulse'
      default: return 'text-gray-400'
    }
  }

  const getThreatColor = (level: SystemState['threatLevel']) => {
    switch (level) {
      case 'LOW': return 'text-green-400'
      case 'MEDIUM': return 'text-yellow-400'
      case 'HIGH': return 'text-orange-400'
      case 'CRITICAL': return 'text-red-500 animate-pulse'
      default: return 'text-gray-400'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-16 bg-gray-900/90 backdrop-blur-md border-b border-cyan-500/20 flex items-center justify-between px-6 relative overflow-hidden"
    >
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-900/20 via-cyan-900/20 to-blue-900/20" />
      
      {/* Left Section - System Status */}
      <div className="flex items-center space-x-6 relative z-10">
        <div className="flex items-center space-x-2">
          <Activity className={clsx('w-4 h-4', getOperationalColor(system.operational))} />
          <span className={clsx('text-xs font-mono uppercase tracking-wider', getOperationalColor(system.operational))}>
            {system.operational}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <Shield className={clsx('w-4 h-4', getThreatColor(system.threatLevel))} />
          <span className={clsx('text-xs font-mono uppercase tracking-wider', getThreatColor(system.threatLevel))}>
            {system.threatLevel}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <AlertTriangle className="text-orange-400 w-4 h-4" />
          <span className="text-orange-400 text-xs font-mono">
            {system.activeIncidents.toString().padStart(2, '0')}
          </span>
        </div>
      </div>

      {/* Center Section - System Metrics */}
      <div className="flex items-center space-x-4 relative z-10">
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Cpu className="text-cyan-400 w-5 h-5" />
            <motion.div
              className="absolute -inset-1 bg-cyan-400/20 rounded-full blur-sm"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          </div>
          <span className="text-cyan-400 text-xs font-mono">
            AI: {system.aiBrainActivity.toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <Zap className="text-yellow-400 w-4 h-4" />
          <span className="text-yellow-400 text-xs font-mono">
            GPU: {system.gpuUtilization.toFixed(1)}%
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <Globe className="text-blue-400 w-4 h-4" />
          <span className="text-blue-400 text-xs font-mono uppercase">
            {system.deploymentMode}
          </span>
        </div>
      </div>

      {/* Right Section - Time and Controls */}
      <div className="flex items-center space-x-4 relative z-10">
        <div className="flex items-center space-x-2">
          <Clock className="text-gray-400 w-4 h-4" />
          <span className="text-gray-400 text-xs font-mono">
            {system.systemTime.toLocaleTimeString()}
          </span>
        </div>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onEmergencyOverride}
          className={clsx(
            'px-3 py-1 rounded text-xs font-mono uppercase tracking-wider transition-all duration-200',
            system.emergencyOverride
              ? 'bg-red-600 text-white animate-pulse'
              : 'bg-red-900/50 text-red-400 border border-red-500/30 hover:bg-red-800/50'
          )}
        >
          {system.emergencyOverride ? 'OVERRIDE ACTIVE' : 'EMERGENCY'}
        </motion.button>

        <div className="flex items-center space-x-1">
          {system.operational === 'ONLINE' ? (
            <Wifi className="text-green-400 w-4 h-4" />
          ) : (
            <WifiOff className="text-red-400 w-4 h-4" />
          )}
        </div>
      </div>

      {/* Scanning line effect */}
      <motion.div
        className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
        animate={{
          x: ['-100%', '100%']
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "linear"
        }}
      />
    </motion.div>
  )
}

export default GlobalCommandBar
