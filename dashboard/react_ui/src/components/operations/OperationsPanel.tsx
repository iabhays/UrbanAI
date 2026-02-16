import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Camera, 
  Settings, 
  Shield, 
  Sliders, 
  Filter,
  ToggleLeft,
  ToggleRight,
  Zap,
  Brain,
  Eye,
  Crosshair
} from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { Camera as CameraType, Plugin } from '../../types'

interface OperationsPanelProps {
  cameras: CameraType[]
  autonomousMode: boolean
  manualOverride: boolean
  plugins: Plugin[]
}

const OperationsPanel: React.FC<OperationsPanelProps> = ({
  cameras,
  autonomousMode,
  manualOverride,
  plugins
}) => {
  const [selectedModel, setSelectedModel] = useState('YOLOv8-L')
  const [riskThreshold, setRiskThreshold] = useState(75)
  const [trackingEnabled, setTrackingEnabled] = useState(true)

  return (
    <div className="h-full flex flex-col space-y-4 p-4 overflow-y-auto">
      {/* Camera Control Center */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-cyan-500/20 p-4"
      >
        <div className="flex items-center space-x-2 mb-4">
          <Camera className="text-cyan-400 w-4 h-4" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Camera Control</h3>
        </div>
        
        <div className="space-y-2">
          {cameras.map((camera) => (
            <motion.div
              key={camera.id}
              whileHover={{ scale: 1.02 }}
              className={clsx(
                'p-2 rounded border text-xs transition-all cursor-pointer',
                camera.status === 'ACTIVE' 
                  ? 'bg-green-900/20 border-green-500/30 text-green-400'
                  : camera.status === 'MAINTENANCE'
                  ? 'bg-yellow-900/20 border-yellow-500/30 text-yellow-400'
                  : 'bg-red-900/20 border-red-500/30 text-red-400'
              )}
            >
              <div className="flex justify-between items-center">
                <span className="font-mono">{camera.id}</span>
                <span className="text-gray-400">{camera.fps} fps</span>
              </div>
              <div className="text-gray-500 text-xs mt-1">{camera.location}</div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Model Selector */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-cyan-500/20 p-4"
      >
        <div className="flex items-center space-x-2 mb-4">
          <Brain className="text-cyan-400 w-4 h-4" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">AI Model</h3>
        </div>
        
        <select 
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full bg-gray-900/50 border border-cyan-500/20 rounded px-3 py-2 text-xs text-gray-300 focus:outline-none focus:border-cyan-400/50"
        >
          <option value="YOLOv8-L">YOLOv8-L (High Accuracy)</option>
          <option value="YOLOv8-M">YOLOv8-M (Balanced)</option>
          <option value="YOLOv8-S">YOLOv8-S (Fast)</option>
          <option value="YOLOv9-E">YOLOv9-E (Experimental)</option>
        </select>
      </motion.div>

      {/* Risk Engine Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-cyan-500/20 p-4"
      >
        <div className="flex items-center space-x-2 mb-4">
          <Shield className="text-cyan-400 w-4 h-4" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Risk Engine</h3>
        </div>
        
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Threshold</span>
              <span className="text-cyan-400">{riskThreshold}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={riskThreshold}
              onChange={(e) => setRiskThreshold(Number(e.target.value))}
              className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
          </div>
        </div>
      </motion.div>

      {/* Tracking Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-cyan-500/20 p-4"
      >
        <div className="flex items-center space-x-2 mb-4">
          <Crosshair className="text-cyan-400 w-4 h-4" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Tracking</h3>
        </div>
        
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">Auto-Tracking</span>
            <button
              onClick={() => setTrackingEnabled(!trackingEnabled)}
              className="text-cyan-400"
            >
              {trackingEnabled ? <ToggleRight className="w-5 h-5" /> : <ToggleLeft className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Autonomous Mode */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-cyan-500/20 p-4"
      >
        <div className="flex items-center space-x-2 mb-4">
          <Zap className="text-cyan-400 w-4 h-4" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Autonomous Mode</h3>
        </div>
        
        <div className={clsx(
          'px-3 py-2 rounded text-xs font-mono text-center',
          autonomousMode 
            ? 'bg-green-900/20 border border-green-500/30 text-green-400'
            : 'bg-red-900/20 border border-red-500/30 text-red-400'
        )}>
          {autonomousMode ? 'ENGAGED' : 'DISENGAGED'}
        </div>
      </motion.div>
    </div>
  )
}

export default OperationsPanel
