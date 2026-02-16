import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Activity, Zap, Network } from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { AIBrainActivity } from '../../types'

interface AIBrainVisualizationProps {
  brainActivity: AIBrainActivity
}

const AIBrainVisualization: React.FC<AIBrainVisualizationProps> = ({ brainActivity }) => {
  const [neuralConnections, setNeuralConnections] = useState<Array<{from: number, to: number, strength: number}>>([])

  useEffect(() => {
    // Generate random neural connections for visualization
    const connections = Array.from({ length: 12 }, (_, i) => ({
      from: Math.floor(Math.random() * 9),
      to: Math.floor(Math.random() * 9),
      strength: Math.random()
    }))
    setNeuralConnections(connections)
  }, [])

  const neurons = Array.from({ length: 9 }, (_, i) => ({
    id: i,
    activation: Math.random() * 100,
    layer: i < 3 ? 'input' : i < 6 ? 'hidden' : 'output'
  }))

  return (
    <div className="space-y-6">
      {/* Neural Network Visualization */}
      <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-6">
        <div className="flex items-center space-x-2 mb-6">
          <Network className="text-cyan-400 w-5 h-5" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Neural Network</h3>
        </div>
        
        <div className="relative h-64">
          {/* Neural Layers */}
          <div className="absolute inset-0 flex justify-between items-center px-8">
            {/* Input Layer */}
            <div className="flex flex-col space-y-8">
              {neurons.slice(0, 3).map((neuron, index) => (
                <motion.div
                  key={`input-${index}`}
                  className="relative"
                  animate={{
                    scale: [1, 1 + neuron.activation / 200, 1],
                    opacity: [0.5 + neuron.activation / 200, 1, 0.5 + neuron.activation / 200]
                  }}
                  transition={{
                    duration: 2 + Math.random() * 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <div className="w-8 h-8 bg-green-500/20 border border-green-400/50 rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full" />
                  </div>
                  <span className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 text-xs text-gray-500 font-mono">
                    I{index + 1}
                  </span>
                </motion.div>
              ))}
            </div>

            {/* Hidden Layer */}
            <div className="flex flex-col space-y-8">
              {neurons.slice(3, 6).map((neuron, index) => (
                <motion.div
                  key={`hidden-${index}`}
                  className="relative"
                  animate={{
                    scale: [1, 1 + neuron.activation / 200, 1],
                    opacity: [0.5 + neuron.activation / 200, 1, 0.5 + neuron.activation / 200]
                  }}
                  transition={{
                    duration: 2 + Math.random() * 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <div className="w-8 h-8 bg-yellow-500/20 border border-yellow-400/50 rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full" />
                  </div>
                  <span className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 text-xs text-gray-500 font-mono">
                    H{index + 1}
                  </span>
                </motion.div>
              ))}
            </div>

            {/* Output Layer */}
            <div className="flex flex-col space-y-8">
              {neurons.slice(6, 9).map((neuron, index) => (
                <motion.div
                  key={`output-${index}`}
                  className="relative"
                  animate={{
                    scale: [1, 1 + neuron.activation / 200, 1],
                    opacity: [0.5 + neuron.activation / 200, 1, 0.5 + neuron.activation / 200]
                  }}
                  transition={{
                    duration: 2 + Math.random() * 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <div className="w-8 h-8 bg-blue-500/20 border border-blue-400/50 rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-blue-400 rounded-full" />
                  </div>
                  <span className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 text-xs text-gray-500 font-mono">
                    O{index + 1}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Neural Connections */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {neuralConnections.map((connection, index) => {
              const fromNeuron = neurons[connection.from]
              const toNeuron = neurons[connection.to]
              const layerPositions = [80, 200, 320] // X positions for layers
              
              const fromX = layerPositions[fromNeuron.layer === 'input' ? 0 : fromNeuron.layer === 'hidden' ? 1 : 2]
              const toX = layerPositions[toNeuron.layer === 'input' ? 0 : toNeuron.layer === 'hidden' ? 1 : 2]
              
              const fromY = 40 + (connection.from % 3) * 80
              const toY = 40 + (connection.to % 3) * 80

              return (
                <motion.line
                  key={index}
                  x1={fromX}
                  y1={fromY}
                  x2={toX}
                  y2={toY}
                  stroke="rgba(34, 211, 238, 0.3)"
                  strokeWidth={connection.strength * 2}
                  animate={{
                    opacity: [0.1, connection.strength, 0.1]
                  }}
                  transition={{
                    duration: 3 + Math.random() * 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              )
            })}
          </svg>
        </div>

        {/* Layer Labels */}
        <div className="flex justify-between mt-4 px-8">
          <span className="text-green-400 text-xs font-mono">INPUT</span>
          <span className="text-yellow-400 text-xs font-mono">HIDDEN</span>
          <span className="text-blue-400 text-xs font-mono">OUTPUT</span>
        </div>
      </div>

      {/* Brain Activity Metrics */}
      <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-4">
        <div className="flex items-center space-x-2 mb-4">
          <Activity className="text-cyan-400 w-5 h-5" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Activity Metrics</h3>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Processing Latency</span>
              <span className={clsx(
                'font-mono',
                brainActivity.processingLatency < 20 ? 'text-green-400' :
                brainActivity.processingLatency < 35 ? 'text-yellow-400' :
                'text-red-400'
              )}>
                {brainActivity.processingLatency.toFixed(1)}ms
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <motion.div
                className={clsx(
                  'h-full',
                  brainActivity.processingLatency < 20 ? 'bg-green-400' :
                  brainActivity.processingLatency < 35 ? 'bg-yellow-400' :
                  'bg-red-400'
                )}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, (brainActivity.processingLatency / 50) * 100)}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Model Confidence</span>
              <span className={clsx(
                'font-mono',
                brainActivity.modelConfidence > 80 ? 'text-green-400' :
                brainActivity.modelConfidence > 60 ? 'text-yellow-400' :
                'text-red-400'
              )}>
                {brainActivity.modelConfidence.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <motion.div
                className={clsx(
                  'h-full',
                  brainActivity.modelConfidence > 80 ? 'bg-green-400' :
                  brainActivity.modelConfidence > 60 ? 'bg-yellow-400' :
                  'bg-red-400'
                )}
                initial={{ width: 0 }}
                animate={{ width: `${brainActivity.modelConfidence}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Decision Flow Visualization */}
      <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-4">
        <div className="flex items-center space-x-2 mb-4">
          <Zap className="text-cyan-400 w-5 h-5" />
          <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Decision Pipeline</h3>
        </div>
        
        <div className="space-y-3">
          {[
            { stage: 'Data Input', status: 'active', progress: 100 },
            { stage: 'Feature Extraction', status: 'active', progress: 85 },
            { stage: 'Pattern Recognition', status: 'active', progress: 70 },
            { stage: 'Threat Assessment', status: 'processing', progress: 45 },
            { stage: 'Decision Output', status: 'pending', progress: 20 }
          ].map((item, index) => (
            <motion.div
              key={item.stage}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center space-x-3"
            >
              <div className={clsx(
                'w-3 h-3 rounded-full',
                item.status === 'active' ? 'bg-green-400 animate-pulse' :
                item.status === 'processing' ? 'bg-yellow-400 animate-pulse' :
                'bg-gray-600'
              )} />
              <span className="text-gray-300 text-xs font-mono flex-1">{item.stage}</span>
              <div className="w-24 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                <motion.div
                  className={clsx(
                    'h-full',
                    item.status === 'active' ? 'bg-green-400' :
                    item.status === 'processing' ? 'bg-yellow-400' :
                    'bg-gray-600'
                  )}
                  initial={{ width: 0 }}
                  animate={{ width: `${item.progress}%` }}
                  transition={{ duration: 1, delay: index * 0.2 }}
                />
              </div>
              <span className="text-gray-500 text-xs w-10 text-right">{item.progress}%</span>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default AIBrainVisualization
