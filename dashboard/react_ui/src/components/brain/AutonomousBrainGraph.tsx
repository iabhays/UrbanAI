import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Activity, Zap, Target, Eye, Cpu, AlertTriangle, CheckCircle } from 'lucide-react'
import { clsx } from '../../utils/clsx'

interface BrainNode {
  id: string
  name: string
  type: 'input' | 'processing' | 'analysis' | 'output'
  status: 'active' | 'processing' | 'idle' | 'error'
  confidence: number
  latency: number
  throughput: number
  position: { x: number; y: number }
  description: string
}

interface BrainConnection {
  from: string
  to: string
  strength: number
  dataFlow: number
  active: boolean
}

interface AutonomousBrainGraphProps {
  className?: string
}

const AutonomousBrainGraph: React.FC<AutonomousBrainGraphProps> = ({ className }) => {
  const [nodes, setNodes] = useState<BrainNode[]>([])
  const [connections, setConnections] = useState<BrainConnection[]>([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const canvasRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Initialize autonomous pipeline nodes
    const initialNodes: BrainNode[] = [
      {
        id: 'video-ingestion',
        name: 'Video Ingestion',
        type: 'input',
        status: 'active',
        confidence: 98.5,
        latency: 12,
        throughput: 850,
        position: { x: 10, y: 20 },
        description: 'Multi-camera video stream processing and frame extraction'
      },
      {
        id: 'detection-model',
        name: 'YOLOv8 Detection',
        type: 'processing',
        status: 'active',
        confidence: 94.2,
        latency: 45,
        throughput: 720,
        position: { x: 30, y: 20 },
        description: 'Object detection and classification using YOLOv8-L model'
      },
      {
        id: 'tracking-engine',
        name: 'DeepSORT Tracking',
        type: 'processing',
        status: 'active',
        confidence: 89.7,
        latency: 28,
        throughput: 680,
        position: { x: 50, y: 20 },
        description: 'Multi-object tracking with ReID and trajectory prediction'
      },
      {
        id: 'pose-extraction',
        name: 'Pose Extraction',
        type: 'analysis',
        status: 'processing',
        confidence: 87.3,
        latency: 35,
        throughput: 450,
        position: { x: 70, y: 20 },
        description: 'Human pose keypoint extraction and analysis'
      },
      {
        id: 'behavior-embedding',
        name: 'Behavior Embedding',
        type: 'analysis',
        status: 'active',
        confidence: 91.8,
        latency: 52,
        throughput: 380,
        position: { x: 30, y: 50 },
        description: 'Behavioral pattern analysis and embedding generation'
      },
      {
        id: 'risk-engine',
        name: 'Risk Assessment',
        type: 'analysis',
        status: 'active',
        confidence: 93.1,
        latency: 18,
        throughput: 420,
        position: { x: 50, y: 50 },
        description: 'Multi-factor risk scoring and threat level assessment'
      },
      {
        id: 'plugin-modules',
        name: 'Plugin Pipeline',
        type: 'processing',
        status: 'active',
        confidence: 96.4,
        latency: 67,
        throughput: 290,
        position: { x: 70, y: 50 },
        description: 'Custom plugin execution and result aggregation'
      },
      {
        id: 'alert-generation',
        name: 'Alert Generation',
        type: 'output',
        status: 'active',
        confidence: 95.7,
        latency: 8,
        throughput: 180,
        position: { x: 40, y: 80 },
        description: 'Intelligent alert generation and prioritization'
      }
    ]

    const initialConnections: BrainConnection[] = [
      { from: 'video-ingestion', to: 'detection-model', strength: 0.95, dataFlow: 850, active: true },
      { from: 'detection-model', to: 'tracking-engine', strength: 0.88, dataFlow: 720, active: true },
      { from: 'tracking-engine', to: 'pose-extraction', strength: 0.82, dataFlow: 680, active: true },
      { from: 'detection-model', to: 'behavior-embedding', strength: 0.91, dataFlow: 450, active: true },
      { from: 'tracking-engine', to: 'risk-engine', strength: 0.94, dataFlow: 420, active: true },
      { from: 'behavior-embedding', to: 'risk-engine', strength: 0.87, dataFlow: 380, active: true },
      { from: 'risk-engine', to: 'plugin-modules', strength: 0.79, dataFlow: 290, active: true },
      { from: 'risk-engine', to: 'alert-generation', strength: 0.93, dataFlow: 180, active: true },
      { from: 'plugin-modules', to: 'alert-generation', strength: 0.85, dataFlow: 150, active: true }
    ]

    setNodes(initialNodes)
    setConnections(initialConnections)
  }, [])

  useEffect(() => {
    // Simulate real-time node updates
    const interval = setInterval(() => {
      setNodes(prevNodes => 
        prevNodes.map(node => ({
          ...node,
          confidence: Math.max(75, Math.min(99, node.confidence + (Math.random() - 0.5) * 2)),
          latency: Math.max(5, Math.min(100, node.latency + (Math.random() - 0.5) * 5)),
          throughput: Math.max(100, Math.min(1000, node.throughput + (Math.random() - 0.5) * 50)),
          status: Math.random() > 0.95 ? 'processing' : 
                  Math.random() > 0.98 ? 'error' : 'active'
        }))
      )

      setConnections(prevConnections =>
        prevConnections.map(conn => ({
          ...conn,
          dataFlow: Math.max(50, Math.min(1000, conn.dataFlow + (Math.random() - 0.5) * 100)),
          active: Math.random() > 0.1
        }))
      )
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'input': return <Eye className="w-4 h-4" />
      case 'processing': return <Cpu className="w-4 h-4" />
      case 'analysis': return <Brain className="w-4 h-4" />
      case 'output': return <AlertTriangle className="w-4 h-4" />
      default: return <Activity className="w-4 h-4" />
    }
  }

  const getNodeColor = (status: string) => {
    switch (status) {
      case 'active': return 'border-green-400 bg-green-400/10'
      case 'processing': return 'border-yellow-400 bg-yellow-400/10'
      case 'error': return 'border-red-400 bg-red-400/10'
      default: return 'border-gray-400 bg-gray-400/10'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-3 h-3 text-green-400" />
      case 'processing': return <Activity className="w-3 h-3 text-yellow-400 animate-pulse" />
      case 'error': return <AlertTriangle className="w-3 h-3 text-red-400" />
      default: return <div className="w-3 h-3 bg-gray-400 rounded-full" />
    }
  }

  return (
    <div className={clsx('relative h-full bg-gray-900/50 rounded-lg border border-cyan-500/20 overflow-hidden', className)}>
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-800/90 backdrop-blur-sm border-b border-cyan-500/20 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="text-cyan-400 w-5 h-5" />
            <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">
              Autonomous AI Pipeline
            </h3>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 text-xs font-mono">LIVE</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <span className="text-gray-400">Nodes:</span>
              <span className="text-cyan-400 font-mono">{nodes.length}</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-gray-400">Connections:</span>
              <span className="text-cyan-400 font-mono">{connections.filter(c => c.active).length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Graph Canvas */}
      <div ref={canvasRef} className="relative h-full pt-16">
        {/* Connection Lines */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          <defs>
            <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(34, 211, 238, 0.3)" />
              <stop offset="100%" stopColor="rgba(59, 130, 246, 0.3)" />
            </linearGradient>
          </defs>
          
          {connections.map((connection, index) => {
            const fromNode = nodes.find(n => n.id === connection.from)
            const toNode = nodes.find(n => n.id === connection.to)
            
            if (!fromNode || !toNode) return null
            
            return (
              <g key={index}>
                <motion.line
                  x1={`${fromNode.position.x}%`}
                  y1={`${fromNode.position.y}%`}
                  x2={`${toNode.position.x}%`}
                  y2={`${toNode.position.y}%`}
                  stroke="url(#connectionGradient)"
                  strokeWidth={Math.max(1, connection.strength * 3)}
                  strokeOpacity={connection.active ? 0.8 : 0.2}
                  animate={{
                    strokeOpacity: connection.active ? [0.3, 0.8, 0.3] : 0.2
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
                
                {/* Data flow particles */}
                {connection.active && (
                  <motion.circle
                    r="2"
                    fill="#22d3ee"
                    animate={{
                      cx: [`${fromNode.position.x}%`, `${toNode.position.x}%`],
                      cy: [`${fromNode.position.y}%`, `${toNode.position.y}%`]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "linear",
                      delay: index * 0.2
                    }}
                  />
                )}
              </g>
            )
          })}
        </svg>

        {/* Nodes */}
        {nodes.map((node) => (
          <motion.div
            key={node.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${node.position.x}%`,
              top: `${node.position.y}%`
            }}
            whileHover={{ scale: 1.05 }}
            onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
          >
            {/* Node Container */}
            <div className={clsx(
              'relative p-3 rounded-lg border-2 cursor-pointer transition-all duration-300',
              getNodeColor(node.status),
              selectedNode === node.id ? 'ring-2 ring-cyan-400 z-20' : ''
            )}>
              {/* Node Icon */}
              <div className="flex items-center justify-center mb-2">
                {getNodeIcon(node.type)}
              </div>
              
              {/* Node Name */}
              <div className="text-xs font-mono text-center text-gray-300 mb-1">
                {node.name}
              </div>
              
              {/* Status Indicator */}
              <div className="flex items-center justify-center">
                {getStatusIcon(node.status)}
              </div>

              {/* Pulse Animation for Active Nodes */}
              {node.status === 'active' && (
                <motion.div
                  className="absolute inset-0 rounded-lg border-2 border-cyan-400"
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 0, 0.5]
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              )}
            </div>

            {/* Node Details Tooltip */}
            <AnimatePresence>
              {selectedNode === node.id && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8, y: 10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.8, y: 10 }}
                  className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 bg-gray-900 border border-cyan-500/30 rounded-lg p-3 z-30"
                >
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Status:</span>
                      <div className="flex items-center space-x-1">
                        {getStatusIcon(node.status)}
                        <span className="text-gray-300 capitalize">{node.status}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Confidence:</span>
                      <span className={clsx(
                        'font-mono',
                        node.confidence > 90 ? 'text-green-400' :
                        node.confidence > 80 ? 'text-yellow-400' :
                        'text-red-400'
                      )}>
                        {node.confidence.toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Latency:</span>
                      <span className={clsx(
                        'font-mono',
                        node.latency < 20 ? 'text-green-400' :
                        node.latency < 50 ? 'text-yellow-400' :
                        'text-red-400'
                      )}>
                        {node.latency.toFixed(0)}ms
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Throughput:</span>
                      <span className="text-cyan-400 font-mono">
                        {node.throughput.toFixed(0)}/s
                      </span>
                    </div>
                    
                    <div className="pt-2 border-t border-gray-700">
                      <p className="text-gray-300 text-xs">{node.description}</p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export default AutonomousBrainGraph
