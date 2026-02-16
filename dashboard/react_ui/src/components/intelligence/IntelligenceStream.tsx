import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  AlertTriangle, 
  Activity, 
  Eye, 
  TrendingUp,
  Clock,
  Zap,
  Target
} from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { Alert, IntelligenceFeed, AIBrainActivity } from '../../types'

interface IntelligenceStreamProps {
  alerts: Alert[]
  intelligence: IntelligenceFeed[]
  aiBrain: AIBrainActivity
}

const IntelligenceStream: React.FC<IntelligenceStreamProps> = ({
  alerts,
  intelligence,
  aiBrain
}) => {
  const [activeTab, setActiveTab] = useState<'alerts' | 'intelligence' | 'brain'>('alerts')

  const criticalAlerts = alerts.filter(alert => 
    alert.severity === 'CRITICAL' || alert.severity === 'EMERGENCY'
  )

  return (
    <div className="h-full flex flex-col">
      {/* Tab Navigation */}
      <div className="flex border-b border-cyan-500/20">
        {[
          { id: 'alerts', label: 'Alerts', icon: AlertTriangle, count: alerts.length },
          { id: 'intelligence', label: 'Intelligence', icon: Brain, count: intelligence.length },
          { id: 'brain', label: 'AI Brain', icon: Activity, count: null }
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
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
            {tab.count !== null && (
              <span className={clsx(
                'px-2 py-0.5 rounded text-xs',
                activeTab === tab.id ? 'bg-cyan-400/20 text-cyan-300' : 'bg-gray-800 text-gray-400'
              )}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {activeTab === 'alerts' && (
            <motion.div
              key="alerts"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-4 space-y-2"
            >
              {alerts.length === 0 ? (
                <div className="text-center py-8">
                  <AlertTriangle className="text-gray-600 w-8 h-8 mx-auto mb-2" />
                  <p className="text-gray-500 text-xs font-mono">No Active Alerts</p>
                </div>
              ) : (
                alerts.slice(0, 20).map((alert, index) => (
                  <motion.div
                    key={alert.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={clsx(
                      'p-3 rounded-lg border text-xs',
                      alert.severity === 'EMERGENCY' ? 'bg-red-900/20 border-red-500/30' :
                      alert.severity === 'CRITICAL' ? 'bg-orange-900/20 border-orange-500/30' :
                      alert.severity === 'WARNING' ? 'bg-yellow-900/20 border-yellow-500/30' :
                      'bg-blue-900/20 border-blue-500/30'
                    )}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className={clsx(
                          'w-4 h-4',
                          alert.severity === 'EMERGENCY' ? 'text-red-400' :
                          alert.severity === 'CRITICAL' ? 'text-orange-400' :
                          alert.severity === 'WARNING' ? 'text-yellow-400' :
                          'text-blue-400'
                        )} />
                        <span className={clsx(
                          'font-mono uppercase',
                          alert.severity === 'EMERGENCY' ? 'text-red-400' :
                          alert.severity === 'CRITICAL' ? 'text-orange-400' :
                          alert.severity === 'WARNING' ? 'text-yellow-400' :
                          'text-blue-400'
                        )}>
                          {alert.severity}
                        </span>
                      </div>
                      <span className="text-gray-500 text-xs">
                        {alert.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    
                    <p className="text-gray-200 font-mono mb-1">{alert.title}</p>
                    <p className="text-gray-400 text-xs mb-2">{alert.description}</p>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">{alert.cameraId}</span>
                      <span className="text-gray-400 text-xs">
                        {alert.confidence.toFixed(1)}% confidence
                      </span>
                    </div>
                  </motion.div>
                ))
              )}
            </motion.div>
          )}

          {activeTab === 'intelligence' && (
            <motion.div
              key="intelligence"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-4 space-y-2"
            >
              {intelligence.length === 0 ? (
                <div className="text-center py-8">
                  <Brain className="text-gray-600 w-8 h-8 mx-auto mb-2" />
                  <p className="text-gray-500 text-xs font-mono">No Intelligence Data</p>
                </div>
              ) : (
                intelligence.slice(0, 20).map((item, index) => (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={clsx(
                        'text-xs font-mono uppercase',
                        item.type === 'THREAT' ? 'text-red-400' :
                        item.type === 'ANOMALY' ? 'text-yellow-400' :
                        item.type === 'BEHAVIOR' ? 'text-blue-400' :
                        item.type === 'CROWD' ? 'text-green-400' :
                        'text-purple-400'
                      )}>
                        {item.type}
                      </span>
                      <span className="text-gray-500 text-xs">
                        {item.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    
                    <p className="text-gray-200 font-mono mb-1">{item.classification}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-500 text-xs">{item.cameraId}</span>
                      <span className="text-cyan-400 text-xs">
                        {item.confidence.toFixed(1)}%
                      </span>
                    </div>
                  </motion.div>
                ))
              )}
            </motion.div>
          )}

          {activeTab === 'brain' && (
            <motion.div
              key="brain"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full p-4"
            >
              <div className="space-y-4">
                {/* AI Brain Activity Visualization */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-4">
                  <div className="flex items-center space-x-2 mb-4">
                    <Brain className="text-cyan-400 w-5 h-5" />
                    <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Neural Activity</h3>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 mb-4">
                    {Array.from({ length: 9 }, (_, i) => (
                      <motion.div
                        key={i}
                        className="h-16 bg-gradient-to-br from-cyan-900/20 to-blue-900/20 rounded border border-cyan-500/20 flex items-center justify-center"
                        animate={{
                          opacity: [0.3, 1, 0.3],
                          scale: [0.95, 1.05, 0.95]
                        }}
                        transition={{
                          duration: 2 + Math.random() * 2,
                          repeat: Infinity,
                          delay: Math.random() * 2
                        }}
                      >
                        <div className="w-2 h-2 bg-cyan-400 rounded-full" />
                      </motion.div>
                    ))}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Processing Latency</span>
                      <span className="text-cyan-400 font-mono">{aiBrain.processingLatency.toFixed(1)}ms</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Model Confidence</span>
                      <span className="text-green-400 font-mono">{aiBrain.modelConfidence.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                {/* Decision Flow */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-4">
                  <div className="flex items-center space-x-2 mb-4">
                    <Target className="text-cyan-400 w-5 h-5" />
                    <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Decision Flow</h3>
                  </div>
                  
                  <div className="space-y-2">
                    {['INPUT', 'PROCESSING', 'DECISION', 'OUTPUT'].map((stage, index) => (
                      <div key={stage} className="flex items-center space-x-3">
                        <div className={clsx(
                          'w-2 h-2 rounded-full',
                          stage === 'INPUT' ? 'bg-green-400' :
                          stage === 'PROCESSING' ? 'bg-yellow-400' :
                          stage === 'DECISION' ? 'bg-orange-400' :
                          'bg-blue-400'
                        )} />
                        <span className="text-gray-400 text-xs font-mono w-20">{stage}</span>
                        <div className="flex-1 h-1 bg-gray-700 rounded">
                          <motion.div
                            className={clsx(
                              'h-full rounded',
                              stage === 'INPUT' ? 'bg-green-400' :
                              stage === 'PROCESSING' ? 'bg-yellow-400' :
                              stage === 'DECISION' ? 'bg-orange-400' :
                              'bg-blue-400'
                            )}
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.random() * 60 + 40}%` }}
                            transition={{ duration: 1, delay: index * 0.2 }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default IntelligenceStream
