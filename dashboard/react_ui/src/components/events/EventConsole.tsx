import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Terminal, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  User,
  Activity,
  Zap
} from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { OperatorAction, Alert } from '../../types'

interface EventConsoleProps {
  operatorActions: OperatorAction[]
  alerts: Alert[]
}

const EventConsole: React.FC<EventConsoleProps> = ({ operatorActions, alerts }) => {
  const [logs, setLogs] = useState<Array<{
    id: string
    timestamp: Date
    type: 'operator' | 'alert' | 'system' | 'ai'
    message: string
    details?: any
    severity: 'info' | 'warning' | 'error' | 'success'
  }>>([])
  const [isPaused, setIsPaused] = useState(false)
  const [filter, setFilter] = useState<'all' | 'operator' | 'alert' | 'system' | 'ai'>('all')
  const consoleRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isPaused) return

    const interval = setInterval(() => {
      const newLog = {
        id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date(),
        type: ['operator', 'alert', 'system', 'ai'][Math.floor(Math.random() * 4)] as any,
        message: generateMockLogMessage(),
        severity: ['info', 'warning', 'error', 'success'][Math.floor(Math.random() * 4)] as any
      }

      setLogs(prev => [newLog, ...prev].slice(0, 100))
    }, 800)

    return () => clearInterval(interval)
  }, [isPaused])

  useEffect(() => {
    // Auto-scroll to bottom
    if (consoleRef.current) {
      consoleRef.current.scrollTop = 0
    }
  }, [logs])

  const generateMockLogMessage = () => {
    const messages = [
      'AI model processed 15 objects in frame',
      'Camera CAM-003 connection restored',
      'Risk threshold exceeded in Zone 4',
      'Operator initiated manual tracking',
      'Plugin YOLOv8 execution completed',
      'System performance optimization triggered',
      'Threat classification updated',
      'Network latency spike detected',
      'Memory usage within normal parameters',
      'Emergency protocol activated',
      'Camera auto-focus recalibrated',
      'Object tracking confidence improved',
      'Security perimeter scan completed',
      'Data backup procedure initiated',
      'System health check passed'
    ]
    return messages[Math.floor(Math.random() * messages.length)]
  }

  const getLogIcon = (type: string) => {
    switch (type) {
      case 'operator': return <User className="w-3 h-3" />
      case 'alert': return <AlertTriangle className="w-3 h-3" />
      case 'system': return <Activity className="w-3 h-3" />
      case 'ai': return <Zap className="w-3 h-3" />
      default: return <Terminal className="w-3 h-3" />
    }
  }

  const getLogColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'text-red-400 border-red-500/30'
      case 'warning': return 'text-yellow-400 border-yellow-500/30'
      case 'success': return 'text-green-400 border-green-500/30'
      default: return 'text-gray-400 border-gray-500/30'
    }
  }

  const filteredLogs = filter === 'all' ? logs : logs.filter(log => log.type === filter)

  return (
    <div className="h-full flex flex-col">
      {/* Console Header */}
      <div className="bg-gray-800/50 backdrop-blur-sm border-b border-cyan-500/20 p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Terminal className="text-cyan-400 w-4 h-4" />
            <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Event Console</h3>
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsPaused(!isPaused)}
              className={clsx(
                'px-2 py-1 rounded text-xs font-mono transition-colors',
                isPaused 
                  ? 'bg-yellow-900/50 text-yellow-400 border border-yellow-500/30'
                  : 'bg-gray-900/50 text-gray-400 border border-gray-500/30 hover:bg-gray-800/50'
              )}
            >
              {isPaused ? 'PAUSED' : 'LIVE'}
            </button>
            
            <button
              onClick={() => setLogs([])}
              className="px-2 py-1 bg-gray-900/50 text-gray-400 border border-gray-500/30 rounded text-xs font-mono hover:bg-gray-800/50 transition-colors"
            >
              CLEAR
            </button>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-1">
          {[
            { id: 'all', label: 'All', count: logs.length },
            { id: 'operator', label: 'Operator', count: logs.filter(l => l.type === 'operator').length },
            { id: 'alert', label: 'Alerts', count: logs.filter(l => l.type === 'alert').length },
            { id: 'system', label: 'System', count: logs.filter(l => l.type === 'system').length },
            { id: 'ai', label: 'AI', count: logs.filter(l => l.type === 'ai').length }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilter(tab.id as any)}
              className={clsx(
                'px-3 py-1 rounded text-xs font-mono transition-all',
                filter === tab.id
                  ? 'bg-cyan-400/10 text-cyan-400 border-b border-cyan-400'
                  : 'text-gray-500 hover:text-gray-300'
              )}
            >
              {tab.label}
              {tab.count > 0 && (
                <span className="ml-1 text-xs opacity-70">({tab.count})</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Console Output */}
      <div 
        ref={consoleRef}
        className="flex-1 overflow-y-auto bg-black/50 font-mono text-xs p-3 space-y-1"
        style={{ fontFamily: 'Monaco, Consolas, "Courier New", monospace' }}
      >
        <AnimatePresence>
          {filteredLogs.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-8"
            >
              <Terminal className="text-gray-600 w-8 h-8 mx-auto mb-2" />
              <p className="text-gray-500 text-xs">No events logged</p>
            </motion.div>
          ) : (
            filteredLogs.map((log, index) => (
              <motion.div
                key={log.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
                className={clsx(
                  'flex items-start space-x-2 p-2 rounded border-l-2',
                  getLogColor(log.severity)
                )}
              >
                <div className="flex items-center space-x-2 flex-shrink-0">
                  {getLogIcon(log.type)}
                  <span className="text-gray-500">
                    {log.timestamp.toLocaleTimeString()}
                  </span>
                </div>
                
                <div className="flex-1">
                  <span className="text-gray-300">{log.message}</span>
                  {log.details && (
                    <div className="text-gray-600 text-xs mt-1">
                      {JSON.stringify(log.details, null, 2)}
                    </div>
                  )}
                </div>

                {log.severity === 'error' && (
                  <XCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
                )}
                {log.severity === 'success' && (
                  <CheckCircle className="w-3 h-3 text-green-400 flex-shrink-0" />
                )}
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Console Footer */}
      <div className="bg-gray-800/50 backdrop-blur-sm border-t border-cyan-500/20 p-2">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-4">
            <span className="text-gray-400">
              Total: {logs.length} events
            </span>
            <span className="text-gray-400">
              Filtered: {filteredLogs.length} events
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <Clock className="text-gray-500 w-3 h-3" />
            <span className="text-gray-400">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EventConsole
