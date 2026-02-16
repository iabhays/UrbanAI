import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import GlobalCommandBar from '../command/GlobalCommandBar'
import OperationsPanel from '../operations/OperationsPanel'
import CommandGrid from '../command/CommandGrid'
import IntelligenceStream from '../intelligence/IntelligenceStream'
import SystemTelemetryStrip from '../telemetry/SystemTelemetryStrip'
import { useWebSocketSimulation } from '../../hooks/useWebSocketSimulation'
import { clsx } from '../../utils/clsx'

const CommandCenterLayout: React.FC = () => {
  const { state, sendMessage, connected } = useWebSocketSimulation()
  const [emergencyOverride, setEmergencyOverride] = useState(false)

  const handleEmergencyOverride = () => {
    setEmergencyOverride(!emergencyOverride)
    sendMessage({
      type: 'SYSTEM',
      timestamp: new Date(),
      data: { emergencyOverride: !emergencyOverride }
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-gray-100 overflow-hidden">
      {/* Global Command Bar */}
      <GlobalCommandBar 
        system={state.system} 
        onEmergencyOverride={handleEmergencyOverride}
      />

      {/* Main Command Grid */}
      <div className="flex h-[calc(100vh-8rem)]">
        {/* Left Operations Panel */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="w-80 bg-gray-900/40 backdrop-blur-sm border-r border-cyan-500/20"
        >
          <OperationsPanel 
            cameras={state.cameras}
            autonomousMode={state.autonomousMode}
            manualOverride={state.manualOverride}
            plugins={state.plugins}
          />
        </motion.div>

        {/* Central Command Grid */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="flex-1 bg-gray-900/20 backdrop-blur-sm"
        >
          <CommandGrid
            cameras={state.cameras}
            alerts={state.alerts}
            selectedCamera={state.selectedCamera}
            intelligence={state.intelligence}
          />
        </motion.div>

        {/* Right Intelligence Stream */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="w-96 bg-gray-900/40 backdrop-blur-sm border-l border-cyan-500/20"
        >
          <IntelligenceStream
            alerts={state.alerts}
            intelligence={state.intelligence}
            aiBrain={state.aiBrain}
          />
        </motion.div>
      </div>

      {/* Bottom System Telemetry Strip */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="h-16 bg-gray-900/60 backdrop-blur-md border-t border-cyan-500/20"
      >
        <SystemTelemetryStrip 
          telemetry={state.telemetry}
          connected={connected}
        />
      </motion.div>

      {/* Connection Status Indicator */}
      <AnimatePresence>
        {!connected && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed top-20 right-4 bg-red-600/90 backdrop-blur-sm text-white px-4 py-2 rounded-lg shadow-lg z-50"
          >
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-sm font-medium">Connection Lost</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Emergency Overlay */}
      <AnimatePresence>
        {emergencyOverride && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-red-900/20 backdrop-blur-sm pointer-events-none z-40"
          >
            <div className="absolute inset-0 border-4 border-red-500/50 animate-pulse" />
            <motion.div
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
              animate={{
                scale: [1, 1.1, 1],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <div className="text-red-500 text-4xl font-bold tracking-wider">
                EMERGENCY OVERRIDE
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default CommandCenterLayout
