import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Monitor, Settings, Maximize2, Grid3x3 } from 'lucide-react'
import { clsx } from '../../utils/clsx'
import OperationsPanel from '../operations/OperationsPanel'
import CommandGrid from '../command/CommandGrid'
import IntelligenceStream from '../intelligence/IntelligenceStream'
import SystemTelemetryStrip from '../telemetry/SystemTelemetryStrip'
import { useWebSocketSimulation } from '../../hooks/useWebSocketSimulation'

interface CommandWallLayoutProps {
  layoutMode?: 'standard' | 'command-wall'
}

const CommandWallLayout: React.FC<CommandWallLayoutProps> = ({ 
  layoutMode = 'command-wall' 
}) => {
  const { state, sendMessage, connected } = useWebSocketSimulation()
  const [activeMonitor, setActiveMonitor] = useState<'left' | 'center' | 'right'>('center')
  const [showLabels, setShowLabels] = useState(true)

  if (layoutMode === 'standard') {
    // Fallback to original layout
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-gray-100 overflow-hidden">
        <div className="h-16 bg-gray-900/90 backdrop-blur-md border-b border-cyan-500/20 flex items-center justify-between px-6">
          <div className="text-cyan-400 text-lg font-mono">SENTIENTCITY AI - Standard Mode</div>
        </div>
        <div className="flex h-[calc(100vh-8rem)]">
          <div className="w-80 bg-gray-900/40 backdrop-blur-sm border-r border-cyan-500/20">
            <OperationsPanel 
              cameras={state.cameras}
              autonomousMode={state.autonomousMode}
              manualOverride={state.manualOverride}
              plugins={state.plugins}
            />
          </div>
          <div className="flex-1 bg-gray-900/20 backdrop-blur-sm">
            <CommandGrid
              cameras={state.cameras}
              alerts={state.alerts}
              selectedCamera={state.selectedCamera}
              intelligence={state.intelligence}
            />
          </div>
          <div className="w-96 bg-gray-900/40 backdrop-blur-sm border-l border-cyan-500/20">
            <IntelligenceStream
              alerts={state.alerts}
              intelligence={state.intelligence}
              aiBrain={state.aiBrain}
            />
          </div>
        </div>
        <div className="h-16 bg-gray-900/60 backdrop-blur-md border-t border-cyan-500/20">
          <SystemTelemetryStrip telemetry={state.telemetry} connected={connected} />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-gray-100 overflow-hidden relative">
      {/* Command Room Header */}
      <div className="h-8 bg-gray-900/95 backdrop-blur-sm border-b border-cyan-500/30 flex items-center justify-between px-4 z-50">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Grid3x3 className="text-cyan-400 w-4 h-4" />
            <span className="text-cyan-400 text-sm font-mono uppercase tracking-wider">
              Command Wall Mode
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-green-400 text-xs font-mono">OPERATIONAL</span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowLabels(!showLabels)}
            className="text-gray-400 hover:text-cyan-400 transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
          
          <div className="text-gray-500 text-xs font-mono">
            {new Date().toLocaleString()}
          </div>
        </div>
      </div>

      {/* Multi-Monitor Layout */}
      <div className="flex h-[calc(100vh-4rem)]">
        {/* LEFT MONITOR */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          className="relative"
        >
          {/* Monitor Frame */}
          <div className="absolute inset-0 bg-gray-800 border-4 border-gray-700 rounded-lg overflow-hidden">
            {/* Monitor Bezel */}
            <div className="absolute top-0 left-0 right-0 h-8 bg-gray-800 border-b border-gray-600 flex items-center justify-between px-3">
              <div className="flex items-center space-x-2">
                <Monitor className="text-gray-400 w-3 h-3" />
                {showLabels && (
                  <span className="text-gray-400 text-xs font-mono">LEFT - OPERATIONS</span>
                )}
              </div>
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-red-500 rounded-full" />
                <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                <div className="w-2 h-2 bg-green-500 rounded-full" />
              </div>
            </div>
            
            {/* Monitor Content */}
            <div className="pt-8 h-full bg-gray-900/95">
              <div className="h-full overflow-hidden">
                <OperationsPanel 
                  cameras={state.cameras}
                  autonomousMode={state.autonomousMode}
                  manualOverride={state.manualOverride}
                  plugins={state.plugins}
                />
              </div>
            </div>
          </div>

          {/* Monitor Shadow */}
          <div className="absolute -bottom-2 left-2 right-2 h-4 bg-black/20 blur-xl rounded-full" />
        </motion.div>

        {/* CENTER MONITOR */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex-1 relative mx-4"
        >
          {/* Monitor Frame */}
          <div className="absolute inset-0 bg-gray-800 border-4 border-gray-700 rounded-lg overflow-hidden">
            {/* Monitor Bezel */}
            <div className="absolute top-0 left-0 right-0 h-8 bg-gray-800 border-b border-gray-600 flex items-center justify-between px-3">
              <div className="flex items-center space-x-2">
                <Monitor className="text-gray-400 w-3 h-3" />
                {showLabels && (
                  <span className="text-gray-400 text-xs font-mono">CENTER - MAIN SURVEILLANCE</span>
                )}
              </div>
              <div className="flex items-center space-x-2">
                <button className="text-gray-400 hover:text-cyan-400 transition-colors">
                  <Maximize2 className="w-3 h-3" />
                </button>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                </div>
              </div>
            </div>
            
            {/* Monitor Content */}
            <div className="pt-8 h-full bg-gray-900/95">
              <div className="h-full">
                <CommandGrid
                  cameras={state.cameras}
                  alerts={state.alerts}
                  selectedCamera={state.selectedCamera}
                  intelligence={state.intelligence}
                />
              </div>
            </div>
          </div>

          {/* Monitor Shadow */}
          <div className="absolute -bottom-2 left-2 right-2 h-4 bg-black/20 blur-xl rounded-full" />
        </motion.div>

        {/* RIGHT MONITOR */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="relative"
        >
          {/* Monitor Frame */}
          <div className="absolute inset-0 bg-gray-800 border-4 border-gray-700 rounded-lg overflow-hidden">
            {/* Monitor Bezel */}
            <div className="absolute top-0 left-0 right-0 h-8 bg-gray-800 border-b border-gray-600 flex items-center justify-between px-3">
              <div className="flex items-center space-x-2">
                <Monitor className="text-gray-400 w-3 h-3" />
                {showLabels && (
                  <span className="text-gray-400 text-xs font-mono">RIGHT - INTELLIGENCE</span>
                )}
              </div>
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-red-500 rounded-full" />
                <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                <div className="w-2 h-2 bg-green-500 rounded-full" />
              </div>
            </div>
            
            {/* Monitor Content */}
            <div className="pt-8 h-full bg-gray-900/95">
              <div className="h-full overflow-hidden">
                <IntelligenceStream
                  alerts={state.alerts}
                  intelligence={state.intelligence}
                  aiBrain={state.aiBrain}
                />
              </div>
            </div>
          </div>

          {/* Monitor Shadow */}
          <div className="absolute -bottom-2 left-2 right-2 h-4 bg-black/20 blur-xl rounded-full" />
        </motion.div>
      </div>

      {/* Bottom Telemetry Strip */}
      <div className="absolute bottom-0 left-0 right-0 h-16 bg-gray-900/95 backdrop-blur-sm border-t border-cyan-500/30 z-40">
        <SystemTelemetryStrip telemetry={state.telemetry} connected={connected} />
      </div>

      {/* Ambient Lighting Effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-1/3 h-full bg-gradient-to-r from-cyan-900/10 to-transparent" />
        <div className="absolute top-0 right-0 w-1/3 h-full bg-gradient-to-l from-blue-900/10 to-transparent" />
      </div>

      {/* Command Room Grid Overlay */}
      {showLabels && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-8 left-0 bottom-16 w-px bg-gray-800/30" />
          <div className="absolute top-8 right-0 bottom-16 w-px bg-gray-800/30" />
          <div className="absolute top-8 left-0 right-0 h-px bg-gray-800/30" />
          <div className="absolute bottom-16 left-0 right-0 h-px bg-gray-800/30" />
        </div>
      )}
    </div>
  )
}

export default CommandWallLayout
