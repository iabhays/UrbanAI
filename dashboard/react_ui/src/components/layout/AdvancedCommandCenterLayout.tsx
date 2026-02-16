import React, { useState } from 'react'
import { motion } from 'framer-motion'
import GlobalCommandBar from '../command/GlobalCommandBar'
import OperationsPanel from '../operations/OperationsPanel'
import CommandGrid from '../command/CommandGrid'
import IntelligenceStream from '../intelligence/IntelligenceStream'
import SystemTelemetryStrip from '../telemetry/SystemTelemetryStrip'
import AutonomousBrainGraph from '../brain/AutonomousBrainGraph'
import RiskHeatmap from '../risk/RiskHeatmap'
import EventConsole from '../events/EventConsole'
import CityIntelligence3D from '../city/CityIntelligence3D'
import OperatorAIPanel from '../collaboration/OperatorAIPanel'
import LLMReasoningPanel from '../reasoning/LLMReasoningPanel'
import EdgeCommandPanel from '../devices/EdgeCommandPanel'
import CommandModulesDropdown from '../navigation'
import { useWebSocketSimulation } from '../../hooks/useWebSocketSimulation'
import { useCommandNavigation } from '../../hooks/useCommandNavigation'
import { clsx } from '../../utils/clsx'

const AdvancedCommandCenterLayout: React.FC = () => {
  const { state, sendMessage, connected } = useWebSocketSimulation()
  const [emergencyOverride, setEmergencyOverride] = useState(false)
  const { activeModule, setActiveModule } = useCommandNavigation()

  const handleEmergencyOverride = () => {
    setEmergencyOverride(!emergencyOverride)
    sendMessage({
      type: 'SYSTEM',
      timestamp: new Date(),
      data: { emergencyOverride: !emergencyOverride }
    })
  }

  const renderRightPanel = () => {
    switch (activeModule) {
      case 'intelligence':
        return <IntelligenceStream alerts={state.alerts} intelligence={state.intelligence} aiBrain={state.aiBrain} />
      case 'brain':
        return <AutonomousBrainGraph />
      case 'risk':
        return <RiskHeatmap heatmap={state.riskHeatmap} />
      case 'events':
        return <EventConsole operatorActions={state.operatorActions} alerts={state.alerts} />
      case 'city3d':
        return <CityIntelligence3D />
      case 'operator':
        return <OperatorAIPanel />
      case 'reasoning':
        return <LLMReasoningPanel />
      case 'edge':
        return <EdgeCommandPanel />
      default:
        return <IntelligenceStream alerts={state.alerts} intelligence={state.intelligence} aiBrain={state.aiBrain} />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-gray-100 overflow-hidden">
      {/* Global Command Bar */}
      <GlobalCommandBar 
        system={state.system} 
        onEmergencyOverride={handleEmergencyOverride}
      />

      {/* Main Content Area */}
      <div className="flex h-[calc(100vh-8rem)]">
        {/* Left Operations Panel */}
        <div className="w-80 bg-gray-900/40 backdrop-blur-sm border-r border-cyan-500/20">
          <OperationsPanel 
            cameras={state.cameras}
            autonomousMode={state.autonomousMode}
            manualOverride={state.manualOverride}
            plugins={state.plugins}
          />
        </div>

        {/* Main Content with Right Sidebar */}
        <div className="flex flex-1">
          {/* Central Command Grid */}
          <div className="flex-1 bg-gray-900/20 backdrop-blur-sm">
            <CommandGrid
              cameras={state.cameras}
              alerts={state.alerts}
              selectedCamera={state.selectedCamera}
              intelligence={state.intelligence}
            />
          </div>

          {/* Right Sidebar with Dropdown and Content */}
          <div className="w-96 bg-gray-900/40 backdrop-blur-sm border-l border-cyan-500/20 flex flex-col">
            {/* Navigation Dropdown */}
            <div className="p-4 border-b border-cyan-500/20">
              <CommandModulesDropdown
                activeModule={activeModule}
                onModuleChange={setActiveModule}
              />
            </div>
            
            {/* Module Content Area - Full Height */}
            <div className="flex-1 overflow-hidden">
              {renderRightPanel()}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom System Telemetry Strip */}
      <div className="h-16 bg-gray-900/60 backdrop-blur-md border-t border-cyan-500/20">
        <SystemTelemetryStrip telemetry={state.telemetry} connected={connected} />
      </div>

      {/* Emergency Overlay */}
      {emergencyOverride && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
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

      </div>
  )
}

export default AdvancedCommandCenterLayout
