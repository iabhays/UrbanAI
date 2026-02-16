import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Map, Activity, Box, Users, MessageSquare, Server, Monitor, Zap, Film } from 'lucide-react'
import { clsx } from '../../utils/clsx'

type CommandModule = 
  | 'intelligence' 
  | 'brain' 
  | 'risk' 
  | 'events' 
  | 'city3d' 
  | 'operator' 
  | 'reasoning' 
  | 'edge'
  | 'videos'

interface CommandModuleItem {
  id: CommandModule
  label: string
  icon: React.ReactNode
  group: 'intelligence' | 'operations' | 'environment'
}

interface CommandModulesDropdownProps {
  activeModule: CommandModule
  onModuleChange: (moduleId: CommandModule) => void
}

const CommandModulesDropdown: React.FC<CommandModulesDropdownProps> = ({ 
  activeModule, 
  onModuleChange 
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const modules: CommandModuleItem[] = [
    // Intelligence Modules
    { id: 'intelligence', label: 'Intelligence', icon: <Zap className="w-4 h-4" />, group: 'intelligence' },
    { id: 'brain', label: 'AI Brain', icon: <Brain className="w-4 h-4" />, group: 'intelligence' },
    { id: 'risk', label: 'Risk Map', icon: <Map className="w-4 h-4" />, group: 'intelligence' },
    { id: 'events', label: 'Events', icon: <Activity className="w-4 h-4" />, group: 'intelligence' },
    { id: 'reasoning', label: 'LLM Reason', icon: <MessageSquare className="w-4 h-4" />, group: 'intelligence' },
    
    // Operations Modules
    { id: 'operator', label: 'AI Operator', icon: <Users className="w-4 h-4" />, group: 'operations' },
    { id: 'edge', label: 'Edge Devices', icon: <Server className="w-4 h-4" />, group: 'operations' },
    
    // Environment Modules
    { id: 'city3d', label: '3D City', icon: <Box className="w-4 h-4" />, group: 'environment' },
    { id: 'videos', label: 'Demo Videos', icon: <Film className="w-4 h-4" />, group: 'environment' }
  ]

  const groupedModules = {
    intelligence: modules.filter(m => m.group === 'intelligence'),
    operations: modules.filter(m => m.group === 'operations'),
    environment: modules.filter(m => m.group === 'environment')
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleModuleClick = (moduleId: CommandModule) => {
    onModuleChange(moduleId)
    setIsOpen(false)
  }

  return (
    <div className="relative" ref={dropdownRef} style={{ zIndex: 9999 }}>
      {/* Dropdown Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={clsx(
          'flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-mono uppercase tracking-wider transition-all duration-200',
          'bg-gray-800/50 border border-cyan-500/30 hover:bg-cyan-600/30 hover:border-cyan-400/50',
          isOpen && 'bg-cyan-600/50 border-cyan-400 text-cyan-300'
        )}
        style={{ position: 'relative', zIndex: 10000 }}
      >
        <Brain className="w-4 h-4 text-cyan-400" />
        <span>Advanced Intelligence</span>
        <div className={clsx(
          'w-4 h-4 transition-transform duration-200',
          isOpen ? 'rotate-180' : ''
        )}>
          <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Dropdown Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="absolute top-full left-0 mt-2"
            style={{ 
              zIndex: 99999,
              width: '320px'
            }}
          >
            <div className="bg-gray-800/95 backdrop-blur-md border border-cyan-500/30 rounded-lg shadow-2xl shadow-cyan-500/10 overflow-hidden"
                 style={{ backgroundColor: 'rgba(31, 41, 55, 0.95)' }}
              >
              {/* Intelligence Section */}
              <div className="border-b border-gray-700/50">
                <div className="px-4 py-2 bg-gray-900/30">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-4 h-4 text-cyan-400" />
                    <span className="text-cyan-400 text-xs font-mono uppercase tracking-wider">Intelligence</span>
                  </div>
                </div>
                <div className="p-2 space-y-1">
                  {groupedModules.intelligence.map((module) => (
                    <button
                      key={module.id}
                      onClick={() => handleModuleClick(module.id)}
                      className={clsx(
                        'w-full flex items-center space-x-3 px-3 py-2 rounded text-xs font-mono transition-all duration-150',
                        activeModule === module.id
                          ? 'bg-cyan-600/30 text-cyan-300 border-l-2 border-cyan-400'
                          : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-700/50'
                      )}
                      style={{ position: 'relative', zIndex: 10 }}
                    >
                      <span className="w-4 h-4 flex items-center justify-center">
                        {module.icon}
                      </span>
                      <span>{module.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Operations Section */}
              <div className="border-b border-gray-700/50">
                <div className="px-4 py-2 bg-gray-900/30">
                  <div className="flex items-center space-x-2">
                    <Users className="w-4 h-4 text-cyan-400" />
                    <span className="text-cyan-400 text-xs font-mono uppercase tracking-wider">Operations</span>
                  </div>
                </div>
                <div className="p-2 space-y-1">
                  {groupedModules.operations.map((module) => (
                    <button
                      key={module.id}
                      onClick={() => handleModuleClick(module.id)}
                      className={clsx(
                        'w-full flex items-center space-x-3 px-3 py-2 rounded text-xs font-mono transition-all duration-150',
                        activeModule === module.id
                          ? 'bg-cyan-600/30 text-cyan-300 border-l-2 border-cyan-400'
                          : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-700/50'
                      )}
                      style={{ position: 'relative', zIndex: 10 }}
                    >
                      <span className="w-4 h-4 flex items-center justify-center">
                        {module.icon}
                      </span>
                      <span>{module.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Environment Section */}
              <div>
                <div className="px-4 py-2 bg-gray-900/30">
                  <div className="flex items-center space-x-2">
                    <Monitor className="w-4 h-4 text-cyan-400" />
                    <span className="text-cyan-400 text-xs font-mono uppercase tracking-wider">Environment</span>
                  </div>
                </div>
                <div className="p-2 space-y-1">
                  {groupedModules.environment.map((module) => (
                    <button
                      key={module.id}
                      onClick={() => handleModuleClick(module.id)}
                      className={clsx(
                        'w-full flex items-center space-x-3 px-3 py-2 rounded text-xs font-mono transition-all duration-150',
                        activeModule === module.id
                          ? 'bg-cyan-600/30 text-cyan-300 border-l-2 border-cyan-400'
                          : 'text-gray-300 hover:text-cyan-400 hover:bg-gray-700/50'
                      )}
                      style={{ position: 'relative', zIndex: 10 }}
                    >
                      <span className="w-4 h-4 flex items-center justify-center">
                        {module.icon}
                      </span>
                      <span>{module.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default CommandModulesDropdown
