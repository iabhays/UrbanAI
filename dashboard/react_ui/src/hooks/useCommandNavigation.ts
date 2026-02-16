import { useState, useCallback } from 'react'

export type CommandModule = 
  | 'intelligence' 
  | 'brain' 
  | 'risk' 
  | 'events' 
  | 'city3d' 
  | 'operator' 
  | 'reasoning' 
  | 'edge'

interface UseCommandNavigationReturn {
  activeModule: CommandModule
  setActiveModule: (module: CommandModule) => void
  isModuleActive: (module: CommandModule) => boolean
}

export const useCommandNavigation = (): UseCommandNavigationReturn => {
  const [activeModule, setActiveModule] = useState<CommandModule>('intelligence')

  const handleModuleChange = useCallback((module: CommandModule) => {
    setActiveModule(module)
  }, [])

  const isModuleActive = useCallback((module: CommandModule) => {
    return activeModule === module
  }, [activeModule])

  return {
    activeModule,
    setActiveModule: handleModuleChange,
    isModuleActive
  }
}
