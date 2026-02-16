import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Camera, AlertTriangle, Activity, Maximize2 } from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { Camera as CameraType, Alert, IntelligenceFeed } from '../../types'

interface CommandGridProps {
  cameras: CameraType[]
  alerts: Alert[]
  selectedCamera: string | null
  intelligence: IntelligenceFeed[]
}

const CommandGrid: React.FC<CommandGridProps> = ({
  cameras,
  alerts,
  selectedCamera,
  intelligence
}) => {
  const [focusedCamera, setFocusedCamera] = useState<string | null>(null)

  const activeCameras = cameras.filter(cam => cam.status === 'ACTIVE')
  const criticalAlerts = alerts.filter(alert => alert.severity === 'CRITICAL' || alert.severity === 'EMERGENCY')

  return (
    <div className="h-full p-4">
      <div className="grid grid-cols-3 gap-4 h-full">
        {/* Main Camera View */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="col-span-2 row-span-2 relative"
        >
          <div className="h-full bg-gray-900/50 rounded-lg border border-cyan-500/20 overflow-hidden relative">
            {/* Camera Feed */}
            {(cameras.find(c => c.id === (focusedCamera || activeCameras[0]?.id))?.videoUrl) ? (
               <video 
                 src={cameras.find(c => c.id === (focusedCamera || activeCameras[0]?.id))?.videoUrl}
                 autoPlay 
                 loop 
                 muted 
                 playsInline
                 className="absolute inset-0 w-full h-full object-cover opacity-80"
               />
            ) : (
              <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="text-cyan-400 w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-gray-500 text-sm font-mono">MAIN CAMERA FEED</p>
                  <p className="text-gray-600 text-xs font-mono mt-2">
                    {focusedCamera || 'CAM-001'} • 30 FPS
                  </p>
                </div>
              </div>
            )}

            {/* Camera Overlay Info */}
            <div className="absolute top-4 left-4 space-y-2">
              <div className="bg-black/60 backdrop-blur-sm px-3 py-1 rounded">
                <p className="text-cyan-400 text-xs font-mono">
                  {focusedCamera || 'CAM-001'} • {activeCameras[0]?.location || 'Main Entrance'}
                </p>
              </div>
              {criticalAlerts.length > 0 && (
                <div className="bg-red-600/80 backdrop-blur-sm px-3 py-1 rounded animate-pulse">
                  <p className="text-white text-xs font-mono">
                    {criticalAlerts.length} CRITICAL ALERTS
                  </p>
                </div>
              )}
            </div>

            {/* Camera Controls */}
            <div className="absolute top-4 right-4 flex space-x-2">
              <button className="bg-black/60 backdrop-blur-sm p-2 rounded hover:bg-black/80 transition-colors">
                <Maximize2 className="text-cyan-400 w-4 h-4" />
              </button>
            </div>

            {/* Bottom Status Bar */}
            <div className="absolute bottom-4 left-4 right-4">
              <div className="bg-black/60 backdrop-blur-sm px-3 py-2 rounded flex justify-between items-center">
                <div className="flex space-x-4">
                  <span className="text-green-400 text-xs font-mono">● LIVE</span>
                  <span className="text-gray-400 text-xs font-mono">30 FPS</span>
                  <span className="text-gray-400 text-xs font-mono">1920x1080</span>
                </div>
                <div className="flex space-x-2">
                  <span className="text-yellow-400 text-xs font-mono">TRACKING: 3</span>
                  <span className="text-cyan-400 text-xs font-mono">AI: ACTIVE</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Secondary Camera Views */}
        {activeCameras.slice(1).map((camera, index) => (
          <motion.div
            key={camera.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index }}
            className="relative group cursor-pointer"
            onClick={() => setFocusedCamera(camera.id)}
          >
            <div className="h-full bg-gray-900/50 rounded-lg border border-cyan-500/20 overflow-hidden hover:border-cyan-400/40 transition-colors relative">
              {/* Camera Feed */}
              {camera.videoUrl ? (
                <video 
                  src={camera.videoUrl}
                  autoPlay 
                  loop 
                  muted 
                  playsInline
                  className="absolute inset-0 w-full h-full object-cover opacity-80"
                />
              ) : (
                <div className="h-full bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="text-gray-600 w-8 h-8 mx-auto mb-2" />
                    <p className="text-gray-500 text-xs font-mono">{camera.id}</p>
                    <p className="text-gray-600 text-xs font-mono mt-1">{camera.fps} FPS</p>
                  </div>
                </div>
              )}

              {/* Camera Info Overlay */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="absolute bottom-2 left-2 right-2">
                  <p className="text-cyan-400 text-xs font-mono">{camera.id}</p>
                  <p className="text-gray-400 text-xs">{camera.location}</p>
                </div>
              </div>

              {/* Risk Indicator */}
              {camera.riskScore > 70 && (
                <div className="absolute top-2 right-2">
                  <div className="bg-red-600/80 backdrop-blur-sm px-2 py-1 rounded">
                    <span className="text-white text-xs font-mono">
                      RISK: {camera.riskScore.toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        ))}

        {/* Intelligence Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="col-span-3 row-span-1"
        >
          <div className="h-full bg-gray-900/50 rounded-lg border border-cyan-500/20 p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Activity className="text-cyan-400 w-4 h-4" />
                <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Live Intelligence Feed</h3>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-green-400 text-xs font-mono">REAL-TIME</span>
              </div>
            </div>
            
            <div className="grid grid-cols-4 gap-4">
              {intelligence.slice(0, 4).map((item, index) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  className="bg-gray-800/50 rounded border border-cyan-500/10 p-3"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className={clsx(
                      'text-xs font-mono uppercase',
                      item.type === 'THREAT' ? 'text-red-400' :
                      item.type === 'ANOMALY' ? 'text-yellow-400' :
                      item.type === 'BEHAVIOR' ? 'text-blue-400' :
                      'text-gray-400'
                    )}>
                      {item.type}
                    </span>
                    <span className="text-gray-500 text-xs">
                      {item.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-gray-300 text-xs font-mono">{item.classification}</p>
                  <p className="text-gray-600 text-xs mt-1">{item.cameraId}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default CommandGrid
