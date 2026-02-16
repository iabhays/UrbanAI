import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { clsx } from '../../utils/clsx'
import { RiskHeatmap } from '../../types'

interface RiskHeatmapProps {
  heatmap: RiskHeatmap
}

const RiskHeatmapComponent: React.FC<RiskHeatmapProps> = ({ heatmap }) => {
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null)

  // Generate mock regions if empty
  const regions = heatmap.regions.length > 0 ? heatmap.regions : [
    {
      id: 'zone-1',
      coordinates: { x: 0, y: 0, width: 33, height: 33 },
      riskLevel: 85,
      trend: 'INCREASING' as const,
      factors: ['Crowd Density', 'Unusual Movement']
    },
    {
      id: 'zone-2',
      coordinates: { x: 33, y: 0, width: 34, height: 33 },
      riskLevel: 45,
      trend: 'STABLE' as const,
      factors: ['Normal Activity']
    },
    {
      id: 'zone-3',
      coordinates: { x: 67, y: 0, width: 33, height: 33 },
      riskLevel: 20,
      trend: 'DECREASING' as const,
      factors: ['Low Activity']
    },
    {
      id: 'zone-4',
      coordinates: { x: 0, y: 33, width: 33, height: 34 },
      riskLevel: 65,
      trend: 'INCREASING' as const,
      factors: ['Suspicious Behavior', 'Loitering']
    },
    {
      id: 'zone-5',
      coordinates: { x: 33, y: 33, width: 34, height: 34 },
      riskLevel: 30,
      trend: 'STABLE' as const,
      factors: ['Monitored Area']
    },
    {
      id: 'zone-6',
      coordinates: { x: 67, y: 33, width: 33, height: 34 },
      riskLevel: 55,
      trend: 'DECREASING' as const,
      factors: ['Clearing Area']
    },
    {
      id: 'zone-7',
      coordinates: { x: 0, y: 67, width: 33, height: 33 },
      riskLevel: 75,
      trend: 'INCREASING' as const,
      factors: ['High Traffic', 'Security Alert']
    },
    {
      id: 'zone-8',
      coordinates: { x: 33, y: 67, width: 34, height: 33 },
      riskLevel: 40,
      trend: 'STABLE' as const,
      factors: ['Patrolled Zone']
    },
    {
      id: 'zone-9',
      coordinates: { x: 67, y: 67, width: 33, height: 33 },
      riskLevel: 25,
      trend: 'DECREASING' as const,
      factors: ['Secure Area']
    }
  ]

  const getRiskColor = (riskLevel: number) => {
    if (riskLevel >= 80) return 'bg-red-500/80 border-red-400'
    if (riskLevel >= 60) return 'bg-orange-500/80 border-orange-400'
    if (riskLevel >= 40) return 'bg-yellow-500/80 border-yellow-400'
    if (riskLevel >= 20) return 'bg-blue-500/80 border-blue-400'
    return 'bg-green-500/80 border-green-400'
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'INCREASING': return <TrendingUp className="w-3 h-3 text-red-400" />
      case 'DECREASING': return <TrendingDown className="w-3 h-3 text-green-400" />
      default: return <Minus className="w-3 h-3 text-gray-400" />
    }
  }

  return (
    <div className="space-y-4">
      {/* Heatmap Grid */}
      <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="text-cyan-400 w-4 h-4" />
            <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">Risk Heatmap</h3>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full" />
            <span className="text-gray-400 text-xs">Low</span>
            <div className="w-2 h-2 bg-yellow-400 rounded-full ml-2" />
            <span className="text-gray-400 text-xs">Med</span>
            <div className="w-2 h-2 bg-red-400 rounded-full ml-2" />
            <span className="text-gray-400 text-xs">High</span>
          </div>
        </div>

        <div className="relative h-64 bg-gray-900/50 rounded-lg border border-gray-700 overflow-hidden">
          {/* Grid Layout */}
          <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 gap-1 p-1">
            {regions.map((region) => (
              <motion.div
                key={region.id}
                className={clsx(
                  'relative cursor-pointer border transition-all duration-300 rounded',
                  getRiskColor(region.riskLevel),
                  selectedRegion === region.id ? 'ring-2 ring-cyan-400 z-10' : ''
                )}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedRegion(region.id === selectedRegion ? null : region.id)}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: parseInt(region.id.split('-')[1]) * 0.1 }}
              >
                {/* Risk Level Overlay */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-white font-bold text-lg">
                    {region.riskLevel}
                  </span>
                  <span className="text-white/80 text-xs">
                    {region.riskLevel >= 80 ? 'CRITICAL' :
                     region.riskLevel >= 60 ? 'HIGH' :
                     region.riskLevel >= 40 ? 'MEDIUM' :
                     region.riskLevel >= 20 ? 'LOW' : 'MINIMAL'}
                  </span>
                </div>

                {/* Trend Indicator */}
                <div className="absolute top-1 right-1">
                  {getTrendIcon(region.trend)}
                </div>

                {/* Hover Details */}
                {selectedRegion === region.id && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 bg-gray-900 border border-cyan-500/30 rounded-lg p-3 z-20 min-w-48"
                  >
                    <div className="text-xs space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Zone:</span>
                        <span className="text-cyan-400 font-mono">{region.id.toUpperCase()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Risk Level:</span>
                        <span className={clsx(
                          'font-mono',
                          region.riskLevel >= 80 ? 'text-red-400' :
                          region.riskLevel >= 60 ? 'text-orange-400' :
                          region.riskLevel >= 40 ? 'text-yellow-400' :
                          'text-blue-400'
                        )}>
                          {region.riskLevel}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">Trend:</span>
                        <div className="flex items-center space-x-1">
                          {getTrendIcon(region.trend)}
                          <span className="text-gray-300 text-xs">{region.trend}</span>
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-400">Factors:</span>
                        <div className="mt-1 space-y-1">
                          {region.factors.map((factor, index) => (
                            <div key={index} className="text-gray-300 text-xs">
                              • {factor}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Animated Risk Pulse Effect */}
          {regions.filter(r => r.riskLevel >= 70).map((region) => (
            <motion.div
              key={`pulse-${region.id}`}
              className="absolute pointer-events-none"
              style={{
                left: `${region.coordinates.x}%`,
                top: `${region.coordinates.y}%`,
                width: `${region.coordinates.width}%`,
                height: `${region.coordinates.height}%`
              }}
              animate={{
                opacity: [0, 0.5, 0],
                scale: [1, 1.1, 1]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <div className={clsx('w-full h-full rounded', getRiskColor(region.riskLevel))} />
            </motion.div>
          ))}
        </div>

        {/* Risk Statistics */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              {regions.filter(r => r.riskLevel >= 70).length}
            </div>
            <div className="text-xs text-gray-400">Critical</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-400">
              {regions.filter(r => r.riskLevel >= 50 && r.riskLevel < 70).length}
            </div>
            <div className="text-xs text-gray-400">High Risk</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {regions.filter(r => r.riskLevel >= 30 && r.riskLevel < 50).length}
            </div>
            <div className="text-xs text-gray-400">Medium Risk</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {regions.filter(r => r.riskLevel < 30).length}
            </div>
            <div className="text-xs text-gray-400">Low Risk</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RiskHeatmapComponent
