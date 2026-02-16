/**
 * SENTIENTCITY AI - Dashboard Home Page
 * Live monitoring overview
 */

import { useState, useEffect } from 'react';
import Head from 'next/head';
import { AlertCard } from '../components/AlertCard';
import { CameraGrid } from '../components/CameraGrid';
import { StatsPanel } from '../components/StatsPanel';
import { RiskHeatmap } from '../components/RiskHeatmap';
import { useAlerts } from '../hooks/useAlerts';
import { useCameras } from '../hooks/useCameras';
import { useSystemHealth } from '../hooks/useSystemHealth';

export default function Dashboard() {
  const { alerts, isLoading: alertsLoading } = useAlerts();
  const { cameras, isLoading: camerasLoading } = useCameras();
  const { health, isLoading: healthLoading } = useSystemHealth();
  
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  return (
    <>
      <Head>
        <title>SENTIENTCITY - Live Monitoring</title>
        <meta name="description" content="Smart City Intelligence Dashboard" />
      </Head>

      <div className="min-h-screen bg-gray-900 text-white">
        {/* Header */}
        <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-cyan-400">SENTIENTCITY</h1>
              <span className="text-gray-400">|</span>
              <span className="text-gray-300">Live Monitoring</span>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* System Status Indicator */}
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                } animate-pulse`} />
                <span className="text-sm text-gray-300">
                  {health?.status === 'healthy' ? 'System Operational' : 'System Issues'}
                </span>
              </div>
              
              {/* Active Cameras */}
              <div className="bg-gray-700 rounded-lg px-3 py-1">
                <span className="text-cyan-400 font-semibold">{cameras?.length || 0}</span>
                <span className="text-gray-400 ml-1">Active Cameras</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="p-6">
          <div className="grid grid-cols-12 gap-6">
            {/* Left Panel - Camera Grid */}
            <div className="col-span-8">
              <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
                <h2 className="text-lg font-semibold mb-4 flex items-center">
                  <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
                  Live Camera Feeds
                </h2>
                <CameraGrid 
                  cameras={cameras || []} 
                  selectedCamera={selectedCamera}
                  onSelectCamera={setSelectedCamera}
                />
              </div>
              
              {/* Risk Heatmap */}
              <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 mt-6">
                <h2 className="text-lg font-semibold mb-4">Risk Heatmap</h2>
                <RiskHeatmap cameras={cameras || []} />
              </div>
            </div>

            {/* Right Panel - Alerts & Stats */}
            <div className="col-span-4 space-y-6">
              {/* Stats */}
              <StatsPanel health={health} cameras={cameras || []} />
              
              {/* Recent Alerts */}
              <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
                <h2 className="text-lg font-semibold mb-4 flex items-center justify-between">
                  <span>Recent Alerts</span>
                  <span className="text-sm text-gray-400">
                    {alerts?.length || 0} active
                  </span>
                </h2>
                
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {alertsLoading ? (
                    <div className="text-center py-8 text-gray-400">
                      Loading alerts...
                    </div>
                  ) : alerts?.length === 0 ? (
                    <div className="text-center py-8 text-gray-400">
                      No active alerts
                    </div>
                  ) : (
                    alerts?.slice(0, 10).map((alert) => (
                      <AlertCard key={alert.alert_id} alert={alert} />
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
