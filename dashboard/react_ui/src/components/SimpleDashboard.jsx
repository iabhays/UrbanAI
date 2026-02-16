import React, { useState } from 'react'
import { useWebSocketSimulation } from '../hooks/useWebSocketSimulation'

const SimpleDashboard = () => {
  const { state, sendMessage, connected } = useWebSocketSimulation()

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      background: 'linear-gradient(to bottom, #0a0a0a, #1a1a1a)',
      color: '#fff',
      fontFamily: 'monospace',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        height: '60px',
        background: 'rgba(0, 20, 40, 0.8)',
        borderBottom: '1px solid #00ffff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <h1 style={{ color: '#00ffff', fontSize: '18px', margin: 0 }}>
            SENTIENTCITY AI COMMAND CENTER
          </h1>
          <div style={{ 
            width: '10px', 
            height: '10px', 
            borderRadius: '50%', 
            background: connected ? '#0f0' : '#f00',
            animation: connected ? 'pulse 2s infinite' : 'none'
          }} />
          <span style={{ fontSize: '12px', color: connected ? '#0f0' : '#f00' }}>
            {connected ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '30px', fontSize: '12px' }}>
          <span style={{ color: '#ff6b6b' }}>
            THREAT: {state.system.threatLevel}
          </span>
          <span style={{ color: '#ffd93d' }}>
            INCIDENTS: {state.system.activeIncidents}
          </span>
          <span style={{ color: '#6bcf7f' }}>
            AI: {state.system.aiBrainActivity.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: 'flex', height: 'calc(100vh - 120px)' }}>
        {/* Left Panel */}
        <div style={{
          width: '300px',
          background: 'rgba(0, 10, 20, 0.6)',
          borderRight: '1px solid #00ffff30',
          padding: '20px',
          overflow: 'auto'
        }}>
          <h2 style={{ color: '#00ffff', fontSize: '14px', marginBottom: '20px' }}>
            OPERATIONS
          </h2>
          
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ color: '#fff', fontSize: '12px', marginBottom: '10px' }}>
              Cameras ({state.cameras.length})
            </h3>
            {state.cameras.slice(0, 4).map(camera => (
              <div key={camera.id} style={{
                background: camera.status === 'ACTIVE' ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)',
                border: `1px solid ${camera.status === 'ACTIVE' ? '#0f0' : '#f00'}`,
                padding: '8px',
                marginBottom: '8px',
                borderRadius: '4px'
              }}>
                <div style={{ fontSize: '10px', color: '#ccc' }}>
                  {camera.id} - {camera.location}
                </div>
                <div style={{ fontSize: '9px', color: '#888' }}>
                  {camera.fps} FPS | Risk: {camera.riskScore.toFixed(0)}%
                </div>
              </div>
            ))}
          </div>

          <div>
            <h3 style={{ color: '#fff', fontSize: '12px', marginBottom: '10px' }}>
              AI Model
            </h3>
            <select style={{
              width: '100%',
              background: '#111',
              border: '1px solid #333',
              color: '#fff',
              padding: '4px',
              fontSize: '10px'
            }}>
              <option>YOLOv8-L (High Accuracy)</option>
              <option>YOLOv8-M (Balanced)</option>
              <option>YOLOv8-S (Fast)</option>
            </select>
          </div>
        </div>

        {/* Center Panel */}
        <div style={{
          flex: 1,
          background: 'rgba(0, 5, 10, 0.4)',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <h2 style={{ color: '#00ffff', fontSize: '14px', marginBottom: '20px' }}>
            SURVEILLANCE GRID
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gridTemplateRows: 'repeat(3, 1fr)',
            gap: '10px',
            flex: 1
          }}>
            {/* Main Camera View */}
            <div style={{
              gridColumn: '1 / 3',
              gridRow: '1 / 3',
              background: 'linear-gradient(45deg, #1a1a2a, #2a2a3a)',
              border: '1px solid #00ffff30',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative'
            }}>
              <div style={{ textAlign: 'center', color: '#666' }}>
                <div style={{ fontSize: '24px', marginBottom: '10px' }}>📹</div>
                <div style={{ fontSize: '12px' }}>MAIN CAMERA FEED</div>
                <div style={{ fontSize: '10px', marginTop: '5px' }}>
                  CAM-001 • 30 FPS • LIVE
                </div>
              </div>
              
              {state.alerts.filter(a => a.severity === 'CRITICAL' || a.severity === 'EMERGENCY').length > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  left: '10px',
                  background: 'rgba(255, 0, 0, 0.8)',
                  color: '#fff',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '10px',
                  animation: 'pulse 2s infinite'
                }}>
                  {state.alerts.filter(a => a.severity === 'CRITICAL' || a.severity === 'EMERGENCY').length} CRITICAL ALERTS
                </div>
              )}
            </div>

            {/* Secondary Camera Views */}
            {state.cameras.slice(1, 7).map((camera, index) => (
              <div key={camera.id} style={{
                background: 'linear-gradient(45deg, #1a1a2a, #2a2a3a)',
                border: '1px solid #00ffff30',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'all 0.3s'
              }}>
                <div style={{ textAlign: 'center', color: '#666' }}>
                  <div style={{ fontSize: '16px', marginBottom: '5px' }}>📹</div>
                  <div style={{ fontSize: '9px' }}>{camera.id}</div>
                  <div style={{ fontSize: '8px', color: '#888' }}>
                    {camera.fps} FPS
                  </div>
                </div>
              </div>
            ))}

            {/* Intelligence Overview */}
            <div style={{
              gridColumn: '1 / 4',
              background: 'rgba(0, 20, 40, 0.3)',
              border: '1px solid #00ffff30',
              borderRadius: '8px',
              padding: '15px'
            }}>
              <h3 style={{ color: '#00ffff', fontSize: '12px', marginBottom: '10px' }}>
                LIVE INTELLIGENCE FEED
              </h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: '10px'
              }}>
                {state.intelligence.slice(0, 4).map((item, index) => (
                  <div key={item.id} style={{
                    background: 'rgba(0, 30, 60, 0.5)',
                    border: '1px solid #00ffff20',
                    padding: '8px',
                    borderRadius: '4px'
                  }}>
                    <div style={{
                      fontSize: '9px',
                      color: item.type === 'THREAT' ? '#ff6b6b' :
                              item.type === 'ANOMALY' ? '#ffd93d' :
                              item.type === 'BEHAVIOR' ? '#4ecdc4' :
                              '#666'
                    }}>
                      {item.type}
                    </div>
                    <div style={{ fontSize: '10px', color: '#ccc', marginTop: '2px' }}>
                      {item.classification}
                    </div>
                    <div style={{ fontSize: '8px', color: '#888' }}>
                      {item.confidence.toFixed(1)}% • {item.cameraId}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div style={{
          width: '350px',
          background: 'rgba(0, 10, 20, 0.6)',
          borderLeft: '1px solid #00ffff30',
          padding: '20px',
          overflow: 'auto'
        }}>
          <h2 style={{ color: '#00ffff', fontSize: '14px', marginBottom: '20px' }}>
            INTELLIGENCE STREAM
          </h2>

          {/* Alerts */}
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ color: '#fff', fontSize: '12px', marginBottom: '10px' }}>
              Alerts ({state.alerts.length})
            </h3>
            <div style={{ maxHeight: '200px', overflow: 'auto' }}>
              {state.alerts.slice(0, 5).map(alert => (
                <div key={alert.id} style={{
                  background: alert.severity === 'EMERGENCY' ? 'rgba(255, 0, 0, 0.2)' :
                              alert.severity === 'CRITICAL' ? 'rgba(255, 100, 0, 0.2)' :
                              alert.severity === 'WARNING' ? 'rgba(255, 200, 0, 0.2)' :
                              'rgba(0, 100, 255, 0.2)',
                  border: `1px solid ${
                    alert.severity === 'EMERGENCY' ? '#f00' :
                    alert.severity === 'CRITICAL' ? '#f60' :
                    alert.severity === 'WARNING' ? '#fa0' :
                    '#06f'
                  }`,
                  padding: '8px',
                  marginBottom: '8px',
                  borderRadius: '4px'
                }}>
                  <div style={{
                    fontSize: '10px',
                    color: alert.severity === 'EMERGENCY' ? '#ff6b6b' :
                            alert.severity === 'CRITICAL' ? '#ffa500' :
                            alert.severity === 'WARNING' ? '#ffd93d' :
                            '#6bcf7f',
                    fontWeight: 'bold'
                  }}>
                    {alert.severity}
                  </div>
                  <div style={{ fontSize: '9px', color: '#ccc', marginTop: '2px' }}>
                    {alert.title}
                  </div>
                  <div style={{ fontSize: '8px', color: '#888', marginTop: '2px' }}>
                    {alert.cameraId} • {alert.confidence.toFixed(1)}% confidence
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Brain Activity */}
          <div>
            <h3 style={{ color: '#fff', fontSize: '12px', marginBottom: '10px' }}>
              AI Brain Activity
            </h3>
            <div style={{
              background: 'rgba(0, 30, 60, 0.3)',
              border: '1px solid #00ffff30',
              padding: '10px',
              borderRadius: '4px'
            }}>
              <div style={{ fontSize: '10px', color: '#6bcf7f', marginBottom: '5px' }}>
                Processing Latency: {state.aiBrain.processingLatency.toFixed(1)}ms
              </div>
              <div style={{ fontSize: '10px', color: '#6bcf7f', marginBottom: '5px' }}>
                Model Confidence: {state.aiBrain.modelConfidence.toFixed(1)}%
              </div>
              <div style={{ fontSize: '10px', color: '#6bcf7f' }}>
                Active Neurons: 8/8
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Status Bar */}
      <div style={{
        height: '60px',
        background: 'rgba(0, 20, 40, 0.8)',
        borderTop: '1px solid #00ffff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px',
        fontSize: '11px'
      }}>
        <div style={{ display: 'flex', gap: '30px' }}>
          <span style={{ color: '#6bcf7f' }}>
            FPS: {state.telemetry.fps}
          </span>
          <span style={{ color: '#ffd93d' }}>
            GPU: {state.telemetry.gpuUsage.toFixed(1)}%
          </span>
          <span style={{ color: '#ff6b6b' }}>
            CPU: {state.telemetry.cpuUsage.toFixed(1)}%
          </span>
          <span style={{ color: '#4ecdc4' }}>
            MEM: {state.telemetry.memoryUsage.toFixed(1)}%
          </span>
        </div>
        
        <div style={{ display: 'flex', gap: '30px' }}>
          <span style={{ color: '#6bcf7f' }}>
            Events: {state.telemetry.eventProcessingRate.toFixed(0)}/s
          </span>
          <span style={{ color: '#ffd93d' }}>
            Tracking: {state.telemetry.trackingStability.toFixed(1)}%
          </span>
          <span style={{ color: '#4ecdc4' }}>
            Latency: {state.telemetry.pluginLatency.toFixed(0)}ms
          </span>
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  )
}

export default SimpleDashboard
