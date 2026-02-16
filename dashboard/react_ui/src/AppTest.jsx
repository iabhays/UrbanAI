import React, { useState } from 'react'
import { useWebSocketSimulation } from './hooks/useWebSocketSimulation'

function App() {
  const { state, sendMessage, connected } = useWebSocketSimulation()

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      background: 'black',
      color: 'cyan',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '20px',
      fontFamily: 'monospace'
    }}>
      <div style={{ textAlign: 'center' }}>
        <h1>SENTIENTCITY AI</h1>
        <p>Command Center Interface</p>
        <p style={{ fontSize: '14px', marginTop: '20px', color: connected ? '#0f0' : '#f00' }}>
          WebSocket: {connected ? 'Connected' : 'Disconnected'}
        </p>
        <p style={{ fontSize: '12px', marginTop: '10px', color: '#666' }}>
          Cameras: {state.cameras.length} | Alerts: {state.alerts.length}
        </p>
      </div>
    </div>
  )
}

export default App
