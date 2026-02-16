import React from 'react'

function AlertPanel({ alerts = [] }) {
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-red-500'
      case 'HIGH':
        return 'bg-orange-500'
      case 'MODERATE':
        return 'bg-yellow-500'
      default:
        return 'bg-blue-500'
    }
  }

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4">Alerts</h2>
      <div className="space-y-4">
        {alerts.length === 0 ? (
          <p className="text-gray-500">No alerts</p>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.alert_id}
              className={`p-4 rounded-lg ${getSeverityColor(alert.severity)} text-white`}
            >
              <div className="font-bold">{alert.severity}</div>
              <div className="text-sm">{alert.explanation}</div>
              <div className="text-xs mt-2 opacity-75">
                {new Date(alert.timestamp).toLocaleString()}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default AlertPanel
