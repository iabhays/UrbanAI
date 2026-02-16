import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function Analytics({ data }) {
  // Placeholder data
  const chartData = [
    { time: '00:00', risk: 0.3 },
    { time: '06:00', risk: 0.4 },
    { time: '12:00', risk: 0.6 },
    { time: '18:00', risk: 0.5 },
    { time: '24:00', risk: 0.4 },
  ]

  return (
    <div className="bg-white shadow rounded-lg p-6 mt-6">
      <h2 className="text-xl font-bold mb-4">Analytics</h2>
      {data && (
        <div className="mb-4">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-500">Total Detections</div>
              <div className="text-2xl font-bold">{data.total_detections}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Risk Score</div>
              <div className="text-2xl font-bold">{data.risk_score.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Alerts</div>
              <div className="text-2xl font-bold">{data.alerts_count}</div>
            </div>
          </div>
        </div>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="risk" stroke="#8884d8" name="Risk Score" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default Analytics
