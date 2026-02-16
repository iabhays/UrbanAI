import React from 'react'

function CameraView({ cameras = [], selectedCamera, onSelectCamera }) {
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h2 className="text-xl font-bold mb-4">Camera Feeds</h2>
      <div className="grid grid-cols-2 gap-4">
        {cameras.map((camera) => (
          <div
            key={camera.id}
            className={`p-4 border rounded-lg cursor-pointer ${
              selectedCamera?.id === camera.id ? 'border-blue-500' : ''
            }`}
            onClick={() => onSelectCamera(camera)}
          >
            <div className="font-bold">{camera.location}</div>
            <div className="text-sm text-gray-500">{camera.id}</div>
            <div className="mt-2 bg-gray-200 h-32 rounded flex items-center justify-center">
              Camera Feed Placeholder
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default CameraView
