import React from 'react';

export const StatsPanel: React.FC<{ health?: any; cameras?: any[] }> = ({ health, cameras = [] }) => {
  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
      <h3 className="font-semibold mb-2">System Stats</h3>
      <div className="text-sm text-gray-300">Status: <span className="font-medium text-white">{health?.status || 'unknown'}</span></div>
      <div className="text-sm text-gray-300 mt-2">Cameras: <span className="font-medium text-white">{cameras.length}</span></div>
    </div>
  );
};

export default StatsPanel;
