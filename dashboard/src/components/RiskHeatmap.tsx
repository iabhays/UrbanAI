import React from 'react';

export const RiskHeatmap: React.FC<{ cameras: any[] }> = ({ cameras = [] }) => {
  return (
    <div className="h-64 bg-gray-700 rounded flex items-center justify-center text-gray-400">Risk heatmap placeholder ({cameras.length} cameras)</div>
  );
};

export default RiskHeatmap;
