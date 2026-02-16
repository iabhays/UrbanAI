import React from 'react';

type Alert = {
  alert_id?: string | number;
  title?: string;
  severity?: string;
  timestamp?: string;
};

export const AlertCard: React.FC<{ alert: Alert }> = ({ alert }) => {
  return (
    <div className="p-3 bg-gray-700 rounded flex items-start space-x-3">
      <div className="w-3 h-3 mt-1 rounded-full bg-red-400" />
      <div className="flex-1">
        <div className="font-semibold text-sm text-white">{alert?.title || `Alert ${alert?.alert_id}`}</div>
        <div className="text-xs text-gray-400">{alert?.severity || 'info'} • {alert?.timestamp || ''}</div>
      </div>
    </div>
  );
};

export default AlertCard;
