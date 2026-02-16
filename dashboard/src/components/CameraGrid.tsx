import React from 'react';

export const CameraGrid: React.FC<{
  cameras: any[];
  selectedCamera?: string | null;
  onSelectCamera?: (id: string | null) => void;
}> = ({ cameras = [], selectedCamera, onSelectCamera }) => {
  return (
    <div className="grid grid-cols-2 gap-3">
      {cameras.length === 0 ? (
        <div className="text-gray-400 p-6">No cameras available</div>
      ) : (
        cameras.map((cam: any) => (
          <div
            key={cam.id || cam.camera_id}
            className={`bg-gray-600 rounded p-2 cursor-pointer border ${selectedCamera === (cam.id || cam.camera_id) ? 'border-cyan-400' : 'border-transparent'}`}
            onClick={() => onSelectCamera && onSelectCamera(cam.id || cam.camera_id)}
          >
            <div className="bg-black h-36 rounded flex items-center justify-center text-gray-400">Camera {cam.name || cam.camera_id || cam.id}</div>
            <div className="text-xs text-gray-300 mt-2">{cam.location || ''}</div>
          </div>
        ))
      )}
    </div>
  );
};

export default CameraGrid;
