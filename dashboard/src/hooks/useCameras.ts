import { useState, useEffect } from 'react';

export function useCameras() {
  const [cameras, setCameras] = useState<any[] | null>(null);
  const [isLoading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setCameras([
        { id: 'cam1', name: 'Main St', location: 'Intersection' },
        { id: 'cam2', name: 'Park', location: 'Central Park' },
      ]);
      setLoading(false);
    }, 200);
  }, []);

  return { cameras, isLoading };
}

export default useCameras;
