import { useState, useEffect } from 'react';

export function useSystemHealth() {
  const [health, setHealth] = useState<any | null>(null);
  const [isLoading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setHealth({ status: 'healthy' });
      setLoading(false);
    }, 100);
  }, []);

  return { health, isLoading };
}

export default useSystemHealth;
