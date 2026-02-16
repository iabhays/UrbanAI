import { useState, useEffect } from 'react';

export function useAlerts() {
  const [alerts, setAlerts] = useState<any[] | null>(null);
  const [isLoading, setLoading] = useState(true);

  useEffect(() => {
    // Placeholder: return empty sample data
    setTimeout(() => {
      setAlerts([
        { alert_id: 'a1', title: 'Crowd density high', severity: 'high', timestamp: 'now' },
      ]);
      setLoading(false);
    }, 200);
  }, []);

  return { alerts, isLoading };
}

export default useAlerts;
