import { useEffect, useMemo, useState, type ReactNode } from 'react';
import type { UnitSystem } from '../utils/units';
import { UnitPreferenceContext } from './UnitPreferenceContext';

const STORAGE_KEY = 'openflight.unit-system';

function readStoredUnitSystem(): UnitSystem {
  if (typeof window === 'undefined') {
    return 'imperial';
  }

  const storedValue = window.localStorage.getItem(STORAGE_KEY);
  return storedValue === 'metric' ? 'metric' : 'imperial';
}

export function UnitPreferenceProvider({ children }: { children: ReactNode }) {
  const [unitSystem, setUnitSystem] = useState<UnitSystem>(readStoredUnitSystem);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, unitSystem);
  }, [unitSystem]);

  const value = useMemo(
    () => ({
      unitSystem,
      setUnitSystem,
    }),
    [unitSystem],
  );

  return <UnitPreferenceContext.Provider value={value}>{children}</UnitPreferenceContext.Provider>;
}
