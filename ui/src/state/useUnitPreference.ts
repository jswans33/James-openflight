import { useContext } from 'react';
import { UnitPreferenceContext } from './UnitPreferenceContext';

export function useUnitPreference() {
  const context = useContext(UnitPreferenceContext);

  if (context === null) {
    throw new Error('useUnitPreference must be used within a UnitPreferenceProvider');
  }

  return context;
}
