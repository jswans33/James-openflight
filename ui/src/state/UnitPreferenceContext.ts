import { createContext } from 'react';
import type { UnitSystem } from '../utils/units';

export interface UnitPreferenceContextValue {
  unitSystem: UnitSystem;
  setUnitSystem: (unitSystem: UnitSystem) => void;
}

export const UnitPreferenceContext = createContext<UnitPreferenceContextValue | null>(null);
