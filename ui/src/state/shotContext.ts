import { createContext } from 'react';
import type { Shot } from '../types/shot';

export interface ShotContextValue {
  latestShot: Shot | null;
  shots: Shot[];
  isNewShot: boolean;
  /** Increments on every new shot — use as React key to force animation remount */
  shotVersion: number;
  addShot: (shot: Shot) => void;
  setShots: (shots: Shot[]) => void;
  clearShots: () => void;
}

export const ShotContext = createContext<ShotContextValue | null>(null);
