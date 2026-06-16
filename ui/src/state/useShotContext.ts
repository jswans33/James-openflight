import { useContext } from 'react';
import { ShotContext, type ShotContextValue } from './shotContext';

export function useShotContext(): ShotContextValue {
  const ctx = useContext(ShotContext);

  if (!ctx) {
    throw new Error('useShotContext must be used within a <ShotProvider>');
  }
  return ctx;
}
