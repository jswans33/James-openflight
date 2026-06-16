import { describe, expect, it } from 'vitest';
import { ALL_CLUBS, CLUBS_BY_TYPE } from './clubs';

describe('clubs data', () => {
  it('flattens every grouped club into ALL_CLUBS', () => {
    const grouped = Object.values(CLUBS_BY_TYPE).flat();
    expect(ALL_CLUBS).toHaveLength(grouped.length);
    expect(ALL_CLUBS).toEqual(grouped);
  });

  it('includes the driver default with a DR label', () => {
    const driver = ALL_CLUBS.find((c) => c.id === 'driver');
    expect(driver).toBeDefined();
    expect(driver?.label).toBe('DR');
  });

  it('has unique club ids', () => {
    const ids = ALL_CLUBS.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
