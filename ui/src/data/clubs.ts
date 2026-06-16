export interface Club {
  id: string;
  label: string;
}

// Clubs grouped by type. Object insertion order is preserved and drives the
// display order in both the ClubPicker dropdown and the ClubSelectScreen.
export const CLUBS_BY_TYPE: Record<string, Club[]> = {
  Irons: [
    { id: '2-iron', label: '2i' },
    { id: '3-iron', label: '3i' },
    { id: '4-iron', label: '4i' },
    { id: '5-iron', label: '5i' },
    { id: '6-iron', label: '6i' },
    { id: '7-iron', label: '7i' },
    { id: '8-iron', label: '8i' },
    { id: '9-iron', label: '9i' },
    { id: 'pw', label: 'PW' },
    { id: 'gw', label: 'GW' },
    { id: 'sw', label: 'SW' },
    { id: 'lw', label: 'LW' },
  ],
  Hybrids: [
    { id: '3-hybrid', label: '3H' },
    { id: '5-hybrid', label: '5H' },
    { id: '7-hybrid', label: '7H' },
    { id: '9-hybrid', label: '9H' },
  ],
  Woods: [
    { id: 'driver', label: 'DR' },
    { id: '3-wood', label: '3W' },
    { id: '5-wood', label: '5W' },
    { id: '7-wood', label: '7W' },
  ],
};

export const ALL_CLUBS: Club[] = Object.values(CLUBS_BY_TYPE).flat();
