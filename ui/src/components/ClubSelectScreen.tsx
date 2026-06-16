import { CLUBS_BY_TYPE } from '../data/clubs';
import './ClubSelectScreen.css';

interface ClubSelectScreenProps {
  /** Currently selected club id; pre-highlighted on the grid. */
  selectedClub: string;
  /** Called with the chosen club id when the user picks a club. */
  onSelect: (club: string) => void;
  /** Called when the user dismisses (X) without changing the current club. */
  onSkip: () => void;
}

/**
 * Full-screen interstitial shown on app load so the user confirms which club
 * they're hitting before the first shot. Dismissible via the X in the corner,
 * which keeps the current (default) club.
 */
export function ClubSelectScreen({ selectedClub, onSelect, onSkip }: ClubSelectScreenProps) {
  return (
    <div className="club-select" role="dialog" aria-modal="true" aria-label="Select your club">
      <div className="club-select__panel">
        <button className="club-select__close" onClick={onSkip} aria-label="Close club selection">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        <h1 className="club-select__title">Select your club</h1>
        <p className="club-select__subtitle">Choose the club you're hitting to start your session.</p>

        {Object.entries(CLUBS_BY_TYPE).map(([type, clubs]) => (
          <div className="club-select__section" key={type}>
            <span className="club-select__section-title">{type}</span>
            <div className="club-select__grid">
              {clubs.map((club) => (
                <button
                  key={club.id}
                  className={`club-select__option ${selectedClub === club.id ? 'club-select__option--selected' : ''}`}
                  onClick={() => onSelect(club.id)}
                >
                  {club.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
