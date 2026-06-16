import { useState } from 'react';
import { ALL_CLUBS, CLUBS_BY_TYPE } from '../data/clubs';
import './ClubPicker.css';

interface ClubPickerProps {
  selectedClub: string;
  onClubChange: (club: string) => void;
}

export function ClubPicker({ selectedClub, onClubChange }: ClubPickerProps) {
  const [isOpen, setIsOpen] = useState(false);

  const selectedLabel = ALL_CLUBS.find((c) => c.id === selectedClub)?.label || 'DR';

  const handleSelect = (clubId: string) => {
    onClubChange(clubId);
    setIsOpen(false);
  };

  return (
    <div className="club-picker">
      <button className="club-picker__trigger" onClick={() => setIsOpen(!isOpen)} aria-expanded={isOpen}>
        <span className="club-picker__label">Club</span>
        <span className="club-picker__value">{selectedLabel}</span>
        <svg
          className={`club-picker__arrow ${isOpen ? 'club-picker__arrow--open' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div className="club-picker__overlay" onClick={() => setIsOpen(false)} />
          <div className="club-picker__dropdown">
            {Object.entries(CLUBS_BY_TYPE).map(([type, clubs]) => (
              <div className="club-picker__section">
                <span className="club-picker__section-title">{type}</span>
                <div className="club-picker__grid">
                  {clubs.map((club) => (
                    <button
                      key={club.id}
                      className={`club-picker__option ${
                        selectedClub === club.id ? 'club-picker__option--selected' : ''
                      }`}
                      onClick={() => handleSelect(club.id)}
                    >
                      {club.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
