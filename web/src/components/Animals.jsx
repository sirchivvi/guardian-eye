import React from 'react';

// Cute Fox Illustration SVG
export const FoxSvg = ({ className = "w-8 h-8", title = "Friendly Fox" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-label={title}>
    {/* Body / head base */}
    <path d="M24 10 L38 24 C38 32 32 40 24 40 C16 40 10 32 10 24 Z" fill="#E07A5F" />
    {/* Ears */}
    <path d="M11 20 L8 7 C13 7 17 11 17 14 Z" fill="#C2542B" />
    <path d="M37 20 L40 7 C35 7 31 11 31 14 Z" fill="#C2542B" />
    <path d="M10 16 L9 9 C12 9 15 12 15 14 Z" fill="#FFF9F0" />
    <path d="M38 16 L39 9 C36 9 33 12 33 14 Z" fill="#FFF9F0" />
    {/* Cheeks */}
    <path d="M10 24 C14 26 18 31 19 36 C14 36 10 32 10 24 Z" fill="#FFF9F0" />
    <path d="M38 24 C34 26 30 31 29 36 C34 36 38 32 38 24 Z" fill="#FFF9F0" />
    {/* Eyes */}
    <circle cx="19" cy="23" r="2" fill="#2D3748" />
    <circle cx="29" cy="23" r="2" fill="#2D3748" />
    <circle cx="18.3" cy="22.3" r="0.7" fill="#FFFFFF" />
    <circle cx="28.3" cy="22.3" r="0.7" fill="#FFFFFF" />
    {/* Nose */}
    <path d="M22 32 C22 31 26 31 26 32 C26 34 22 34 22 32 Z" fill="#2D3748" />
  </svg>
);

// Cute Elephant Illustration SVG
export const ElephantSvg = ({ className = "w-8 h-8", title = "Gentle Elephant" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-label={title}>
    {/* Body */}
    <ellipse cx="26" cy="26" rx="14" ry="12" fill="#76A284" />
    {/* Ear */}
    <ellipse cx="16" cy="22" rx="7" ry="9" fill="#5B8266" />
    <ellipse cx="16" cy="22" rx="4.5" ry="6" fill="#A4C4AF" opacity="0.6" />
    {/* Trunk */}
    <path d="M36 24 C40 26 43 32 40 36 C37 39 34 37 35 34 C36 32 35 29 32 28" stroke="#5B8266" strokeWidth="3.5" strokeLinecap="round" />
    {/* Eye */}
    <circle cx="28" cy="20" r="1.8" fill="#2D3748" />
    <circle cx="27.4" cy="19.4" r="0.6" fill="#FFFFFF" />
    {/* Legs */}
    <rect x="18" y="34" width="4.5" height="7" rx="2" fill="#5B8266" />
    <rect x="28" y="34" width="4.5" height="7" rx="2" fill="#5B8266" />
  </svg>
);

// Cute Owl Illustration SVG
export const OwlSvg = ({ className = "w-8 h-8", title = "Wise Owl" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-label={title}>
    {/* Body */}
    <path d="M24 8 C16 8 12 15 12 28 C12 36 17 40 24 40 C31 40 36 36 36 28 C36 15 32 8 24 8 Z" fill="#9C6644" />
    {/* Tummy */}
    <ellipse cx="24" cy="30" rx="7" ry="8" fill="#F4E8C1" />
    {/* Eyes patches */}
    <circle cx="19" cy="19" r="6" fill="#FFF9F0" />
    <circle cx="29" cy="19" r="6" fill="#FFF9F0" />
    <circle cx="19" cy="19" r="3.5" fill="#F59E0B" />
    <circle cx="29" cy="19" r="3.5" fill="#F59E0B" />
    <circle cx="19" cy="19" r="2" fill="#2D3748" />
    <circle cx="29" cy="19" r="2" fill="#2D3748" />
    <circle cx="18.3" cy="18.3" r="0.7" fill="#FFFFFF" />
    <circle cx="28.3" cy="18.3" r="0.7" fill="#FFFFFF" />
    {/* Beak */}
    <polygon points="24,21 22,25 26,25" fill="#E07A5F" />
    {/* Feather tufts / ears */}
    <path d="M14 9 L18 13" stroke="#7F5539" strokeWidth="2.5" strokeLinecap="round" />
    <path d="M34 9 L30 13" stroke="#7F5539" strokeWidth="2.5" strokeLinecap="round" />
  </svg>
);

// Cute Deer Illustration SVG
export const DeerSvg = ({ className = "w-8 h-8", title = "Forest Deer" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} aria-label={title}>
    <path d="M24 16 C18 16 14 21 14 27 C14 34 18 39 24 39 C30 39 34 34 34 27 C34 21 30 16 24 16 Z" fill="#DDB892" />
    <path d="M18 11 L13 6 M18 11 L16 4 M18 11 L22 7" stroke="#7F5539" strokeWidth="2" strokeLinecap="round" />
    <path d="M30 11 L35 6 M30 11 L32 4 M30 11 L26 7" stroke="#7F5539" strokeWidth="2" strokeLinecap="round" />
    <circle cx="20" cy="25" r="2" fill="#2D3748" />
    <circle cx="28" cy="25" r="2" fill="#2D3748" />
    <circle cx="19.3" cy="24.3" r="0.7" fill="#FFFFFF" />
    <circle cx="27.3" cy="24.3" r="0.7" fill="#FFFFFF" />
    <ellipse cx="24" cy="30" rx="2" ry="1.2" fill="#2D3748" />
  </svg>
);

// Cute Leaf / Sprout
export const SproutSvg = ({ className = "w-5 h-5" }) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <path d="M12 21 C12 17 12 13 12 9 C12 5 16 3 20 4 C21 8 19 12 15 13" stroke="#5B8266" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="#E2ECE5" />
    <path d="M12 14 C9 13 6 11 5 8 C7 5 11 6 12 9" stroke="#5B8266" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="#C4D9CC" />
  </svg>
);

// Cute Camera Trap SVG
export const CameraTrapSvg = ({ className = "w-8 h-8" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <rect x="8" y="14" width="32" height="24" rx="6" fill="#5B8266" />
    <rect x="18" y="10" width="12" height="5" rx="2" fill="#4A6E55" />
    <circle cx="24" cy="26" r="7" fill="#FFF9F0" />
    <circle cx="24" cy="26" r="4.5" fill="#2D3748" />
    <circle cx="22.5" cy="24.5" r="1.5" fill="#FFFFFF" />
    <circle cx="34" cy="19" r="1.8" fill="#FDE68A" />
  </svg>
);

// Cute Microphone / Acoustic SVG
export const AcousticMicSvg = ({ className = "w-8 h-8" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <rect x="18" y="10" width="12" height="18" rx="6" fill="#D96B43" />
    <path d="M14 22 C14 28 18 32 24 32 C30 32 34 28 34 22" stroke="#2D3748" strokeWidth="2.5" strokeLinecap="round" />
    <path d="M24 32 L24 38 M18 38 L30 38" stroke="#2D3748" strokeWidth="2.5" strokeLinecap="round" />
    <circle cx="38" cy="14" r="2" fill="#F59E0B" />
    <circle cx="41" cy="20" r="1.5" fill="#F59E0B" />
  </svg>
);

// Shield / Ranger Badge SVG
export const RangerShieldSvg = ({ className = "w-8 h-8" }) => (
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <path d="M24 6 L38 12 V24 C38 33 24 42 24 42 C24 42 10 33 10 24 V12 Z" fill="#5B8266" />
    <path d="M24 10 L34 14.5 V23.5 C34 30.5 24 37 24 37 C24 37 14 30.5 14 23.5 V14.5 Z" fill="#76A284" />
    <circle cx="24" cy="23" r="5" fill="#FFF9F0" />
    <circle cx="24" cy="23" r="2.5" fill="#2D3748" />
  </svg>
);
