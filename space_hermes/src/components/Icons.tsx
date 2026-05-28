interface P {
  className?: string;
}
const base = (className = "w-4 h-4") => ({
  className,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export const Plus = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);
export const Trash = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" />
  </svg>
);
export const Send = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M5 12l14-7-7 14-2-5-5-2z" />
  </svg>
);
export const Stop = ({ className }: P) => (
  <svg {...base(className)}>
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);
export const Download = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" />
  </svg>
);
export const External = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M14 4h6v6M20 4l-9 9M19 14v5H5V5h5" />
  </svg>
);
export const Code = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M8 7l-5 5 5 5M16 7l5 5-5 5" />
  </svg>
);
export const Eye = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);
export const Gear = ({ className }: P) => (
  <svg {...base(className)}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19 12a7 7 0 00-.1-1.3l2-1.5-2-3.4-2.3 1a7 7 0 00-2.2-1.3L14 2h-4l-.4 2.5a7 7 0 00-2.2 1.3l-2.3-1-2 3.4 2 1.5A7 7 0 005 12a7 7 0 00.1 1.3l-2 1.5 2 3.4 2.3-1a7 7 0 002.2 1.3L10 22h4l.4-2.5a7 7 0 002.2-1.3l2.3 1 2-3.4-2-1.5A7 7 0 0019 12z" />
  </svg>
);
export const Chip = ({ className }: P) => (
  <svg {...base(className)}>
    <rect x="7" y="7" width="10" height="10" rx="1.5" />
    <path d="M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2" />
  </svg>
);
export const Bolt = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z" />
  </svg>
);
export const Copy = ({ className }: P) => (
  <svg {...base(className)}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a2 2 0 012-2h10" />
  </svg>
);
export const Satellite = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M5 13l-2 2 4 4 2-2M11 7L9 9M7 11l6 6M13 5l6 6M16 8l3-3M9 3l2 2M19 15a4 4 0 01-4 4" />
  </svg>
);
export const Refresh = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M21 12a9 9 0 11-3-6.7M21 4v5h-5" />
  </svg>
);
export const Check = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M5 13l4 4L19 7" />
  </svg>
);
export const X = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

// ---- agent icons ----------------------------------------------------------
export const Wrench = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M14.7 6.3a4 4 0 00-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 005.4-5.4l-2.6 2.6-2.4-.6-.6-2.4 2.6-2.6z" />
  </svg>
);
export const Sparkles = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M12 3l1.8 4.7L18.5 9.5 13.8 11.3 12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3z" />
    <path d="M18 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8.8-2z" />
  </svg>
);
export const Brain = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M9 4a3 3 0 00-3 3 3 3 0 00-1 5.8A3 3 0 008 17a3 3 0 003 1V4.5A2.5 2.5 0 009 4z" />
    <path d="M15 4a3 3 0 013 3 3 3 0 011 5.8A3 3 0 0116 17a3 3 0 01-3 1V4.5A2.5 2.5 0 0115 4z" />
  </svg>
);
export const ListTodo = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M9 6h11M9 12h11M9 18h11" />
    <path d="M4 5l1.5 1.5L8 4M4 11l1.5 1.5L8 10" />
    <rect x="3.5" y="16" width="3" height="3" rx="0.5" />
  </svg>
);
export const Search = ({ className }: P) => (
  <svg {...base(className)}>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3" />
  </svg>
);
export const Clock = ({ className }: P) => (
  <svg {...base(className)}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);
export const Bot = ({ className }: P) => (
  <svg {...base(className)}>
    <rect x="4" y="8" width="16" height="11" rx="3" />
    <path d="M12 8V4M9 13h.01M15 13h.01M9 16h6" />
    <path d="M2 13h2M20 13h2" />
  </svg>
);
export const Play = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M7 5l12 7-12 7z" />
  </svg>
);
export const ChevronR = ({ className }: P) => (
  <svg {...base(className)}>
    <path d="M9 6l6 6-6 6" />
  </svg>
);
