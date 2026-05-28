import type { Command } from "../lib/commands";

export function SlashMenu({ commands, onPick }: { commands: Command[]; onPick: (name: string) => void }) {
  if (commands.length === 0) return null;
  return (
    <div className="absolute bottom-full mb-2 left-0 right-0 glass rounded-xl border border-[var(--color-edge-2)] overflow-hidden max-h-60 overflow-y-auto">
      {commands.map((c) => (
        <button
          key={c.name}
          onMouseDown={(e) => {
            e.preventDefault();
            onPick(c.name);
          }}
          className="w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-[var(--color-panel-2)] transition"
        >
          <span className="font-[var(--font-mono)] text-[12.5px] text-[var(--color-pi-2)]">{c.name}</span>
          {c.arg && <span className="font-[var(--font-mono)] text-[11px] text-[var(--color-ink-faint)]">{c.arg}</span>}
          <span className="ml-auto text-[11px] text-[var(--color-ink-faint)] truncate">{c.description}</span>
        </button>
      ))}
    </div>
  );
}
