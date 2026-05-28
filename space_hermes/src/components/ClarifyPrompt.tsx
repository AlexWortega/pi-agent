import { useState } from "react";
import { Bot } from "./Icons";

export interface ClarifyRequest {
  question: string;
  options?: string[];
  resolve: (answer: string) => void;
}

export function ClarifyPrompt({ req }: { req: ClarifyRequest }) {
  const [text, setText] = useState("");
  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 shrink-0 rounded-lg grid place-items-center bg-gradient-to-br from-[var(--color-pi)] to-[var(--color-soyuz)] mt-0.5">
        <Bot className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0 rounded-2xl border border-[var(--color-pi)]/40 bg-[var(--color-pi)]/5 p-3">
        <div className="text-[10px] uppercase tracking-wider text-[var(--color-pi-2)] mb-1">Hermes needs a hint</div>
        <div className="text-[13.5px] text-[var(--color-ink)] mb-2.5">{req.question}</div>
        {req.options && req.options.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {req.options.map((o) => (
              <button key={o} className="btn btn-ghost" onClick={() => req.resolve(o)}>
                {o}
              </button>
            ))}
          </div>
        ) : (
          <div className="flex gap-2">
            <input
              autoFocus
              className="field text-[13px]"
              placeholder="Type your answer…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && text.trim()) req.resolve(text.trim());
              }}
            />
            <button className="btn btn-primary shrink-0" disabled={!text.trim()} onClick={() => req.resolve(text.trim())}>
              Send
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
