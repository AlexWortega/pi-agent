import { useState } from "react";
import type { ToolCall, ToolResult } from "../types";
import { Wrench, Check, X, Refresh, ChevronR } from "./Icons";

export function ToolCallCard({ call, result }: { call: ToolCall; result?: ToolResult }) {
  const [open, setOpen] = useState(false);
  const running = !result;
  const ok = result?.ok;

  return (
    <div className="tool-card mt-2 text-[12.5px]">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-2.5 py-2 text-left hover:bg-[var(--color-panel-2)]/40 transition"
      >
        <ChevronR className={`w-3 h-3 text-[var(--color-ink-faint)] transition ${open ? "rotate-90" : ""}`} />
        <span className="w-5 h-5 grid place-items-center rounded-md bg-[var(--color-pi)]/15 text-[var(--color-pi-2)] shrink-0">
          <Wrench className="w-3 h-3" />
        </span>
        <span className="font-[var(--font-mono)] text-[var(--color-ink)]">{call.name}</span>
        <span className="ml-auto flex items-center gap-1.5 text-[11px]">
          {running ? (
            <span className="inline-flex items-center gap-1 text-[var(--color-pi-2)]">
              <Refresh className="w-3 h-3 spin" /> running
            </span>
          ) : ok ? (
            <span className="inline-flex items-center gap-1 text-[var(--color-mint)]">
              <Check className="w-3 h-3" /> {result?.ms != null ? `${result.ms} ms` : "done"}
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-[var(--color-soyuz)]">
              <X className="w-3 h-3" /> error
            </span>
          )}
        </span>
      </button>
      {open && (
        <div className="border-t border-[var(--color-edge)]">
          <div className="px-2.5 pt-1.5 text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)]">arguments</div>
          <pre>{JSON.stringify(call.arguments, null, 2)}</pre>
          {result && (
            <>
              <div className="px-2.5 pt-1.5 text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)]">
                {ok ? "result" : "error"}
              </div>
              <pre className={ok ? "" : "text-[var(--color-soyuz)]"}>
                {typeof result.content === "string" ? result.content : JSON.stringify(result.content, null, 2)}
              </pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
