/**
 * Stop-sequence handling for local generation. Kept free of any wllama import
 * so it stays unit-testable in plain Node (the wllama wasm can't load there).
 *
 * Small models love to keep going past their turn and fabricate the tool result
 * or the next role themselves; we cut output at the earliest such marker.
 */
export const STOP_SEQUENCES = [
  "<tool_response>",
  "</tool_response>",
  "<|im_end|>",
  "<|im_start|>",
  "<|endoftext|>",
];

/** If any stop sequence appears, return the text truncated before it; else null. */
export function cutAtStop(text: string, stops: string[] = STOP_SEQUENCES): string | null {
  let best = -1;
  for (const s of stops) {
    const i = text.indexOf(s);
    if (i !== -1 && (best === -1 || i < best)) best = i;
  }
  return best === -1 ? null : text.slice(0, best);
}
