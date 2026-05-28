/** Split a model response into its reasoning chain and the visible answer.
 *  Handles both <think>…</think> and Hermes' <scratch_pad>…</scratch_pad>. */
export function splitThink(raw: string): { think: string; answer: string } {
  let think = "";
  let answer = raw;

  for (const tag of ["think", "scratch_pad", "scratchpad"]) {
    const closed = answer.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "i"));
    if (closed) {
      think = (think ? think + "\n" : "") + closed[1].trim();
      answer = answer.replace(closed[0], "").trim();
    }
  }
  if (think) return { think, answer };

  // open but not yet closed (streaming): everything after the tag is thought
  const open = answer.match(/<(?:think|scratch_?pad)>([\s\S]*)$/i);
  if (open) {
    think = open[1].trim();
    answer = answer.slice(0, open.index).trim();
  }
  return { think, answer };
}

/** Remove <tool_call>…</tool_call> blocks (and a trailing unterminated one)
 *  so what's left is the assistant's plain visible prose. */
export function stripToolCalls(text: string): string {
  return text
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "")
    .replace(/<tool_call>[\s\S]*$/i, "")
    .trim();
}

export interface ParsedHtml {
  title: string;
  html: string;
}

/** Pull the last complete ```html …``` block out of an answer. */
export function extractHtmlArtifact(answer: string): ParsedHtml | null {
  const fence = /```(?:html|HTML)\s*\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;
  let last: string | null = null;
  while ((match = fence.exec(answer)) !== null) {
    last = match[1];
  }
  // also catch an un-terminated trailing block while streaming
  if (!last) {
    const open = answer.match(/```(?:html|HTML)\s*\n([\s\S]*)$/);
    if (open && /<\w+/.test(open[1])) last = open[1];
  }
  if (!last) return null;

  const html = last.trim();
  const titleMatch = html.match(/<title>([^<]+)<\/title>/i);
  const h1Match = html.match(/<h1[^>]*>([^<]+)<\/h1>/i);
  const title = (titleMatch?.[1] || h1Match?.[1] || "Untitled").trim();
  return { title, html };
}

const escapeHtml = (s: string) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

/** Minimal, dependency-free markdown → HTML for chat bubbles. */
export function renderMarkdown(src: string): string {
  const blocks: string[] = [];
  // pull out fenced code first so its contents are never md-parsed
  const withPlaceholders = src.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, lang, code) => {
    const label = lang ? `<div class="text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)] mb-1">${escapeHtml(lang)}</div>` : "";
    blocks.push(`<pre>${label}<code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return ` BLOCK${blocks.length - 1} `;
  });

  const lines = withPlaceholders.split("\n");
  let out = "";
  let inList = false;
  const flushList = () => {
    if (inList) {
      out += "</ul>";
      inList = false;
    }
  };

  const inline = (t: string) =>
    escapeHtml(t)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(?<!\*)\*(?!\*)([^*]+)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-[var(--color-pi-2)] underline">$1</a>');

  for (const line of lines) {
    const ph = line.match(/^ BLOCK(\d+) $/);
    if (ph) {
      flushList();
      out += blocks[Number(ph[1])];
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) {
        out += "<ul>";
        inList = true;
      }
      out += `<li>${inline(line.replace(/^\s*[-*]\s+/, ""))}</li>`;
      continue;
    }
    flushList();
    const h = line.match(/^(#{1,3})\s+(.*)/);
    if (h) {
      const tag = `h${h[1].length}`;
      out += `<${tag}>${inline(h[2])}</${tag}>`;
    } else if (line.trim() === "") {
      out += "";
    } else {
      out += `<p>${inline(line)}</p>`;
    }
  }
  flushList();
  return out;
}
