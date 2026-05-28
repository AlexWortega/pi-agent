import type { Tool } from "../registry";

/** Strip HTML to readable text (best-effort, DOMParser-based). */
function htmlToText(html: string): string {
  try {
    const doc = new DOMParser().parseFromString(html, "text/html");
    doc.querySelectorAll("script,style,noscript,svg,nav,footer,header").forEach((n) => n.remove());
    const text = doc.body?.textContent ?? "";
    return text.replace(/\n{3,}/g, "\n\n").replace(/[ \t]{2,}/g, " ").trim();
  } catch {
    return html;
  }
}

export const webExtractTool: Tool = {
  name: "web_extract",
  description:
    "Fetch a URL and return its main text content as markdown/plain text. Many sites block cross-origin requests; when a direct fetch is blocked it retries through a CORS reader proxy.",
  web: true,
  parameters: {
    type: "object",
    properties: { url: { type: "string" } },
    required: ["url"],
  },
  async run(args, ctx) {
    let url = String(args?.url ?? "").trim();
    if (!/^https?:\/\//i.test(url)) url = "https://" + url;

    // 1) direct fetch (works for CORS-open hosts)
    try {
      const r = await fetch(url, { signal: ctx.signal });
      if (r.ok) {
        const ct = r.headers.get("content-type") || "";
        const body = await r.text();
        const text = ct.includes("html") ? htmlToText(body) : body;
        return { url, via: "direct", text: text.slice(0, 6000) };
      }
    } catch {
      /* fall through to proxy */
    }

    // 2) reader proxy fallback (returns markdown, CORS-open)
    const proxy = ctx.settings.readerProxy;
    if (proxy) {
      try {
        const r = await fetch(proxy + url, { signal: ctx.signal });
        if (r.ok) {
          const text = await r.text();
          return { url, via: "reader-proxy", text: text.slice(0, 6000) };
        }
      } catch {
        /* ignore */
      }
    }

    throw new Error("could not fetch the URL (blocked by CORS and the reader proxy failed)");
  },
};
