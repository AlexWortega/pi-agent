import type { Tool } from "../registry";
import { getKeenableKey } from "../../lib/settings";
import { LOG_API } from "../../config";
import { clientId } from "../../lib/logger";

/** Real search routing:
 *   1. user's BYO keenable key (Tools panel) → direct keenable, no shared quota
 *   2. Railway /api/search proxy (server holds a shared keenable key,
 *      rate-limits per browser + per IP, never leaks the key)
 *   3. DuckDuckGo Instant Answer (CORS-open, definition-level only)
 *   4. Wikipedia REST summary (final fallback) */
export const webSearchTool: Tool = {
  name: "web_search",
  description:
    "Search the live web. Prefers a user-supplied keenable.ai key (Tools panel), falls back to the shared Railway proxy, then to DuckDuckGo Instant Answer + Wikipedia summary.",
  web: true,
  parameters: {
    type: "object",
    properties: { query: { type: "string" }, limit: { type: "number", description: "max hits, default 6" } },
    required: ["query"],
  },
  async run(args, ctx) {
    const q = String(args?.query ?? "").trim();
    if (!q) throw new Error("query is empty");
    const limit = Math.min(10, Math.max(1, Number(args?.limit) || 6));

    // 1) BYO keenable key — direct, no shared quota burn
    const byoKey = ctx.settings.keenableApiKey?.trim();
    if (byoKey) {
      try {
        const r = await fetch("https://api.keenable.ai/v1/search", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-Key": byoKey },
          body: JSON.stringify({ query: q }),
          signal: ctx.signal,
        });
        if (r.ok) {
          const d: any = await r.json();
          const results = (d.results ?? []).slice(0, limit).map((x: any) => ({
            title: x.title,
            url: x.url,
            snippet: x.snippet || x.description || "",
          }));
          if (results.length) return { source: "keenable (your key)", query: q, results };
        }
      } catch {
        /* fall through to proxy */
      }
    }

    // 2) Railway /api/search proxy — shared key, rate-limited per client
    if (LOG_API) {
      try {
        const r = await fetch(`${LOG_API}/api/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, client_id: clientId() }),
          signal: ctx.signal,
        });
        if (r.ok) {
          const d: any = await r.json();
          if (Array.isArray(d.results) && d.results.length) {
            return { source: "keenable (shared proxy)", query: q, results: d.results.slice(0, limit) };
          }
        } else if (r.status === 429) {
          const d: any = await r.json().catch(() => ({}));
          return { source: "rate-limited", note: d.error || "rate limit hit", retry_in_s: d.retry_in_s };
        }
        // 503 / 502 → fall through to keyless
      } catch {
        /* fall through */
      }
    }

    // 3) DuckDuckGo Instant Answer (CORS-open)
    try {
      const r = await fetch(
        `https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_html=1&skip_disambig=1`,
        { signal: ctx.signal },
      );
      if (r.ok) {
        const d: any = await r.json();
        const related = Array.isArray(d.RelatedTopics)
          ? d.RelatedTopics.filter((t: any) => t.Text)
              .slice(0, 5)
              .map((t: any) => ({ text: t.Text, url: t.FirstURL }))
          : [];
        if (d.AbstractText || related.length) {
          return {
            source: "duckduckgo",
            answer: d.AbstractText || d.Answer || "",
            abstractUrl: d.AbstractURL || undefined,
            heading: d.Heading || undefined,
            related,
          };
        }
      }
    } catch {
      /* fall through */
    }

    // 4) Wikipedia REST summary
    try {
      const wr = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(q)}`, {
        signal: ctx.signal,
      });
      if (wr.ok) {
        const w: any = await wr.json();
        if (w.extract) return { source: "wikipedia", title: w.title, answer: w.extract, url: w.content_urls?.desktop?.page };
      }
    } catch {
      /* ignore */
    }

    return {
      source: "none",
      answer: "",
      note:
        "All search backends returned nothing. Add a keenable.ai key in the Tools panel, or try web_extract on a specific URL.",
    };
  },
};
