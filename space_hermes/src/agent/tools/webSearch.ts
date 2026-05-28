import type { Tool } from "../registry";
import { getKeenableKey } from "../../lib/settings";

/** Real search via keenable.ai (when a key is configured), with CORS-open
 *  fallbacks (DuckDuckGo Instant Answer + Wikipedia REST) when it isn't. */
export const webSearchTool: Tool = {
  name: "web_search",
  description:
    "Search the live web. Uses keenable.ai when a key is configured (full results: title, url, snippet) and falls back to DuckDuckGo Instant Answer + Wikipedia summary otherwise.",
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

    // 1) keenable.ai (preferred — real search results)
    const key = getKeenableKey();
    if (key) {
      try {
        const r = await fetch("https://api.keenable.ai/v1/search", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-Key": key },
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
          if (results.length) return { source: "keenable", query: q, results };
        }
      } catch {
        /* fall through */
      }
    }

    // 2) DuckDuckGo Instant Answer
    try {
      const r = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_html=1&skip_disambig=1`, {
        signal: ctx.signal,
      });
      if (r.ok) {
        const d: any = await r.json();
        const related = Array.isArray(d.RelatedTopics)
          ? d.RelatedTopics.filter((t: any) => t.Text).slice(0, 5).map((t: any) => ({ text: t.Text, url: t.FirstURL }))
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

    // 3) Wikipedia REST summary
    try {
      const wr = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(q)}`, { signal: ctx.signal });
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
      note: "No result. Add a keenable.ai key in the Tools panel for real search, or try web_extract on a specific URL.",
    };
  },
};
