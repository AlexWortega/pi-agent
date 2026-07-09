/**
 * Inline workspace-relative <link rel="stylesheet"> and <script src> references
 * into an HTML document.
 *
 * The preview pane renders index.html via iframe `srcDoc`, where relative URLs
 * resolve against the host page — i.e. never against the OPFS virtual
 * filesystem. Cloud models legitimately split projects into index.html +
 * css/js; without inlining, the preview would render the app unstyled and
 * inert. External URLs (http/https/data/protocol-relative) are left untouched.
 */

export type ReadWorkspaceFile = (absPath: string) => Promise<string | null>;

function isExternal(url: string): boolean {
  return /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(url);
}

/** Resolve a relative href against the preview file's directory (POSIX, "/"-rooted). */
function resolveHref(href: string, baseDir: string): string {
  const clean = href.split(/[?#]/)[0];
  const joined = clean.startsWith("/") ? clean : `${baseDir}/${clean}`;
  const out: string[] = [];
  for (const seg of joined.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      out.pop();
      continue;
    }
    out.push(seg);
  }
  return "/" + out.join("/");
}

function escapeClosingTags(code: string, tag: string): string {
  // Prevent an embedded "</style>"/"</script>" sequence from terminating the
  // inline block early.
  return code.replace(new RegExp(`</${tag}`, "gi"), `<\\/${tag}`);
}

export async function inlineAssets(
  html: string,
  baseDir: string,
  readFile: ReadWorkspaceFile,
): Promise<string> {
  let out = html;

  // <link … href="x.css" …> (any attribute order; only stylesheet rels)
  const linkTags = [...out.matchAll(/<link\b[^>]*>/gi)];
  for (const m of linkTags) {
    const tag = m[0];
    if (!/rel=["']?stylesheet["']?/i.test(tag)) continue;
    const href = tag.match(/href=["']([^"']+)["']/i)?.[1];
    if (!href || isExternal(href)) continue;
    const css = await readFile(resolveHref(href, baseDir));
    if (css === null) continue;
    out = out.replace(tag, `<style>\n${escapeClosingTags(css, "style")}\n</style>`);
  }

  // <script … src="x.js" …></script>
  const scriptTags = [...out.matchAll(/<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*>\s*<\/script>/gi)];
  for (const m of scriptTags) {
    const [tag, src] = m;
    if (isExternal(src)) continue;
    const js = await readFile(resolveHref(src, baseDir));
    if (js === null) continue;
    const isModule = /type=["']?module["']?/i.test(tag);
    out = out.replace(
      tag,
      `<script${isModule ? ' type="module"' : ""}>\n${escapeClosingTags(js, "script")}\n</script>`,
    );
  }

  return out;
}
