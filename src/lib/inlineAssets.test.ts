import { describe, it, expect } from "vitest";
import { inlineAssets } from "./inlineAssets";

const files: Record<string, string> = {
  "/workspace/css/style.css": "body { color: red; }",
  "/workspace/js/app.js": "console.log('hi');",
  "/workspace/js/mod.js": "export const x = 1;",
  "/workspace/tricky.js": "const s = '</script>';",
};

const read = async (p: string) => files[p] ?? null;

describe("inlineAssets", () => {
  it("inlines relative stylesheets and scripts", async () => {
    const html = `<html><head><link rel="stylesheet" href="css/style.css"></head>
<body><script src="js/app.js"></script></body></html>`;
    const out = await inlineAssets(html, "/workspace", read);
    expect(out).toContain("<style>\nbody { color: red; }\n</style>");
    expect(out).toContain("<script>\nconsole.log('hi');\n</script>");
    expect(out).not.toContain('href="css/style.css"');
    expect(out).not.toContain('src="js/app.js"');
  });

  it("preserves type=module and handles ./ and attribute order", async () => {
    const html = `<script type="module" src="./js/mod.js"></script><link href="css/style.css" rel="stylesheet">`;
    const out = await inlineAssets(html, "/workspace", read);
    expect(out).toContain('<script type="module">\nexport const x = 1;\n</script>');
    expect(out).toContain("<style>");
  });

  it("leaves external and missing references untouched", async () => {
    const html = `<link rel="stylesheet" href="https://cdn.example.com/a.css">
<script src="//cdn.example.com/b.js"></script>
<script src="js/missing.js"></script>
<link rel="icon" href="favicon.ico">`;
    const out = await inlineAssets(html, "/workspace", read);
    expect(out).toBe(html);
  });

  it("escapes embedded closing tags so the inline block survives", async () => {
    const out = await inlineAssets(`<script src="tricky.js"></script>`, "/workspace", read);
    expect(out).toContain("<\\/script>");
    // exactly one real closing tag
    expect(out.match(/<\/script>/g)!.length).toBe(1);
  });
});
