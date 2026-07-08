/**
 * The agent's tool set, backed by the in-browser virtual filesystem.
 *
 * These are lean AgentTool objects (no pi-tui renderers — our React UI renders
 * from the AgentEvent stream) that preserve pi's tool schemas, descriptions and
 * — for edit — the exact apply/diff semantics. `bash` is intentionally a stub:
 * there is no shell in the browser (the chosen trade-off), so it returns a
 * clear, model-actionable error.
 */
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { type Static, Type } from "typebox";
import type { FsBackend } from "../fs/backend";
import { dirname, resolveToCwd } from "../fs/path";
import {
  applyEditsToNormalizedContent,
  detectLineEnding,
  type Edit,
  generateDiffString,
  generateUnifiedPatch,
  normalizeToLF,
  restoreLineEndings,
  stripBom,
} from "./editDiff";
import { withFileMutationQueue } from "./mutationQueue";

const MAX_LINES = 2000;
const MAX_BYTES = 50 * 1024;

function textResult(text: string, details: unknown = undefined): AgentToolResult<unknown> {
  return { content: [{ type: "text", text }], details };
}

/** Head truncation by line and byte limits (whichever hits first). */
function truncateHead(content: string): { content: string; truncated: boolean; outputLines: number; totalLines: number } {
  const lines = content.split("\n");
  const totalLines = lines.length;
  let kept = lines.slice(0, MAX_LINES);
  let truncated = kept.length < totalLines;
  let out = kept.join("\n");
  if (new TextEncoder().encode(out).length > MAX_BYTES) {
    // Trim lines until under the byte budget.
    while (kept.length > 1 && new TextEncoder().encode(kept.join("\n")).length > MAX_BYTES) {
      kept = kept.slice(0, Math.floor(kept.length * 0.9) || 1);
    }
    out = kept.join("\n");
    truncated = true;
  }
  return { content: out, truncated, outputLines: kept.length, totalLines };
}

/* -------------------------------------------------------------------- read --- */

const readSchema = Type.Object({
  path: Type.String({ description: "Path to the file to read (relative or absolute)" }),
  offset: Type.Optional(Type.Number({ description: "Line number to start reading from (1-indexed)" })),
  limit: Type.Optional(Type.Number({ description: "Maximum number of lines to read" })),
});

function createReadTool(fs: FsBackend, cwd: string): AgentTool<typeof readSchema> {
  return {
    name: "read",
    label: "read",
    description: `Read the contents of a text file. Output is truncated to ${MAX_LINES} lines or ${MAX_BYTES / 1024}KB (whichever is hit first). Use offset/limit for large files.`,
    parameters: readSchema,
    async execute(_id, { path, offset, limit }: Static<typeof readSchema>) {
      const abs = resolveToCwd(path, cwd);
      const stat = await fs.stat(abs);
      if (!stat) throw new Error(`File not found: ${path}`);
      if (stat.kind !== "file") throw new Error(`Not a file: ${path}`);

      const raw = await fs.readText(abs);
      const allLines = raw.split("\n");
      const startLine = offset ? Math.max(0, offset - 1) : 0;
      if (startLine >= allLines.length) {
        throw new Error(`Offset ${offset} is beyond end of file (${allLines.length} lines total)`);
      }
      const endLine = limit !== undefined ? Math.min(startLine + limit, allLines.length) : allLines.length;
      const selected = allLines.slice(startLine, endLine).join("\n");
      const trunc = truncateHead(selected);
      let text = trunc.content;
      if (trunc.truncated) {
        const shownEnd = startLine + trunc.outputLines;
        text += `\n\n[Showing lines ${startLine + 1}-${shownEnd} of ${allLines.length}. Use offset=${shownEnd + 1} to continue.]`;
      } else if (endLine < allLines.length) {
        text += `\n\n[${allLines.length - endLine} more lines in file. Use offset=${endLine + 1} to continue.]`;
      }
      return textResult(text);
    },
  };
}

/* ------------------------------------------------------------------- write --- */

const writeSchema = Type.Object({
  path: Type.String({ description: "Path to the file to write (relative or absolute)" }),
  content: Type.String({ description: "Content to write to the file" }),
});

function createWriteTool(fs: FsBackend, cwd: string): AgentTool<typeof writeSchema> {
  return {
    name: "write",
    label: "write",
    description:
      "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
    parameters: writeSchema,
    async execute(_id, { path, content }: Static<typeof writeSchema>) {
      const abs = resolveToCwd(path, cwd);
      return withFileMutationQueue(abs, async () => {
        await fs.mkdirp(dirname(abs));
        await fs.writeText(abs, content);
        return textResult(`Successfully wrote ${content.length} bytes to ${path}`);
      });
    },
  };
}

/* -------------------------------------------------------------------- edit --- */

const replaceEditSchema = Type.Object(
  {
    oldText: Type.String({
      description:
        "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
    }),
    newText: Type.String({ description: "Replacement text for this targeted edit." }),
  },
  { additionalProperties: false },
);

const editSchema = Type.Object(
  {
    path: Type.String({ description: "Path to the file to edit (relative or absolute)" }),
    edits: Type.Array(replaceEditSchema, {
      description:
        "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits.",
    }),
  },
  { additionalProperties: false },
);

export interface EditToolDetails {
  diff: string;
  patch: string;
  firstChangedLine?: number;
}

/** Mirror pi's prepareArguments: tolerate edits sent as a JSON string / legacy single-edit shape. */
function prepareEditArguments(input: unknown): Static<typeof editSchema> {
  if (!input || typeof input !== "object") return input as Static<typeof editSchema>;
  const args = input as Record<string, unknown>;
  if (typeof args.edits === "string") {
    try {
      const parsed = JSON.parse(args.edits);
      if (Array.isArray(parsed)) args.edits = parsed;
    } catch {
      /* leave as-is; validation will surface it */
    }
  }
  if (typeof args.oldText === "string" && typeof args.newText === "string") {
    const edits = Array.isArray(args.edits) ? [...(args.edits as Edit[])] : [];
    edits.push({ oldText: args.oldText as string, newText: args.newText as string });
    const { oldText: _o, newText: _n, ...rest } = args;
    return { ...rest, edits } as Static<typeof editSchema>;
  }
  return args as Static<typeof editSchema>;
}

function createEditTool(fs: FsBackend, cwd: string): AgentTool<typeof editSchema, EditToolDetails> {
  return {
    name: "edit",
    label: "edit",
    description:
      "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. Merge nearby changes into one edit. Keep oldText as small as possible while still unique.",
    parameters: editSchema,
    prepareArguments: prepareEditArguments,
    async execute(_id, input: Static<typeof editSchema>) {
      const { path, edits } = input;
      if (!Array.isArray(edits) || edits.length === 0) {
        throw new Error("Edit tool input is invalid. edits must contain at least one replacement.");
      }
      const abs = resolveToCwd(path, cwd);
      return withFileMutationQueue(abs, async () => {
        const stat = await fs.stat(abs);
        if (!stat) throw new Error(`Could not edit file: ${path}. File not found.`);

        const rawContent = await fs.readText(abs);
        const { bom, text: content } = stripBom(rawContent);
        const originalEnding = detectLineEnding(content);
        const normalizedContent = normalizeToLF(content);
        const { baseContent, newContent } = applyEditsToNormalizedContent(normalizedContent, edits, path);

        const finalContent = bom + restoreLineEndings(newContent, originalEnding);
        await fs.writeText(abs, finalContent);

        const diffResult = generateDiffString(baseContent, newContent);
        const patch = generateUnifiedPatch(path, baseContent, newContent);
        return {
          content: [{ type: "text", text: `Successfully replaced ${edits.length} block(s) in ${path}.` }],
          details: { diff: diffResult.diff, patch, firstChangedLine: diffResult.firstChangedLine },
        };
      });
    },
  };
}

/* --------------------------------------------------------------------- ls ---- */

const lsSchema = Type.Object({
  path: Type.Optional(Type.String({ description: "Directory to list (defaults to the project root)" })),
});

function createLsTool(fs: FsBackend, cwd: string): AgentTool<typeof lsSchema> {
  return {
    name: "ls",
    label: "ls",
    description: "List files and directories at a path (defaults to the project root).",
    parameters: lsSchema,
    async execute(_id, { path }: Static<typeof lsSchema>) {
      const abs = path ? resolveToCwd(path, cwd) : cwd;
      const entries = await fs.list(abs);
      if (entries.length === 0) return textResult("(empty)");
      const text = entries.map((e) => (e.kind === "directory" ? `${e.name}/` : e.name)).join("\n");
      return textResult(text);
    },
  };
}

/* ------------------------------------------------------------------- bash ---- */

const bashSchema = Type.Object({
  command: Type.String({ description: "Shell command (NOT available — use the file tools instead)" }),
});

// Soyuz is trained to reach for bash; if it's not registered the loop returns a
// bare "Tool bash not found". Register a stub that redirects it to the real
// file tools so it adapts instead of stalling.
function createBashStubTool(): AgentTool<typeof bashSchema> {
  return {
    name: "bash",
    label: "bash",
    description: "Run a shell command. NOT available in the browser — use ls/read/edit/write instead.",
    parameters: bashSchema,
    async execute() {
      throw new Error(
        "There is no shell in this browser environment. Use the file tools instead: `ls` to list files, `read` to view a file, `edit` for a small change, `write` to create/overwrite a file.",
      );
    },
  };
}

/* -------------------------------------------------------------------- grep --- */

const grepSchema = Type.Object({
  pattern: Type.String({ description: "Regular expression to search for (JavaScript syntax)" }),
  path: Type.Optional(Type.String({ description: "File or directory to search in (defaults to the project root)" })),
  ignoreCase: Type.Optional(Type.Boolean({ description: "Case-insensitive search (default false)" })),
  literal: Type.Optional(Type.Boolean({ description: "Treat pattern as a literal string, not a regex (default false)" })),
  maxResults: Type.Optional(Type.Number({ description: "Maximum number of matching lines to return (default 100)" })),
});

const GREP_MAX_RESULTS = 100;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function createGrepTool(fs: FsBackend, cwd: string): AgentTool<typeof grepSchema> {
  return {
    name: "grep",
    label: "grep",
    description:
      "Search file contents for a regular expression. Returns matching lines as path:line: text. Searches the whole project by default; pass path to narrow to a file or directory.",
    parameters: grepSchema,
    async execute(_id, { pattern, path, ignoreCase, literal, maxResults }: Static<typeof grepSchema>) {
      const source = literal ? escapeRegExp(pattern) : pattern;
      let re: RegExp;
      try {
        re = new RegExp(source, ignoreCase ? "i" : undefined);
      } catch (e) {
        throw new Error(`Invalid regex: ${e instanceof Error ? e.message : String(e)}`);
      }
      const abs = path ? resolveToCwd(path, cwd) : cwd;
      const stat = await fs.stat(abs);
      if (!stat) throw new Error(`Path not found: ${path ?? cwd}`);
      const files = stat.kind === "file" ? [abs] : await fs.walk(abs);
      const limit = Math.max(1, maxResults ?? GREP_MAX_RESULTS);

      const out: string[] = [];
      let truncated = false;
      for (const file of files) {
        let text: string;
        try {
          text = await fs.readText(file);
        } catch {
          continue; // unreadable/binary — skip
        }
        const lines = text.split("\n");
        for (let i = 0; i < lines.length; i++) {
          if (re.test(lines[i])) {
            const rel = file.startsWith(cwd + "/") ? file.slice(cwd.length + 1) : file;
            const line = lines[i].length > 500 ? lines[i].slice(0, 500) + "…" : lines[i];
            out.push(`${rel}:${i + 1}: ${line}`);
            if (out.length >= limit) {
              truncated = true;
              break;
            }
          }
        }
        if (truncated) break;
      }
      if (out.length === 0) return textResult("No matches found.");
      let text = out.join("\n");
      if (truncated) text += `\n\n[Truncated at ${limit} matches. Narrow the pattern or pass a path.]`;
      return textResult(text);
    },
  };
}

/* -------------------------------------------------------------------- find --- */

const findSchema = Type.Object({
  pattern: Type.String({
    description: 'Glob-style filename pattern, e.g. "*.ts", "src/**/*.css", "index*". Matched against project-relative paths.',
  }),
  path: Type.Optional(Type.String({ description: "Directory to search under (defaults to the project root)" })),
});

/** Convert a glob to a regex: ** = any path, * = any chars within a segment, ? = one char. */
function globToRegExp(glob: string): RegExp {
  let out = "";
  for (let i = 0; i < glob.length; i++) {
    const ch = glob[i];
    if (ch === "*") {
      if (glob[i + 1] === "*") {
        out += ".*";
        i++;
        if (glob[i + 1] === "/") i++; // "**/" also matches zero directories
      } else {
        out += "[^/]*";
      }
    } else if (ch === "?") {
      out += "[^/]";
    } else {
      out += escapeRegExp(ch);
    }
  }
  return new RegExp(`(^|/)${out}$`);
}

function createFindTool(fs: FsBackend, cwd: string): AgentTool<typeof findSchema> {
  return {
    name: "find",
    label: "find",
    description: "Find files by name using a glob pattern (*, **, ?). Returns project-relative paths.",
    parameters: findSchema,
    async execute(_id, { pattern, path }: Static<typeof findSchema>) {
      const abs = path ? resolveToCwd(path, cwd) : cwd;
      const re = globToRegExp(pattern);
      const files = await fs.walk(abs);
      const matches = files
        .map((f) => (f.startsWith(cwd + "/") ? f.slice(cwd.length + 1) : f))
        .filter((rel) => re.test(rel));
      if (matches.length === 0) return textResult("No files found.");
      return textResult(matches.join("\n"));
    },
  };
}

export function buildTools(fs: FsBackend, cwd: string): AgentTool<any>[] {
  return [
    createReadTool(fs, cwd),
    createWriteTool(fs, cwd),
    createEditTool(fs, cwd),
    createLsTool(fs, cwd),
    createGrepTool(fs, cwd),
    createFindTool(fs, cwd),
    createBashStubTool(),
  ];
}
