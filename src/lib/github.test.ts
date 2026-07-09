import { describe, it, expect } from "vitest";
import { computeGitBlobSha, diffWorkspace, parseRepoRef, type RepoMeta } from "./github";
import { getFsBackend } from "../pi/fs/backend";

describe("parseRepoRef", () => {
  it("parses owner/repo forms", () => {
    expect(parseRepoRef("copy/v86")).toEqual({ owner: "copy", repo: "v86", branch: undefined });
    expect(parseRepoRef("copy/v86@wip")).toEqual({ owner: "copy", repo: "v86", branch: "wip" });
    expect(parseRepoRef("copy/v86.git")).toEqual({ owner: "copy", repo: "v86", branch: undefined });
  });

  it("parses github urls", () => {
    expect(parseRepoRef("https://github.com/copy/v86")).toEqual({ owner: "copy", repo: "v86", branch: undefined });
    expect(parseRepoRef("github.com/copy/v86/tree/master")).toEqual({ owner: "copy", repo: "v86", branch: "master" });
    expect(parseRepoRef("https://github.com/copy/v86/tree/master/src/browser")).toEqual({
      owner: "copy",
      repo: "v86",
      branch: "master",
    });
    expect(parseRepoRef("https://github.com/copy/v86.git")).toEqual({ owner: "copy", repo: "v86", branch: undefined });
  });

  it("rejects garbage", () => {
    expect(parseRepoRef("")).toBeNull();
    expect(parseRepoRef("just-a-name")).toBeNull();
  });
});

describe("computeGitBlobSha", () => {
  it("matches git hash-object", async () => {
    // $ echo -n 'hello world' | git hash-object --stdin
    expect(await computeGitBlobSha("hello world")).toBe("95d09f2b10159347eece71399a7e2e907ea3df4f");
    // $ printf '' | git hash-object --stdin
    expect(await computeGitBlobSha("")).toBe("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
  });
});

describe("diffWorkspace", () => {
  it("classifies changed, added and deleted files", async () => {
    const fs = getFsBackend();
    const ROOT = "/workspace";
    try {
      await fs.remove(ROOT);
    } catch {
      /* fresh */
    }
    await fs.writeText(`${ROOT}/kept.txt`, "same\n");
    await fs.writeText(`${ROOT}/mod.txt`, "new content\n");
    await fs.writeText(`${ROOT}/fresh.txt`, "added\n");

    const meta: RepoMeta = {
      owner: "o",
      repo: "r",
      branch: "main",
      headSha: "h",
      treeSha: "t",
      files: {
        "kept.txt": await computeGitBlobSha("same\n"),
        "mod.txt": await computeGitBlobSha("old content\n"),
        "gone.txt": await computeGitBlobSha("bye\n"),
      },
      skipped: [],
      importedAt: 0,
    };

    const diff = await diffWorkspace(meta, fs, ROOT);
    expect(diff.changed.map((f) => f.path)).toEqual(["mod.txt"]);
    expect(diff.added.map((f) => f.path)).toEqual(["fresh.txt"]);
    expect(diff.deleted).toEqual(["gone.txt"]);
  });
});
