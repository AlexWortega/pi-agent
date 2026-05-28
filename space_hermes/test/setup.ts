import { afterEach, beforeEach } from "vitest";

// Every test starts with a fresh browser storage so module-level stores
// (memory, skills, todos, scheduler, settings) don't bleed between tests.
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});
afterEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});
