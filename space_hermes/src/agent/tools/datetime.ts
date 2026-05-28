import type { Tool } from "../registry";

export const datetimeTool: Tool = {
  name: "datetime",
  description: "Get the current date and time. Optionally pass an IANA timezone like 'Europe/London'.",
  parameters: {
    type: "object",
    properties: { timezone: { type: "string", description: "IANA tz name, optional" } },
  },
  async run(args) {
    const now = new Date();
    const tz = args?.timezone;
    let local: string;
    try {
      local = now.toLocaleString("en-US", tz ? { timeZone: tz } : undefined);
    } catch {
      local = now.toLocaleString();
    }
    return {
      iso: now.toISOString(),
      local,
      timezone: tz || Intl.DateTimeFormat().resolvedOptions().timeZone,
      unix: Math.floor(now.getTime() / 1000),
      weekday: now.toLocaleDateString("en-US", { weekday: "long" }),
    };
  },
};
