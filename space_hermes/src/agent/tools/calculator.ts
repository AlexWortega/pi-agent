import type { Tool } from "../registry";

const FUNCS: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  abs: Math.abs,
  ln: Math.log,
  log: Math.log10,
  exp: Math.exp,
  round: Math.round,
  floor: Math.floor,
  ceil: Math.ceil,
};
const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E };

/** Tiny safe arithmetic evaluator (no eval/Function): tokenise → shunting-yard
 *  → evaluate RPN. Supports + - * / % ^, parentheses, unary minus, the funcs
 *  above and pi/e. */
function evaluate(expr: string): number {
  const tokens = expr.match(/\d+\.?\d*|\.\d+|[a-zA-Z_]+|[()+\-*/%^,]/g);
  if (!tokens) throw new Error("empty expression");

  const out: (number | string)[] = [];
  const ops: string[] = [];
  const prec: Record<string, number> = { "+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3, u: 4 };
  const rightAssoc = new Set(["^", "u"]);
  let prev: string | null = null;

  const popWhile = (cond: (top: string) => boolean) => {
    while (ops.length && cond(ops[ops.length - 1])) out.push(ops.pop()!);
  };

  for (const t of tokens) {
    if (/^(\d|\.)/.test(t)) {
      out.push(parseFloat(t));
      prev = "num";
    } else if (t in CONSTS) {
      out.push(CONSTS[t]);
      prev = "num";
    } else if (t in FUNCS) {
      ops.push(t);
      prev = "func";
    } else if (t === "(") {
      ops.push(t);
      prev = "(";
    } else if (t === ")") {
      popWhile((top) => top !== "(");
      if (!ops.length) throw new Error("mismatched parentheses");
      ops.pop(); // discard "("
      if (ops.length && ops[ops.length - 1] in FUNCS) out.push(ops.pop()!);
      prev = "num";
    } else {
      // operator
      let op = t;
      if (op === "-" && (prev === null || prev === "(" || prev === "op")) op = "u"; // unary minus
      popWhile((top) => top in prec && (rightAssoc.has(op) ? prec[top] > prec[op] : prec[top] >= prec[op]));
      ops.push(op);
      prev = "op";
    }
  }
  while (ops.length) {
    const op = ops.pop()!;
    if (op === "(") throw new Error("mismatched parentheses");
    out.push(op);
  }

  const st: number[] = [];
  for (const tok of out) {
    if (typeof tok === "number") st.push(tok);
    else if (tok === "u") st.push(-st.pop()!);
    else if (tok in FUNCS) st.push(FUNCS[tok](st.pop()!));
    else {
      const b = st.pop()!;
      const a = st.pop()!;
      st.push(
        tok === "+" ? a + b : tok === "-" ? a - b : tok === "*" ? a * b : tok === "/" ? a / b : tok === "%" ? a % b : Math.pow(a, b),
      );
    }
  }
  if (st.length !== 1 || !isFinite(st[0])) throw new Error("could not evaluate");
  return st[0];
}

export const calculatorTool: Tool = {
  name: "calculator",
  description: "Evaluate a math expression (supports + - * / % ^, parentheses, sqrt/sin/cos/ln/log/abs… and pi/e).",
  parameters: {
    type: "object",
    properties: { expression: { type: "string", description: "e.g. '840 * 0.125' or 'sqrt(2) * pi'" } },
    required: ["expression"],
  },
  async run(args) {
    const expr = String(args?.expression ?? "");
    const result = evaluate(expr);
    return { expression: expr, result };
  },
};
