import { diffLines } from "diff";

export type SplitCell = { text: string; kind: "same" | "del" | "add" | "empty"; no: number | null };
export type SplitRow = { left: SplitCell; right: SplitCell };

const EMPTY: SplitCell = { text: "", kind: "empty", no: null };

function lines(value: string): string[] {
  const xs = value.split("\n");
  if (xs.length > 1 && xs.at(-1) === "") xs.pop();
  return xs;
}

/** Side-by-side rows from a line diff; removed/added runs are paired line by line. */
export function splitRows(a: string, b: string): SplitRow[] {
  const parts = diffLines(a, b);
  const rows: SplitRow[] = [];
  let l = 1;
  let r = 1;
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i]!;
    if (!p.added && !p.removed) {
      for (const t of lines(p.value)) rows.push({ left: { text: t, kind: "same", no: l++ }, right: { text: t, kind: "same", no: r++ } });
      continue;
    }
    if (p.removed) {
      const dels = lines(p.value);
      const next = parts[i + 1];
      const adds = next?.added ? lines(next.value) : [];
      if (next?.added) i++;
      const n = Math.max(dels.length, adds.length);
      for (let k = 0; k < n; k++) {
        const d = dels[k];
        const a2 = adds[k];
        rows.push({
          left: d !== undefined ? { text: d, kind: "del", no: l++ } : EMPTY,
          right: a2 !== undefined ? { text: a2, kind: "add", no: r++ } : EMPTY,
        });
      }
      continue;
    }
    for (const t of lines(p.value)) rows.push({ left: EMPTY, right: { text: t, kind: "add", no: r++ } });
  }
  return rows;
}
