import { describe, expect, it } from "vitest";
import { splitRows } from "./diffRows";
import { layoutGraph } from "@/components/run/layout";

describe("splitRows", () => {
  it("pairs changed lines and keeps unchanged ones aligned", () => {
    const rows = splitRows("a\nb\nc\n", "a\nB\nc\nd\n");
    expect(rows.map((r) => [r.left.kind, r.right.kind])).toEqual([
      ["same", "same"],
      ["del", "add"],
      ["same", "same"],
      ["empty", "add"],
    ]);
    expect(rows[1]?.left.text).toBe("b");
    expect(rows[1]?.right.text).toBe("B");
    expect(rows[3]?.right.no).toBe(4);
  });
});

describe("layoutGraph", () => {
  it("lays out left to right by dependency", () => {
    const pos = layoutGraph(
      ["a", "b", "c"],
      [
        { source_node_id: "a", target_node_id: "b" },
        { source_node_id: "b", target_node_id: "c" },
      ],
    );
    expect(pos.a!.x).toBeLessThan(pos.b!.x);
    expect(pos.b!.x).toBeLessThan(pos.c!.x);
  });
});
