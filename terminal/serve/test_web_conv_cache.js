// Monitor's Conversation cache card (serve/web/app.js: convCacheView + renderConvCache), run in Node.
//   node serve/test_web_conv_cache.js
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const appSrc = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8");
assert.ok(appSrc.includes("renderConvCache(m.conversation_cache, m.conversation_parent_parking)"),
          "render() must pass both conversation_cache and conversation_parent_parking");

function sliceFn(src, startSig, endMarker) {
  const start = src.indexOf(startSig);
  const end = src.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `missing block ${startSig}`);
  return src.slice(start, end);
}

const helpersSrc = [
  sliceFn(appSrc, "const esc = ", "\nconst fmt = "),
  sliceFn(appSrc, "function since(t)", "\nfunction renderTotals("),
  sliceFn(appSrc, "function facts(el, rows)", "\n// INFO cvec="),
].join("\n");

const ids = ["cc-card", "cc-bars", "cc-slots-text", "cc-slots-bar", "cc-mem-text", "cc-mem-bar",
             "cc-sum", "cc-facts", "cc-note"];
const dom = Object.fromEntries(ids.map((id) => [id, {hidden: false, textContent: "", innerHTML: "", style: {width: ""}}]));
const sandbox = {
  $: (id) => dom[id],
  icon: (name) => `<svg>${name}</svg>`,
  fmt: (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d})),
  gb: (b, d = 1) => (b == null ? "–" : sandbox.fmt(b / 1073741824, d)),
  Date, Number, Math, Array, Boolean, String, Object,
};
const ctx = vm.createContext(sandbox);
vm.runInContext(helpersSrc, ctx);

// 1. Parent RAM parking overrides per-replica r0 placeholders and surfaces peer_reads
const GiB = 1073741824;
ctx.renderConvCache(
  {enabled: true, budget_mib: 20480, slots: 10, parked: 0, bytes: 0, parks: 0, restores: 0, evictions: 0,
   last_event: "parked", last_tokens: 12000, last_at: Date.now() / 1000 - 30,
   requests: 10, requests_reused: 7, reused_tokens: 45000, prompt_tokens: 60000, last_reused: 8000, last_prompt: 9000},
  {enabled: true, capacity_bytes: 40 * GiB, max_entries: 20, charged_bytes: 12 * GiB, entries: 3,
   published: 5, reads: 3, peer_reads: 2, evictions: 1}
);
assert.equal(dom["cc-card"].hidden, false);
assert.equal(dom["cc-bars"].hidden, false);
assert.equal(dom["cc-slots-text"].textContent, "3 / 20");
assert.equal(dom["cc-slots-bar"].style.width, "15%");
assert.equal(dom["cc-mem-text"].textContent, "12.0 / 40.0 GB");
assert.equal(dom["cc-mem-bar"].style.width, "30%");
assert.equal(dom["cc-sum"].textContent, "7 of 10 requests reused part of their prompt");
assert.ok(dom["cc-facts"].innerHTML.includes("<dt>Parked / restored</dt><dd>5 / 3 · 2 cross-replica · 1 evicted</dd>"),
          dom["cc-facts"].innerHTML);
assert.ok(!dom["cc-facts"].innerHTML.includes("Last switch"), "parent mode must not show stale replica-0 heap switch");
assert.ok(dom["cc-note"].textContent.includes("Shared RAM parking across all replicas"));

// 2. Parent parking with 0 peer_reads still shows "0 cross-replica" so the transfer metric stays visible
ctx.renderConvCache(
  {enabled: true, requests: 0},
  {enabled: true, capacity_bytes: 40 * GiB, max_entries: 20, charged_bytes: 0, entries: 0,
   published: 1, reads: 0, peer_reads: 0, evictions: 0}
);
assert.ok(dom["cc-facts"].innerHTML.includes("<dt>Parked / restored</dt><dd>1 / 0 · 0 cross-replica</dd>"));

// 3. Legacy multi-replica / disk handoff uses c.total and total slot/RAM budget
ctx.renderConvCache(
  {enabled: true, budget_mib: 8192, slots: 8, parked: 0, bytes: 0, parks: 0, restores: 0, evictions: 0,
   replicas: [{replica: 0}, {replica: 1}],
   total: {parked: 4, bytes: 16 * GiB, parks: 6, restores: 5, peer_reads: 3, evictions: 1}},
  null
);
assert.equal(dom["cc-slots-text"].textContent, "4 / 16");
assert.equal(dom["cc-mem-text"].textContent, "16.0 / 16.0 GB");
assert.ok(dom["cc-facts"].innerHTML.includes("<dt>Parked / restored</dt><dd>6 / 5 · 3 cross-replica · 1 evicted</dd>"));

// 4. Single-engine disabled hides bars and omits Parked / restored
ctx.renderConvCache({enabled: false, requests: 2, requests_reused: 1, reused_tokens: 100, prompt_tokens: 200}, null);
assert.equal(dom["cc-card"].hidden, false);
assert.equal(dom["cc-bars"].hidden, true);
assert.ok(!dom["cc-facts"].innerHTML.includes("Parked / restored"));

// 5. Missing both objects hides the card
ctx.renderConvCache(null, null);
assert.equal(dom["cc-card"].hidden, true);

console.log("ok web conv cache");
