// Assertions for the shared motion/format/HUD layer (run: node lib/ui.test.js).
'use strict';
const assert = require('assert');
const ui = require('./ui');
const { makeTheme } = require('./theme');

// formatting primitives
assert.strictEqual(ui.fmtTok(950), '950');
assert.strictEqual(ui.fmtTok(1500), '1.5k');
assert.strictEqual(ui.fmtNum(0.3), '0.300');
assert.strictEqual(ui.fmtNum(12.345), '12.35');
assert.strictEqual(ui.fmtDur(500), '500ms');
assert.strictEqual(ui.fmtDur(4200), '4.2s');
assert.strictEqual(ui.fmtDur(95000), '1m35s');
assert.strictEqual(ui.sparkline([1, 2, 3]).length, 3);
assert.strictEqual(ui.sparkline([5]), '');                 // need >= 2 points

// pipeline normalization accepts an array OR a {stages} wrapper
assert.deepStrictEqual(ui.normalizePipeline([{ name: 'A', state: 'done' }]), [{ name: 'A', state: 'done' }]);
assert.deepStrictEqual(ui.normalizePipeline({ stages: [{ name: 'A' }] }), [{ name: 'A' }]);
assert.strictEqual(ui.normalizePipeline(null), null);

// hudRows: a pipeline row + a status row, as semantic segments (no color codes)
{
  const hud = { start: Date.now(), tokens: 1500, actions: 3, activeRole: 'Quantizer',
    pipeline: [{ name: 'Quantizer', state: 'active' }, { name: 'Evaluator', state: 'pending' }],
    metric: { step: 5, total: 10, loss: 0.3, histLoss: [1, 0.5, 0.3] } };
  const rows = ui.hudRows(hud, '*');
  assert.strictEqual(rows.length, 2);
  assert.strictEqual(rows[0].kind, 'pipeline');
  assert.strictEqual(rows[1].kind, 'status');
  // the active stage's glyph is the spinner; its name is colored by role
  assert.ok(rows[0].segs.some((s) => s.t === '*'));
  assert.ok(rows[0].segs.some((s) => s.t === 'Quantizer' && s.c === 'brightYellow'));
  // status row carries who + step + tokens
  const txt = rows[1].segs.map((s) => s.t).join('');
  assert.ok(txt.includes('Quantizer') && txt.includes('step 5/10') && txt.includes('1.5k tok'));
}

// no pipeline -> only the status row
assert.strictEqual(ui.hudRows({ start: Date.now() }, '*').length, 1);

// painters: ANSI paints with color + truncates to width; mono paints plain text
{
  const dark = makeTheme('dark'); const mono = makeTheme('mono');
  const segs = [{ t: 'hello ', c: 'green' }, { t: 'world', c: 'red' }];
  const ansi = ui.paintRowAnsi(segs, dark, 80);
  assert.ok(ansi.includes('\x1b[32m') && ansi.includes('hello'));
  assert.strictEqual(ui.paintRowAnsi(segs, mono, 80), 'hello world');     // mono = plain
  // width budget: total visible chars never exceed maxw
  const clipped = ui.paintRowAnsi(segs, mono, 8);
  assert.strictEqual(clipped.length, 8);
  assert.ok(clipped.endsWith('…'));
  // markup painter escapes carets and respects width
  const esc = (s) => String(s).replace(/\^/g, '^^');
  const mk = ui.paintRowMk([{ t: 'a^b', c: 'cyan' }], dark, esc, 80);
  assert.ok(mk.includes('a^^b') && mk.includes('^c'));
}

// runSummary: compact end-of-turn line
{
  const segs = ui.runSummary({ start: Date.now() - 5000, actions: 7, tokens: 3100 });
  const txt = segs.map((s) => s.t).join('');
  assert.ok(txt.startsWith('done') && txt.includes('7 actions') && txt.includes('3.1k tok'));
}

console.log('ui.test.js: all assertions passed');
