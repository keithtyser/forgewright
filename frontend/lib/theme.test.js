// Assertions for the shared color/theme layer (run: node lib/theme.test.js).
'use strict';
const assert = require('assert');
const { makeTheme, resolveThemeName, escMarkup, mkVlen, ansiVlen } = require('./theme');

// dark is the proven palette: must stay byte-identical so the classic UI does not change
const dark = makeTheme('dark');
assert.strictEqual(dark.ansi('cyan'), '\x1b[36m');
assert.strictEqual(dark.ansi('brightWhite'), '\x1b[97m');
assert.strictEqual(dark.reset, '\x1b[0m');
assert.strictEqual(dark.mk('brightCyan'), '^C');
assert.strictEqual(dark.paint('green', 'hi'), '\x1b[32mhi\x1b[0m');

// role -> color name -> sequence, one source of truth (shared with lib/render)
assert.strictEqual(dark.ansi(dark.roleColor('Quantizer')), '\x1b[93m');   // brightYellow
assert.strictEqual(dark.roleColor('nope'), 'white');

// mono emits NO color anywhere (accessibility / piping / NO_COLOR)
const mono = makeTheme('mono');
assert.strictEqual(mono.ansi('red'), '');
assert.strictEqual(mono.mk('red'), '');
assert.strictEqual(mono.reset, '');
assert.strictEqual(mono.paint('red', 'x'), 'x');
assert.strictEqual(mono.paintMk('red', 'x'), 'x');

// light swaps near-white text for dark ink so it is readable on a light background
const light = makeTheme('light');
assert.strictEqual(light.ansi('brightWhite'), '\x1b[30m');

// theme name resolution: explicit > FORGEWRIGHT_THEME > NO_COLOR > dark
assert.strictEqual(resolveThemeName('light'), 'light');
{
  const save = { t: process.env.FORGEWRIGHT_THEME, n: process.env.NO_COLOR };
  delete process.env.FORGEWRIGHT_THEME; delete process.env.NO_COLOR;
  assert.strictEqual(resolveThemeName(), 'dark');
  process.env.NO_COLOR = '1';
  assert.strictEqual(resolveThemeName(), 'mono');
  process.env.FORGEWRIGHT_THEME = 'light';
  assert.strictEqual(resolveThemeName(), 'light');   // explicit env beats NO_COLOR
  if (save.t == null) delete process.env.FORGEWRIGHT_THEME; else process.env.FORGEWRIGHT_THEME = save.t;
  if (save.n == null) delete process.env.NO_COLOR; else process.env.NO_COLOR = save.n;
}

// markup safety + measurement helpers
assert.strictEqual(escMarkup('a^b^^c'), 'a^^b^^^^c');
assert.strictEqual(mkVlen('^Rhi^:'), 2);          // ^R and ^: are zero-width
assert.strictEqual(mkVlen('a^^b'), 3);            // ^^ is one literal caret
assert.strictEqual(ansiVlen('\x1b[31mhi\x1b[0m'), 2);

console.log('theme.test.js: all assertions passed');
