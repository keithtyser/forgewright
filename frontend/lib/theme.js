// One color system for BOTH front-ends (classic ANSI + full-screen terminal-kit). The renderer
// (lib/render.js) speaks in semantic color NAMES; this module resolves a name to a raw ANSI
// sequence OR a terminal-kit markup prefix, per the active theme. That is the single source of
// truth for color, so the two surfaces can never drift, and it adds real theming:
//   FORGEWRIGHT_THEME=dark|light|mono   (default: dark; NO_COLOR -> mono).
// 'mono' emits no color at all (accessibility / dumb terminals / piping).
'use strict';

const { ROLE_COLOR } = require('./render');

// color name -> ANSI SGR foreground. 'dark' is the proven palette (kept byte-identical so the
// classic UI does not visually change); 'light' swaps the near-white/gray tones for dark ink.
const ANSI_DARK = {
  reset: '\x1b[0m', bold: '\x1b[1m',
  gray: '\x1b[90m', white: '\x1b[37m', brightWhite: '\x1b[97m',
  cyan: '\x1b[36m', brightCyan: '\x1b[96m', green: '\x1b[32m', brightGreen: '\x1b[92m',
  red: '\x1b[31m', brightRed: '\x1b[91m', yellow: '\x1b[33m', brightYellow: '\x1b[93m',
  blue: '\x1b[34m', brightBlue: '\x1b[94m', magenta: '\x1b[35m', brightMagenta: '\x1b[95m',
};
const ANSI_LIGHT = Object.assign({}, ANSI_DARK, {
  white: '\x1b[30m', brightWhite: '\x1b[30m', gray: '\x1b[90m',
  brightCyan: '\x1b[36m', brightGreen: '\x1b[32m', brightYellow: '\x1b[33m',
  brightBlue: '\x1b[34m', brightMagenta: '\x1b[35m', brightWhite_: '\x1b[30m',
});

// color name -> terminal-kit markup prefix (^x). Theme-independent shape; 'mono' blanks them.
const MK = {
  gray: '^K', white: '^w', brightWhite: '^W', cyan: '^c', brightCyan: '^C',
  green: '^g', brightGreen: '^G', red: '^r', brightRed: '^R', yellow: '^y',
  brightYellow: '^Y', blue: '^b', brightBlue: '^B', magenta: '^m', brightMagenta: '^M',
};

function resolveThemeName(explicit) {
  const name = (explicit || process.env.FORGEWRIGHT_THEME || '').toLowerCase();
  if (name === 'dark' || name === 'light' || name === 'mono') return name;
  if (process.env.NO_COLOR != null && process.env.NO_COLOR !== '') return 'mono';
  return 'dark';
}

function makeTheme(explicit) {
  const name = resolveThemeName(explicit);
  const mono = name === 'mono';
  const ansiTable = name === 'light' ? ANSI_LIGHT : ANSI_DARK;
  const ansi = (color) => (mono ? '' : (ansiTable[color] || ansiTable.white));
  const mk = (color) => (mono ? '' : (MK[color] || '^w'));
  const reset = mono ? '' : '\x1b[0m';
  const bold = mono ? '' : '\x1b[1m';
  const mkReset = mono ? '' : '^:';
  const mkBold = mono ? '' : '^+';
  return {
    name, mono,
    ansi, mk, reset, bold, mkReset, mkBold,
    // role -> its semantic color name (single source: lib/render.ROLE_COLOR)
    roleColor: (role) => ROLE_COLOR[role] || 'white',
    // paint a string in ANSI (dynamic/untrusted text is safe in raw ANSI)
    paint: (color, text, b) => mono ? String(text) : (ansi(color) + (b ? bold : '') + text + reset),
    // paint a string as terminal-kit markup (caller passes already caret-escaped text)
    paintMk: (color, escaped, b) => mono ? String(escaped) : (mk(color) + (b ? mkBold : '') + escaped + mkReset),
  };
}

// caret-escape untrusted text so it cannot inject terminal-kit markup (^ -> ^^)
const escMarkup = (s) => String(s == null ? '' : s).replace(/\^/g, '^^');
// visible length of a terminal-kit markup string (^x are 0-width; ^^ is one literal caret)
function mkVlen(s) {
  s = String(s); let n = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '^') { i += 1; if (s[i] === '^') n += 1; } else { n += 1; }
  }
  return n;
}
// visible length of an ANSI string (SGR codes stripped)
const ANSI_RE = /\x1b\[[0-9;]*m/g;
const ansiVlen = (s) => String(s).replace(ANSI_RE, '').length;

module.exports = { makeTheme, resolveThemeName, escMarkup, mkVlen, ansiVlen, ANSI_RE, MK };
