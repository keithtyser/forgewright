// Shared presentation primitives for BOTH front-ends: spinner/sparkline motion, number/token/
// duration formatting, and a STRUCTURED HUD model. hudRows() returns rows of semantic segments
// ({ t: text, c: colorName, b: bold }); each surface paints them with its own width-budgeted
// painter (ANSI or terminal-kit markup) so the swarm HUD is defined exactly once.
'use strict';

const { roleColor } = require('./render');

const GLYPHS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPARK = '▁▂▃▄▅▆▇█';

function sparkline(a) {
  if (!a || a.length < 2) return '';
  const lo = Math.min.apply(null, a), hi = Math.max.apply(null, a), rng = (hi - lo) || 1;
  return a.map((v) => SPARK[Math.min(7, Math.floor((v - lo) / rng * 7.999))]).join('');
}
const fmtTok = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n));
const fmtNum = (v) => { const n = Number(v); return Number.isFinite(n) ? (Math.abs(n) < 1 ? n.toFixed(3) : n.toFixed(2)) : String(v); };
function fmtDur(ms) {
  ms = Number(ms) || 0;
  if (ms < 1000) return Math.round(ms) + 'ms';
  if (ms < 60000) return (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + 's';
  const m = Math.floor(ms / 60000), s = Math.round((ms % 60000) / 1000);
  return m + 'm' + (s ? s + 's' : '');
}

// stage state -> { glyph, color }; `spin` is the current spinner frame for the active stage.
function stageGlyph(state, spin) {
  if (state === 'done') return { glyph: '✓', color: 'green' };
  if (state === 'failed') return { glyph: '✗', color: 'red' };
  if (state === 'active') return { glyph: spin, color: 'magenta' };
  return { glyph: '◌', color: 'gray' };
}

function normalizePipeline(pipeline) {
  if (!pipeline) return null;
  const stages = Array.isArray(pipeline) ? pipeline : pipeline.stages;
  return (stages && stages.length) ? stages : null;
}

// Build the HUD as rows of semantic segments. Pure: no color codes, no width logic.
// hud: { start, tokens, actions, pipeline, metric, activeRole, lastAction }; spin: a glyph char.
function hudRows(hud, spin) {
  const rows = [];
  const stages = normalizePipeline(hud.pipeline);
  if (stages) {
    const segs = [{ t: '  ' }];
    stages.forEach((s, idx) => {
      if (idx) segs.push({ t: '   ' });
      const g = stageGlyph(s.state, spin);
      segs.push({ t: g.glyph, c: g.color });
      segs.push({ t: ' ' });
      const lit = s.state === 'active' || s.state === 'done' || s.state === 'failed';
      segs.push({ t: s.name, c: lit ? roleColor(s.name) : 'gray' });
    });
    rows.push({ kind: 'pipeline', segs });
  }

  const m = hud.metric;
  const el = Math.round((Date.now() - (hud.start || Date.now())) / 1000);
  const who = (hud.activeRole && hud.activeRole !== 'agent') ? hud.activeRole : 'working';
  const segs = [{ t: '  ' }, { t: spin, c: 'magenta' }, { t: ' ' }, { t: who, c: roleColor(hud.activeRole || '') }];
  const sep = () => segs.push({ t: ' · ', c: 'gray' });
  if (m && m.step != null) { sep(); segs.push({ t: 'step ' + m.step + (m.total ? '/' + m.total : ''), c: 'brightWhite' }); }
  if (m && m.loss != null) {
    sep(); segs.push({ t: 'loss ' + fmtNum(m.loss), c: 'gray' });
    const sp = sparkline(m.histLoss); if (sp) { segs.push({ t: ' ' }); segs.push({ t: sp, c: 'cyan' }); }
  }
  if (m && m.reward != null) {
    sep(); segs.push({ t: 'rwd ' + fmtNum(m.reward), c: 'gray' });
    const sp = sparkline(m.histReward); if (sp) { segs.push({ t: ' ' }); segs.push({ t: sp, c: 'green' }); }
  }
  if (m && m.grad_norm != null) { sep(); segs.push({ t: 'grad ' + fmtNum(m.grad_norm), c: 'gray' }); }
  if (!m && hud.lastAction) { sep(); segs.push({ t: String(hud.lastAction), c: 'gray' }); }
  if (hud.actions) { sep(); segs.push({ t: hud.actions + ' actions', c: 'gray' }); }
  sep(); segs.push({ t: el + 's', c: 'gray' });
  if (hud.tokens) { sep(); segs.push({ t: '↑' + fmtTok(hud.tokens) + ' tok', c: 'gray' }); }
  rows.push({ kind: 'status', segs });
  return rows;
}

// A compact end-of-turn summary line (segments): "done · 42s · 7 actions · ↑3.1k tok".
function runSummary(hud) {
  const el = Math.round((Date.now() - (hud.start || Date.now())) / 1000);
  const segs = [{ t: 'done', c: 'gray' }, { t: ' · ' + el + 's', c: 'gray' }];
  if (hud.actions) segs.push({ t: ' · ' + hud.actions + ' actions', c: 'gray' });
  if (hud.tokens) segs.push({ t: ' · ↑' + fmtTok(hud.tokens) + ' tok', c: 'gray' });
  return segs;
}

// --- width-budgeted painters: segments -> a single line, truncated to `maxw` visible cols -----

function _truncate(text, room) {
  if (text.length <= room) return { text, used: text.length };
  if (room <= 1) return { text: '…'.slice(0, Math.max(0, room)), used: Math.max(0, room) };
  return { text: text.slice(0, room - 1) + '…', used: room };
}

// ANSI painter (dynamic text painted in raw ANSI, never through markup)
function paintRowAnsi(segs, theme, maxw) {
  let vis = 0; let out = '';
  for (const s of segs) {
    if (vis >= maxw) break;
    const { text, used } = _truncate(String(s.t == null ? '' : s.t), maxw - vis);
    if (!text) continue;
    out += theme.paint(s.c || 'white', text, s.b);
    vis += used;
  }
  return out;
}

// terminal-kit markup painter (text caret-escaped by theme.paintMk's caller via escMarkup)
function paintRowMk(segs, theme, escMarkup, maxw) {
  let vis = 0; let out = '';
  for (const s of segs) {
    if (vis >= maxw) break;
    const { text, used } = _truncate(String(s.t == null ? '' : s.t), maxw - vis);
    if (!text) continue;
    out += theme.paintMk(s.c || 'white', escMarkup(text), s.b);
    vis += used;
  }
  return out;
}

module.exports = {
  GLYPHS, SPARK, sparkline, fmtTok, fmtNum, fmtDur, stageGlyph, normalizePipeline,
  hudRows, runSummary, paintRowAnsi, paintRowMk,
};
