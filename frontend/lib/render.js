// Pure rendering of backend event objects into colored transcript segments.
// Dependency-free (no terminal-kit) so it can be unit-checked headlessly; the TUI maps
// each segment's color name onto terminal-kit's chainable colors.
//
// formatEvent returns an ARRAY of { text, color } lines (possibly empty), so a single
// event (e.g. an assistant turn that both says something AND calls tools) renders as
// several attributed lines: the model's thinking, then each tool call, etc.
'use strict';

const ROLE_COLOR = {
  Director: 'brightCyan',
  DataCurator: 'brightBlue',
  SFTTrainer: 'brightGreen',
  RLTrainer: 'green',
  Abliterator: 'brightMagenta',
  Quantizer: 'brightYellow',
  ServingOptimizer: 'yellow',
  Evaluator: 'brightWhite',
  Publisher: 'brightRed',
  Merger: 'cyan',
};

function roleColor(role) {
  return ROLE_COLOR[role] || 'white';
}

function clip(s, n) {
  s = String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

// Turn markdown-ish assistant text into clean terminal lines: KEEP the line structure
// (so plans and tables stay readable), drop table-rule lines, and strip the noisy markers
// (#, **bold**, `code`). Collapses runs of blank lines.
function mdToLines(md) {
  const out = [];
  for (let line of String(md == null ? '' : md).replace(/\r/g, '').split('\n')) {
    if (/^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$/.test(line)) continue;   // table rule row
    line = line
      .replace(/\*\*(.+?)\*\*/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/^#{1,6}\s*/, '')
      .replace(/\s+$/, '');
    if (line.length > 240) line = line.slice(0, 239) + '…';
    if (line === '' && out[out.length - 1] === '') continue;          // collapse blank runs
    out.push(line);
  }
  while (out.length && out[out.length - 1] === '') out.pop();
  return out;
}

function argsSummary(args) {
  if (!args || typeof args !== 'object') return '';
  const s = JSON.stringify(args);
  return s === '{}' ? '' : clip(s, 160);
}

// Pick the most MEANINGFUL number from a gate's metrics for the one-line badge: a score-like
// metric first (what a human reads), then any number. Returns [key, value] or null.
const _METRIC_PREF = ['score', 'accuracy', 'pass_rate', 'capability', 'refusal_rate_harmful',
  'speedup', 'output_speedup', 'reward', 'final_loss', 'loss'];
function headlineMetric(m) {
  if (!m || typeof m !== 'object') return null;
  for (const k of _METRIC_PREF) if (typeof m[k] === 'number') return [k, m[k]];
  for (const k of Object.keys(m)) if (typeof m[k] === 'number') return [k, m[k]];
  return null;
}
// an optional pre-formatted duration string (the surface times the step + formats via ui.fmtDur)
const dur = (obj) => (obj && obj.dur ? '  ' + obj.dur : '');

// Returns [{ text, color }, ...] for the transcript (empty array to skip).
function formatEvent(obj) {
  const role = obj.role || '';
  const tag = role ? '[' + role + '] ' : '';
  const out = [];
  switch (obj.type) {
    case 'ready':
      return [{ text: '· backend ready ·', color: 'gray' }];
    case 'bye':
      return [{ text: '· session ended ·', color: 'gray' }];
    case 'assistant': {
      const lines = mdToLines(obj.content);
      lines.forEach((ln, i) => out.push({ text: (i === 0 ? tag : '  ') + ln, color: roleColor(role) }));
      const calls = Array.isArray(obj.tool_calls) ? obj.tool_calls : [];
      if (calls.length) out.push({ text: '  → calling ' + calls.join(', '), color: 'cyan' });
      const u = obj.usage || {};
      if (u.total_tokens) out.push({ text: '    (' + u.total_tokens + ' tokens)', color: 'gray' });
      return out;
    }
    case 'tool': {
      const ok = obj.ok !== false;
      const mark = ok ? '✓' : '✗';
      const name = obj.tool ? obj.tool : '';
      const a = argsSummary(obj.args);
      const head = ('  ' + mark + ' ' + tag + name + (a ? ' ' + a : '') + dur(obj)).trimEnd();
      out.push({ text: head, color: ok ? 'cyan' : 'red' });
      const body = clip(obj.output, 600);
      if (body) out.push({ text: '    ' + body, color: 'gray' });
      return out;
    }
    case 'pipeline': {
      const stages = Array.isArray(obj.stages) ? obj.stages : [];
      return [{ text: '◆ plan ' + stages.join(' → '), color: 'brightCyan' }];
    }
    case 'stage': {
      if (obj.state === 'done') return [{ text: '✓ ' + obj.name + ' complete' + dur(obj), color: 'green' }];
      if (obj.state === 'failed') return [{ text: '✗ ' + obj.name + ' failed' + dur(obj), color: 'red' }];
      if (obj.state === 'retry') return [{ text: '↻ ' + obj.name + ' retry (attempt ' + (obj.attempt || '?') + ')', color: 'yellow' }];
      return [];   // 'active' is shown live in the HUD, not the transcript
    }
    case 'repair': {
      const ch = obj.changes && typeof obj.changes === 'object'
        ? Object.keys(obj.changes).map((k) => k + '=' + obj.changes[k]).join(', ') : '';
      const why = obj.reason ? '  (' + clip(obj.reason, 80) + ')' : '';
      return [{ text: '↻ repair ' + (obj.name || '') + ': ' + ch + why, color: 'yellow' }];
    }
    case 'artifact': {
      const par = Array.isArray(obj.parents) && obj.parents.length
        ? ' ← ' + obj.parents.map((p) => String(p).slice(-6)).join(', ') : '';
      const hm = headlineMetric(obj.metrics);
      const met = hm ? '  ' + hm[0] + ' ' + (Math.round(hm[1] * 1000) / 1000) : '';
      const tag = obj.passed === false ? ' ✗' : (obj.passed === true ? ' ✓' : '');
      return [{ text: '◇ ' + obj.kind + '#' + String(obj.id || '').slice(-6) + par + met + tag + dur(obj), color: roleColor(role) }];
    }
    case 'metric':
    case 'budget':
      return [];   // live HUD-only events; no transcript line
    case 'graph': {
      const nodes = Array.isArray(obj.nodes) ? obj.nodes : [];
      if (!nodes.length) return [{ text: 'provenance graph: (empty)', color: 'gray' }];
      const lines = [{ text: 'provenance graph (' + nodes.length + ' artifacts)', color: 'white' }];
      nodes.forEach((n) => {
        const t = n.passed === false ? ' ✗' : (n.passed === true ? ' ✓' : '');
        const sc = (n.score != null) ? '  ' + (Math.round(n.score * 1000) / 1000) : '';
        const par = (n.parents && n.parents.length) ? ' ← ' + n.parents.map((p) => String(p).slice(-6)).join(', ') : '';
        lines.push({ text: '  ' + n.kind + '#' + String(n.id || '').slice(-6) + par + sc + t, color: roleColor(n.produced_by || '') });
      });
      return lines;
    }
    case 'models': {
      const list = Array.isArray(obj.available) ? obj.available : [];
      const lines = [{ text: 'models (' + (obj.source || '') + ')', color: 'white' }];
      if (obj.note) lines.push({ text: '  ' + obj.note, color: 'gray' });
      list.forEach((mid) => {
        const cur = obj.current && (obj.current === mid || String(obj.current).endsWith(':' + mid));
        lines.push({ text: '  ' + (cur ? '● ' : '· ') + mid + (cur ? ' (current)' : ''), color: cur ? 'green' : 'gray' });
      });
      return lines;
    }
    case 'progress':
      return [{ text: '  ' + tag + clip(obj.text, 500), color: 'gray' }];
    case 'approval_request':
      return [{
        text: '  ⚠ APPROVAL NEEDED: ' + obj.tool + ' (' + (obj.risk || '') + ') ' +
          clip(JSON.stringify(obj.args || {}), 300),
        color: 'brightYellow',
      }];
    case 'done':
      return obj.ok === false
        ? [{ text: '· error: ' + clip(obj.error, 400) + ' ·', color: 'red' }]
        : [{ text: '· done ·', color: 'gray' }];
    default:
      return [{ text: tag + clip(JSON.stringify(obj), 300), color: 'gray' }];
  }
}

module.exports = { formatEvent, roleColor, argsSummary, headlineMetric, ROLE_COLOR };
