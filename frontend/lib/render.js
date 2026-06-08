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
  Publisher: 'red',
};

function roleColor(role) {
  return ROLE_COLOR[role] || 'white';
}

function clip(s, n) {
  s = String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function argsSummary(args) {
  if (!args || typeof args !== 'object') return '';
  const s = JSON.stringify(args);
  return s === '{}' ? '' : clip(s, 160);
}

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
      const c = clip(obj.content, 2000);
      if (c) out.push({ text: tag + c, color: roleColor(role) });
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
      const head = ('  ' + mark + ' ' + tag + name + (a ? ' ' + a : '')).trimEnd();
      out.push({ text: head, color: ok ? 'cyan' : 'red' });
      const body = clip(obj.output, 600);
      if (body) out.push({ text: '    ' + body, color: 'gray' });
      return out;
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

module.exports = { formatEvent, roleColor, argsSummary, ROLE_COLOR };
