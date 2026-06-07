// Pure rendering of backend event objects into colored transcript lines.
// Dependency-free (no terminal-kit) so it can be unit-checked headlessly; the TUI maps
// the returned color name onto terminal-kit's chainable colors.
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

// Returns { text, color } for a transcript line, or null to skip the event.
function formatEvent(obj) {
  const role = obj.role || '';
  const tag = role ? '[' + role + '] ' : '';
  switch (obj.type) {
    case 'ready':
      return { text: '· backend ready ·', color: 'gray' };
    case 'bye':
      return { text: '· session ended ·', color: 'gray' };
    case 'assistant': {
      const c = clip(obj.content, 2000);
      return c ? { text: tag + c, color: roleColor(role) } : null;
    }
    case 'tool': {
      const ok = obj.ok !== false;
      const mark = ok ? '✓' : '✗';
      const name = obj.tool ? obj.tool + ' ' : '';
      const body = clip(obj.output, 500);
      const text = ('  ' + mark + ' ' + tag + name + (body ? '· ' + body : '')).trimEnd();
      return { text, color: ok ? 'gray' : 'red' };
    }
    case 'progress':
      return { text: '  ' + tag + clip(obj.text, 500), color: 'gray' };
    case 'approval_request':
      return {
        text: '  ⚠ APPROVAL NEEDED: ' + obj.tool + ' (' + (obj.risk || '') + ') ' +
          clip(JSON.stringify(obj.args || {}), 300),
        color: 'brightYellow',
      };
    case 'done':
      return obj.ok === false
        ? { text: '· error: ' + clip(obj.error, 300) + ' ·', color: 'red' }
        : { text: '· done ·', color: 'gray' };
    default:
      return { text: tag + clip(JSON.stringify(obj), 300), color: 'gray' };
  }
}

module.exports = { formatEvent, roleColor, ROLE_COLOR };
