#!/usr/bin/env node
// Forgewright TUI: a single Claude-Code-style conversational terminal for the
// post-training swarm. It spawns the Python backend (`python -m forgewright serve`),
// streams its role-tagged JSON events into one transcript with real markdown rendering,
// and surfaces approvals at one prompt. The swarm stays entirely on the backend.
//
// Built with terminal-kit by Cedric Ronvel (MIT) + marked-terminal. See NOTICE.
//
//   forgewright [--brain <b>]      interactive TUI (spawns the backend)
//   forgewright --render-test       headless: read JSON events on stdin, print them
'use strict';

const readline = require('readline');
const { formatEvent } = require('./lib/render');

// --- ANSI paint (dynamic text goes through here, NOT terminal-kit's term(), so model
//     output containing ^ or % cannot break the markup) ----------------------------
const A = { r: '\x1b[0m', b: '\x1b[1m', dim: '\x1b[90m', cyan: '\x1b[36m', green: '\x1b[32m',
  red: '\x1b[31m', yellow: '\x1b[33m', white: '\x1b[97m' };
const ROLE_FG = {
  Director: '\x1b[96m', DataCurator: '\x1b[94m', SFTTrainer: '\x1b[92m', RLTrainer: '\x1b[32m',
  Abliterator: '\x1b[95m', Quantizer: '\x1b[93m', ServingOptimizer: '\x1b[33m',
  Evaluator: '\x1b[97m', Publisher: '\x1b[91m',
};
const ROLE_BG = {
  Director: '\x1b[46m', DataCurator: '\x1b[44m', SFTTrainer: '\x1b[42m', RLTrainer: '\x1b[42m',
  Abliterator: '\x1b[45m', Quantizer: '\x1b[43m', ServingOptimizer: '\x1b[43m',
  Evaluator: '\x1b[47m', Publisher: '\x1b[41m',
};
const w = (s) => process.stdout.write(s);
const clip = (s, n) => { s = String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n - 1) + '…' : s; };

// --- headless render test (no terminal-kit, no child) ------------------------------
function renderTest() {
  const rl = readline.createInterface({ input: process.stdin });
  rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let obj; try { obj = JSON.parse(line); } catch (e) { return; }
    for (const r of formatEvent(obj)) w('[' + r.color + '] ' + r.text + '\n');
  });
}

// --- interactive TUI ---------------------------------------------------------------
function interactive(brain) {
  const { spawn } = require('child_process');
  const term = require('terminal-kit').terminal;
  const { Marked } = require('marked');
  const { markedTerminal } = require('marked-terminal');
  const marked = new Marked(markedTerminal({ tab: 2, reflowText: true, width: Math.max(60, (term.width || 100) - 4) }));

  const renderMarkdown = (text) => {
    let s;
    try { s = String(marked.parse(String(text))); } catch (e) { s = String(text); }
    return s.replace(/\s+$/, '').split('\n').map((l) => '  ' + l).join('\n');
  };
  const badge = (role) => {
    const bg = ROLE_BG[role] || '\x1b[100m';
    w('\n' + bg + '\x1b[30m\x1b[1m ' + (role || 'agent') + ' ' + A.r + '\n');
  };

  function renderEvent(obj) {
    switch (obj.type) {
      case 'ready':
        w(A.dim + '\n  ready. describe a post-training job, or ask a question.\n' + A.r); break;
      case 'bye':
        w(A.dim + '\n  session ended.\n' + A.r); break;
      case 'assistant': {
        const content = String(obj.content || '').trim();
        if (content) { badge(obj.role); w(renderMarkdown(content) + '\n'); }
        const calls = Array.isArray(obj.tool_calls) ? obj.tool_calls : [];
        if (calls.length) w('  ' + A.cyan + '↳ ' + calls.join(', ') + A.r + '\n');
        break;
      }
      case 'tool': {
        const ok = obj.ok !== false;
        const rc = ROLE_FG[obj.role] || '';
        const mark = ok ? A.green + '✓' : A.red + '✗';
        const a = obj.args && Object.keys(obj.args).length ? ' ' + A.dim + clip(JSON.stringify(obj.args), 120) + A.r : '';
        w('  ' + mark + A.r + ' ' + rc + (obj.tool || '') + A.r + a + '\n');
        const out = String(obj.output || '').trim();
        if (out) w('      ' + A.dim + clip(out, 500) + A.r + '\n');
        break;
      }
      case 'progress':
        w('      ' + A.dim + clip(obj.text, 400) + A.r + '\n'); break;
      case 'done':
        if (obj.ok === false) w('\n  ' + A.red + '✗ ' + clip(obj.error, 400) + A.r + '\n'); break;
      default: break;
    }
  }

  // backend
  const python = process.env.FORGEWRIGHT_PYTHON || 'python3';
  const args = ['-m', 'forgewright', 'serve'];
  if (brain) args.push('--brain', brain);
  const child = spawn(python, args, { stdio: ['pipe', 'pipe', 'inherit'], shell: process.platform === 'win32' });
  child.on('error', (e) => {
    w(A.red + 'forgewright: could not start the backend (' + python + ' -m forgewright serve): ' + e.message + A.r + '\n' +
      'Install it (pip install -e .) and/or set FORGEWRIGHT_PYTHON to your venv python.\n');
    process.exit(1);
  });

  let busy = false, awaitingApproval = false, spinner = null;
  const send = (obj) => child.stdin.write(JSON.stringify(obj) + '\n');

  async function startThinking() {
    try { stopThinking(); spinner = await term.spinner('impulse'); term(A.dim + ' working…' + A.r); } catch (e) { spinner = null; }
  }
  function stopThinking() {
    try { if (spinner) { spinner.animate(false); spinner = null; term.column(1); term.eraseLine(); } } catch (e) { spinner = null; }
  }

  function prompt() {
    if (busy || awaitingApproval) return;
    w('\n' + A.b + A.green + '› ' + A.r);
    term.inputField({ cancelable: true }, (err, input) => {
      w('\n');
      if (input && input.trim()) { busy = true; send({ type: 'user_msg', text: input.trim() }); startThinking(); }
      else prompt();
    });
  }

  function handleApproval(obj) {
    awaitingApproval = true; stopThinking();
    w('\n  ' + A.yellow + '⚠ approval: ' + (obj.tool || 'command') + (obj.risk ? ' (' + obj.risk + ')' : '') + A.r + '\n');
    if (obj.args && Object.keys(obj.args).length) w('  ' + A.dim + clip(JSON.stringify(obj.args), 200) + A.r + '\n');
    const items = ['approve once', 'approve all ' + (obj.tool || ''), 'YOLO: bypass all', 'deny'];
    const decisions = ['yes', 'all', 'yolo', 'no'];
    term.singleLineMenu(items, { selectedIndex: 0, style: term.gray, selectedStyle: term.brightWhite.bgGreen }, (err, resp) => {
      w('\n');
      send({ type: 'approval_response', decision: decisions[(resp && resp.selectedIndex != null) ? resp.selectedIndex : 3] });
      awaitingApproval = false;
      if (busy) startThinking();
    });
  }

  const rl = readline.createInterface({ input: child.stdout });
  rl.on('line', (line) => {
    line = line.trim(); if (!line) return;
    let obj; try { obj = JSON.parse(line); } catch (e) { return; }
    if (obj.type === 'approval_request') { handleApproval(obj); return; }
    if (obj.type === 'done') busy = false;     // before render so the spinner does not restart
    stopThinking();
    renderEvent(obj);
    if (busy && !awaitingApproval && obj.type !== 'done') startThinking();
    if (obj.type === 'ready' || obj.type === 'done') prompt();
    if (obj.type === 'bye') { stopThinking(); term.processExit(0); }
  });
  child.on('exit', (code) => { stopThinking(); term.processExit(code || 0); });
  term.on('key', (name) => {
    if (name === 'CTRL_C') { send({ type: 'shutdown' }); try { child.stdin.end(); } catch (e) {} term.processExit(0); }
  });

  banner(term);
  w('\n' + A.dim + '  post-training swarm · one chat, the swarm works behind it' + A.r + '\n');
  w(A.dim + '  starting backend…  (Ctrl-C to quit)' + A.r + '\n');
}

function banner(term) {
  const width = (term && term.width) || 100;
  const GRAD = ['\x1b[38;5;87m', '\x1b[38;5;81m', '\x1b[38;5;75m', '\x1b[38;5;69m', '\x1b[38;5;63m', '\x1b[38;5;33m'];
  let art = null;
  try {
    const figlet = require('figlet');
    for (const font of ['ANSI Shadow', 'Slant', 'Small']) {
      const t = figlet.textSync('forgewright', { font });
      const wmax = Math.max.apply(null, t.split('\n').map((l) => l.length));
      if (wmax <= width - 2) { art = t; break; }
    }
  } catch (e) { art = null; }
  w('\n');
  if (art) {
    const lines = art.replace(/\s+$/, '').split('\n');
    lines.forEach((l, i) => w('  ' + (GRAD[i] || GRAD[GRAD.length - 1]) + l + A.r + '\n'));
  } else {
    w('  ' + A.b + GRAD[0] + 'forgewright' + A.r + '\n');   // fallback for tiny terminals
  }
}

const argv = process.argv.slice(2);
if (argv.includes('--render-test')) {
  renderTest();
} else {
  const bi = argv.indexOf('--brain');
  interactive(bi >= 0 ? argv[bi + 1] : null);
}
