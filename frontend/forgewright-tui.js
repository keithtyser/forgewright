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
const fs = require('fs');
const os = require('os');
const path = require('path');
const { formatEvent } = require('./lib/render');

// --- persisted credentials (shared with the Python backend) ------------------------
// Written by the first-run setup wizard so the user configures their brain once. Same file
// the backend reads (forgewright/credentials.py): ~/.forgewright/credentials.json.
const OPENROUTER_DEFAULT_MODEL = 'openrouter:deepseek/deepseek-v4-pro';
// OpenAI models reachable over the Codex (ChatGPT-login) Responses API, newest first. The
// wizard offers these plus a free-text "Other" entry, so any model id (including ones newer
// than this list) still works without a code change.
const CODEX_MODELS = [
  'gpt-5.5-codex',
  'gpt-5.5',
  'gpt-5.1-codex-max',
  'gpt-5.1-codex',
  'gpt-5.1',
  'gpt-5-codex',
  'gpt-5',
  'gpt-5-mini',
];
function fwHome() { return process.env.FORGEWRIGHT_HOME || path.join(os.homedir(), '.forgewright'); }
function credsPath() { return path.join(fwHome(), 'credentials.json'); }
function codexAuthPath() {
  const base = process.env.CODEX_HOME || path.join(os.homedir(), '.codex');
  return path.join(base, 'auth.json');
}
function loadCreds() {
  try { return JSON.parse(fs.readFileSync(credsPath(), 'utf8')) || {}; } catch (e) { return {}; }
}
function saveCreds(creds) {
  try {
    fs.mkdirSync(fwHome(), { recursive: true });
    fs.writeFileSync(credsPath(), JSON.stringify(creds, null, 2), 'utf8');
    try { fs.chmodSync(credsPath(), 0o600); } catch (e) {}
  } catch (e) { w(A.red + '  could not save credentials: ' + e.message + A.r + '\n'); }
}
function envFromCreds(creds) {
  const env = {};
  if (creds.openrouter_api_key) env.OPENROUTER_API_KEY = creds.openrouter_api_key;
  if (creds.anthropic_api_key) env.ANTHROPIC_API_KEY = creds.anthropic_api_key;
  if (creds.openai_api_key) env.OPENAI_API_KEY = creds.openai_api_key;
  return env;
}
// Returns { brain, env } if a brain is already configured (flag > saved creds > env), else null.
function resolveBrainConfig(brainArg) {
  const creds = loadCreds();
  const env = envFromCreds(creds);
  const haveOR = process.env.OPENROUTER_API_KEY || env.OPENROUTER_API_KEY;
  const brain = brainArg || creds.brain || (haveOR ? OPENROUTER_DEFAULT_MODEL : null);
  return brain ? { brain, env } : null;
}

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

  const renderMarkdownLines = (text) => {
    let s;
    try { s = String(marked.parse(String(text))); } catch (e) { s = String(text); }
    return s.replace(/\s+$/, '').split('\n');
  };
  const dot = (color) => color + '●' + A.r;
  const roleDot = (role) => ROLE_FG[role] || '\x1b[96m';
  const roleLabel = (role) => (role && role !== 'agent') ? A.dim + role + ' ' + A.r : '';
  const primaryArg = (obj) => {
    const a = obj.args;
    if (!a || typeof a !== 'object') return clip(a, 90);
    if (a.command) return clip(a.command, 90);
    if (a.args) return clip(a.args, 90);
    if (a.recipe) return clip(a.recipe + (a.params ? ' ' + JSON.stringify(a.params) : ''), 90);
    const ks = Object.keys(a);
    return ks.length ? clip(JSON.stringify(a), 90) : '';
  };

  // pick a headline metric from an artifact's gate metrics (score-like first, else any number)
  const headlineMetric = (m) => {
    if (!m || typeof m !== 'object') return null;
    const pref = ['score', 'accuracy', 'pass_rate', 'refusal_rate_harmful', 'reward', 'loss'];
    for (const k of pref) if (typeof m[k] === 'number') return [k, m[k]];
    for (const k of Object.keys(m)) if (typeof m[k] === 'number') return [k, m[k]];
    return null;
  };

  // Claude-Code-style: `●` action bullets (colored by role/status) with `⎿` result lines, plus
  // swarm-native lines (◆ plan, ◇ artifact lineage). Most cases also update live HUD state.
  function renderEvent(obj) {
    switch (obj.type) {
      case 'ready':
        w('\n' + A.dim + '  ready. describe a post-training job, or ask a question.' + A.r + '\n'); break;
      case 'bye':
        w('\n' + A.dim + '  session ended.' + A.r + '\n'); break;
      case 'assistant': {
        const content = String(obj.content || '').trim();
        if (content) {
          const lines = renderMarkdownLines(content);
          w('\n' + dot(roleDot(obj.role)) + ' ' + roleLabel(obj.role) + (lines[0] || '') + '\n');
          for (let i = 1; i < lines.length; i++) w('  ' + lines[i] + '\n');
        }
        if (obj.role && obj.role !== 'agent') hud.activeRole = obj.role;
        if (obj.usage && obj.usage.total_tokens) hud.tokens = obj.usage.total_tokens;
        break;
      }
      case 'tool': {
        const ok = obj.ok !== false;
        const arg = primaryArg(obj);
        w(dot(ok ? A.green : A.red) + ' ' + roleLabel(obj.role) + (obj.tool || '') +
          (arg ? A.dim + '(' + arg + ')' + A.r : '') + '\n');
        const out = String(obj.output || '').trim();
        if (out) w('  ' + A.dim + '⎿  ' + clip(out, 400) + A.r + '\n');
        if (obj.role && obj.role !== 'agent') hud.activeRole = obj.role;
        hud.lastAction = clip((obj.tool || '') + (arg ? '(' + arg + ')' : ''), 48);
        break;
      }
      case 'pipeline': {
        const stages = Array.isArray(obj.stages) ? obj.stages : [];
        hud.pipeline = { stages: stages.map((n) => ({ name: n, state: 'pending' })) };
        hud.activeRole = 'Director';
        const map = stages.map((n) => (roleDot(n) + n + A.r)).join(A.dim + ' → ' + A.r);
        w('\n' + '\x1b[96m' + '◆' + A.r + ' ' + A.dim + 'plan ' + A.r + map + '\n');
        break;
      }
      case 'stage': {
        if (hud.pipeline && hud.pipeline.stages[obj.index]) hud.pipeline.stages[obj.index].state = obj.state;
        if (obj.state === 'active') { hud.activeRole = obj.name; hud.metric = null; hud.lastAction = ''; }
        if (obj.state === 'done') w(A.green + '✓' + A.r + ' ' + A.dim + obj.name + ' complete' + A.r + '\n');
        if (obj.state === 'failed') w(A.red + '✗' + A.r + ' ' + A.red + obj.name + ' failed' + A.r + '\n');
        break;
      }
      case 'artifact': {
        const col = roleDot(obj.role);
        const tag = obj.passed === false ? A.red + ' ✗' + A.r : (obj.passed === true ? A.green + ' ✓' + A.r : '');
        const par = Array.isArray(obj.parents) && obj.parents.length
          ? A.dim + ' ← ' + obj.parents.map(shortId).join(', ') + A.r : '';
        const hm = headlineMetric(obj.metrics);
        const met = hm ? A.dim + '  ' + hm[0] + ' ' + A.r + A.white + (Math.round(hm[1] * 1000) / 1000) + A.r : '';
        w(col + '◇' + A.r + ' ' + col + obj.kind + A.r + A.dim + '#' + shortId(obj.id) + A.r + par + met + tag + '\n');
        hud.lastAction = 'produced ' + obj.kind + ' #' + shortId(obj.id);
        break;
      }
      case 'progress': {
        w('  ' + A.dim + '⎿  ' + clip(obj.text, 400) + A.r + '\n');
        const txt = String(obj.text || '');
        const sm = txt.match(/step\s+(\d+)\s*\/\s*(\d+)/i);
        const lm = txt.match(/loss[:=\s]+([0-9]*\.?[0-9]+)/i);
        if (sm || lm) {
          hud.metric = hud.metric || { step: 0, total: 0, loss: null, hist: [] };
          if (sm) { hud.metric.step = +sm[1]; hud.metric.total = +sm[2]; }
          if (lm) { hud.metric.loss = lm[1]; hud.metric.hist.push(parseFloat(lm[1])); if (hud.metric.hist.length > 24) hud.metric.hist.shift(); }
        }
        break;
      }
      case 'done':
        if (obj.ok === false) w('\n' + dot(A.red) + ' ' + A.red + clip(obj.error, 400) + A.r + '\n'); break;
      default: break;
    }
  }

  // backend (spawned after the brain is configured; can be restarted by /login)
  const python = process.env.FORGEWRIGHT_PYTHON || 'python3';
  let child = null, rl = null, currentBrain = null, restarting = false;
  let busy = false, awaitingApproval = false;
  const send = (obj) => { if (child && child.stdin.writable) child.stdin.write(JSON.stringify(obj) + '\n'); };

  function startBackend(cfg) {
    currentBrain = cfg.brain;
    const args = ['-m', 'forgewright', 'serve'];
    if (cfg.brain) args.push('--brain', cfg.brain);
    child = spawn(python, args, {
      stdio: ['pipe', 'pipe', 'inherit'],
      shell: process.platform === 'win32',
      env: Object.assign({}, process.env, cfg.env || {}),
    });
    child.on('error', (e) => {
      w(A.red + 'forgewright: could not start the backend (' + python + ' -m forgewright serve): ' + e.message + A.r + '\n' +
        'Install it (pip install -e .) and/or set FORGEWRIGHT_PYTHON to your venv python.\n');
      process.exit(1);
    });
    child.on('exit', (code) => {
      if (restarting) return;          // a /login restart will respawn; don't exit the app
      stopStatus(); term.processExit(code || 0);
    });
    rl = readline.createInterface({ input: child.stdout });
    rl.on('line', onLine);
  }

  // --- the Swarm HUD: a live, multi-line panel that hovers above the prompt while the swarm
  //     works and clears for your turn. It shows the pipeline (each specialist's stage state),
  //     the active specialist, training step/loss with a sparkline, elapsed, and tokens.
  //     It redraws in place (cursor-up + erase-below), the same flicker-free trick the old
  //     one-line spinner used, just extended to N lines. Every line is width-budgeted so it
  //     never wraps (wrapping would break the erase math).
  const GLYPHS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  const SPARK = '▁▂▃▄▅▆▇█';
  const PLAIN = !!process.env.FORGEWRIGHT_PLAIN;   // escape hatch: minimal one-line status
  const hud = { timer: null, start: 0, i: 0, prevLines: 0, tokens: 0,
    pipeline: null, metric: null, activeRole: null, lastAction: '' };
  const fmtTok = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n));
  const shortId = (id) => String(id || '').slice(-6);
  const sparkline = (a) => {
    if (!a || a.length < 2) return '';
    const lo = Math.min.apply(null, a), hi = Math.max.apply(null, a), rng = (hi - lo) || 1;
    return a.map((v) => SPARK[Math.min(7, Math.floor((v - lo) / rng * 7.999))]).join('');
  };
  const stageGlyph = (state, spin) =>
    state === 'done' ? A.green + '✓' + A.r :
    state === 'failed' ? A.red + '✗' + A.r :
    state === 'active' ? '\x1b[35m' + spin + A.r : A.dim + '◌' + A.r;

  // width-budgeted line builder: add(text,color) counts text length; addRaw(s,vis) appends
  // pre-colored content with an explicit visible width. Truncates to fit maxw.
  function lineBuilder(maxw) {
    let vis = 0; const parts = [];
    return {
      add(text, color) {
        if (vis >= maxw) return;
        let t = String(text == null ? '' : text);
        if (vis + t.length > maxw) t = t.slice(0, Math.max(0, maxw - vis - 1)) + '…';
        vis += t.length; parts.push(color ? color + t + A.r : t);
      },
      addRaw(s, visLen) { if (vis + visLen <= maxw) { vis += visLen; parts.push(s); } },
      str() { return parts.join(''); },
    };
  }

  function hudLines() {
    const maxw = Math.max(24, (term.width || 80) - 1);
    const spin = GLYPHS[hud.i % GLYPHS.length];
    const el = Math.round((Date.now() - hud.start) / 1000);
    const out = [''];                                  // a blank gap above the panel
    const title = '─── swarm ';
    out.push(A.dim + (title + '─'.repeat(Math.max(0, maxw - title.length))).slice(0, maxw) + A.r);

    // pipeline row (only when a recipe is running and the terminal is wide enough)
    if (hud.pipeline && hud.pipeline.stages.length && maxw >= 40) {
      const lb = lineBuilder(maxw); lb.add('  ');
      hud.pipeline.stages.forEach((s, idx) => {
        if (idx) lb.add('   ');
        lb.addRaw(stageGlyph(s.state, spin), 1); lb.add(' ');
        const lit = s.state === 'active' || s.state === 'done' || s.state === 'failed';
        lb.add(s.name, lit ? roleDot(s.name) : A.dim);
      });
      out.push(lb.str());
    }

    // status row: active specialist · (metric | last action) · elapsed · tokens
    const lb = lineBuilder(maxw); lb.add('  ');
    lb.addRaw('\x1b[35m' + spin + A.r, 1); lb.add(' ');
    const who = (hud.activeRole && hud.activeRole !== 'agent') ? hud.activeRole : 'working';
    lb.add(who, roleDot(hud.activeRole || ''));
    if (hud.metric && hud.metric.total) {
      lb.add(' · '); lb.add('step ' + hud.metric.step + '/' + hud.metric.total, A.white);
      if (hud.metric.loss != null) {
        lb.add(' · loss ' + hud.metric.loss, A.dim);
        const sp = sparkline(hud.metric.hist);
        if (sp) { lb.add(' '); lb.addRaw(A.cyan + sp + A.r, sp.length); }
      }
    } else if (hud.lastAction) {
      lb.add(' · '); lb.add(hud.lastAction, A.dim);
    }
    lb.add(' · '); lb.add(el + 's', A.dim);
    if (hud.tokens) { lb.add(' · '); lb.add('↑' + fmtTok(hud.tokens) + ' tok', A.dim); }
    out.push(lb.str());
    return out;
  }

  function hudDraw() {
    if (!busy || awaitingApproval) return;
    const lines = PLAIN
      ? ['', A.dim + GLYPHS[hud.i % GLYPHS.length] + ' working… (' +
         Math.round((Date.now() - hud.start) / 1000) + 's' +
         (hud.tokens ? ' · ↑' + fmtTok(hud.tokens) : '') + ')' + A.r]
      : hudLines();
    w(lines.join('\n') + '\n');
    hud.prevLines = lines.length;
  }
  function hudClear() {
    if (hud.prevLines > 0) { w('\x1b[' + hud.prevLines + 'A\x1b[J'); hud.prevLines = 0; }
  }
  function hudPaint() { hudClear(); hudDraw(); }
  function hudResetTurn() {
    hud.start = Date.now(); hud.i = 0; hud.tokens = 0;
    hud.pipeline = null; hud.metric = null; hud.activeRole = null; hud.lastAction = '';
  }
  function startStatus() {
    if (!hud.start) hud.start = Date.now();
    if (hud.timer) return;
    hud.timer = setInterval(() => { hud.i++; hudPaint(); }, 120);
    hudDraw();
  }
  function stopStatus() {
    if (hud.timer) { clearInterval(hud.timer); hud.timer = null; }
    hudClear();
  }

  function prompt() {
    if (busy || awaitingApproval) return;
    // a clear, unmistakable "your turn" prompt
    w('\n' + A.dim + '─── your turn ' + '─'.repeat(Math.max(0, ((term.width || 80) - 16))) + A.r + '\n');
    w(A.green + A.b + '❯ ' + A.r);
    term.inputField({ cancelable: true }, (err, input) => {
      w('\n');
      const t = (input || '').trim();
      if (!t) return prompt();
      if (t[0] === '/') return handleSlash(t);
      busy = true; send({ type: 'user_msg', text: t }); hudResetTurn(); startStatus();
    });
  }

  function showHelp() {
    w('\n' + A.b + '  commands' + A.r + '\n');
    w('  ' + A.cyan + '/login' + A.r + A.dim + '   reconfigure or refresh your brain (OpenRouter key / Codex login)' + A.r + '\n');
    w('  ' + A.cyan + '/brain' + A.r + A.dim + '   show the brain in use' + A.r + '\n');
    w('  ' + A.cyan + '/help' + A.r + A.dim + '    this help' + A.r + '\n');
    w('  ' + A.cyan + '/quit' + A.r + A.dim + '    exit (or Ctrl-C)' + A.r + '\n');
  }

  function handleSlash(cmd) {
    const c = cmd.toLowerCase();
    if (c === '/login' || c === '/auth' || c === '/refresh') { relogin(); return; }
    if (c === '/quit' || c === '/exit') { send({ type: 'shutdown' }); try { child.stdin.end(); } catch (e) {} term.processExit(0); return; }
    if (c === '/brain') { w('  ' + A.dim + 'brain: ' + (currentBrain || '(default)') + A.r + '\n'); return prompt(); }
    if (c === '/help' || c === '/?') { showHelp(); return prompt(); }
    w('  ' + A.dim + 'unknown command ' + cmd + ' — try /help' + A.r + '\n');
    return prompt();
  }

  // Re-run the setup wizard, then restart the backend with the new credentials so a fresh
  // API key / Codex login takes effect without leaving the app.
  async function relogin() {
    const cfg = await setupWizard(term);
    if (!cfg) return prompt();
    restarting = true;
    try { rl.close(); } catch (e) {}
    const old = child;
    old.once('exit', () => { restarting = false; w(A.dim + '  restarting backend…' + A.r + '\n'); startBackend(cfg); });
    try { old.stdin.end(); } catch (e) {}
    try { old.kill(); } catch (e) {}
  }

  function handleApproval(obj) {
    awaitingApproval = true; stopStatus();
    w('\n' + A.yellow + '─── approval needed ' + '─'.repeat(Math.max(0, ((term.width || 80) - 22))) + A.r + '\n');
    w('  ' + A.yellow + A.b + '⚠ ' + (obj.tool || 'command') + (obj.risk ? ' (' + obj.risk + ')' : '') + A.r + '\n');
    if (obj.args && Object.keys(obj.args).length) w('  ' + A.dim + clip(JSON.stringify(obj.args), 200) + A.r + '\n');
    w('  ' + A.dim + '↑/↓ to choose · enter to confirm' + A.r + '\n');
    // singleColumnMenu navigates with UP/DOWN + ENTER (singleLineMenu used left/right, which
    // is what tripped people up). Vertical also matches Claude Code's approval prompt.
    const items = ['approve once', 'approve all ' + (obj.tool || ''), 'YOLO: bypass all permissions', 'deny'];
    const decisions = ['yes', 'all', 'yolo', 'no'];
    term.singleColumnMenu(items, {
      selectedIndex: 0, cancelable: false, leftPadding: '    ', selectedLeftPadding: '  ❯ ',
      style: term.gray, selectedStyle: term.brightWhite.bgGreen,
    }, (err, resp) => {
      w('\n');
      send({ type: 'approval_response', decision: decisions[(resp && resp.selectedIndex != null) ? resp.selectedIndex : 3] });
      awaitingApproval = false;
      if (busy) startStatus();
    });
  }

  function onLine(line) {
    line = line.trim(); if (!line) return;
    let obj; try { obj = JSON.parse(line); } catch (e) { return; }
    if (obj.type === 'approval_request') { handleApproval(obj); return; }
    if (obj.type === 'done') busy = false;     // before render so the spinner does not restart
    stopStatus();
    renderEvent(obj);
    // restart the spinner only if more work is coming (a tool call, or an assistant turn
    // that requested tools). A final text answer with no tool_calls ends the turn -> no flash.
    const finalAnswer = obj.type === 'assistant' &&
      (!Array.isArray(obj.tool_calls) || obj.tool_calls.length === 0);
    if (busy && !awaitingApproval && obj.type !== 'done' && !finalAnswer) startStatus();
    if (obj.type === 'ready' || obj.type === 'done') prompt();
    if (obj.type === 'bye') { stopStatus(); term.processExit(0); }
  }

  term.on('key', (name) => {
    if (name === 'CTRL_C') { send({ type: 'shutdown' }); try { child && child.stdin.end(); } catch (e) {} term.processExit(0); }
  });

  // --- startup: banner, then ensure a brain is configured (wizard on first run) ----
  banner(term);
  w('\n' + A.dim + '  post-training swarm · one chat, the swarm works behind it' + A.r + '\n');
  (async () => {
    let cfg = resolveBrainConfig(brain);
    if (!cfg) cfg = await setupWizard(term);
    if (!cfg) { w(A.red + '  no brain configured; exiting.' + A.r + '\n'); term.processExit(1); return; }
    w(A.dim + '  starting backend…  (/help for commands · Ctrl-C to quit)' + A.r + '\n');
    startBackend(cfg);
  })();
}

// First-run (and /login) setup: pick a brain and persist it. Resolves to { brain, env } or
// null if the user backs out. Uses a vertical menu (up/down + enter) and hidden input.
function setupWizard(term) {
  return new Promise((resolve) => {
    w('\n' + A.b + '  Set up your brain' + A.r + A.dim + ' — how the agent connects to a model.' + A.r + '\n');
    w(A.dim + '  ↑/↓ to choose · enter to confirm' + A.r + '\n\n');
    const items = ['OpenRouter API key  (hosted models)', 'Codex  (ChatGPT login / pick a GPT-5 model)'];
    term.singleColumnMenu(items, {
      selectedIndex: 0, cancelable: true, leftPadding: '    ', selectedLeftPadding: '  ❯ ',
      style: term.gray, selectedStyle: term.brightWhite.bgGreen,
    }, (err, resp) => {
      w('\n');
      const idx = (resp && resp.selectedIndex != null && !resp.canceled) ? resp.selectedIndex : -1;
      if (idx === 0) return wizardOpenRouter(term, resolve);
      if (idx === 1) return wizardCodex(term, resolve);
      resolve(null);   // canceled
    });
  });
}

function wizardOpenRouter(term, resolve) {
  w('  Paste your OpenRouter API key ' + A.dim + '(hidden; get one at openrouter.ai/keys)' + A.r + '\n');
  w(A.green + A.b + '  ❯ ' + A.r);
  term.inputField({ echoChar: '*', cancelable: true }, (err, input) => {
    w('\n');
    const key = (input || '').trim();
    if (!key) { w(A.red + '  no key entered.' + A.r + '\n'); return resolve(setupWizard(term)); }
    const creds = loadCreds();
    creds.brain = OPENROUTER_DEFAULT_MODEL;
    creds.openrouter_api_key = key;
    saveCreds(creds);
    w(A.green + '  ✓ saved to ' + credsPath() + ' · change anytime with /login' + A.r + '\n');
    resolve({ brain: creds.brain, env: { OPENROUTER_API_KEY: key } });
  });
}

function wizardCodex(term, resolve) {
  const authed = (() => { try { return fs.existsSync(codexAuthPath()); } catch (e) { return false; } })();
  if (authed) {
    w(A.green + '  ✓ Codex login found at ' + codexAuthPath() + A.r + '\n');
  } else {
    w('  No Codex login yet. In another terminal run ' + A.cyan + 'codex login' + A.r + ' (ChatGPT account),\n');
    w('  then run ' + A.cyan + '/login' + A.r + ' here again. Picking a model anyway so you can do that now.\n');
  }
  // let the user pick which OpenAI model to drive Codex with
  w('\n  Choose an OpenAI model for Codex:\n');
  w(A.dim + '  ↑/↓ to choose · enter to confirm' + A.r + '\n\n');
  const saved = loadCreds().codex_model;
  const items = CODEX_MODELS.concat(['Other (type a model id)…']);
  const sel = Math.max(0, CODEX_MODELS.indexOf(saved));
  term.singleColumnMenu(items, {
    selectedIndex: sel, cancelable: true, leftPadding: '    ', selectedLeftPadding: '  ❯ ',
    style: term.gray, selectedStyle: term.brightWhite.bgGreen,
  }, (err, resp) => {
    w('\n');
    const idx = (resp && resp.selectedIndex != null && !resp.canceled) ? resp.selectedIndex : -1;
    if (idx === -1) return resolve(null);                         // backed out
    if (idx === items.length - 1) {                              // "Other": free-text entry
      w('  Enter a model id ' + A.dim + '(e.g. gpt-5.5-codex)' + A.r + '\n');
      w(A.green + A.b + '  ❯ ' + A.r);
      term.inputField({ cancelable: true }, (e2, input) => {
        w('\n');
        finishCodex(resolve, (input || '').trim() || CODEX_MODELS[0]);
      });
      return;
    }
    finishCodex(resolve, CODEX_MODELS[idx]);
  });
}

function finishCodex(resolve, model) {
  const brain = 'oauth-codex:' + model;
  const creds = loadCreds();
  creds.brain = brain;
  creds.codex_model = model;
  delete creds.openrouter_api_key;
  saveCreds(creds);
  w(A.green + '  ✓ Codex model ' + A.b + model + A.r + A.green + ' saved · change anytime with /login' + A.r + '\n');
  resolve({ brain: brain, env: {} });
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
