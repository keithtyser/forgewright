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
        if (obj.step != null) hud.agentStep = obj.step;     // the enforced agent-loop step budget
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
        sessionGraph.push({ id: obj.id, kind: obj.kind, produced_by: obj.role,
          parents: Array.isArray(obj.parents) ? obj.parents : [],
          passed: obj.passed, score: (headlineMetric(obj.metrics) || [null, null])[1] });
        break;
      }
      case 'metric': {
        // structured training telemetry from the backend metric_tap (no frontend regex)
        const keep = (a, v) => { a.push(+v); if (a.length > 28) a.shift(); return a; };
        hud.metric = hud.metric || { step: null, total: null, loss: null, reward: null,
          grad_norm: null, lr: null, histLoss: [], histReward: [] };
        const x = hud.metric;
        if (obj.step != null) x.step = obj.step;
        if (obj.total != null) x.total = obj.total;
        if (obj.loss != null) { x.loss = obj.loss; keep(x.histLoss, obj.loss); }
        if (obj.reward != null) { x.reward = obj.reward; keep(x.histReward, obj.reward); }
        if (obj.grad_norm != null) x.grad_norm = obj.grad_norm;
        if (obj.lr != null) x.lr = obj.lr;
        break;
      }
      case 'budget':
        hud.budget = obj; break;
      case 'progress':
        w('  ' + A.dim + '⎿  ' + clip(obj.text, 400) + A.r + '\n'); break;
      case 'graph':
        renderGraph(Array.isArray(obj.nodes) ? obj.nodes : []); break;
      case 'models':
        renderModels(obj); break;
      case 'done':
        if (obj.ok === false) w('\n' + dot(A.red) + ' ' + A.red + clip(obj.error, 400) + A.r + '\n'); break;
      default: break;
    }
  }

  // /graph: draw the session's provenance DAG as an indented tree (roots -> children).
  function renderGraph(nodes) {
    if (!nodes.length) { w('\n' + A.dim + '  no artifacts in the registry yet.' + A.r + '\n'); return; }
    const byId = {}; nodes.forEach((n) => { byId[n.id] = n; });
    const kidsOf = {}; nodes.forEach((n) => (n.parents || []).forEach((p) => { (kidsOf[p] = kidsOf[p] || []).push(n.id); }));
    const isRoot = (n) => !(n.parents || []).some((p) => byId[p]);
    const roots = nodes.filter(isRoot);
    w('\n' + A.b + '  provenance graph' + A.r + A.dim + '  (' + nodes.length + ' artifacts)' + A.r + '\n');
    const seen = new Set();
    const line = (n, prefix, connector) => {
      const col = roleDot(n.produced_by || '');
      const tag = n.passed === false ? A.red + ' ✗' + A.r : (n.passed === true ? A.green + ' ✓' + A.r : '');
      const score = (n.score != null) ? A.dim + '  ' + (Math.round(n.score * 1000) / 1000) + A.r : '';
      const by = n.produced_by ? A.dim + ' · ' + n.produced_by + A.r : '';
      w('  ' + A.dim + prefix + connector + A.r + col + n.kind + A.r + A.dim + '#' + shortId(n.id) + A.r + by + score + tag + '\n');
    };
    const walk = (id, prefix, isLast, depth) => {
      if (seen.has(id)) return; seen.add(id);
      const n = byId[id]; if (!n) return;
      line(n, prefix, depth === 0 ? '' : (isLast ? '└─ ' : '├─ '));
      const kids = (kidsOf[id] || []).filter((k) => byId[k]);
      const childPrefix = prefix + (depth === 0 ? '' : (isLast ? '   ' : '│  '));
      kids.forEach((k, i) => walk(k, childPrefix, i === kids.length - 1, depth + 1));
    };
    roots.forEach((r, i) => walk(r.id, '', i === roots.length - 1, 0));
    nodes.forEach((n) => { if (!seen.has(n.id)) line(n, '', ''); });   // orphans (parents off-window)
  }

  // /models: list models, marking the current one; falls back to the curated Codex list.
  function renderModels(obj) {
    const cur = obj.current || currentBrain || '';
    w('\n' + A.b + '  models' + A.r + (obj.source ? A.dim + '  (' + obj.source + ')' + A.r : '') + '\n');
    let list = Array.isArray(obj.available) ? obj.available : [];
    if (obj.note) w('  ' + A.dim + obj.note + A.r + '\n');
    if (!list.length && obj.source === 'error') { w('  ' + A.dim + 'curated fallback:' + A.r + '\n'); list = CODEX_MODELS; }
    list.forEach((mid) => {
      const isCur = cur && (cur === mid || cur.endsWith(':' + mid));
      w('  ' + (isCur ? A.green + '● ' : A.dim + '· ') + A.r + (isCur ? A.green + mid + ' (current)' + A.r : mid) + '\n');
    });
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
  //     It animates by overwriting each line in place (clear-line, never erase-below), so it
  //     does not flicker. Every line is width-budgeted so it never wraps (a wrap would throw
  //     off the fixed-line in-place redraw).
  const GLYPHS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  const SPARK = '▁▂▃▄▅▆▇█';
  const PLAIN = !!process.env.FORGEWRIGHT_PLAIN;   // escape hatch: minimal one-line status
  const hud = { timer: null, start: 0, i: 0, prevLines: 0, tokens: 0, agentStep: null,
    pipeline: null, metric: null, budget: null, activeRole: null, lastAction: '' };
  // session provenance graph, accumulated from `artifact` events for /graph
  const sessionGraph = [];
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

    // status row: active specialist · (metrics | last action) · elapsed · tokens
    const m = hud.metric;
    const lb = lineBuilder(maxw); lb.add('  ');
    lb.addRaw('\x1b[35m' + spin + A.r, 1); lb.add(' ');
    const who = (hud.activeRole && hud.activeRole !== 'agent') ? hud.activeRole : 'working';
    lb.add(who, roleDot(hud.activeRole || ''));
    if (m && m.step != null) {
      lb.add(' · '); lb.add('step ' + m.step + (m.total ? '/' + m.total : ''), A.white);
    }
    if (m && m.loss != null) {
      lb.add(' · '); lb.add('loss ' + fmtNum(m.loss), A.dim);
      const sp = sparkline(m.histLoss); if (sp) { lb.add(' '); lb.addRaw(A.cyan + sp + A.r, sp.length); }
    }
    if (m && m.reward != null) {
      lb.add(' · '); lb.add('rwd ' + fmtNum(m.reward), A.dim);
      const sp = sparkline(m.histReward); if (sp) { lb.add(' '); lb.addRaw(A.green + sp + A.r, sp.length); }
    }
    if (m && m.grad_norm != null) { lb.add(' · '); lb.add('grad ' + fmtNum(m.grad_norm), A.dim); }
    if (!m && hud.lastAction) { lb.add(' · '); lb.add(hud.lastAction, A.dim); }
    lb.add(' · '); lb.add(el + 's', A.dim);
    if (hud.tokens) { lb.add(' · '); lb.add('↑' + fmtTok(hud.tokens) + ' tok', A.dim); }
    out.push(lb.str());

    // guardrails row: the governor's safety envelope, with a live step gauge vs the step cap
    if (hud.budget && maxw >= 40) {
      const b = hud.budget;
      const lb2 = lineBuilder(maxw); lb2.add('  ');
      lb2.add('guardrails ', A.dim);
      // the agent-step budget is the one cap that is actually enforced; show it as a live gauge
      if (hud.agentStep != null && b.max_steps) {
        const frac = Math.max(0, Math.min(1, hud.agentStep / b.max_steps));
        const W = 10, fill = Math.round(frac * W);
        const near = frac >= 0.85;
        lb2.addRaw((near ? A.yellow : A.green) + '▰'.repeat(fill) + A.r + A.dim + '▱'.repeat(W - fill) + A.r, W);
        lb2.add(' ' + hud.agentStep + '/' + b.max_steps + ' steps', near ? A.yellow : A.dim);
      } else {
        lb2.add('≤' + b.max_steps + ' steps', A.dim);
      }
      lb2.add('  ·  ', A.dim);
      lb2.add('≤' + b.max_gpu_hours + ' GPU·h  ·  ≤$' + b.max_cost_usd + '  ·  ≤' + b.max_wall_clock_hours + 'h wall', A.dim);
      out.push(lb2.str());
    }
    return out;
  }
  const fmtNum = (v) => { const n = Number(v); return Number.isFinite(n) ? (Math.abs(n) < 1 ? n.toFixed(3) : n.toFixed(2)) : String(v); };

  // The HUD updates IN PLACE to avoid flicker: the per-frame animation overwrites each line
  // with `\x1b[2K` (clear-line) at its fixed position. It never uses `\x1b[J` (erase-below),
  // which blanks the whole block each frame and is what caused the flicker. Invariant: after
  // any draw, the cursor sits at the END of the last HUD line (no trailing newline).
  function curLines() {
    if (!PLAIN) return hudLines();
    return ['', A.dim + GLYPHS[hud.i % GLYPHS.length] + ' working… (' +
      Math.round((Date.now() - hud.start) / 1000) + 's' +
      (hud.tokens ? ' · ↑' + fmtTok(hud.tokens) : '') + ')' + A.r];
  }
  function hudDraw() {                       // fresh draw at the current cursor position
    if (!busy || awaitingApproval) return;
    const lines = curLines();
    w(lines.join('\n'));                     // no trailing newline -> cursor at end of last line
    hud.prevLines = lines.length;
  }
  function hudRedraw() {                     // in-place overwrite (the animation path; no flicker)
    if (!busy || awaitingApproval) return;
    if (hud.prevLines === 0) return hudDraw();
    const lines = curLines();
    if (lines.length !== hud.prevLines) { hudClear(); return hudDraw(); }  // structure changed
    let s = '\r' + (hud.prevLines > 1 ? '\x1b[' + (hud.prevLines - 1) + 'A' : '');  // to block top
    for (let i = 0; i < lines.length; i++) s += '\x1b[2K' + lines[i] + (i < lines.length - 1 ? '\n\r' : '');
    w(s);                                    // cursor back at end of last line
    hud.prevLines = lines.length;
  }
  function hudClear() {                      // erase the block; leave cursor at its top (col 0)
    if (hud.prevLines <= 0) return;
    let s = '\r\x1b[2K';
    for (let i = 1; i < hud.prevLines; i++) s += '\x1b[1A\x1b[2K';
    w(s);
    hud.prevLines = 0;
  }
  function hudResetTurn() {
    hud.start = Date.now(); hud.i = 0; hud.tokens = 0; hud.agentStep = null;
    hud.pipeline = null; hud.metric = null; hud.budget = null; hud.activeRole = null; hud.lastAction = '';
  }
  function startStatus() {
    if (!hud.start) hud.start = Date.now();
    if (hud.timer) return;
    hudDraw();
    hud.timer = setInterval(() => { hud.i++; hudRedraw(); }, 140);
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
    w('  ' + A.cyan + '/graph' + A.r + A.dim + '   show the provenance DAG of artifacts this session' + A.r + '\n');
    w('  ' + A.cyan + '/models' + A.r + A.dim + '  list models the current brain can reach' + A.r + '\n');
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
    // backend commands: send and let the resulting events + `done` drive back to the prompt
    if (c === '/graph') { send({ type: 'command', name: 'graph' }); return; }
    if (c === '/models') { send({ type: 'command', name: 'models' }); return; }
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
