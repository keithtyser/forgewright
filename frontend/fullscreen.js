// Full-screen Forgewright TUI on terminal-kit's document model: a status bar, a scrollable
// transcript pane (mouse-wheel + keys), a live HUD strip, and an always-focused input. Reuses
// lib/render.formatEvent for event -> colored segments, mapped to terminal-kit markup (untrusted
// text is caret-escaped). The classic scrolling UI stays the default until this is proven; this
// is opt-in via FORGEWRIGHT_FULLSCREEN=1 (or --fullscreen).
'use strict';
const { spawn } = require('child_process');
const readline = require('readline');
const { formatEvent } = require('./lib/render');
const creds = require('./lib/creds');

// color-name (from lib/render) -> terminal-kit markup prefix
const MARKUP = {
  gray: '^K', white: '^w', brightWhite: '^W', cyan: '^c', brightCyan: '^C',
  green: '^g', brightGreen: '^G', red: '^r', brightRed: '^R', yellow: '^y',
  brightYellow: '^Y', blue: '^b', brightBlue: '^B', magenta: '^m', brightMagenta: '^M',
};
const escMarkup = (s) => String(s == null ? '' : s).replace(/\^/g, '^^');   // safe for model text
const mk = (seg) => (MARKUP[seg.color] || '^w') + escMarkup(seg.text) + '^:';

const GLYPHS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPARK = '▁▂▃▄▅▆▇█';
const ROLE_MK = {
  Director: '^C', DataCurator: '^B', SFTTrainer: '^G', RLTrainer: '^g', Abliterator: '^M',
  Quantizer: '^Y', ServingOptimizer: '^y', Evaluator: '^W', Publisher: '^R', Merger: '^c',
};
const fmtTok = (n) => (n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n));
const sparkline = (a) => {
  if (!a || a.length < 2) return '';
  const lo = Math.min.apply(null, a), hi = Math.max.apply(null, a), rng = (hi - lo) || 1;
  return a.map((v) => SPARK[Math.min(7, Math.floor((v - lo) / rng * 7.999))]).join('');
};

function run(brainArg) {
  const cfg = creds.resolveBrainConfig(brainArg);
  if (!cfg) {
    process.stderr.write(
      'No brain configured yet. Run the classic UI once to set up:\n' +
      '  FORGEWRIGHT_CLASSIC=1 forgewright   (pick OpenRouter / Codex), then retry full-screen.\n');
    process.exit(1);
    return;
  }

  const termkit = require('terminal-kit');
  const term = termkit.terminal;

  let child = null, rl = null, document = null, transcript = null, hud = null, input = null;
  let busy = false, awaitingApproval = false, lastCtrlC = 0, timer = null;
  const state = { start: 0, i: 0, tokens: 0, actions: 0, pipeline: null, metric: null,
                  activeRole: null, lastAction: '' };

  function teardown(code) {
    if (timer) { clearInterval(timer); timer = null; }
    try { term.grabInput(false); } catch (e) {}
    try { term.fullscreen(false); } catch (e) {}
    try { child && child.stdin.end(); } catch (e) {}
    term.processExit(code || 0);
  }

  // --- layout -----------------------------------------------------------------------
  function buildUI() {
    term.fullscreen(true);
    term.grabInput({ mouse: 'button' });
    document = term.createDocument({ palette: new termkit.Palette() });
    const W = term.width, H = term.height;

    new termkit.Text({
      parent: document, x: 0, y: 0, contentHasMarkup: true,
      content: '^C^+ forgewright ^:^K· ' + escMarkup(cfg.brain) + '^:',
    });
    transcript = new termkit.TextBox({
      parent: document, x: 0, y: 1, width: W, height: H - 4,
      scrollable: true, vScrollBar: true, wordWrap: true, contentHasMarkup: true, content: '',
    });
    hud = new termkit.TextBox({
      parent: document, x: 0, y: H - 3, width: W, height: 1, contentHasMarkup: true, content: '',
    });
    new termkit.Text({
      parent: document, x: 0, y: H - 2, contentHasMarkup: true,
      content: '^K' + '─'.repeat(Math.max(0, W)) + '^:',
    });
    input = new termkit.InlineInput({
      parent: document, x: 0, y: H - 1, width: W,
      prompt: { content: '^G^+❯ ^:', contentHasMarkup: true },
      history: creds.loadHistory(),
      autoComplete: ['/graph', '/models', '/login', '/brain', '/help', '/quit'],
      autoCompleteMenu: true, autoCompleteHint: true,
    });
    input.on('submit', onSubmit);
    document.giveFocusTo(input);
  }

  function appendLine(line) {
    try { transcript.appendLog(line); } catch (e) { /* element may be torn down */ }
  }
  function note(markupText) { appendLine(markupText); }

  // --- HUD --------------------------------------------------------------------------
  function hudContent() {
    const spin = GLYPHS[state.i % GLYPHS.length];
    const el = Math.round((Date.now() - state.start) / 1000);
    const parts = [];
    if (state.pipeline && state.pipeline.length) {
      parts.push(state.pipeline.map((s) => {
        const g = s.state === 'done' ? '^g✓' : s.state === 'failed' ? '^r✗'
          : s.state === 'active' ? '^M' + spin : '^K◌';
        return g + ' ' + (ROLE_MK[s.name] || '^w') + s.name + '^:';
      }).join('^K · ^:'));
    }
    const who = (state.activeRole && state.activeRole !== 'agent') ? state.activeRole : 'working';
    let line = '^M' + spin + '^: ' + (ROLE_MK[state.activeRole] || '^w') + who + '^:';
    const m = state.metric;
    if (m && m.step != null) line += '^K · ^:step ' + m.step + (m.total ? '/' + m.total : '');
    if (m && m.loss != null) line += '^K · ^:loss ' + m.loss + ' ^c' + sparkline(m.histLoss) + '^:';
    if (m && m.reward != null) line += '^K · ^:rwd ' + m.reward + ' ^g' + sparkline(m.histReward) + '^:';
    if (!m && state.lastAction) line += '^K · ' + escMarkup(state.lastAction) + '^:';
    if (state.actions) line += '^K · ' + state.actions + ' actions^:';
    line += '^K · ' + el + 's^:';
    if (state.tokens) line += '^K · ↑' + fmtTok(state.tokens) + ' tok^:';
    return parts.length ? (parts.join('') + '\n' + line) : line;
  }
  function drawHud() { try { hud.setContent(busy ? hudContent() : '', true); } catch (e) {} }
  function startTimer() { if (!timer) timer = setInterval(() => { state.i++; if (busy) drawHud(); }, 140); }
  function resetTurn() {
    state.start = Date.now(); state.i = 0; state.tokens = 0; state.actions = 0;
    state.pipeline = null; state.metric = null; state.activeRole = null; state.lastAction = '';
  }

  // --- backend ----------------------------------------------------------------------
  function send(obj) { try { if (child && child.stdin.writable) child.stdin.write(JSON.stringify(obj) + '\n'); } catch (e) {} }

  function startBackend() {
    const python = process.env.FORGEWRIGHT_PYTHON || 'python3';
    const args = ['-m', 'forgewright', 'serve'];
    if (cfg.brain) args.push('--brain', cfg.brain);
    child = spawn(python, args, {
      stdio: ['pipe', 'pipe', 'inherit'], shell: process.platform === 'win32',
      env: Object.assign({}, process.env, cfg.env || {}),
    });
    child.on('error', (e) => { note('^rbackend failed to start: ' + escMarkup(e.message) + '^:'); });
    child.on('exit', (code) => teardown(code || 0));
    rl = readline.createInterface({ input: child.stdout });
    rl.on('line', onLine);
  }

  function onLine(line) {
    line = line.trim(); if (!line) return;
    let obj; try { obj = JSON.parse(line); } catch (e) { return; }
    if (obj.type === 'approval_request') { showApproval(obj); return; }
    if (obj.type === 'done') busy = false;
    updateHudState(obj);
    for (const seg of formatEvent(obj)) appendLine(mk(seg));
    if (obj.type === 'bye') return teardown(0);
    if (busy) startTimer();
    drawHud();
  }

  function updateHudState(obj) {
    switch (obj.type) {
      case 'assistant':
        if (obj.role && obj.role !== 'agent') state.activeRole = obj.role;
        if (obj.usage && obj.usage.total_tokens) state.tokens = obj.usage.total_tokens;
        break;
      case 'tool':
        if (obj.role && obj.role !== 'agent') state.activeRole = obj.role;
        state.actions += 1;
        state.lastAction = String((obj.tool || '')).slice(0, 48);
        break;
      case 'pipeline':
        state.pipeline = (obj.stages || []).map((n) => ({ name: n, state: 'pending' }));
        state.activeRole = 'Director'; break;
      case 'stage':
        if (state.pipeline && state.pipeline[obj.index]) state.pipeline[obj.index].state = obj.state;
        if (obj.state === 'active') { state.activeRole = obj.name; state.metric = null; }
        break;
      case 'metric': {
        const keep = (a, v) => { a.push(+v); if (a.length > 28) a.shift(); return a; };
        const m = state.metric || (state.metric = { step: null, total: null, loss: null, reward: null, histLoss: [], histReward: [] });
        if (obj.step != null) m.step = obj.step;
        if (obj.total != null) m.total = obj.total;
        if (obj.loss != null) { m.loss = obj.loss; keep(m.histLoss, obj.loss); }
        if (obj.reward != null) { m.reward = obj.reward; keep(m.histReward, obj.reward); }
        break;
      }
      default: break;
    }
  }

  // --- input + approval + interrupt -------------------------------------------------
  function onSubmit(value) {
    const t = (value || '').trim();
    try { input.setContent('', false); } catch (e) {}
    try { document.giveFocusTo(input); } catch (e) {}
    if (!t) return;
    creds.appendHistory(t);
    note('^G^+❯ ^:' + escMarkup(t));
    if (t[0] === '/') return handleSlash(t);
    const wasBusy = busy;
    if (!wasBusy) { busy = true; resetTurn(); }
    send({ type: 'user_msg', text: t });
    if (wasBusy) note('^K  ↳ queued — runs after the current turn^:');
    if (!wasBusy) startTimer();
    drawHud();
  }

  function handleSlash(cmd) {
    const c = cmd.toLowerCase();
    if (c === '/quit' || c === '/exit') return teardown(0);
    if (c === '/brain') return note('^Kbrain: ' + escMarkup(cfg.brain) + '^:');
    if (c === '/help' || c === '/?') {
      note('^W^+commands^:');
      ['/graph  provenance DAG', '/models  models the brain can reach', '/brain  current brain',
       '/quit  exit  (Ctrl-C interrupts a run, twice quits)'].forEach((l) => note('^K  ' + l + '^:'));
      return;
    }
    if (c === '/graph' || c === '/models') {
      const wasBusy = busy; if (!wasBusy) { busy = true; resetTurn(); }
      send({ type: 'command', name: c.slice(1) });
      if (!wasBusy) startTimer();
      return;
    }
    if (c === '/login') return note('^y/login from full-screen lands next; for now: FORGEWRIGHT_CLASSIC=1 forgewright^:');
    note('^Kunknown command ' + escMarkup(cmd) + ' — try /help^:');
  }

  function showApproval(obj) {
    awaitingApproval = true;
    note('^Y^+⚠ approval needed: ^:^Y' + escMarkup(obj.tool || 'command') + (obj.risk ? ' (' + escMarkup(obj.risk) + ')' : '') + '^:');
    if (obj.args && Object.keys(obj.args).length) note('^K  ' + escMarkup(JSON.stringify(obj.args).slice(0, 200)) + '^:');
    const items = [
      { content: 'approve once', value: 'yes' },
      { content: 'approve all ' + (obj.tool || ''), value: 'all' },
      { content: 'YOLO: bypass all permissions', value: 'yolo' },
      { content: 'deny', value: 'no' },
    ];
    const menu = new termkit.ColumnMenu({
      parent: document, x: 2, y: Math.max(2, term.height - 6),
      buttonFocusAttr: { bgColor: 'green', color: 'white', bold: true }, items,
    });
    document.giveFocusTo(menu);
    menu.on('submit', (value) => {
      try { menu.destroy(); } catch (e) {}
      send({ type: 'approval_response', decision: value || 'no' });
      awaitingApproval = false;
      try { document.giveFocusTo(input); } catch (e) {}
    });
  }

  function onCtrlC() {
    const now = Date.now();
    if (now - lastCtrlC < 2000) return teardown(0);
    lastCtrlC = now;
    if (busy) { send({ type: 'interrupt' }); note('^Y  ⎿ interrupting… (Ctrl-C again to quit)^:'); }
    else note('^K  (Ctrl-C again to quit)^:');
  }

  // --- start ------------------------------------------------------------------------
  try {
    buildUI();
    term.on('key', (name) => { if (name === 'CTRL_C') onCtrlC(); });
    note('^Kstarting backend…  (type anytime · /help · Ctrl-C interrupts, twice quits)^:');
    startBackend();
  } catch (e) {
    teardown(1);
    throw e;   // dispatcher catches and falls back to classic
  }
}

module.exports = { run };
