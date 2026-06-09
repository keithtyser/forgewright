// Full-screen Forgewright TUI on terminal-kit's document model: a status bar, a scrollable
// transcript pane (mouse-wheel + keys), a live HUD strip, and an always-focused input. Reuses
// lib/render.formatEvent for event -> colored segments, mapped to terminal-kit markup (untrusted
// text is caret-escaped). The classic scrolling UI stays the default until this is proven; this
// is opt-in via FORGEWRIGHT_FULLSCREEN=1 (or --fullscreen).
'use strict';
const { spawn } = require('child_process');
const readline = require('readline');
const { formatEvent, roleColor } = require('./lib/render');
const creds = require('./lib/creds');
const { makeTheme, escMarkup, mkVlen } = require('./lib/theme');
const ui = require('./lib/ui');

function run(brainArg) {
  let cfg = creds.resolveBrainConfig(brainArg);   // may be null on first run -> in-app wizard

  const termkit = require('terminal-kit');
  const term = termkit.terminal;

  // one color system (FORGEWRIGHT_THEME / NO_COLOR) shared with the classic UI + render layer
  const theme = makeTheme();
  const mk = (seg) => theme.paintMk(seg.color, escMarkup(seg.text));        // event segment -> markup
  const roleMk = (role) => theme.mk(roleColor(role));                       // role -> markup prefix

  let child = null, rl = null, document = null, transcript = null, hud = null, input = null;
  let busy = false, awaitingApproval = false, lastCtrlC = 0, timer = null;
  const state = { start: 0, i: 0, tokens: 0, actions: 0, pipeline: null, metric: null,
                  activeRole: null, lastAction: '', stageStart: 0 };

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
      content: theme.paintMk('brightCyan', ' forgewright ', true) + theme.paintMk('gray', '· ' + escMarkup(cfg ? cfg.brain : 'configure your brain')),
    });
    // layout reserves TWO rows for the HUD (pipeline row + status row) so neither is clipped
    transcript = new termkit.TextBox({
      parent: document, x: 0, y: 1, width: W, height: Math.max(1, H - 5),
      scrollable: true, vScrollBar: true, wordWrap: true, contentHasMarkup: true, content: '',
    });
    hud = new termkit.TextBox({
      parent: document, x: 0, y: H - 4, width: W, height: 2, contentHasMarkup: true, content: '',
    });
    new termkit.Text({
      parent: document, x: 0, y: H - 2, contentHasMarkup: true,
      content: theme.mk('gray') + '─'.repeat(Math.max(0, W)) + theme.mkReset,
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

  const buffer = [];
  function appendLine(line, replay) {
    if (!replay) { buffer.push(line); if (buffer.length > 2000) buffer.shift(); }
    try { transcript.appendLog(line); } catch (e) { /* element may be torn down */ }
  }
  function note(markupText) { appendLine(markupText); }

  // visible length of a markup string (shared with the theme layer)
  const vlen = mkVlen;
  // a clean rounded box appended into the scrolling transcript (terminal-kit-style framing)
  function panel(title, lines, accent) {
    accent = accent || '^C';
    const maxw = Math.max(28, (term.width || 80) - 2);
    const inner = Math.min(maxw - 2, Math.max(vlen(title) + 4, Math.max.apply(null, [0].concat(lines.map(vlen))), 28));
    appendLine(accent + '╭─ ^+' + title + '^: ' + accent + '─'.repeat(Math.max(1, inner - vlen(title) - 2)) + '╮^:');
    for (const ln of lines) appendLine(accent + '│ ^:' + ln + ' '.repeat(Math.max(0, inner - vlen(ln))) + accent + ' │^:');
    appendLine(accent + '╰' + '─'.repeat(inner + 2) + '╯^:');
  }

  // rebuild the document on resize, preserving the recent transcript
  function relayout() {
    try { if (document) document.destroy(); } catch (e) {}
    buildUI();
    for (const l of buffer.slice(-500)) appendLine(l, true);
    drawHud();
  }

  // --- /graph + /models as framed panels in the transcript --------------------------
  const sid = (id) => String(id || '').slice(-6);
  function renderGraphBox(nodes) {
    if (!nodes || !nodes.length) { panel('provenance graph', ['^Kno artifacts in the registry yet^:']); return; }
    const byId = {}; nodes.forEach((n) => { byId[n.id] = n; });
    const kids = {}; nodes.forEach((n) => (n.parents || []).forEach((p) => { (kids[p] = kids[p] || []).push(n.id); }));
    const roots = nodes.filter((n) => !(n.parents || []).some((p) => byId[p]));
    const seen = new Set(); const out = [];
    const fmt = (n, prefix, conn) => {
      const col = roleMk(n.produced_by);
      const tag = n.passed === false ? ' ^r✗^:' : (n.passed === true ? ' ^g✓^:' : '');
      const score = (n.score != null) ? '  ^K' + (Math.round(n.score * 1000) / 1000) + '^:' : '';
      const by = n.produced_by ? ' ^K· ' + n.produced_by + '^:' : '';
      return '^K' + prefix + conn + '^:' + col + escMarkup(n.kind) + '^:^K#' + sid(n.id) + '^:' + by + score + tag;
    };
    const walk = (id, prefix, isLast, depth) => {
      if (seen.has(id)) return; seen.add(id); const n = byId[id]; if (!n) return;
      out.push(fmt(n, prefix, depth === 0 ? '' : (isLast ? '└─ ' : '├─ ')));
      const ks = (kids[id] || []).filter((k) => byId[k]);
      const cp = prefix + (depth === 0 ? '' : (isLast ? '   ' : '│  '));
      ks.forEach((k, i) => walk(k, cp, i === ks.length - 1, depth + 1));
    };
    roots.forEach((r, i) => walk(r.id, '', i === roots.length - 1, 0));
    nodes.forEach((n) => { if (!seen.has(n.id)) out.push(fmt(n, '', '')); });
    panel('provenance graph  (' + nodes.length + ' artifacts)', out);
  }
  function renderModelsBox(obj) {
    const cur = obj.current || (cfg && cfg.brain) || '';
    let list = Array.isArray(obj.available) ? obj.available : [];
    const out = [];
    if (obj.note) out.push('^K' + escMarkup(obj.note) + '^:');
    if (!list.length && obj.source === 'error') { out.push('^Kcurated fallback:^:'); list = creds.CODEX_MODELS; }
    list.forEach((mid) => {
      const isCur = cur && (cur === mid || cur.endsWith(':' + mid));
      out.push(isCur ? '^g● ' + escMarkup(mid) + ' (current)^:' : '^K· ^:' + escMarkup(mid));
    });
    panel('models' + (obj.source ? '  (' + obj.source + ')' : ''), out.length ? out : ['^K(none)^:']);
  }

  // --- HUD (painted from the shared structured model in lib/ui) ----------------------
  function hudContent() {
    const spin = ui.GLYPHS[state.i % ui.GLYPHS.length];
    const W = Math.max(20, term.width || 80);
    return ui.hudRows(state, spin).map((r) => ui.paintRowMk(r.segs, theme, escMarkup, W)).join('\n');
  }
  function drawHud() { try { hud.setContent(busy ? hudContent() : '', true); } catch (e) {} }
  function startTimer() { if (!timer) timer = setInterval(() => { state.i++; if (busy) drawHud(); }, 120); }
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
    if (obj.type === 'graph') renderGraphBox(obj.nodes);
    else if (obj.type === 'models') renderModelsBox(obj);
    else for (const seg of formatEvent(obj)) appendLine(mk(seg));
    // a persistent end-of-turn summary (elapsed / actions / tokens) in the transcript
    if (obj.type === 'done' && obj.ok !== false && (state.actions || state.tokens)) {
      appendLine(ui.paintRowMk(ui.runSummary(state), theme, escMarkup, term.width || 80));
    }
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
        if (obj.state === 'active') { state.activeRole = obj.name; state.metric = null; state.stageStart = Date.now(); }
        else if (state.stageStart && (obj.state === 'done' || obj.state === 'failed')) {
          obj.dur = ui.fmtDur(Date.now() - state.stageStart);   // shown by formatEvent
        }
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
    if (c === '/login' || c === '/auth' || c === '/refresh') {
      return runWizard((c2) => { cfg = c2; relayout(); note('^Krestarting backend…^:'); restartBackend(); });
    }
    note('^Kunknown command ' + escMarkup(cmd) + ' — try /help^:');
  }

  // --- setup wizard (first run + /login) --------------------------------------------
  function runWizard(done) {
    note('^W^+Set up your brain^: ^K(↑/↓ · enter)^:');
    const menu = new termkit.ColumnMenu({
      parent: document, x: 2, y: Math.max(2, term.height - 7),
      buttonFocusAttr: { bgColor: 'green', color: 'white', bold: true },
      items: [{ content: 'OpenRouter API key', value: 'openrouter' },
              { content: 'Codex (ChatGPT login)', value: 'codex' }],
    });
    document.giveFocusTo(menu);
    menu.on('submit', (v) => {
      try { menu.destroy(); } catch (e) {}
      if (menuValue(v) === 'codex') wizardCodex(done); else wizardOpenRouter(done);
    });
  }
  function wizardOpenRouter(done) {
    note('^Kpaste your OpenRouter API key, then enter:^:');
    const ask = new termkit.InlineInput({
      parent: document, x: 0, y: term.height - 1, width: term.width,
      prompt: { content: '^G❯ ^:', contentHasMarkup: true },
    });
    document.giveFocusTo(ask);
    ask.on('submit', (key) => {
      try { ask.destroy(); } catch (e) {}
      key = (key || '').trim();
      if (!key) { note('^rno key entered^:'); return runWizard(done); }
      const c = creds.loadCreds();
      c.brain = creds.OPENROUTER_DEFAULT_MODEL; c.openrouter_api_key = key;
      creds.saveCreds(c);
      done({ brain: c.brain, env: { OPENROUTER_API_KEY: key } });
    });
  }
  function wizardCodex(done) {
    note('^Kpick a Codex model (run `codex login` first if you have not):^:');
    const items = creds.CODEX_MODELS.map((m) => ({ content: m, value: m }));
    const menu = new termkit.ColumnMenu({
      parent: document, x: 2, y: Math.max(2, term.height - 12),
      buttonFocusAttr: { bgColor: 'green', color: 'white', bold: true }, items,
    });
    document.giveFocusTo(menu);
    menu.on('submit', (v) => {
      try { menu.destroy(); } catch (e) {}
      const model = menuValue(v) || creds.CODEX_MODELS[0];
      const c = creds.loadCreds();
      c.brain = 'oauth-codex:' + model; c.codex_model = model; delete c.openrouter_api_key;
      creds.saveCreds(c);
      done({ brain: c.brain, env: {} });
    });
  }

  function restartBackend() {
    try { if (rl) rl.close(); } catch (e) {}
    const old = child;
    if (old) { old.removeAllListeners('exit'); old.once('exit', () => startBackend()); try { old.stdin.end(); } catch (e) {} try { old.kill(); } catch (e) {} }
    else startBackend();
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
      send({ type: 'approval_response', decision: menuValue(value) || 'no' });
      awaitingApproval = false;
      try { document.giveFocusTo(input); } catch (e) {}
    });
  }

  // terminal-kit ColumnMenu may emit the item's value or the item object; accept both.
  function menuValue(v) { return (v && typeof v === 'object') ? (v.value != null ? v.value : v.content) : v; }

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
    term.on('resize', () => { try { relayout(); } catch (e) {} });
    if (cfg) {
      note('^Kstarting backend…  (type anytime · /help · Ctrl-C interrupts, twice quits)^:');
      startBackend();
    } else {
      runWizard((c) => { cfg = c; relayout(); note('^Kstarting backend…^:'); startBackend(); });
    }
  } catch (e) {
    teardown(1);
    throw e;   // dispatcher catches and falls back to classic
  }
}

module.exports = { run };
