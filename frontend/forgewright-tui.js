#!/usr/bin/env node
// Forgewright TUI: a single Claude-Code-style conversational terminal for the
// post-training swarm. It spawns `forgewright serve` as the backend, streams its
// role-tagged JSON events into one transcript, and surfaces approvals at one prompt.
// The swarm (Director + specialists) stays entirely on the Python backend.
//
// Built with terminal-kit by Cedric Ronvel (MIT). See NOTICE.
//
// Modes:
//   node forgewright-tui.js [--brain <b>]   interactive TUI (spawns `forgewright serve`)
//   node forgewright-tui.js --render-test    headless: read JSON events on stdin, print them
'use strict';

const readline = require('readline');
const { formatEvent } = require('./lib/render');

// --- headless render test (no terminal-kit, no child): verify the event contract ---
function renderTest() {
  const rl = readline.createInterface({ input: process.stdin });
  rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let obj;
    try { obj = JSON.parse(line); } catch (e) { return; }
    for (const r of formatEvent(obj)) process.stdout.write('[' + r.color + '] ' + r.text + '\n');
  });
}

// --- interactive TUI ---------------------------------------------------------------
function interactive(brain) {
  const { spawn } = require('child_process');
  const term = require('terminal-kit').terminal;

  // spawn the Python backend via the module (no command-name clash with this `forgewright` TUI).
  // Point FORGEWRIGHT_PYTHON at your forgewright venv python if it is not the default python3.
  const python = process.env.FORGEWRIGHT_PYTHON || 'python3';
  const args = ['-m', 'forgewright', 'serve'];
  if (brain) args.push('--brain', brain);
  const child = spawn(python, args, { stdio: ['pipe', 'pipe', 'inherit'], shell: process.platform === 'win32' });
  child.on('error', (e) => {
    process.stderr.write(
      'forgewright: could not start the Python backend (' + python + ' -m forgewright serve): ' + e.message + '\n' +
      'Install it (pip install -e .) and/or set FORGEWRIGHT_PYTHON to your venv python, e.g.\n' +
      '  export FORGEWRIGHT_PYTHON=~/projects/forgewright/.venv/bin/python\n');
    process.exit(1);
  });

  let busy = false;
  let awaitingApproval = false;
  let spinner = null;

  const send = (obj) => child.stdin.write(JSON.stringify(obj) + '\n');

  async function startThinking() {
    try {
      stopThinking();
      spinner = await term.spinner('impulse');
      term.gray(' working…');
    } catch (e) { spinner = null; }
  }
  function stopThinking() {
    try {
      if (spinner) { spinner.animate(false); spinner = null; term.column(1); term.eraseLine(); }
    } catch (e) { spinner = null; }
  }

  const show = (obj) => {
    stopThinking();
    for (const r of formatEvent(obj)) { term[r.color](r.text); term('\n'); }
    if (busy && !awaitingApproval && obj.type !== 'done') startThinking();
  };

  function prompt() {
    if (busy || awaitingApproval) return;
    term.brightGreen('\nyou › ');
    term.inputField({ cancelable: true }, (err, input) => {
      term('\n');
      if (input && input.trim()) { busy = true; send({ type: 'user_msg', text: input.trim() }); startThinking(); }
      else prompt();
    });
  }

  function handleApproval(obj) {
    awaitingApproval = true;
    stopThinking();
    for (const r of formatEvent(obj)) { term[r.color](r.text); term('\n'); }
    const tool = obj.tool || 'command';
    const items = ['approve once', 'approve all ' + tool, 'YOLO: bypass all', 'deny'];
    const decisions = ['yes', 'all', 'yolo', 'no'];
    term.yellow('approve ' + tool + '? ');
    term.singleLineMenu(items, { selectedIndex: 0 }, (err, resp) => {
      term('\n');
      const i = (resp && resp.selectedIndex != null) ? resp.selectedIndex : 3;
      send({ type: 'approval_response', decision: decisions[i] });
      awaitingApproval = false;
      if (busy) startThinking();
    });
  }

  const rl = readline.createInterface({ input: child.stdout });
  rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let obj;
    try { obj = JSON.parse(line); } catch (e) { return; }
    if (obj.type === 'approval_request') { handleApproval(obj); return; }
    if (obj.type === 'done') busy = false;   // set before show() so the spinner does not restart
    show(obj);
    if (obj.type === 'ready') prompt();
    if (obj.type === 'done') prompt();
    if (obj.type === 'bye') { stopThinking(); term.processExit(0); }
  });

  child.on('exit', (code) => { term.processExit(code || 0); });

  term.on('key', (name) => {
    if (name === 'CTRL_C') { send({ type: 'shutdown' }); try { child.stdin.end(); } catch (e) {} term.processExit(0); }
  });

  term.fullscreen(false);
  term.brightCyan('Forgewright'); term(' · post-training swarm · one chat, the swarm works behind it\n');
  term.gray('starting backend (forgewright serve)…\n');
}

const argv = process.argv.slice(2);
if (argv.includes('--render-test')) {
  renderTest();
} else {
  const bi = argv.indexOf('--brain');
  interactive(bi >= 0 ? argv[bi + 1] : null);
}
