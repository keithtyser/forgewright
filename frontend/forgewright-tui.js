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
    const r = formatEvent(obj);
    if (r) process.stdout.write('[' + r.color + '] ' + r.text + '\n');
  });
}

// --- interactive TUI ---------------------------------------------------------------
function interactive(brain) {
  const { spawn } = require('child_process');
  const term = require('terminal-kit').terminal;

  const args = ['serve'];
  if (brain) args.push('--brain', brain);
  const child = spawn('forgewright', args, { stdio: ['pipe', 'pipe', 'inherit'], shell: process.platform === 'win32' });

  let busy = false;
  let awaitingApproval = false;

  const send = (obj) => child.stdin.write(JSON.stringify(obj) + '\n');

  const show = (obj) => {
    const r = formatEvent(obj);
    if (r) { term[r.color](r.text); term('\n'); }
  };

  function prompt() {
    if (busy || awaitingApproval) return;
    term.brightGreen('\nyou › ');
    term.inputField({ cancelable: true }, (err, input) => {
      term('\n');
      if (input && input.trim()) { busy = true; send({ type: 'user_msg', text: input.trim() }); }
      else prompt();
    });
  }

  function handleApproval(obj) {
    awaitingApproval = true;
    show(obj);
    term.yellow('approve? [y/N] ');
    term.yesOrNo({ yes: ['y', 'Y'], no: ['n', 'N', 'ENTER'] }, (err, yes) => {
      term('\n');
      send({ type: 'approval_response', approved: !!yes });
      awaitingApproval = false;
    });
  }

  const rl = readline.createInterface({ input: child.stdout });
  rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let obj;
    try { obj = JSON.parse(line); } catch (e) { return; }
    if (obj.type === 'approval_request') { handleApproval(obj); return; }
    show(obj);
    if (obj.type === 'ready') prompt();
    if (obj.type === 'done') { busy = false; prompt(); }
    if (obj.type === 'bye') { term.processExit(0); }
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
