// Persisted brain credentials + brain resolution, shared by both front-ends (classic + full-screen)
// and the Python backend (forgewright/credentials.py): ~/.forgewright/credentials.json.
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');

const OPENROUTER_DEFAULT_MODEL = 'openrouter:deepseek/deepseek-v4-pro';
// OpenAI models reachable over the Codex (ChatGPT-login) Responses API, newest first; the wizard
// also offers a free-text entry so newer ids work without a code change.
const CODEX_MODELS = [
  'gpt-5.5-codex', 'gpt-5.5', 'gpt-5.1-codex-max', 'gpt-5.1-codex',
  'gpt-5.1', 'gpt-5-codex', 'gpt-5', 'gpt-5-mini',
];

function fwHome() { return process.env.FORGEWRIGHT_HOME || path.join(os.homedir(), '.forgewright'); }
function credsPath() { return path.join(fwHome(), 'credentials.json'); }
function historyPath() { return path.join(fwHome(), 'history'); }
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
    return true;
  } catch (e) { return false; }
}
function envFromCreds(creds) {
  const env = {};
  if (creds.openrouter_api_key) env.OPENROUTER_API_KEY = creds.openrouter_api_key;
  if (creds.anthropic_api_key) env.ANTHROPIC_API_KEY = creds.anthropic_api_key;
  if (creds.openai_api_key) env.OPENAI_API_KEY = creds.openai_api_key;
  return env;
}
// { brain, env } if a brain is already configured (flag > saved creds > env), else null.
function resolveBrainConfig(brainArg) {
  const creds = loadCreds();
  const env = envFromCreds(creds);
  const haveOR = process.env.OPENROUTER_API_KEY || env.OPENROUTER_API_KEY;
  const brain = brainArg || creds.brain || (haveOR ? OPENROUTER_DEFAULT_MODEL : null);
  return brain ? { brain, env } : null;
}
// command history (newest last), persisted across sessions
function loadHistory() {
  try {
    return fs.readFileSync(historyPath(), 'utf8').split('\n').map((s) => s.trim()).filter(Boolean).slice(-500);
  } catch (e) { return []; }
}
function appendHistory(line) {
  if (!line || !line.trim()) return;
  try { fs.mkdirSync(fwHome(), { recursive: true }); fs.appendFileSync(historyPath(), line.trim() + '\n', 'utf8'); } catch (e) {}
}

module.exports = {
  OPENROUTER_DEFAULT_MODEL, CODEX_MODELS, fwHome, credsPath, codexAuthPath,
  loadCreds, saveCreds, envFromCreds, resolveBrainConfig, loadHistory, appendHistory,
};
