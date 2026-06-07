# Forgewright TUI

A single Claude-Code-style conversational terminal for the Forgewright post-training swarm.
You chat here; the swarm (Director plus specialists) runs entirely on the Python backend.

Built with [terminal-kit](https://www.terminal-kit.com/) by Cedric Ronvel (MIT). See `NOTICE`.

## How it works

The TUI spawns `forgewright serve` as a child process and speaks newline-delimited JSON
with it:

- backend to UI: `assistant`, `tool`, `progress`, `approval_request`, `done` events,
  each tagged with the specialist `role` so one transcript stays attributable.
- UI to backend: `user_msg` (your turns) and `approval_response` (approve or deny at the
  single prompt). The swarm is never an agent-management surface; it is one chat.

## Run

```bash
npm install
npm start                 # interactive TUI; spawns `forgewright serve`
npm start -- --brain openrouter:deepseek/deepseek-v4-pro
```

## Verify the event contract without a TTY

```bash
# pipe backend events in, see how they render (no terminal-kit, no child process):
echo '{"type":"assistant","role":"Director","content":"plan: DataCurator -> SFTTrainer"}' \
  | node forgewright-tui.js --render-test
```

`lib/render.js` (`formatEvent`) is dependency-free and holds the pure rendering logic.
