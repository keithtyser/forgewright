# Forgewright terminal UI

The Node UI starts the Python backend and displays its events. Install the Python
package first, then use Node 24 and the checked-in lockfile:

```bash
npm ci --ignore-scripts
npm start
npm start -- --brain YOUR_CONFIGURED_PROVIDER
```

After `npm install -g .`, the command is `forgewright-tui`. Python owns the separate
`forgewright` command. Set `FORGEWRIGHT_PYTHON` to select a Python environment.
Use `forgewright-tui --help` for supported UI options; use Python's `run` command
for a positional goal, resume and backend options.

The UI and backend exchange newline-delimited JSON. Approval responses must carry
the current request ID. EOF denies pending approvals. A completed transport turn
can still report an incomplete task through `done.ok=false` and `done.reason`.

Set `NO_COLOR=1` for plain color output, `FORGEWRIGHT_REDUCED_MOTION=1` for slower
animation, or `FORGEWRIGHT_PLAIN=1` for compact status. Use `--fullscreen` for the
optional full-screen view. Terminal-control sequences are removed from event text.

Run `npm test` for rendering, theme, UI and sanitization checks. These helper tests
do not replace PTY, keyboard-only, resize or screen-reader testing.

Built with [terminal-kit](https://www.terminal-kit.com/) by Cedric Ronvel (MIT).
See NOTICE.
