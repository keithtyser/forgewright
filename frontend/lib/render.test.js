// Dependency-free assertions for the pure render layer (run: `npm test`). Covers the
// swarm-native event types so their transcript shape can't silently regress.
'use strict';
const assert = require('assert');
const { formatEvent } = require('./render');

function only(obj) { return formatEvent(obj).map((r) => r.text); }

// pipeline -> one plan line listing the stages
assert.deepStrictEqual(
  only({ type: 'pipeline', stages: ['DataCurator', 'SFTTrainer', 'Evaluator'] }),
  ['◆ plan DataCurator → SFTTrainer → Evaluator']
);

// stage: active is HUD-only (no transcript line); done/failed are recorded
assert.deepStrictEqual(only({ type: 'stage', name: 'SFTTrainer', state: 'active' }), []);
assert.deepStrictEqual(only({ type: 'stage', name: 'SFTTrainer', state: 'done' }), ['✓ SFTTrainer complete']);
assert.deepStrictEqual(only({ type: 'stage', name: 'Evaluator', state: 'failed' }), ['✗ Evaluator failed']);

// artifact: lineage badge with short ids, headline metric, and pass/fail tag
assert.deepStrictEqual(
  only({ type: 'artifact', role: 'Evaluator', kind: 'eval', id: 'eval-20260607-def456',
    parents: ['adapter-20260607-xyz789'], metrics: { score: 0.94 }, passed: true }),
  ['◇ eval#def456 ← xyz789  score 0.94 ✓']
);
assert.deepStrictEqual(
  only({ type: 'artifact', role: 'DataCurator', kind: 'dataset', id: 'dataset-1-abc123', parents: [], metrics: {} }),
  ['◇ dataset#abc123']
);

// color attribution by producer role
assert.strictEqual(
  formatEvent({ type: 'artifact', role: 'SFTTrainer', kind: 'adapter', id: 'adapter-1-q', parents: [] })[0].color,
  'brightGreen'
);

// metric/budget are live HUD-only -> no transcript lines
assert.deepStrictEqual(only({ type: 'metric', loss: 0.3, step: 5 }), []);
assert.deepStrictEqual(only({ type: 'budget', max_steps: 80 }), []);

// graph renders a header + one line per node (root-to-leaf order preserved by caller)
{
  const out = only({ type: 'graph', nodes: [
    { id: 'dataset-1-aaa111', kind: 'dataset', produced_by: 'DataCurator', parents: [] },
    { id: 'eval-1-bbb222', kind: 'eval', produced_by: 'Evaluator', parents: ['adapter-1-ccc333'], score: 0.94, passed: true },
  ] });
  assert.strictEqual(out[0], 'provenance graph (2 artifacts)');
  assert.ok(out[2].includes('eval#bbb222') && out[2].includes('← ccc333') && out[2].includes('0.94') && out[2].includes('✓'));
}

// models marks the current one
{
  const out = only({ type: 'models', available: ['gpt-5.5-codex', 'gpt-5'], current: 'oauth-codex:gpt-5', source: 'probe' });
  assert.ok(out.some((l) => l.includes('gpt-5 (current)')));
}

// artifact: the headline metric prefers a meaningful key (capability) over an arbitrary one
assert.deepStrictEqual(
  only({ type: 'artifact', role: 'Quantizer', kind: 'model', id: 'model-1-zzz999',
    parents: [], metrics: { exit_code: 0, capability: 0.91 } }),
  ['◇ model#zzz999  capability 0.91']
);

// duration: a pre-formatted obj.dur is appended to tool / stage / artifact lines (info clarity)
assert.deepStrictEqual(only({ type: 'stage', name: 'Quantizer', state: 'done', dur: '12.3s' }),
  ['✓ Quantizer complete  12.3s']);
{
  const head = only({ type: 'tool', tool: 'launch_job', ok: true, args: {}, dur: '2.0s' })[0];
  assert.ok(head.includes('launch_job') && head.endsWith('2.0s'));
}
assert.ok(only({ type: 'artifact', kind: 'model', id: 'm-1-aaa111', parents: [], dur: '40s' })[0].endsWith('40s'));

console.log('render.test.js: all assertions passed');
