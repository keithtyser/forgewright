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

console.log('render.test.js: all assertions passed');
