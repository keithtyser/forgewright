'use strict';
const assert = require('assert');
const { sanitize } = require('./sanitize');
assert.deepStrictEqual(sanitize({ content: '\x1b[2Jvisible\x1b]52;c;clipboard\x07\nnext' }),
                       { content: 'visible\nnext' });
assert.deepStrictEqual(sanitize({ args: ['\x00safe', 3], ok: false }), { args: ['safe', 3], ok: false });
console.log('sanitize.test.js: all assertions passed');
