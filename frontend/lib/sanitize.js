'use strict';
const { stripVTControlCharacters } = require('node:util');

function sanitize(value) {
  if (typeof value === 'string') {
    return stripVTControlCharacters(value).replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, '');
  }
  if (Array.isArray(value)) return value.map(sanitize);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, sanitize(v)]));
  }
  return value;
}
module.exports = { sanitize };
