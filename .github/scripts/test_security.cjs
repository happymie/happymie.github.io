'use strict';

// Fast tests for the action-data parser and URL boundary. No DOM or network is used.
// Browser tests must separately check the rendered, sanitized document and SVGs.
const test = require('node:test');
const assert = require('node:assert/strict');
const security = require('../../site-assets/security.js');

test('the public security API is immutable', () => {
  assert.equal(Object.isFrozen(security), true);
});

test('representative migrated reference controls remain supported', () => {
  const examples = [
    ['toggleSidebarToc()', 'toggleSidebarToc', []],
    ["switchWorkspace('explorer')", 'switchWorkspace', ['explorer']],
    ["selectArchNode('architecture_model_006_u_net', 0)", 'selectArchNode', ['architecture_model_006_u_net', 0]],
    ["stepArchNode('architecture_model_006_u_net', -1)", 'stepArchNode', ['architecture_model_006_u_net', -1]],
    ["filterInteractiveTable('table-main', this.value)", 'filterInteractiveTable', ['table-main', {$ref: 'this.value'}]],
    ["sortInteractiveTable('table-main', 3)", 'sortInteractiveTable', ['table-main', 3]],
    ["showConstellationTooltip(event, 'architecture_model_006_u_net')", 'showConstellationTooltip', [{$ref: 'event'}, 'architecture_model_006_u_net']],
    ['hideConstellationTooltip()', 'hideConstellationTooltip', []],
    ["copyElementText('deploy-code', 'COPIED')", 'copyElementText', ['deploy-code', 'COPIED']],
    ["loadPresetComparison(['model-a', 'model-b'])", 'loadPresetComparison', [['model-a', 'model-b']]],
  ];
  for (const [source, name, args] of examples) {
    assert.deepEqual(security.parseCalls(source), {calls: [{name, args}], self: false, prevent: false}, source);
  }
});

test('modal background guards and return-false retain their meaning', () => {
  assert.deepEqual(security.parseCalls('if (event.target === this) closeCompareModal();'), {
    calls: [{name: 'closeCompareModal', args: []}], self: true, prevent: false,
  });
  assert.deepEqual(security.parseCalls("closeGlobalSearchModal(); jumpToModelCard('model-a'); return false;"), {
    calls: [
      {name: 'closeGlobalSearchModal', args: []},
      {name: 'jumpToModelCard', args: ['model-a']},
    ], self: false, prevent: true,
  });
});

test('quotes and Unicode escapes are literal values, not executable statements', () => {
  const result = security.parseCalls(String.raw`jumpToSection('literal\'quote; alert(1)'); setLanguageMode('\u007a\u0068')`);
  assert.equal(result.calls[0].args[0], "literal'quote; alert(1)");
  assert.equal(result.calls[1].args[0], 'zh');
});

test('literal numbers, booleans, null, and control references are parsed explicitly', () => {
  assert.deepEqual(security.parseCalls('selectPipelineStage(-1, .5, true, false, null, this, event, this.value)').calls[0].args,
    [-1, .5, true, false, null, {$ref: 'this'}, {$ref: 'event'}, {$ref: 'this.value'}]);
});

test('arbitrary JavaScript, property access, and nested execution are rejected', () => {
  const malicious = [
    'alert(1)', 'window.alert(1)', 'constructor.constructor("return globalThis")()',
    '__proto__.polluted(true)', 'scrollToTop.constructor("alert(1)")()',
    'scrollToTop(); alert(1)', 'scrollToTop.call(window)',
    'jumpToSection(alert(1))', 'jumpToSection(new String("x"))',
    'jumpToSection({toString: alert})', 'jumpToSection({...this})',
    'jumpToSection(this.ownerDocument)', "jumpToSection(this['value'])",
    'jumpToSection(event.currentTarget)', 'jumpToSection(globalThis)',
    'jumpToSection(this.value.constructor)', 'jumpToSection((0, alert)(1))',
    'jumpToSection(`template ${alert(1)}`)', 'jumpToSection("x" + "y")',
    'jumpToSection([].constructor)', 'jumpToSection([1, ...event])',
    'jumpToSection(/a/.exec("a"))', 'jumpToSection(1e300)',
    'jumpToSection(Infinity)', 'jumpToSection(NaN)',
    'if (alert(1)) scrollToTop()', 'if (event.target !== this) scrollToTop()',
    'if (event.target === this) { scrollToTop() }',
    'scrollToTop() && alert(1)', 'scrollToTop()\nalert(1)',
    'scrollToTop(); return false; alert(1)', 'return false;',
    String.raw`jumpToSection('\x27')`, String.raw`jumpToSection('\u00ZZ')`,
    'jumpToSection("raw\nnewline")', 'jumpToSection("unclosed)',
    'jumpToSection(trueevil)', 'jumpToSection(this.valueOf())',
  ];
  for (const source of malicious) assert.throws(() => security.parseCalls(source), undefined, source);
});

test('parser budgets reject excessive source, calls, nesting, and argument counts', () => {
  for (const source of [
    ' '.repeat(4097), 'scrollToTop();'.repeat(9),
    `jumpToSection(${Array(9).fill('0').join(',')})`,
    `loadPresetComparison([${Array(33).fill('0').join(',')}])`,
    'loadPresetComparison([[[[0]]]])', '', '   ',
  ]) assert.throws(() => security.parseCalls(source), undefined, source.slice(0, 80));
});

test('navigation URLs are canonical HTTPS or local development HTTP', () => {
  assert.equal(security.safeURL('/biomed-ai-atlas/'), 'https://happymie.github.io/biomed-ai-atlas/');
  assert.equal(security.safeURL('https://arxiv.org/abs/1234.56789'), 'https://arxiv.org/abs/1234.56789');
  assert.equal(security.safeURL('//github.com/happymie/happymie.github.io'), 'https://github.com/happymie/happymie.github.io');
  assert.equal(security.safeURL('#reading'), 'https://happymie.github.io/#reading');
  const original = globalThis.location;
  try {
    globalThis.location = {href: 'http://127.0.0.1:8000/biomed-ai-atlas/'};
    assert.equal(security.safeURL('assets/figure.svg', {asset: true}), 'http://127.0.0.1:8000/biomed-ai-atlas/assets/figure.svg');
    assert.equal(security.safeURL('http://elsewhere.example/'), '#');
  } finally {
    if (original === undefined) delete globalThis.location;
    else globalThis.location = original;
  }
});

test('executable schemes, embedded credentials, and control characters are rejected', () => {
  for (const value of [
    'javascript:alert(1)', ' JAVASCRIPT:alert(1)', 'vbscript:msgbox(1)',
    'data:text/html,<script>alert(1)</script>', 'blob:https://happymie.github.io/id',
    'file:///etc/passwd', 'http://elsewhere.example/',
    'https://happymie.github.io@elsewhere.example/', 'https://user:pass@example.org/',
    'java\tscript:alert(1)', 'https://example.org/\npath', 'https://example.org/\x00path',
    '', ' ', null, {},
  ]) assert.equal(security.safeURL(value), '#', String(value));
});

test('asset URLs cannot leave the current origin', () => {
  assert.equal(security.safeURL('/biomed-ai-atlas/assets/figure.svg', {asset: true}),
    'https://happymie.github.io/biomed-ai-atlas/assets/figure.svg');
  for (const value of [
    'https://elsewhere.example/image.svg', '//elsewhere.example/image.svg',
    'https://happymie.github.io.elsewhere.example/image.svg',
    'https://happymie.github.io:8443/image.svg',
  ]) assert.equal(security.safeURL(value, {asset: true}), '#', value);
});

test('HTML escaping preserves text without creating elements or attributes', () => {
  assert.equal(security.escapeHTML('<img src="x" onerror=\'attack()\'> & 医学'),
    '&lt;img src=&quot;x&quot; onerror=&#39;attack()&#39;&gt; &amp; 医学');
  assert.equal(security.escapeHTML(null), '');
});

test('invalid or oversized document fragments fail before DOM parsing', () => {
  assert.throws(() => security.fragment({html: 'not a string'}), /Invalid model document/);
  assert.throws(() => security.fragment('x'.repeat(600001)), /Invalid model document/);
});
