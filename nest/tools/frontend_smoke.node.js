// METARGON NEST — frontend DOM smoke test (node, no deps).
//
// Loads the real inline script from nest/static/index.html and executes it
// inside a minimal DOM stub backed by snapshot data shaped exactly like the
// NEST backend API. Asserts the runtime behaviours that previously broke:
//   1. initial welcome message appended to the conversation panel
//   2. world canvas receives drawing calls (renders)
//   3. memory panel HTML contains verified/generated statistics
//   4. typing + Enter (or Send) posts a user message and receives a reply
//   5. control buttons are wired (clicking "start" calls /api/control)
'use strict';

const fs = require('fs');
const path = require('path');

const HTML_PATH = path.join(__dirname, '..', 'static', 'index.html');
const html = fs.readFileSync(HTML_PATH, 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error('no inline script'); process.exit(1); }
const scriptSrc = m[1];

// ---- snapshot data mirroring the backend API shapes ----
const SNAPSHOT = {
  status: 'idle',
  episode_index: 0,
  episode_successes: 0,
  global_steps: 0,
  steps_this_episode: 0,
  challenge: { pool_tag: 'train', difficulty: 'easy',
               focus_door: 'D1', seed: 0, description: 'open door D1' },
  world: {
    present: true,
    pos: [0, 4],
    objs: { A: [2, 4], B: [4, 3] },
    plates: { P1: 'none', P2: 'none' },
    switches: { S1: false, S2: false },
    doors: { D1: false, D2: false },
    goal: [6, 0],
    steps: 0,
    max_steps: 80,
    finished: false,
    last_events: [],
    last_action: null,
    last_valid: null,
    door_cell: { D1: [3, 2], D2: [5, 1] },
    switch_cell: { S1: [0, 1], S2: [4, 4] },
    plate_cell: { P1: [3, 4], P2: [1, 1] },
    walls: [[0, 2], [1, 2], [2, 2], [4, 2], [5, 2], [6, 2], [5, 0]],
    topology: 'TP1',
  },
  memory: {
    hypotheses: [{
      id: 1, door: 'D1', antecedent_text: 'empty(P1)',
      status: 'verified', sup: 10, con: 0, confidence: 0.9,
      from_earlier_episodes: false,
    }],
    stats: { verified: 1, contradicted: 0, hypothesis: 1, predicted: 0 },
    generated: 12,
    step: 40,
    evidence_bank_size: 0,
    bank_retrievals: 0,
    current_epoch: 0,
    verified_hypotheses: 1,
  },
  trace: { memory_actions: 2, memory_correct: 1, memory_incorrect: 0,
           hypothesis_retrieved: 4, cross_epoch_transfer_retrieved: 0 },
  notice: '',
  worker_error: null,
  messages_tail: [],
};

const REPLY = {
  command: { intent: 'hello', text: 'Hello, Metargon!' },
  reply: "Hello! I'm Metargon, a little learning agent. I explore the Academy world, learn door rules from evidence, and remember what I discover. You can ask me what I remember, tell me to open a door, or ask for help.",
  snapshot: SNAPSHOT,
};

// ---- minimal DOM stub ----
let ctxCalls = 0;
const ctx2d = new Proxy({}, {
  get(t, p) {
    if (p === 'canvas') return { width: 420, height: 300 };
    if (!(p in t)) {
      t[p] = (...a) => { ctxCalls += 1; };
    }
    return t[p];
  },
});

class FakeEl {
  constructor(id) {
    this.id = id;
    this.textContent = '';
    this.innerHTML = '';
    this.className = '';
    this.value = '';
    this.dataset = {};
    this.scrollTop = 0;
    this.children = [];
    this._handlers = {};
    this.onclick = null;
    this.style = {};
    this.classList = {
      _set: new Set(),
      add(...names) { names.forEach(n => this._set.add(n)); },
      remove(...names) { names.forEach(n => this._set.delete(n)); },
      contains(n) { return this._set.has(n); },
      toggle(n, force) {
        const on = force !== undefined ? force : !this._set.has(n);
        if (on) this._set.add(n); else this._set.delete(n);
        return on;
      },
    };
  }
  appendChild(el) { this.children.push(el); return el; }
  addEventListener(type, fn) { (this._handlers[type] = this._handlers[type] || []).push(fn); }
  dispatch(type, ev) { (this._handlers[type] || []).forEach(fn => fn(ev)); }
  getContext() { return ctx2d; }
}

const els = {};
['status-pill', 'chat', 'input', 'send', 'cv', 'world-meta', 'memory',
 'api-badge', 'notice-bar', 'challenge']
  .forEach(id => { els[id] = new FakeEl(id); });

const controlButtons = ['start', 'pause', 'continue', 'step', 'stop',
                        'reset', 'save', 'restore', 'fresh']
  .map(op => { const b = new FakeEl('btn-' + op); b.dataset.op = op; return b; });

const chips = ['Hello, Metargon!', 'What do you remember?', 'Open door D1']
  .map(text => { const c = new FakeEl('chip'); c.dataset.text = text; return c; });

global.document = {
  getElementById(id) { return els[id] || null; },
  querySelectorAll(sel) {
    if (sel === '#controls button') return controlButtons;
    if (sel === '#chips .chip') return chips;
    return [];
  },
  createElement(tag) {
    const el = new FakeEl('created-' + tag);
    if (tag === 'canvas') el.getContext = () => ctx2d;
    return el;
  },
};

global.fetch = async (url, opts) => {
  const body = opts && opts.body ? JSON.parse(opts.body) : {};
  let payload = SNAPSHOT;
  if (url === '/api/command') payload = REPLY;
  else if (url === '/api/control') payload = SNAPSHOT;
  else if (url === '/api/config') payload = { api_mode: 'none' };
  return { json: async () => payload };
};

global.setInterval = () => 0;
// keep real timers so async tick()/fetch can settle in the test
global.setTimeout = global.setTimeout || setTimeout;

// surface async errors for diagnosis
process.on('unhandledRejection', (err) => {
  console.error('UNHANDLED REJECTION:', err && err.stack || err);
  process.exitCode = 1;
});

// ---- execute the page script ----
try {
  (0, eval)(scriptSrc);
} catch (e) {
  console.error('SCRIPT EXECUTION THREW:', e.stack || e);
  process.exit(1);
}

// allow the async tick() -> fetch -> render pipeline to complete
(async () => {
  // call render synchronously to surface any render error
  try {
    globalThis.__nest.render(SNAPSHOT);
  } catch (e) {
    console.error('DIRECT RENDER THREW:', e.stack || e);
    process.exitCode = 1;
  }
  try {
    await globalThis.__nest.tick();
  } catch (e) {
    console.error('TICK THREW:', e.stack || e);
    process.exitCode = 1;
  }
  await new Promise(r => setTimeout(r, 20));
  runAssertions();
})();

function runAssertions() {
const chatTexts = els.chat.children.map(c => c.textContent || '');
const ok = (cond, msg) => {
  if (!cond) { console.error('ASSERT FAILED: ' + msg); process.exitCode = 1; }
  console.log((cond ? 'PASS ' : 'FAIL ') + msg);
};

ok(chatTexts.some(t => t.includes("Hello! I'm Metargon")),
   'initial welcome message appears in conversation');
ok(ctxCalls > 0, 'world canvas received drawing calls (' + ctxCalls + ')');
ok(els['world-meta'].innerHTML.includes('steps 0/80'),
   'world meta shows real steps summary');
ok(els.memory.innerHTML.includes('verified') && els.memory.innerHTML.includes('generated'),
   'memory panel shows verified/generated statistics');
ok(els.memory.innerHTML.includes('empty(P1)'),
   'memory panel lists the hypothesis');
ok(els.challenge.innerHTML.includes('Challenge'),
   'challenge card renders');
ok(els['api-badge'].textContent.includes('API:'),
   'api badge renders');

// send via Enter
els.input.value = 'Hello, Metargon!';
els.input.dispatch('keydown', { key: 'Enter' });
const afterEnter = els.chat.children.map(c => c.textContent || '');
ok(afterEnter.some(t => t.includes('Hello, Metargon!')), 'user message appears');
ok(afterEnter.some(t => t.includes("I'm Metargon")), 'bot reply appears after send');

// chip click sends a message
const chip0 = chips[0];
ok(typeof chip0.onclick === 'function', 'chip is wired');
chip0.onclick();
const afterChip = els.chat.children.map(c => c.textContent || '');
ok(afterChip.some(t => t.includes('Hello, Metargon!')), 'chip sends user message');

// control button wiring
const startBtn = controlButtons.find(b => b.dataset.op === 'start');
ok(typeof startBtn.onclick === 'function', 'start button is wired');
startBtn.onclick();

console.log('ctxCalls total:', ctxCalls);
process.exit(process.exitCode || 0);
}
