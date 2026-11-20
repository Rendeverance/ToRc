// ui.js - what every screen shares: the page's state, and the small pieces the screens are built from.

const $ = (id) => document.getElementById(id);
const NL = String.fromCharCode(10);
const num = (n) => Number(n).toLocaleString('en-GB');
const baseName = (file) => String(file).split(/[\\/]/).pop();

// A new element with a class and, when given, its text.
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const S = {
  info: null, // the app's version, and whose NPU this machine has
  settings: null,
  switches: [],
  basicTools: [],
  loaded: null, // the loaded model's name
  loadedWith: null, // the settings it was loaded with: a change after that waits for the next Load
  shareAt: [], // where other machines reach the loaded model's server, when Settings shares it
  busy: false, // a reply is being written, or a compact is running
  job: false, // the tuner or the benchmark is running
  view: 'chat',
  chat: null,
  chats: [], // what the list shows: { id, title, updated, model }
  primers: [],
  servers: [], // the tool servers' states
  toolfunnel: '', // the installed version
  toolfunnelLatest: '', // the version npm has, once asked
  files: [], // attached and not yet sent: { name, text }
  onJobLine: null, // the open job sheet's sink for what the job prints
  log: [], // what the engine and the tool servers have printed
};

function newChat() {
  const now = Date.now();
  return {
    // the time to the millisecond, so two chats started in the same second don't share a file
    id: new Date(now).toISOString().slice(0, 23).replace(/[T:.]/g, '-'),
    title: '',
    created: now,
    updated: now,
    model: '',
    primerId: S.settings ? S.settings.primer : '',
    turns: [],
    compact: null, // { upto, note }: the turns before `upto` are stood for by `note`
    used: 0, // tokens of context the chat holds, as the server last counted them
  };
}

// Changes settings: the shell merges the change into what is saved and hands back the whole.
async function setSettings(patch) {
  // one save at a time, in the order they were asked for: two at once could finish the other way round, and the
  // older change would be the one kept
  const turn = (S.saving || Promise.resolve()).then(() => window.torc.settingsSet(patch));
  S.saving = turn.catch(() => {});
  const mine = (S.saves = (S.saves || 0) + 1);
  const saved = await turn;
  // a save that didn't reach the disk: the change still holds until the app closes, and the person is told
  if (saved && saved.saveError) {
    toast(`The settings couldn't be saved, so this change lasts until the app closes (${saved.saveError})`);
    delete saved.saveError;
  }
  // only the last save asked for gives the settings shown: an earlier one's answer lacks the changes made since (a
  // second press of the Thinking button, say), and the last one's has them all
  if (mine === S.saves) S.settings = saved;
  showPending();
  return saved;
}

// The settings the engine reads when it starts, with the name a person knows each by. A change to one is saved at
// once but only reaches the engine at the next Load; the rest (sampling, the longest reply, tools) apply to the next
// reply by themselves.
const LOAD_TIME = [
  ['lastModel', 'the model file'],
  ['ctx', 'context'],
  ['threads', 'CPU threads'],
  ['npu', 'NPU'],
  ['gpu', 'GPU'],
  ['worker', 'second machine'],
  ['asWorker', 'second machine'],
  ['workerBind', 'second machine'],
  ['port', 'port'],
  ['binDir', 'engine folder'],
  ['share', 'sharing on your network'],
  ['apiKey', 'API key'],
  ['saveChats', 'saving chats'],
  ['env', 'advanced settings'],
];

// What has changed since the loaded model was loaded, by name; nothing when no model is loaded.
function changedSinceLoad() {
  if (!S.loaded || !S.loadedWith) return [];
  const names = [];
  for (const [key, name] of LOAD_TIME) {
    const now = JSON.stringify(S.settings[key] ?? null);
    if (now !== JSON.stringify(S.loadedWith[key] ?? null) && !names.includes(name)) names.push(name);
  }
  return names;
}

// Says so wherever it shows: the side pane, the top bar, and the note at the top of Load a model.
function showPending() {
  if (S.chat) renderPane();
  const note = $('pendingNote');
  if (!note) return;
  const changed = changedSinceLoad();
  note.hidden = !changed.length;
  const what = changed.join(', ');
  note.firstChild.textContent = `Changed since ${S.loaded} was loaded: ${what}. Load it again to use them.`;
}

function toast(text) {
  const t = el('div', 'toast', text);
  document.body.append(t);
  setTimeout(() => t.remove(), 3200);
}

// The context the running server was started with; the Load screen's box only counts from the next load
function ctxSize() {
  return (S.loaded && S.loadedCtx) || S.settings.ctx;
}

function setStatus(text, busy) {
  $('status').textContent = text;
  $('bar').classList.toggle('busy', !!busy);
}

// Opens a sheet over the page and returns its body to fill.
function sheet(title, lead, wide) {
  // while the tuner or the benchmark runs its sheet stays up: another sheet gets a body that is never shown
  if (S.job) {
    toast(WAIT_FOR_JOB);
    return el('div');
  }
  const wrap = el('div', 'sheet');
  const card = el('div', wide ? 'sheet-card wide' : 'sheet-card');
  const body = el('div', 'sheet-body');
  const close = el('button', 'sheet-close', '×');
  close.onclick = closeSheet;
  body.append(el('h2', 'title', title));
  if (lead) body.append(el('p', 'lead', lead));
  card.append(close, body);
  wrap.append(card);
  wrap.onclick = (event) => {
    if (event.target === wrap) closeSheet();
  };
  $('sheet').replaceChildren(wrap);
  $('sheet').dataset.name = title;
  return body;
}

function closeSheet() {
  // a running job's sheet holds its Stop and its output
  if (S.job) return;
  $('sheet').replaceChildren();
  delete $('sheet').dataset.name;
}

const sheetIs = (title) => $('sheet').dataset.name === title;

function toggle(on, change) {
  const label = el('label', 'switch');
  const input = el('input');
  input.type = 'checkbox';
  input.checked = !!on;
  input.onchange = () => change(input.checked);
  label.append(input, el('i'));
  return label;
}

// A text or number box. A number box that is emptied or holds something else keeps the value it had.
function field(type, value, change, extra) {
  const input = el('input');
  input.type = type;
  input.value = value ?? '';
  Object.assign(input, extra || {});
  input.onchange = () => {
    if (type !== 'number') return change(input.value);
    const n = Number(input.value);
    if (input.value.trim() === '' || !Number.isFinite(n)) {
      input.value = value ?? '';
      return;
    }
    change(n);
  };
  return input;
}

function select(options, value, change) {
  const s = el('select');
  for (const [v, label] of options) {
    const o = el('option', '', label);
    o.value = v;
    s.append(o);
  }
  s.value = value;
  s.onchange = () => change(s.value);
  return s;
}

// A settings row: a name, one line about it, an `i` that opens the full explanation under it, and a control.
// Returns the row, and the explanation's row after it when there is one.
function row(name, hint, control, explanation) {
  const r = el('div', 'row');
  const label = el('div', 'lb');
  label.append(el('b', '', name));
  if (hint) label.append(el('span', '', hint));
  r.append(label);
  const out = [r];
  if (explanation) {
    const info = el('button', 'info', 'i');
    info.title = 'What this does';
    const more = el('div', 'row explain');
    more.hidden = true;
    more.append(el('b', '', `${name}. `), document.createTextNode(explanation));
    info.onclick = () => {
      more.hidden = !more.hidden;
      info.classList.toggle('on', !more.hidden);
    };
    r.append(info);
    out.push(more);
  }
  if (control) r.append(control);
  return out;
}

// A titled group of rows.
function group(title, rows) {
  const f = document.createDocumentFragment();
  f.append(el('h3', '', title));
  const g = el('div', 'rows');
  for (const r of rows.flat()) g.append(r);
  f.append(g);
  return f;
}

// A button that asks once more before it acts: the first press changes its label, the second does it.
function confirmButton(cls, label, question, act) {
  const b = el('button', cls, label);
  b.onclick = () => {
    if (b.textContent === label) {
      b.textContent = question;
      b.classList.add('sure');
      setTimeout(() => {
        b.textContent = label;
        b.classList.remove('sure');
      }, 3000);
      return;
    }
    act();
  };
  return b;
}

// A fenced code block with Copy and Save buttons; lang is the word after the opening fence (it names the file type).
function codeBlock(code, lang) {
  const wrap = el('div', 'codewrap');
  const acts = el('div', 'code-acts');
  const copy = el('button', 'btn', 'Copy');
  copy.onclick = async () => {
    let done = false;
    try {
      await navigator.clipboard.writeText(code);
      done = true;
    } catch (_) {
      // the older way, for a webview that refuses the clipboard API
      try {
        const t = el('textarea');
        t.value = code;
        document.body.append(t);
        t.select();
        done = document.execCommand('copy');
        t.remove();
      } catch (_) {}
    }
    toast(done ? 'Copied' : 'Could not copy: select the code and press Ctrl+C');
  };
  const save = el('button', 'btn', 'Save');
  save.onclick = async () => {
    const named = /^[a-z0-9+#-]{1,12}$/i.test(lang) && !/^(text|plain|plaintext)$/i.test(lang);
    const r = await window.torc.saveText(code, `code.${named ? lang : 'txt'}`);
    if (r.ok) toast(`Saved to ${r.path}`);
    else if (!r.cancelled) toast(`Not saved: ${r.error}`);
  };
  acts.append(copy, save);
  wrap.append(acts, el('pre', 'code', code));
  return wrap;
}

// A reply's text with the little Markdown models write most: fenced code, `code` and **bold**. Built as nodes,
// never as HTML, so nothing a model writes can run.
function rich(text) {
  const out = el('div', 'body');
  String(text)
    .split('```')
    .forEach((part, i) => {
      if (i % 2) {
        const firstLine = part.indexOf(NL);
        const code = firstLine >= 0 ? part.slice(firstLine + 1) : part;
        const lang = firstLine > 0 ? part.slice(0, firstLine).trim() : '';
        out.append(codeBlock(code.replace(/\n$/, ''), lang));
        return;
      }
      for (const piece of part.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/)) {
        if (piece.length > 4 && piece.startsWith('**') && piece.endsWith('**'))
          out.append(el('b', '', piece.slice(2, -2)));
        else if (piece.length > 2 && piece.startsWith('`') && piece.endsWith('`'))
          out.append(el('code', '', piece.slice(1, -1)));
        else if (piece) out.append(document.createTextNode(piece));
      }
    });
  return out;
}
