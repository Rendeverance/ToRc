// pane.js - the side pane of the chat view: new chat and load, what is loaded and how, then the saved chats.

function renderPane() {
  const pane = $('pane');
  pane.replaceChildren(paneActions(), modelCard(), el('h3', 'pane-h', 'Chats'), chatList());
  setModel();
}

function paneActions() {
  const acts = el('div', 'pane-acts');
  const fresh = el('button', 'btn primary', 'New chat');
  fresh.onclick = startNew;
  const load = el('button', 'btn', S.loaded ? 'Change model' : 'Load a model');
  load.onclick = openLoader;
  acts.append(fresh, load);
  return acts;
}

// The loaded model and the settings it runs with, the chat's use of the context, its primer, and compacting.
function modelCard() {
  const s = S.settings;
  const chat = S.chat;
  const card = el('div', 'pane-card');

  const head = el('div', 'pane-model');
  head.append(el('span', S.loaded ? 'dot good' : 'dot'), el('b', '', S.loaded || 'No model loaded'));
  if (S.loaded) {
    const off = el('button', 'unload', 'Unload');
    off.title = 'Stop the engine and free the memory';
    off.onclick = unload;
    head.append(off);
  }
  card.append(head);
  // nothing loads by itself when ToRc starts: the model last chosen is one press away
  if (!S.loaded && s.lastModel && !S.job) {
    const again = el('button', 'btn primary pane-load', `Load ${baseName(s.lastModel).replace(/\.torc$/i, '')}`);
    again.title = 'Load it with the settings on Load a model';
    again.onclick = loadNow;
    card.append(again);
  }
  // a setting the engine reads at start, changed since this model was loaded
  const changed = changedSinceLoad();
  if (changed.length) {
    const note = el('div', 'pane-pending');
    note.append(el('span', '', `Changed since loading: ${changed.join(', ')}`));
    const again = el('button', 'btn', 'Load again');
    again.title = 'Load the model again so the changes take effect';
    again.onclick = loadNow;
    note.append(again);
    card.append(note);
  }

  const fact = (name, value, title) => {
    const r = el('div', 'kv');
    r.append(el('span', '', name), el('b', 'num', value));
    if (title) r.title = title;
    card.append(r);
  };
  // what the loaded model runs with, which a change waiting for Load doesn't alter
  const run = (S.loaded && S.loadedWith) || s;
  const used = chat.used || 0;
  const percent = Math.min(100, Math.round((100 * used) / ctxSize()));
  fact('Context', `${num(used)} of ${num(ctxSize())}`);
  const meter = el('div', `meter ${contextLevel().level}`);
  const fill = el('i');
  fill.style.width = `${percent}%`;
  meter.append(fill);
  card.append(meter);
  fact('CPU threads', String(run.threads));
  fact('NPU', npuState(run));
  fact('GPU for long prompts', run.gpu ? 'on' : 'off');
  fact('Machines', S.asWorker ? 'two, this the second' : run.worker ? 'two' : 'one');
  if (S.loaded && !S.asWorker) {
    // the shell shares it only with a key
    const isShared = run.share && !!(run.apiKey || '').trim();
    const at = S.shareAt || [];
    const shared = at.length ? `shared, ${at[0].replace(/ \(.*\)$/, '')}` : 'shared, every network';
    const how = isShared
      ? `Other machines can use the engine’s API, with the API key from Settings, at this machine’s address on ` +
        (at.length ? `any of its networks:${NL}${at.join(NL)}${NL}` : 'any of its networks. ') +
        'Settings turns it off'
      : 'Only this machine can use the engine’s API. Settings can share it on your network';
    fact('API', isShared ? shared : 'this machine only', how);
    // the engine's status page, when TORC_STATUS was on at Load: its address, which opens it in the browser
    const env = run.env || {};
    if (env.TORC_STATUS === '1') {
      // as the engine reads it: a whole number from 1 to 65535, or the API's port
      const n = Number(String(env.TORC_STATUS_PORT || '').trim());
      const own = Number.isInteger(n) && n > 0 && n < 65536 ? n : 0;
      const at = own || run.port;
      // other machines reach it on its own port with TORC_STATUS_NET, or on the API's when the API is shared too
      const others = env.TORC_STATUS_NET === '1' && (own || isShared);
      const page = el('button', 'kv kv-btn');
      page.append(el('span', '', 'Status page'), el('b', 'num', `127.0.0.1:${at}`));
      page.title =
        'Opens the engine’s status page in your browser. ' +
        (!others
          ? 'Only this machine can open it.'
          : env.TORC_STATUS_KEY !== '1'
            ? `Other machines open it at this machine’s address on port ${at}.`
            : (run.apiKey || '').trim()
              ? `Other machines open it at this machine’s address on port ${at}, with the API key as the password.`
              : 'Other machines are refused it: it asks for the API key and none is set.');
      page.onclick = async () => {
        const r = await window.torc.openStatus(at);
        if (!r.ok) toast(`The status page did not open: ${r.error}`);
      };
      card.append(page);
    }
  }
  fact('Tools', s.tools === 'basic' ? 'basic' : 'ToolFunnel');

  const primer = el('div', 'kv kv-pick');
  const options = [['', 'None'], ...S.primers.map((p) => [p.id, p.name]), ['+', 'Manage primers…']];
  const picker = select(options, primerOf(chat) ? chat.primerId : '', (value) => {
    if (value === '+') {
      picker.value = primerOf(chat) ? chat.primerId : '';
      return openPrimers();
    }
    chat.primerId = value;
    if (chat.turns.length) keep();
  });
  picker.title = 'What the model is told before this chat starts';
  primer.append(el('span', '', 'Primer'), picker);
  card.append(primer);

  const compacting = el('div', 'pane-compact');
  const now = el('button', 'btn', 'Compact');
  now.title = 'Summarise the chat so far and carry on from the summary';
  now.onclick = () => compact(false);
  const auto = el('label', 'tick');
  const box = el('input');
  box.type = 'checkbox';
  box.checked = s.autoCompact;
  box.onchange = () => setSettings({ autoCompact: box.checked });
  auto.append(box, document.createTextNode('Auto compact'));
  auto.title = 'Compact by itself when the context is 90% full';
  compacting.append(now, auto);
  card.append(compacting);

  const target = el('div', 'kv kv-pick');
  const share = field(
    'number',
    s.compactShare,
    (v) => setSettings({ compactShare: Math.min(50, Math.max(5, Math.round(v))) }).then(renderPane),
    { min: 5, max: 50, step: 5, className: 'share-box' },
  );
  share.title = 'How long a compact summary aims to be, as a share of the context the chat holds';
  const unit = el('span', 'share');
  unit.append(share, document.createTextNode('%'));
  target.append(el('span', '', 'Compact target'), unit);
  card.append(target);
  return card;
}

function npuState(run) {
  if (S.info.npu === 'snapdragon') return (run || S.settings).npu ? 'on' : 'off';
  return S.info.npu === 'none' ? 'none' : 'not supported yet';
}

function chatList() {
  const list = el('div', 'chats');
  const current = S.chat;
  // a chat that has turns but is not saved yet still gets its row
  const unsaved = current.turns.length && !S.chats.some((c) => c.id === current.id);
  const all = unsaved ? [{ id: current.id, title: current.title, updated: current.updated }, ...S.chats] : S.chats;
  if (!all.length) {
    const none = S.settings.saveChats ? 'Your chats will be listed here.' : 'Saving chats is off in Settings.';
    list.append(el('div', 'muted pane-none', none));
  }
  for (const c of all) list.append(chatRow(c));
  return list;
}

function chatRow(c) {
  const r = el('div', c.id === S.chat.id ? 'chat-row on' : 'chat-row');
  const open = el('button', 'chat-pick');
  const day = new Date(c.updated).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  open.append(el('b', '', c.title || 'New chat'), el('span', '', day));
  open.onclick = () => openChat(c.id);
  const remove = confirmButton('chat-del', '×', 'Delete?', async () => {
    if (S.busy && c.id === S.chat.id) return toast('Wait for the reply to finish, or press Stop');
    await window.torc.chatDel(c.id);
    S.chats = S.chats.filter((k) => k.id !== c.id);
    if (c.id === S.chat.id) S.chat = newChat();
    render();
  });
  remove.title = 'Delete this chat';
  r.append(open, remove);
  return r;
}

// The model shown in the top bar, which is where it lives in the terminal view.
function setModel() {
  $('modelName').textContent = S.loaded || 'No model loaded';
  $('dot').className = S.loaded ? 'dot good' : 'dot';
  const waiting = changedSinceLoad().length ? ' · changes wait for Load' : '';
  $('modelNote').textContent = S.loaded ? `ctx ${num(S.chat.used || 0)} of ${num(ctxSize())}${waiting}` : '';
  $('btnLoad').textContent = S.loaded ? 'Change model' : 'Load a model';
}

function startNew() {
  if (S.busy) return toast('Wait for the reply to finish, or press Stop');
  S.chat = newChat();
  $('speeds').textContent = '';
  render();
  $('input').focus();
}

async function openChat(id) {
  if (id === S.chat.id) return;
  if (S.busy) return toast('Wait for the reply to finish, or press Stop');
  const chat = await window.torc.chatGet(id);
  if (!chat) return toast('That chat could not be read');
  S.chat = Object.assign(newChat(), chat);
  $('speeds').textContent = '';
  render();
}

async function unload() {
  if (S.busy) return toast('Wait for the reply to finish, or press Stop');
  if (S.job) return toast(WAIT_FOR_JOB);
  setStatus('Unloading', true);
  await window.torc.unload();
  S.loaded = null;
  S.asWorker = false;
  setStatus('No model loaded', false);
  render();
}
