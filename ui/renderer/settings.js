// settings.js - the Settings sheet: chats, appearance, where the engine is, the benchmark, settings to share and the
// log.

function openSettings() {
  const s = S.settings;
  const body = sheet('Settings', 'These apply to every model.');

  // the folder chats go to: the one chosen, or ToRc's own (chats beside the app when it runs from its folder)
  const folder = el('button', 'btn', s.chatDir || S.info.chatsDir || 'Choose a folder');
  folder.onclick = async () => {
    const [picked] = await window.torc.pick('folder', S.settings.chatDir || S.info.chatsDir);
    if (!picked) return;
    await setSettings({ chatDir: picked });
    folder.textContent = picked;
    S.chats = await window.torc.chats();
    renderPane();
  };
  const saveChats = toggle(s.saveChats, (v) => setSettings({ saveChats: v }).then(renderPane));
  const autoCompact = toggle(s.autoCompact, (v) => setSettings({ autoCompact: v }).then(renderPane));
  const modelSettings = el('button', 'btn', 'Go to Load a model');
  modelSettings.onclick = openLoader;
  const advancedSettings = el('button', 'btn', 'Open');
  advancedSettings.onclick = () => openAdvanced();
  body.append(
    group('Model', [
      row(
        'Context, threads, NPU and GPU',
        'On the Load a model screen, with the model you load',
        modelSettings,
        'The context, the CPU threads, the NPU and GPU switches and Auto-tune belong to the model you load, so ' +
          'they live on the Load a model screen. This button only takes you there. A change there is saved at ' +
          'once and takes effect when you press Load.',
      ),
      row(
        'Advanced settings',
        'The engine’s switches, used from the next Load',
        advancedSettings,
        'The same sheet as Advanced… under Load a model. A change takes effect when you press Load.',
      ),
    ]),
    group('Chats', [
      row(
        'Save chats',
        'Keep conversations on this machine',
        saveChats,
        'When on, each chat is written to the folder below so you can come back to it from the list on the left. ' +
          'When off, a chat is gone when you close ToRc, and the engine keeps nothing of it on disk either (from ' +
          'the next time a model is loaded). The chats are plain JSON files and stay on this machine.',
      ),
      row('Folder', 'Where saved chats go. Click to choose another', folder),
      row(
        'Summary at most',
        'The longest a compact summary is asked to be, in words',
        field('number', s.compactMaxWords, (v) => setSettings({ compactMaxWords: Math.max(150, Math.round(v)) }), {
          min: 150,
          step: 250,
        }),
        'A compact asks for a summary of the share of the context set as Compact target in the left pane (15% to ' +
        'start). This caps it, since a long summary takes a while to write and leaves less room for the chat ' +
        'that follows. Both are aims the model is given, and it can come in shorter.',
      ),
      row(
        'Auto compact',
        'Compact a chat by itself when its context is 90% full',
        autoCompact,
        'A chat cannot hold more than the model’s context. Compacting has the model write a summary of the chat ' +
        'so far and carry on from that, which frees the context. The whole chat stays on screen and on disk. You ' +
        'can also compact by hand with the Compact button, or /compact in the terminal view.',
      ),
    ]),
  );

  const firstView = select(
    [
      ['chat', 'Chat'],
      ['terminal', 'Terminal'],
    ],
    s.firstView,
    (v) => setSettings({ firstView: v }),
  );
  body.append(
    group('Appearance', [
      row(
        'Open in',
        'The view ToRc starts in',
        firstView,
        'Chat is the plain conversation view. Terminal shows the same conversation as a dark, monospaced stream. ' +
        'You can switch at any time from the top bar.',
      ),
    ]),
  );

  const binDir = field('text', s.binDir, (v) => setSettings({ binDir: v.trim() }), { className: 'wide-box' });
  const browse = el('button', 'btn', 'Browse');
  browse.onclick = async () => {
    const [picked] = await window.torc.pick('folder', S.settings.binDir);
    if (!picked) return;
    await setSettings({ binDir: picked });
    binDir.value = picked;
  };
  const engineDir = el('span', 'ctl');
  engineDir.append(binDir, browse);
  const port = field('number', s.port, (v) => setSettings({ port: Math.round(v) }), { min: 1, max: 65535 });
  // sharing always comes with a key: the first time it is turned on, one is made
  const share = toggle(s.share, async (v) => {
    await setSettings(v && !S.settings.apiKey ? { share: true, apiKey: newApiKey() } : { share: v });
    keyBox.value = S.settings.apiKey || '';
    showKey(v);
  });
  const keyBox = field('text', s.apiKey || '', () => {}, { readOnly: true, className: 'wide-box key-box' });
  const copy = el('button', 'btn', 'Copy');
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(S.settings.apiKey);
      toast('The key is copied');
    } catch (_) {
      keyBox.select();
      toast('The key is selected: press Ctrl+C to copy it');
    }
  };
  const fresh = confirmButton('btn', 'New key', 'Replace the key?', async () => {
    await setSettings({ apiKey: newApiKey() });
    keyBox.value = S.settings.apiKey;
    toast('New key made. Machines using the old one need this one after the next Load');
  });
  const keyCtl = el('span', 'ctl');
  keyCtl.append(keyBox, copy, fresh);
  const keyRow = row(
    'API key',
    'Other machines need it, this one doesn’t',
    keyCtl,
    'Another machine’s requests must carry this key, the way any OpenAI or Anthropic client sends one: paste it ' +
      'where your app or agent asks for an API key (it goes as "Authorization: Bearer <key>" or "x-api-key: ' +
      '<key>"). Requests from this machine, ToRc itself included, need no key. New key replaces it from the next ' +
      'Load, and anything using the old one is refused after that. Keep it like a password.',
  );
  const showKey = (on) => {
    keyRow[0].hidden = !on;
    if (!on) keyRow[1].hidden = true;
  };
  const log = el('button', 'btn', 'Show the log');
  log.onclick = () => showLog();
  body.append(
    group('Engine', [
      row(
        'Engine folder',
        'Where torc-server is',
        engineDir,
        'The folder holding torc-server. After "cargo build --release" that is target/release in the ToRc folder. ' +
          'Used from the next Load.',
      ),
      row(
        'Port',
        'The port the engine’s API listens on',
        port,
        'ToRc, and any app or agent you point at the engine, reaches it on this port: http://127.0.0.1:8080/v1 ' +
          'by default. Change it if something else is using 8080. Used from the next Load.',
      ),
      row(
        'Share on my network',
        'Off: this machine only. On: other machines can use the API too',
        share,
        'Off, the engine answers this machine only (127.0.0.1). On, it listens on every network this machine is ' +
          'on (0.0.0.0), so other machines can use its OpenAI and Anthropic APIs at this machine’s address and ' +
          'the port above, with the API key below; the side pane shows the address once the model is loaded. ' +
          'The key is all that guards it and the traffic is not encrypted, so only turn it on for a network you ' +
          'trust. Windows may ask once whether to let torc-server through its firewall. Used from the next Load.',
      ),
      keyRow,
      row('Log', 'What the engine and the tool servers have printed', log),
    ]),
  );
  showKey(!!s.share);

  const bench = el('button', 'btn', 'Run the benchmark');
  bench.onclick = () => openJob('bench');
  body.append(
    group('Benchmark', [
      row(
        'Measure this machine',
        'The standard prompts, saved as a CSV you can send in',
        bench,
        'Sends the standard prompts that fit the loaded context (a short one, then 1k, 4k and 8k tokens, 16k ' +
          'from a context of 18,000 and 32k from 34,816) and records the engine’s own read and write speeds. The ' +
          'file holds your hardware’s names and the speeds, no host name and no paths. Each row also keeps the ' +
          'settings the engine ran with, so a result can be read back in below.',
      ),
    ]),
  );

  const saveSet = el('button', 'btn', 'Save…');
  saveSet.onclick = saveProfile;
  const readSet = el('button', 'btn', 'Choose a file…');
  readSet.onclick = readProfile;
  body.append(
    group('Share settings', [
      row(
        'Save these settings',
        'What the loaded model runs with on this machine, as a file',
        saveSet,
        'Writes the settings the engine is using now (the prompt block, the work split, the threads, the NPU’s ' +
          'hot experts and the switches it was started with), with this machine’s CPU and memory, as a small ' +
          'JSON file. Someone with the same model file can read it in.',
      ),
      row(
        'Read settings in',
        'From a settings file or a benchmark CSV, for the loaded model',
        readSet,
        'Reads a saved settings file, or the settings a benchmark CSV kept, into the loaded model on this machine, ' +
          'then loads the model again with them. They stay until you run Auto-tune, which measures this machine’s ' +
          'own (the NPU’s hot experts stay as read in). Settings measured on a different model file are refused.',
      ),
    ]),
  );
}

// A new API key: 24 random bytes, as letters and digits another app can take in a paste.
function newApiKey() {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  const text = btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return `torc-${text}`;
}

// The loaded model's settings, saved as a file to keep or pass on.
async function saveProfile() {
  if (!S.loaded || S.asWorker) return toast('Load a model first');
  if (S.busy) return toast(WAIT_FOR_REPLY);
  if (S.job) return toast(WAIT_FOR_JOB);
  const r = await window.torc.engineSettings();
  if (!r.ok) return toast(r.error);
  const name = `${S.loaded}-settings.json`.replace(/[\\/:*?"<>|]/g, '_');
  const w = await window.torc.saveText(JSON.stringify(r.profile, null, 2) + '\n', name);
  if (w && w.ok) toast(`Saved to ${w.path}`);
  else if (w && !w.cancelled) toast(w.error);
}

// A CSV's rows as lists of cells, quoted cells and all.
function csvRows(text) {
  const rows = [[]];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') {
      rows[rows.length - 1].push(cell);
      cell = '';
    } else if (c === '\n') {
      rows[rows.length - 1].push(cell.replace(/\r$/, ''));
      cell = '';
      rows.push([]);
    } else cell += c;
  }
  if (cell) rows[rows.length - 1].push(cell);
  return rows.filter((r) => r.length > 1);
}

// The settings in a settings file, or in a benchmark CSV: the last row measured on this model file (`key`, the loaded
// model's), else the last row that kept settings at all, with the switches that row was run with.
function profileFrom(text, key) {
  const t = text.trim();
  if (t.startsWith('{')) {
    const p = JSON.parse(t);
    if (!p.torc_settings) throw new Error('This is not a ToRc settings file');
    return p;
  }
  const rows = csvRows(t);
  const head = rows[0] || [];
  const at = head.indexOf('torc_settings');
  const sw = head.indexOf('torc_switches');
  if (at < 0) throw new Error('This file keeps no settings (a benchmark from before ToRc 0.1.4, or not a benchmark)');
  const kept = rows.slice(1).filter((r) => r[at]);
  if (!kept.length) throw new Error('No row in this file kept its settings');
  const parsed = kept.map((r) => ({ p: JSON.parse(r[at]), sw: sw >= 0 ? r[sw] || '' : '' }));
  const pick = [...parsed].reverse().find((x) => x.p.model_key === key) || parsed[parsed.length - 1];
  pick.p.switches = {};
  for (const kv of pick.sw.split(' ')) {
    const e = kv.indexOf('=');
    if (e > 0) pick.p.switches[kv.slice(0, e)] = kv.slice(e + 1);
  }
  return pick.p;
}

// Switches a settings file brings that this machine does not take from it: what the tune file now holds (read in
// through it, so Auto-tune can replace them later) and the app's own.
const NOT_READ_IN = [
  'TORC_BLOCK', 'TORC_NPU_HOT', 'TORC_NPU_GUIDED', 'TORC_PIN', 'TORC_DECODE_THREADS', 'TORC_MARKS',
  'TORC_DEVTOOLS_PORT',
];

// The switches a settings file may set on this machine: how the engine runs (kernels, the cache's reads, warming the
// file, drafts, the two-machine split), none that changes the replies (apart from the NPU and GPU toggles, read in by
// readProfile, which aren't bit-exact), makes a model that fits stream from disk, or names a folder, a file, a library
// or an address. Each takes a plain number or one word, so a file passed on by someone else can change how fast the
// engine runs but can never point it at anything on the disk or the network.
const READ_IN = [
  'TORC_Q4X4', 'TORC_ATTN_TILE', 'TORC_WARM', 'TORC_MTP', 'TORC_BLOCK_LONG', 'TORC_CHECKPOINT_MIN',
  'TORC_X86_PF_BLOCKS', 'TORC_MIX_MM', 'TORC_IO_THREADS', 'TORC_IO_DEPTH', 'TORC_IO_PIECE_KB', 'TORC_IO_SORT',
  'TORC_OWN_LARGE', 'TORC_OWN_PREFETCH', 'TORC_LARGE_PAGES', 'TORC_EXPERTS_LARGE', 'TORC_KLAY_LARGE', 'TORC_GPU_CHECK',
  'TORC_PIPE_SPLIT',
];

// Whether a switch from someone else's file may be set here (READ_IN, with a plain value).
function readableSwitch(k, v) {
  return READ_IN.includes(k) && typeof v === 'string' && /^([0-9]{1,6}(\.[0-9]{1,3})?|on|off|auto|pure|wait)$/.test(v);
}

// The layout a profile was measured in, in words.
function layoutName(l) {
  const [m, t] = String(l || '').split('-');
  return `${m === 'two' ? 'two machines' : 'one machine'}, ${t === 'npu' ? 'NPU on' : 'NPU off'}`;
}

// A settings file (or a benchmark CSV) read into the loaded model: the engine writes its values for its next start,
// the switches it may take go into this model's settings, and the model loads again with them.
async function readProfile() {
  if (!S.loaded || S.asWorker) return toast('Load the model the settings are for first');
  if (S.busy) return toast(WAIT_FOR_REPLY);
  if (S.job) return toast(WAIT_FOR_JOB);
  const [file] = await window.torc.pick('text');
  if (!file) return;
  const f = await window.torc.readText(file);
  if (!f.ok) return toast(f.error);
  const mine = await window.torc.engineSettings();
  let p;
  try {
    p = profileFrom(f.text, mine.ok ? mine.profile.model_key : '');
  } catch (e) {
    return toast(e.message);
  }
  const r = await window.torc.engineSettingsSet(p);
  if (!r.ok) return toast(r.error);
  const env = { ...S.settings.env };
  const patch = {};
  const taken = [];
  const left = [];
  for (const [k, v] of Object.entries(p.switches || {})) {
    if (k === 'TORC_NPU' && v === '1') {
      if (S.info.npu === 'snapdragon') patch.npu = true;
    } else if (k === 'TORC_GPU_ATTN' && v === '1') patch.gpu = true;
    else if (readableSwitch(k, v)) {
      env[k] = v;
      taken.push(`${k}=${v}`);
    } else if (!NOT_READ_IN.includes(k)) left.push(k);
  }
  patch.env = env;
  await setSettings(patch);
  const also = taken.length ? ` Switches set: ${taken.join(', ')}.` : '';
  const skipped = left.length ? ` Not taken: ${left.join(', ')}.` : '';
  const where = r.layout && r.layout !== r.running ? ` They apply when you run with ${layoutName(r.layout)}.` : '';
  toast(`Read in ${(r.written || []).join(', ')}.${where}${also}${skipped} Loading the model again with them.`);
  loadNow();
}
