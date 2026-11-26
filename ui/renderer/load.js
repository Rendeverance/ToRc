// load.js - the Load a model sheet, and the sheet that runs the tuner or the benchmark and shows what it prints.

const WAIT_FOR_REPLY = 'Wait for the reply to finish, or press Stop';
const WAIT_FOR_JOB = 'Wait for the tuner or the benchmark to finish, or stop it';

function openLoader() {
  if (S.busy) return toast(WAIT_FOR_REPLY);
  const s = S.settings;
  const body = sheet(
    'Load a model',
    'The main settings. Each is saved as you change it and takes effect when you press Load. Auto-tune ' +
      'measures this machine and sets the rest.',
  );
  // changes made since the model now running was loaded, said again as they are made
  const pending = el('div', 'note warn pending-note');
  pending.id = 'pendingNote';
  pending.hidden = true;
  const again = el('button', 'btn', 'Load now');
  again.onclick = loadNow;
  pending.append(el('span', ''), again);
  body.append(pending);
  showPending();

  const file = el('button', 'btn', s.lastModel ? baseName(s.lastModel) : 'Choose a file');
  file.onclick = async () => {
    const [picked] = await window.torc.pick('model');
    if (!picked) return;
    if (/\.gguf$/i.test(picked)) return openConvert(picked);
    await setSettings({ lastModel: picked });
    file.textContent = baseName(picked);
    noteFor(picked);
  };
  const context = field('number', s.ctx, (v) => setSettings({ ctx: Math.max(1024, Math.round(v)) }), {
    min: 1024,
    step: 1024,
  });
  const longest = field('number', s.replyMax, (v) => setSettings({ replyMax: Math.max(1024, Math.round(v)) }), {
    min: 1024,
    step: 1024,
  });
  const threadChoices = ['auto', 2, 4, 6, 8, 10, 12, 16].map((v) => [String(v), String(v)]);
  const threads = select(threadChoices, String(s.threads), (v) => setSettings({ threads: v }));
  body.append(
    group('Model', [
      row(
        'Model file',
        'A .torc file, or a GGUF to convert once',
        file,
        'ToRc has its own model format. Choose a GGUF and ToRc converts it once, to a .torc file beside it (or in ' +
          'models, when ToRc runs from its own folder). Models keep their own licences.',
      ),
      row(
        'Context',
        'The most tokens a chat can hold',
        context,
        'How much text the model can see at once: the prompt and its reply together. A bigger context takes more ' +
          'memory as it fills. 16384 is a good start; go higher for long documents.',
      ),
      row(
        'CPU threads',
        'Auto uses every core of the fastest kind',
        threads,
        'Auto is usually right. On a CPU with fast and slow cores it uses the fast ones. Set a number to hold ' +
        'some cores back for other work.',
      ),
    ]),
  );

  // how replies are written: their length and how each token is picked
  const modes = [
    ['recommended', 'Recommended for this model'],
    ['code', 'Coding (the maker’s preset, where there is one)'],
    ['greedy', 'Greedy (always the likeliest token)'],
    ['custom', 'Custom'],
  ];
  const custom = el('div', 'rows');
  const note = el('div', 'note', '');
  // what Recommended means for the chosen model, said again when another file is chosen here
  const noteFor = (path) =>
    path &&
    window.torc.modelArch(path).then((m) => (note.textContent = `Recommended for this model: ${samplingNote(m)}`));
  const showCustom = () => (custom.hidden = (S.settings.sampling || 'recommended') !== 'custom');
  const mode = select(modes, s.sampling || 'recommended', (v) => setSettings({ sampling: v }).then(showCustom));
  const num = (key, value, step) => field('number', value, (v) => setSettings({ [key]: v }), { min: 0, step });
  custom.append(
    ...row('Temperature', 'Higher is more varied; 0 is greedy', num('sTemp', s.sTemp ?? 0.7, 0.05)),
    ...row('Top p', 'Only the likeliest tokens making up this share', num('sTopP', s.sTopP ?? 0.95, 0.05)),
    ...row('Top k', 'Only this many of the likeliest tokens; 0 for no limit (slower)', num('sTopK', s.sTopK ?? 20, 1)),
    ...row('Presence penalty', 'Above 0 discourages repeating any word', num('sPresence', s.sPresence ?? 0, 0.1)),
    ...row('Frequency penalty', 'Above 0 discourages a word more each time', num('sFrequency', s.sFrequency ?? 0, 0.1)),
  );
  showCustom();
  noteFor(s.lastModel);
  body.append(
    group('Replies', [
      row(
        'Longest reply',
        'The most tokens one reply may take, thinking included',
        longest,
        'A reply stops here even if the model has more to say. Thinking counts too, and a model can think for ' +
          'thousands of tokens before it answers, so for long code or documents raise it. It also has to fit in ' +
          'the context with the chat so far. 16384 by default.',
      ),
      row(
        'Sampling',
        'How each token of a reply is picked',
        mode,
        'Recommended uses what the model’s maker publishes on its model card, with thinking and without; Coding ' +
          'uses the maker’s preset for precise coding where there is one. Greedy always takes the likeliest token, ' +
          'which repeats exactly but can loop in long thinking. Custom uses the values you set.',
      ),
    ]),
    note,
    custom,
  );

  const worker = field('text', s.worker, (v) => setSettings({ worker: v.trim() }), { placeholder: 'none' });
  const waitOn = row(
    'Wait for the first machine on',
    'The network the first machine reaches this one on',
    listenChoice(),
    'Only for two machines: this is where the second machine waits for the first, not where your chat app or ' +
      'agent reaches the engine (that is Share on my network, in Settings). The second machine has no password, ' +
      'so it waits on one network only: pick the one the first machine is on (a USB4 or Ethernet link, or your ' +
      'home network). Every network is there if you need it, but then anyone on any of them can use it.',
  );
  body.append(
    group('This machine', [
      row(
        'NPU',
        npuHint() || 'Reads prompts faster',
        npuSwitch(),
        'Current support includes Snapdragon X NPUs. Intel and AMD NPU support is planned when I have access to ' +
        'suitable hardware or contributors can assist. Puts part of reading a ' +
        'prompt on the NPU. It isn’t bit-exact, so it only touches prompts, never replies. On one machine it ' +
        'also learns which experts your work uses and reads faster from the second session.',
      ),
      row(
        'GPU for long prompts',
        'A Vulkan GPU takes a share of a long prompt’s attention',
        toggle(s.gpu, (v) => setSettings({ gpu: v })),
        'Helps from a few thousand tokens of prompt and grows with length. Replies always stay on the CPU. It ' +
        'needs an engine built with the GPU feature. If the GPU turns out to be no gain, the engine switches it ' +
        'off for that model.',
      ),
      row(
        'Second machine',
        'Address of a worker, as host:port. Leave empty for one machine',
        worker,
        'Two machines can share one model: this one runs the first layers, the worker the rest. Start the worker ' +
          'first (on that machine, tick "This is the second machine" below and load the same model). Its address ' +
          'is shown there once it is waiting. It has no password, so use a link you trust.',
      ),
      row(
        'This is the second machine',
        'Load waits here for the first machine to connect',
        toggle(s.asWorker, (v) => setSettings({ asWorker: v }).then(() => showWaitOn(v))),
        'Tick this on the machine that helps, not the one you chat on. Load then holds the later layers of the ' +
          'model and waits; the status line shows the address to type into the first machine’s Second machine ' +
          'box. Both machines need the same model file, and the same context or more here. Chat on the first one.',
      ),
      waitOn,
    ]),
  );
  // only a second machine waits for the first, so the choice shows when this is one
  const showWaitOn = (on) => {
    waitOn[0].hidden = !on;
    if (!on) waitOn[1].hidden = true;
  };
  showWaitOn(!!s.asWorker);

  const acts = el('div', 'row-actions');
  const tune = el('button', 'btn', 'Auto-tune');
  tune.onclick = () => openJob('tune');
  const advanced = el('button', 'btn', 'Advanced…');
  advanced.onclick = () => openAdvanced();
  const load = el('button', 'btn primary', 'Load');
  load.onclick = loadNow;
  acts.append(tune, advanced, load);
  body.append(acts);
}

// This machine's addresses with their adapters, likeliest first (the shell ranks them), and every network last. Until
// it is changed, the choice is the first, which is also what the shell falls back to when the one picked before has
// gone (another network, a new lease).
function listenChoice() {
  const s = el('select');
  s.onchange = () => setSettings({ workerBind: s.value });
  // asking the system for its networks takes a moment
  s.append(el('option', '', 'Finding this machine’s networks…'));
  window.torc.addresses().then((list) => {
    s.replaceChildren();
    const hosts = list.map((i) => i.address);
    const options = list.map((i) => [i.address, `${i.address} (${i.name})`]);
    for (const [v, label] of [...options, ['any', 'Every network']]) {
      const o = el('option', '', label);
      o.value = v;
      s.append(o);
    }
    const was = S.settings.workerBind;
    s.value = was === 'any' || hosts.includes(was) ? was : hosts[0] || 'any';
  });
  return s;
}

// The NPU switch only does something on a Snapdragon X machine so far, and says so plainly anywhere else.
function npuHint() {
  if (S.info.npu === 'intel') return 'This machine has an Intel NPU. Intel NPU support is not written yet.';
  if (S.info.npu === 'amd') return 'This machine has an AMD NPU. AMD NPU support is not written yet.';
  return S.info.npu === 'none' ? 'No supported NPU on this machine' : '';
}

function npuSwitch() {
  const usable = S.info.npu === 'snapdragon';
  const t = toggle(S.settings.npu && usable, (v) => setSettings({ npu: v }).then(renderPane));
  if (usable) return t;
  t.querySelector('input').disabled = true;
  t.classList.add('off');
  if (S.info.npu === 'none') return t;
  const wrap = el('span', 'ctl');
  wrap.append(el('span', 'badge warn', 'not supported yet'), t);
  return wrap;
}

async function loadNow() {
  if (S.busy) return toast(WAIT_FOR_REPLY);
  if (S.job) return toast(WAIT_FOR_JOB);
  if (!S.settings.lastModel) return toast('Choose a model file first');
  closeSheet();
  const was = S.loaded;
  S.loaded = null;
  setStatus('Loading the model', true);
  renderPane();
  const asked = JSON.parse(JSON.stringify(S.settings));
  const r = await window.torc.load();
  S.asWorker = !!(r.ok && r.worker);
  if (r.ok) {
    S.loadedWith = asked;
    // where other machines reach it, said in the side pane: this machine's own addresses with the port
    S.shareAt = [];
    if (asked.share && !r.worker) {
      // the likeliest first (the shell ranks them), every one in the line's tooltip
      const mine = await window.torc.addresses();
      S.shareAt = mine.map((i) => `${i.address}:${asked.port} (${i.name})`);
    }
  }
  if (r.ok && r.worker) {
    S.loaded = r.name;
    const where = r.addresses && r.addresses.length ? r.addresses.join(' or ') : 'this machine, port 50052';
    setStatus(`Second machine: waiting for the first one. Its Second machine box: ${where}`, false);
  } else if (r.ok) {
    S.loadedCtx = S.settings.ctx;
    S.modelArch = await window.torc.modelArch(S.settings.lastModel);
    S.loaded = r.name;
    setStatus('Idle', false);
  } else {
    // refused before anything was stopped: the model that was loaded still is
    if (r.kept) S.loaded = was;
    setStatus(`The model did not load: ${r.error}`, false);
    toast(r.error);
  }
  render();
}

// A GGUF the user chose: what it is, whether this engine runs it, and where the converted file goes; then the
// conversion as a job, and the new file chosen.
async function openConvert(src) {
  const c = await window.torc.ggufCheck(src);
  const lead = 'ToRc reads its own format. A GGUF is converted once, then it loads like any model.';
  const body = sheet('Convert a GGUF', lead);
  if (!c.ok) return body.append(el('div', 'note warn', c.error));
  const gb = (c.bytes / 2 ** 30).toFixed(1);
  if (!c.supported) {
    const why = c.why
      ? `This model won't run in ToRc yet: ${c.why}.`
      : `This is a ${c.arch} model, which ToRc doesn't run yet. It runs: ${c.archs}.`;
    return body.append(el('div', 'note warn', why));
  }
  const out = field('text', c.out, () => {});
  const free = c.free == null ? '' : ` (${(c.free / 2 ** 30).toFixed(1)} GB free)`;
  const short = c.free != null && c.free < c.bytes;
  body.append(
    group('Convert', [
      row('The model', `${baseName(src)}: ${c.arch}, ${gb} GB`, el('span')),
      row('Save as', `The converted file is the same size as the GGUF: it needs ${gb} GB${free}`, out),
    ]),
  );
  if (short) body.append(el('div', 'note warn', 'There is not enough space there. Free some, or save it elsewhere.'));
  const acts = el('div', 'row-actions');
  const go = el('button', 'btn primary', 'Convert');
  go.onclick = async () => {
    const r = await openJob('convert', { src, dst: out.value.trim() });
    if (r && r.ok) await setSettings({ lastModel: out.value.trim() });
  };
  acts.append(go);
  body.append(acts);
}

// The tuner, the benchmark or a conversion, with what it prints shown as it runs.
async function openJob(kind, args) {
  const tuning = kind === 'tune';
  const converting = kind === 'convert';
  if (S.busy) return toast(WAIT_FOR_REPLY);
  if (S.job) return toast(WAIT_FOR_JOB);
  if (tuning && !S.settings.lastModel) return toast('Choose a model file first');
  if (kind === 'bench' && !S.loaded) return toast('Load a model first');

  const leads = {
    tune:
      'Measures this machine with this model and keeps the fastest settings. It takes a few minutes. ' +
      'Leave the machine alone while it runs.',
    bench:
      'Sends the standard prompts to the loaded model and writes the speeds to a CSV. ' +
      'Leave the machine alone while it runs.',
    convert: 'Converts the GGUF into a ToRc file. Every tensor is checked as it is written.',
  };
  const titles = { tune: 'Auto-tune', bench: 'Benchmark', convert: 'Convert a GGUF' };
  const body = sheet(titles[kind], leads[kind], true);
  const log = el('pre', 'joblog');
  const acts = el('div', 'row-actions');
  const stop = el('button', 'btn', 'Stop');
  stop.onclick = () => window.torc.jobStop();
  acts.append(stop);
  body.append(log, acts);
  S.onJobLine = (text) => {
    log.textContent += text;
    log.scrollTop = log.scrollHeight;
  };

  setStatus({ tune: 'Tuning', bench: 'Measuring', convert: 'Converting' }[kind], true);
  const was = S.loaded;
  if (tuning) {
    // the tuner needs the model to itself, so the engine is stopped for it
    S.loaded = null;
    renderPane();
  }
  // nothing else may load, chat or start another job until it ends, and its sheet stays open for its Stop
  S.job = true;
  const r = await window.torc.job(kind, args || {});
  S.job = false;
  if (tuning && r.kept) {
    // it failed before the engine was stopped, so the model is still loaded
    S.loaded = was;
    renderPane();
  }
  S.onJobLine = null;
  stop.remove();
  log.textContent += r.ok ? `${NL}Done.` : `${NL}${r.error || `Stopped (exit ${r.code}).`}`;
  log.scrollTop = log.scrollHeight;
  setStatus('Idle', false);
  if (tuning && r.ok) for (const k of toAsk()) body.insertBefore(consent(k), acts);
  if ((tuning || converting) && r.ok) {
    const again = el('button', 'btn primary', converting ? 'Auto-tune it' : 'Load the model');
    again.onclick = converting ? () => openJob('tune') : loadNow;
    acts.append(again);
  }
  return r;
}

// What Auto-tune asks about when it finishes: the NPU on a Snapdragon X while it is off, and the GPU once its switch
// is on, each until it has been answered once (settings.asked). Both round differently from the CPU.
function toAsk() {
  const asked = S.settings.asked || {};
  const kinds = [];
  if (S.info.npu === 'snapdragon' && !S.settings.npu && !asked.npu) kinds.push('npu');
  if (S.settings.gpu && !asked.gpu) kinds.push('gpu');
  return kinds;
}

// The question before the NPU or the GPU is used. Using it is the highlighted choice, since it is faster and the
// effect is slight, but the user decides. No number: how far the odds move depends on the model, and means nothing
// to someone choosing.
function consent(kind) {
  const npu = kind === 'npu';
  const box = el('div', 'note consent');
  box.append(
    el('b', '', npu ? 'Use the NPU?' : 'Use the GPU for long prompts?'),
    el(
      'p',
      '',
      'It reads prompts faster, with a slight effect on quality from rounding: the results are not exact with the ' +
        'CPU’s, or between platforms and vendors. Replies are still written on the CPU.',
    ),
  );
  const btns = el('div', 'row-actions');
  const answer = async (on) => {
    // after any save still in flight, so two answers in a row both stay remembered
    await S.saving;
    await setSettings({ [kind]: on, asked: { ...(S.settings.asked || {}), [kind]: true } });
    renderPane();
    const what = npu ? 'the NPU' : 'the GPU for long prompts';
    btns.replaceChildren(el('span', '', on ? `Using ${what} from the next Load.` : 'CPU only. Nothing else changes.'));
  };
  const yes = el('button', 'btn primary', npu ? 'Use the NPU' : 'Use the GPU');
  const no = el('button', 'btn', 'CPU only');
  yes.onclick = () => answer(true);
  no.onclick = () => answer(false);
  btns.append(yes, no);
  box.append(btns);
  return box;
}

// What the engine and the tool servers have printed, for when something goes wrong.
function showLog(title, text) {
  const body = sheet(
    title || 'Engine log',
    'What the engine and the tool servers have printed since ToRc started.',
    true,
  );
  const log = el('pre', 'joblog tall', text || S.log.join('') || 'Nothing yet.');
  body.append(log);
  log.scrollTop = log.scrollHeight;
}
