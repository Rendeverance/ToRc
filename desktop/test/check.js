// check.js - the app's live checks: the real app, a real model, driven through WebView2's DevTools port the way a
// person uses it. Windows only (the DevTools port is WebView2's). The app runs portable from a new folder in the
// system's temp folder, so nothing of yours is touched.
//
//   node desktop/test/check.js <model.torc> [--app <torc-desktop.exe>] [--bin <engine folder>] [--two] [--wake]
//                              [--toolfunnel] [--gguf <file.gguf>]
//     --app         the app to check (default: desktop/target/release/torc-desktop.exe)
//     --bin         the engine's folder (default: target/release)
//     --two         also this machine as the second of two, with a first machine started here
//     --wake        also the wake tool, which waits a minute for its timer
//     --toolfunnel  also ToolFunnel: installed from npm, used, and uninstalled (needs the network)
//     --gguf <file> also a GGUF converted in the app (into the folder's models) and loaded (needs its size free)
const { spawn, execSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const net = require('net');
const http = require('http');

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const repo = path.join(__dirname, '..', '..');
const model = args[0];
const exe = opt('app', path.join(repo, 'desktop', 'target', 'release', 'torc-desktop.exe'));
const bin = opt('bin', path.join(repo, 'target', 'release'));
const PORT = 9333;
if (!model || model.startsWith('--') || process.platform !== 'win32') {
  console.log('usage (Windows): node desktop/test/check.js <model.torc> [--app exe] [--bin folder] [--two] [--wake]');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (name, ok, detail) => {
  results.push(ok);
  const more = detail === undefined ? '' : `(${String(detail).slice(0, 150)})`;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${more}`);
};

// the portable folder: the app, its data folder with the settings, and mcps with the test's own MCP server
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'torc-check-'));
for (const d of ['data', 'mcps', 'chats', 'models']) fs.mkdirSync(path.join(root, d));
const app = path.join(root, 'ToRc.exe');
fs.copyFileSync(exe, app);
fs.copyFileSync(path.join(__dirname, 'echo-mcp.js'), path.join(root, 'mcps', 'echo-mcp.js'));
const settings = { lastModel: model, binDir: bin, port: 8090, ctx: 16384, basic: { compact: true, wake: true } };
fs.writeFileSync(path.join(root, 'data', 'settings.json'), JSON.stringify(settings));

(async () => {
  const proc = spawn(app, [], { env: { ...process.env, TORC_DEVTOOLS_PORT: String(PORT) }, stdio: 'ignore' });
  let target;
  for (let i = 0; i < 60 && !target; i++) {
    await sleep(500);
    try {
      // the app's page, not the splash in front of it
      const pages = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      target = pages.find((t) => t.type === 'page' && t.url.endsWith('index.html'));
    } catch (_) {}
  }
  if (!target) return finish(proc, 'the window did not open');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const waiting = new Map();
  // anything the page throws and does not catch, from start to end
  const thrown = [];
  ws.onmessage = (m) => {
    const j = JSON.parse(m.data);
    if (j.method === 'Runtime.exceptionThrown') {
      const d = j.params.exceptionDetails;
      thrown.push((d.exception && d.exception.description) || d.text);
    }
    if (j.id && waiting.has(j.id)) waiting.get(j.id)(j), waiting.delete(j.id);
  };
  // runs an expression in the page and gives back its value
  const page = (expr) =>
    new Promise((resolve) => {
      const n = ++id;
      waiting.set(n, (j) => resolve(j.result && j.result.result ? j.result.result.value : undefined));
      const params = { expression: `(async () => (${expr}))()`, awaitPromise: true, returnByValue: true };
      ws.send(JSON.stringify({ id: n, method: 'Runtime.evaluate', params }));
    });
  ws.send(JSON.stringify({ id: ++id, method: 'Runtime.enable' }));
  for (let i = 0; i < 40 && !(await page('!!(window.S && S.info)')); i++) await sleep(250);
  const last = () => page('S.chat.turns[S.chat.turns.length - 1] || null');
  // clicks the first button under `sel` whose text starts with `text`, as a person would
  const click = (sel, text) =>
    page(
      `[...document.querySelectorAll(${JSON.stringify(sel)})]` +
        `.find((b) => b.textContent.startsWith(${JSON.stringify(text)})).click()`,
    );
  const say = async (text) => {
    await page(`(async () => { $('input').value = ${JSON.stringify(text)}; await send(); })()`);
    return last();
  };
  // a turn that may ask to use a tool: every ask is allowed until the reply ends
  const sayAllowing = async (text) => {
    await page(`(() => { $('input').value = ${JSON.stringify(text)}; send(); })()`);
    let asked = 0;
    await sleep(500);
    for (let i = 0; i < 1200 && (await page('S.busy')); i++) {
      if (await page("!!document.querySelector('#asks .ask .btn.primary')")) {
        asked++;
        await page("document.querySelector('#asks .ask .btn.primary').click()");
      }
      await sleep(250);
    }
    return { turn: await last(), asked };
  };
  const readOf = (turn) => Number(((turn && turn.speeds) || '').match(/read ([\d,]+) new/)?.[1]?.replace(/,/g, ''));

  // ---- a model, a chat, the built-in tools
  await page('loadNow()');
  check('the model loads', !!(await page('S.loaded')), await page("$('status').textContent"));
  let turn = await say('What is 12 times 12? Answer with the number only.');
  check('a reply comes back', !!turn && turn.text.includes('144'), turn && turn.text.slice(0, 40));
  check('the engine reports its speeds', !!turn && /tok\/s/.test(turn.speeds || ''), turn && turn.speeds);
  turn = await say('Use the calc tool to work out 123456789 * 3 + 1, then give the result.');
  const calc = !!turn && (turn.used || []).some((u) => u.name === 'calc');
  const sum = calc && turn.text.replace(/,/g, '').includes('370370368');
  check('the model calls calc', sum, turn && turn.text.slice(0, 60));
  turn = await say('Thanks. One word: was that a big number?');
  check('the server reads only the new part of the chat', readOf(turn) > 0 && readOf(turn) < 400, readOf(turn));
  let r = await sayAllowing('Use the ctx tool, then tell me how many tokens this chat has used.');
  const ctx = r.turn && (r.turn.used || []).find((u) => u.name === 'ctx');
  const used = !!ctx && /of 16384 tokens used/.test(ctx.result || '');
  check('the model reads its own context use', used, ctx && ctx.result);

  await page('compact(false)');
  const note1 = await page('S.chat.compact && S.chat.compact.note');
  check('the chat compacts', !!note1, await page("$('status').textContent"));
  turn = await say('What two numbers did I first ask you to multiply? One line.');
  check('the model answers from the summary', !!turn && turn.text.includes('12'), turn && turn.text.slice(0, 80));
  const cid = await page('S.chat.id');
  const turns = await page('S.chat.turns.length');
  await page('startNew()');
  await page(`openChat(${JSON.stringify(cid)})`);
  check('a saved chat reopens', (await page('S.chat.turns.length')) === turns, `${turns} turns`);
  r = await sayAllowing('Use the compact tool now, then say OK.');
  let compacted = null;
  for (let i = 0; i < 120 && !compacted; i++) {
    await sleep(500);
    compacted = await page('(!S.busy && S.chat.compact && S.chat.compact.upto > 0 && S.chat.compact.note) || null');
  }
  check('the model compacts its own chat', !!compacted, r.turn && r.turn.text.slice(0, 60));

  // ---- Stop, attachments, a message too big
  await page('startNew()');
  await page("(() => { $('input').value = 'Count from 1 to 300 in words.'; send(); })()");
  // Stop once the answer has begun: a Stop while it is still thinking leaves nothing, and the turn is not kept
  for (let i = 0; i < 240; i++) {
    if (await page("((S.chat.turns[S.chat.turns.length - 1] || {}).text || '').length > 20")) break;
    await sleep(250);
  }
  await page('window.torc.stop()');
  await sleep(3000);
  const stopped = await page(
    "({ busy: S.busy, status: $('status').textContent, input: $('input').value, " +
      "speeds: (S.chat.turns[S.chat.turns.length - 1] || {}).speeds || '' })",
  );
  // kept and marked stopped, or (nothing written yet) dropped with the message back in the box: both by design
  const kept = /stopped/.test(stopped.speeds);
  const back = stopped.status === 'Stopped' && /Count from 1 to 300/.test(stopped.input);
  const how = kept ? stopped.speeds : back ? 'nothing written yet, the message back in the box' : stopped.status;
  check('Stop ends a reply', !stopped.busy && (kept || back), how);
  if (back) await page("$('input').value = ''");
  const note = path.join(root, 'stock.txt');
  fs.writeFileSync(note, 'Stock list\n\nwidgets: 412\nsprockets: 77\ngaskets: 1,930\nflanges: 6\n');
  await page('startNew()');
  await page(`attachPaths([${JSON.stringify(note)}])`);
  turn = await say('Which item in the attached file has the smallest count? One line.');
  check('an attached file reaches the model', !!turn && /flange/i.test(turn.text), turn && turn.text.slice(0, 60));
  const before = await page('S.chat.turns.length');
  await page(`(async () => { $('input').value = ${JSON.stringify('word '.repeat(40000))}; await send(); })()`);
  const refused = (await page('S.chat.turns.length')) === before;
  check('a message too big for the context is refused before it is sent', refused);
  await page("$('input').value = ''");

  // ---- an MCP server of your own, kept in mcps
  const pidFile = path.join(root, 'echo.pid');
  await page(`addServer('echo', ${JSON.stringify(`node echo-mcp.js "${pidFile}"`)}, false)`);
  let up = null;
  for (let i = 0; i < 80 && !up; i++) {
    await sleep(250);
    up = await page("(S.servers || []).find((k) => k.name === 'echo' && k.state === 'connected') || null");
  }
  await page('closeSheet()');
  check('a server in mcps connects by its name alone', !!up && up.tools.length === 1, up ? '1 tool' : '');
  await page('startNew()');
  r = await sayAllowing('Use the shout tool on the text "quiet please" and tell me exactly what it gave back.');
  const shouted = !!r.turn && r.turn.text.includes('QUIET PLEASE');
  check('the model uses it once you allow it', r.asked > 0 && shouted, `${r.asked} asked`);
  turn = await say('Thanks. Answer in one word, without tools: was it in capital letters?');
  const fresh = readOf(turn);
  check('after a tool call the next message reads only the new part', fresh > 0 && fresh < 200, fresh);
  const pid = fs.existsSync(pidFile) ? Number(fs.readFileSync(pidFile, 'utf8')) : 0;
  await page("removeServer(S.settings.mcp.find((m) => m.name === 'echo'))");
  await page('closeSheet()');
  let gone = false;
  for (let i = 0; i < 20 && !gone; i++) {
    await sleep(250);
    try {
      process.kill(pid, 0);
    } catch (_) {
      gone = true;
    }
  }
  check('removing it stops its process', pid > 0 && gone, `pid ${pid}`);

  // ---- Auto-tune that cannot start, and the benchmark
  const binNow = await page('S.settings.binDir');
  await page(`setSettings({ binDir: ${JSON.stringify(fs.mkdtempSync(path.join(root, 'nobin-')))} })`);
  await page("openJob('tune')");
  const log = "(document.querySelector('.joblog') || {}).textContent || ''";
  check('an Auto-tune that cannot start leaves the model loaded', !!(await page('S.loaded')), await page(log));
  await page(`setSettings({ binDir: ${JSON.stringify(binNow)} })`);
  await page('closeSheet()');
  await page("openJob('bench')");
  const csv = path.join(root, 'data', 'results', 'torc-bench.csv');
  const rows = fs.existsSync(csv) ? fs.readFileSync(csv, 'utf8').trim().split('\n').slice(1) : [];
  const sizes = rows.map((row) => (row.match(/,(alice|1k|4k|8k|16k|32k),\d/) || [])[1]);
  const all = ['alice', '1k', '4k', '8k'].every((s) => sizes.includes(s));
  check('the benchmark writes a row for each prompt size', all, sizes.join(' ') || (await page(log)).slice(-150));
  await page('closeSheet()');

  // ---- settings to share: the benchmark keeps them, the engine gives them, and a file for another model is refused
  // (refused before the tune file is touched, so the check writes nothing of yours)
  const head = fs.existsSync(csv) ? fs.readFileSync(csv, 'utf8').split('\n')[0] : '';
  const lastRow = rows[rows.length - 1] || '';
  const keeps = head.split(',').includes('torc_settings') && lastRow.includes('in_use');
  check('each benchmark row keeps the settings the engine ran with', keeps, head.slice(-60));
  const prof = await page('window.torc.engineSettings()');
  const p = prof && prof.profile;
  const plain = !!p && Object.values(p.switches || {}).every((v) => !/[\\/:]/.test(v) || v.startsWith('<'));
  const gives = !!p && p.in_use.block > 0 && plain;
  check('the engine gives its settings, with no path or address in them', gives, JSON.stringify(p && p.in_use));
  const foreign = JSON.stringify({ ...(p || {}), model_key: 'ffffffffffffffff' });
  const other = await page(`window.torc.engineSettingsSet(${foreign})`);
  const refusedOther = !!other && !other.ok && /another model/.test(other.error);
  check('settings measured on another model file are refused', refusedOther, other && other.error);
  await page('openSettings()');
  const sheetText = await page("document.querySelector('.sheet').textContent");
  check('Settings has the Share settings group', /Share settings/.test(sheetText));
  await page('closeSheet()');

  // ---- what a new user does first: with a model loaded, a new context typed in and Load pressed again, by the buttons
  const reload = async (n) => {
    await page('openLoader()');
    await page(
      `(() => { const i = document.querySelector('.sheet input[type=number]'); i.value = '${n}'; ` +
        "i.dispatchEvent(new Event('change')); })()",
    );
    await click('.sheet .row-actions .btn', 'Load');
    for (let i = 0; i < 1200 && !(await page('S.loaded')); i++) await sleep(250);
    // the engine that was replaced is noticed stopping within a second: past that, a loaded model stays loaded
    await sleep(3000);
    return page(
      "({ loaded: S.loaded, ctx: S.loadedCtx, status: $('status').textContent, note: $('modelNote').textContent })",
    );
  };
  await page('openSettings()');
  await click('.sheet .btn', 'Open');
  check('Settings leads to the context and the other model settings', !!(await page("sheetIs('Load a model')")));
  await page('closeSheet()');
  for (const n of [128000, 16384]) {
    const st = await reload(n);
    const held = !!st.loaded && st.ctx === n && !/stopped/i.test(st.status);
    check(`Load again with the context at ${n} keeps the model loaded`, held, `${st.status} | ${st.note}`);
    await click('#pane .pane-acts .btn', 'New chat');
    const empty = "(document.querySelector('.empty') || {}).textContent";
    const fresh = await page(`({ turns: S.chat.turns.length, empty: ${empty} })`);
    check('New chat opens an empty chat with the model ready', fresh.turns === 0 && /Model loaded/.test(fresh.empty));
    turn = await say('What is 7 times 8? Answer with the number only.');
    check('and it answers', !!turn && turn.text.includes('56'), turn && turn.text.slice(0, 40));
  }

  // ---- every screen opens from its own button, and the view switch goes both ways
  const opened = [];
  const buttons = [
    ['btnPrimers', 'Primers'],
    ['btnTools', 'Tools'],
    ['btnSettings', 'Settings'],
    ['btnLoad', 'Load a model'],
  ];
  for (const [b, title] of buttons) {
    await page(`$('${b}').click()`);
    await sleep(500);
    if (await page(`sheetIs(${JSON.stringify(title)})`)) opened.push(title);
    if (title !== 'Load a model') await page('closeSheet()');
  }
  const rec = await page("[...document.querySelectorAll('.sheet .note')].map((n) => n.textContent).join(' ')");
  const said = /Recommended for this model: .+temperature/.test(rec);
  check('the Load sheet says what Recommended sampling is for this model', said, rec);
  await click('.sheet .row-actions .btn', 'Advanced');
  await sleep(500);
  if (await page("sheetIs('Advanced settings')")) opened.push('Advanced settings');
  await page('closeSheet()');
  const views = [];
  for (const v of ['terminal', 'chat']) {
    await page(`document.querySelector('#view button[data-v=${v}]').click()`);
    views.push(await page('S.view'));
  }
  const every = opened.length === 5 && views.join() === 'terminal,chat';
  check('every screen opens from its button', every, opened.join(', '));

  // ---- a code block in a reply: its code without the fence's language word, with Copy and Save
  const fence = '```';
  const reply = JSON.stringify(`Here:\n${fence}js\nlet x = 1;\n${fence}`);
  const cb = await page(
    `(() => { const b = rich(${reply}); return { code: b.querySelector('pre.code').textContent, ` +
      "acts: [...b.querySelectorAll('.code-acts .btn')].map((x) => x.textContent).join() }; })()",
  );
  const shown = !!cb && cb.code === 'let x = 1;' && cb.acts === 'Copy,Save';
  check('a code block shows its code with Copy and Save', shown, JSON.stringify(cb));

  // ---- a primer, written and used through the Primers sheet's buttons
  await page('startNew()');
  await page("$('btnPrimers').click()");
  await page(
    "(() => { document.querySelector('.primer-name').value = 'Fruit'; " +
      "document.querySelector('.primer-text').value = 'End every reply with the word BANANA in capitals.'; " +
      "[...document.querySelectorAll('.sheet .row-actions .btn')].find((b) => b.textContent === 'Add').click(); })()",
  );
  await sleep(500);
  await page(
    "(() => { const r = [...document.querySelectorAll('.sheet .row')].find((x) => x.textContent.includes('Fruit')); " +
      "[...r.querySelectorAll('.btn')].find((b) => b.textContent === 'Use in this chat').click(); })()",
  );
  await page('closeSheet()');
  turn = await say('What is 2 plus 2?');
  check('a primer added in Primers steers the chat', !!turn && /BANANA/.test(turn.text), turn && turn.text.slice(-60));
  const fruit = await page("(S.primers.find((p) => p.name === 'Fruit') || {}).id");
  if (fruit) await page(`window.torc.primerDel(${JSON.stringify(fruit)})`);
  await page('startNew()');

  // ---- the wake tool
  if (args.includes('--wake')) {
    await page('startNew()');
    const ask1 = 'Use the wake tool to wake yourself in 1 minute with the message "check the kettle". Then say set.';
    r = await sayAllowing(ask1);
    const wake = r.turn && (r.turn.used || []).find((u) => u.name === 'wake');
    check('the model sets a wake-up', !!wake && /in 1 minute /.test(wake.result || ''), wake && wake.result);
    let woke = null;
    for (let i = 0; i < 200 && !woke; i++) {
      await sleep(1000);
      woke = await page("(S.chat.turns.find((t) => t.role === 'user' && /A wake-up you set/.test(t.text)) || {}).text");
    }
    await sleep(1000);
    for (let i = 0; i < 600 && (await page('S.busy')); i++) await sleep(500);
    // a model that thinks on and on about a note it can't act on is the model's affair, not the app's: stopped
    if (await page('S.busy')) await page('window.torc.stop()');
    for (let i = 0; i < 20 && (await page('S.busy')); i++) await sleep(500);
    turn = await last();
    const answered = !!turn && turn.role === 'assistant' && !!(turn.text || turn.thinking || (turn.used || []).length);
    check('the wake-up comes into its chat and is answered', !!woke && answered, `${woke} | ${turn && turn.text}`);
  }

  // ---- ToolFunnel: installed into mcps, a server added through it, its tools listed, uninstalled
  if (args.includes('--toolfunnel')) await toolfunnel();

  await page('unload()');
  await sleep(1000);
  check('the model unloads', !(await page('S.loaded')));

  // ---- this machine as the second of two, with the first machine here too
  if (args.includes('--two')) await twoMachines();
  // ---- a GGUF converted in the app, then loaded
  if (opt('gguf')) await convert(opt('gguf'));
  finish(proc);

  async function convert(src) {
    const c = await page(`window.torc.ggufCheck(${JSON.stringify(src)})`);
    const into = path.join(root, 'models');
    check('a GGUF is read before converting', !!c && c.ok && c.supported && c.out.startsWith(into), c && c.arch);
    const r = await page(`openJob('convert', ${JSON.stringify({ src, dst: c.out })})`);
    check('the app converts it', !!r && r.ok && fs.existsSync(c.out), r && (r.error || 'done'));
    await page('closeSheet()');
    await page(`setSettings({ lastModel: ${JSON.stringify(c.out)} })`);
    await page('loadNow()');
    await page('startNew()');
    const t = await say('What is 12 times 12? Answer with the number only.');
    check('the converted model loads and answers', !!t && t.text.includes('144'), t && t.text.slice(0, 40));
    await page('unload()');
  }

  async function toolfunnel() {
    const latest = await page('window.torc.tfLatest()');
    const inst = await page('window.torc.tfInstall()');
    const have = await page('window.torc.tfHave()');
    const prog = path.join(root, 'mcps', 'toolfunnel', `pkg-${have}`, 'bin', 'toolfunnel.js');
    check('ToolFunnel installs into mcps', !!inst && inst.ok && have === latest && fs.existsSync(prog), have);
    await page("setSettings({ tools: 'toolfunnel' })");
    let tf = null;
    for (let i = 0; i < 80 && !tf; i++) {
      await page('refreshServers()');
      await sleep(250);
      tf = await page("(S.servers || []).find((k) => k.name === 'toolfunnel' && k.state === 'connected') || null");
    }
    check('ToolFunnel runs as the tool server', !!tf && tf.tools.length > 0, tf && `${tf.tools.length} tools`);
    await page(`addServer('echo', ${JSON.stringify(`${process.execPath} echo-mcp.js echo.pid`)}, true)`);
    await page('closeSheet()');
    const copy = path.join(root, 'mcps', 'toolfunnel', 'home', 'mcp', 'servers', 'echo', 'echo-mcp.js');
    check('a server added through it is copied into its folder', fs.existsSync(copy));
    await page('startNew()');
    r = await sayAllowing('Use the shout tool on the text "quiet please" and tell me exactly what it gave back.');
    const through = !!r.turn && r.turn.text.includes('QUIET PLEASE');
    check('the model uses it through ToolFunnel', through, r.turn && r.turn.text);
    await page('openTools()');
    let lists = null;
    for (let i = 0; i < 60 && !(lists && lists.mcps && lists.mcps.length); i++) {
      await sleep(500);
      lists = await page('({ tools: S.tfTools, mcps: S.tfMcps })');
    }
    const echo = lists && (lists.mcps || []).find((u) => u.id === 'echo');
    check('the Tools screen lists its tools', !!lists && (lists.tools || []).length > 0, lists && lists.tools.length);
    check('and its servers with their tools', !!echo && (echo.exposed || []).some((e) => e.tool === 'shout'));
    await page('closeSheet()');
    await page("removeServer(S.settings.mcp.find((m) => m.name === 'echo'))");
    await page("setSettings({ tools: 'basic' })");
    await page('refreshServers()');
    const un = await page('window.torc.tfUninstall()');
    check('ToolFunnel uninstalls', !!un && un.ok && !(await page('window.torc.tfHave()')), JSON.stringify(un));
  }

  async function twoMachines() {
    await page('setSettings({ asWorker: true })');
    await page('loadNow()');
    const status = await page("$('status').textContent");
    check('the app waits as the second machine', !!(await page('S.asWorker')) && /:50052/.test(status), status);
    const shown = (status.match(/(\d+\.\d+\.\d+\.\d+):50052/) || [])[1] || '127.0.0.1';
    const loopback = await new Promise((resolve) => {
      const c = net.connect(50052, '127.0.0.1', () => (c.destroy(), resolve(true)));
      c.on('error', () => resolve(false));
    });
    const only = shown !== '127.0.0.1' && !loopback;
    check('it listens only on the address it shows', only, `${shown}, loopback ${loopback}`);
    const env = { ...process.env, TORC_PIPE: `${shown}:50052` };
    const server = path.join(bin, 'torc-server.exe');
    const first = spawn(server, [model, '18431', 'auto', '16384'], { env, windowsHide: true });
    let firstLog = '';
    first.stderr.on('data', (d) => (firstLog = (firstLog + d).slice(-3000)));
    const up = () =>
      new Promise((resolve) => {
        const opts = { host: '127.0.0.1', port: 18431, path: '/v1/models', timeout: 2000 };
        http.get(opts, (res) => (res.resume(), resolve(res.statusCode === 200))).on('error', () => resolve(false));
      });
    let ok = false;
    for (let i = 0; i < 300 && !ok; i++) {
      await sleep(1000);
      ok = await up();
    }
    const answer = ok ? await ask('What is 12 times 12? Answer with the number only.') : '';
    const pipe = firstLog.includes(`worker at ${shown}:50052`);
    check('the first machine answers through it', ok && answer.includes('144') && pipe, answer.slice(0, 30));
    first.kill();
    await page('unload()');
    await page('setSettings({ asWorker: false })');
    await sleep(1000);
    check('unloading stops the second machine', !(await page('S.loaded')) && !(await page('S.asWorker')));
  }

  function ask(text) {
    const messages = [{ role: 'user', content: text }];
    const plain = { enable_thinking: false };
    const body = JSON.stringify({ model: 'local', max_tokens: 40, messages, chat_template_kwargs: plain });
    return new Promise((resolve) => {
      const opts = { host: '127.0.0.1', port: 18431, path: '/v1/chat/completions', method: 'POST' };
      const req = http.request({ ...opts, headers: { 'Content-Type': 'application/json' } }, (res) => {
        let s = '';
        res.on('data', (d) => (s += d));
        res.on('end', () => {
          try {
            resolve(JSON.parse(s).choices[0].message.content || '');
          } catch (_) {
            resolve(s.slice(0, 100));
          }
        });
      });
      req.on('error', (e) => resolve(String(e.message)));
      req.end(body);
    });
  }

  function finish(p, why) {
    if (why) check('the run finished', false, why);
    else check('nothing was thrown in the page', !thrown.length, thrown[0]);
    console.log(`${results.filter(Boolean).length} of ${results.length} checks passed (folder ${root})`);
    try {
      execSync(`taskkill /PID ${p.pid} /T /F`, { stdio: 'ignore' });
    } catch (_) {}
    process.exit(results.every(Boolean) ? 0 : 1);
  }
})().catch((e) => {
  console.log(`FAIL  the run finished  (${e.message})`);
  process.exit(1);
});
