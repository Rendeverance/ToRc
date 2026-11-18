// torc-bench.js - measures a running server at the standard prompt sizes and writes the results as CSV, one row a
// size, ready to send in for the results table (RESULTS.md).
//
//   node torc-bench.js <port> [options]
//     --sizes alice,1k,4k,8k,16k     which prompts (default: alice,1k,4k,8k,16k; 32k is there if you have the context)
//     --runs 2                       measured runs a size, after one warm-up run that is not counted (default 2)
//     --model "Qwen3.6-35B-A3B Q4_0" the model's name and quant, as you want it shown
//     --settings "threads 12, GPU attention on"          how the server was started, in your own words
//     --name yourname                who to credit (optional)
//     --out torc-bench.csv           the file to write (default torc-bench.csv, rows are added to it)
//     --hwscan path/to/hwscan        where hwscan is, if it is not next to the server (it fills in the hardware)
//     --host 192.168.1.20            the server's address, if it listens somewhere else than this machine (TORC_HOST)
//     --key <key>                    the server's API key, for one on another machine that asks for it (better: set
//                                    TORC_API_KEY instead, which other users of this machine can't see in its process list)
//
// Start the server first with the settings you want measured, and with enough context for the largest size you ask
// for (4096 more than the size is plenty). Nothing else should be running: another program taking even one core
// can cost 10%.
//
// The prompts are fixed texts (lb_*.txt here, public domain: the opening of Alice's Adventures in Wonderland and of
// Pride and Prejudice, Project Gutenberg #11 and #1342), each sent after a run line with the time in it so that no
// server can reuse the prompt before. Replies are greedy, thinking off, 128 tokens (1024 at 32k).
// Speeds are the server's own timings (ToRc and llama.cpp both report them), in tokens a second:
//   read  = prompt tokens / the time to read the prompt        write = reply tokens / the time to write them
// Each figure in the CSV is the median of the measured runs.
//
// The CSV holds what you see in it and nothing else: the hardware's names and sizes, no host name, no paths. With a
// ToRc server, each row also keeps the settings the server ran with (its /torc/settings: the values in use, the layout
// and the model file's key), so the desktop app can read a result's settings back in (AUTOTUNE.md, Sharing settings).
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const { execFileSync } = require('child_process');

const args = process.argv.slice(2);
const port = Number(args[0]);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i > 0 && args[i + 1] !== undefined ? args[i + 1] : d; };
if (!port) {
  console.error('usage: node torc-bench.js <port> [--sizes alice,1k,4k,8k,16k] [--runs 2] [--model name] [--settings text] [--name you] [--out file] [--hwscan path] [--host address]');
  process.exit(2);
}
const FILES = { alice: 'lb_alice.txt', '1k': 'lb_p1k.txt', '4k': 'lb_p4k.txt', '8k': 'lb_p8192.txt', '16k': 'lb_p16k.txt', '32k': 'lb_p32k.txt' };
const sizes = opt('sizes', 'alice,1k,4k,8k,16k').split(',').map((s) => s.trim()).filter(Boolean);
for (const s of sizes) if (!FILES[s]) { console.error(`unknown size ${s} (sizes: ${Object.keys(FILES).join(', ')})`); process.exit(2); }
const runs = Math.max(1, Number(opt('runs', 2)));
const out = opt('out', 'torc-bench.csv');

// the key a server on another machine may ask for (only another machine's requests need it)
const key = opt('key', process.env.TORC_API_KEY || '');
const auth = key ? { Authorization: `Bearer ${key}` } : {};

// one request; resolves to the reply's JSON
function ask(pathname, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const req = http.request({ host: opt('host', '127.0.0.1'), port, path: pathname, method: body ? 'POST' : 'GET',
      headers: { ...(data ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } : {}), ...auth } }, (res) => {
      let buf = '';
      res.setEncoding('utf8');
      res.on('data', (d) => { buf += d; });
      res.on('end', () => { try { resolve(JSON.parse(buf)); } catch (_e) { reject(new Error(`HTTP ${res.statusCode}: ${buf.slice(0, 200)}`)); } });
    });
    req.on('error', reject);
    req.setTimeout(3600000, () => req.destroy(new Error('no reply in an hour')));
    req.end(data);
  });
}

// the text of a size, the same bytes on every system (a Windows checkout may have turned the line ends into CR LF)
function prompt(size, label) {
  const text = fs.readFileSync(path.join(__dirname, FILES[size]), 'utf8').replace(/\r\n/g, '\n').trim();
  return size === 'alice' ? `Run ${label} ${Date.now()}\n\n${text}\n\nContinue the story.` : `Run ${label} ${Date.now()}\n\n${text}`;
}

async function once(size, label) {
  const j = await ask('/v1/chat/completions', { model: 'local', max_tokens: size === '32k' ? 1024 : 128, temperature: 0, top_k: 1,
    stream: false, cache_prompt: false, messages: [{ role: 'user', content: prompt(size, label) }],
    chat_template_kwargs: { enable_thinking: false } });
  const t = j.timings;
  if (!t || !t.prompt_per_second) throw new Error(j.error ? JSON.stringify(j.error).slice(0, 200) : 'the server sent no timings (ToRc and llama.cpp do)');
  return { tokens: t.prompt_n, read: t.prompt_per_second, write: t.predicted_per_second, reply: t.predicted_n };
}

const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };

// the hardware: from hwscan when it can be found (names and sizes only), else what the system itself reports
function hardware() {
  const h = { cpu: os.cpus()[0] ? os.cpus()[0].model.trim() : '', cores: '', threads: os.cpus().length, ram_gb: Math.round(os.totalmem() / 2 ** 30),
    ram_speed: '', gpu: '', npu: '', os: `${os.platform()} ${os.release()}`, power: '' };
  const exe = process.platform === 'win32' ? 'hwscan.exe' : 'hwscan';
  const tried = [opt('hwscan', ''), path.join(__dirname, '..', '..', 'target', 'release', exe), exe].filter(Boolean);
  for (const p of tried) {
    try {
      const j = JSON.parse(execFileSync(p, ['--json', '--no-bench'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 120000 }));
      const inv = j.inventory || {};
      if (inv.cpu) { h.cpu = inv.cpu.brand || h.cpu; h.cores = inv.cpu.physical_cores || ''; h.threads = inv.cpu.logical_cpus || h.threads; }
      if (inv.memory) {
        h.ram_gb = Math.round((inv.memory.installed_bytes || inv.memory.visible_bytes || 0) / 2 ** 30) || h.ram_gb;
        const type = (inv.memory.modules || []).map((m) => m.memory_type).find(Boolean) || '';
        h.ram_speed = inv.memory.speed_mts ? `${type} ${inv.memory.speed_mts} MT/s`.trim() : type;
      }
      h.gpu = ((inv.gpus || {}).gpus || []).map((g) => `${g.name} (${g.kind}, driver ${g.driver_info})`).join('; ');
      h.npu = ((inv.npus || {}).npus || []).map((n) => n.name || String(n)).join('; ');
      if (j.power) h.power = [j.power.on_ac === false ? 'on battery' : 'on AC', j.power.active_scheme_name].filter(Boolean).join(', ');
      return h;
    } catch (_e) { /* try the next place */ }
  }
  console.error('hwscan not found: the CSV will have the CPU and RAM size only. Point --hwscan at it for the rest.');
  return h;
}

(async () => {
  // what the server says it is: ToRc's own description when it has one, else the model list
  let engine = '', served = '';
  try { const m = await ask('/v1/models'); served = m.data && m.data[0] ? m.data[0].id : ''; engine = m.data && m.data[0] && m.data[0].owned_by === 'torc' ? 'ToRc' : (m.data && m.data[0] && m.data[0].owned_by) || ''; }
  catch (e) { console.error(`no server answers on port ${port}: ${e.message}`); process.exit(1); }
  // the TORC_* settings in this shell: the server's own, if it was started from the same shell. A setting that holds a
  // path or a network address is written by name only, so the CSV never carries a folder name or an address, and
  // the API key never at all
  const ADDRESSES = ['TORC_HOST', 'TORC_PIPE', 'TORC_PIPE_BIND'];
  const isPath = (v) => /[\\/:]/.test(v);
  const shown = (k) =>
    k === 'TORC_API_KEY' ? `${k}=<key>` : ADDRESSES.includes(k) ? `${k}=<address>` : isPath(process.env[k]) ? `${k}=<path>` : `${k}=${process.env[k]}`;
  const env = Object.keys(process.env).filter((k) => k.startsWith('TORC_')).sort().map(shown).join(' ');
  // the settings the server runs with, ToRc's own (another server has none)
  let tuned = '';
  if (engine === 'ToRc') {
    try {
      const p = await ask('/torc/settings');
      if (p && p.torc_settings) tuned = JSON.stringify({ torc_settings: p.torc_settings, model_key: p.model_key, layout: p.layout, in_use: p.in_use, chosen: p.chosen });
    } catch (_e) { /* an older server: no settings to keep */ }
  }
  const hw = hardware();
  const rows = [];
  for (const size of sizes) {
    process.stderr.write(`${size}: warm-up`);
    try {
      await once(size, `warm-${size}`);
      const got = [];
      for (let i = 0; i < runs; i++) { got.push(await once(size, `${size}-${i}`)); process.stderr.write(`, run ${i + 1}: read ${got[i].read.toFixed(1)} write ${got[i].write.toFixed(1)}`); }
      process.stderr.write('\n');
      rows.push({ size, tokens: got[0].tokens, read: median(got.map((g) => g.read)), write: median(got.map((g) => g.write)), reply: got[0].reply });
    } catch (e) { process.stderr.write(`\n${size}: ${e.message}\n`); }
  }
  if (!rows.length) process.exit(1);
  const head = ['date', 'name', 'engine', 'model', 'settings', 'torc_switches', 'size', 'prompt_tokens', 'read_tps', 'write_tps', 'reply_tokens', 'runs',
    'cpu', 'cores', 'threads', 'ram_gb', 'ram', 'gpu', 'npu', 'os', 'power', 'torc_settings'];
  const esc = (v) => (/[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : String(v));
  const date = new Date().toISOString().slice(0, 10);
  const lines = rows.map((r) => [date, opt('name', ''), engine, opt('model', served), opt('settings', ''), env, r.size, r.tokens, r.read.toFixed(1), r.write.toFixed(1),
    r.reply, runs, hw.cpu, hw.cores, hw.threads, hw.ram_gb, hw.ram_speed, hw.gpu, hw.npu, hw.os, hw.power, tuned].map(esc).join(','));
  const fresh = !fs.existsSync(out);
  // a file started before the settings column: its header gets the column, its old rows stay as they are
  if (!fresh) {
    const old = fs.readFileSync(out, 'utf8');
    const nl = old.indexOf('\n');
    const first = nl < 0 ? old : old.slice(0, nl);
    if (!first.split(',').includes('torc_settings')) fs.writeFileSync(out, first.replace(/\r$/, '') + ',torc_settings' + (nl < 0 ? '\n' : old.slice(nl)));
  }
  fs.appendFileSync(out, (fresh ? head.join(',') + '\n' : '') + lines.join('\n') + '\n');
  console.log(`${rows.length} rows added to ${out}`);
})();
