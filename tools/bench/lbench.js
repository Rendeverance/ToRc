// lbench.js - one fixed request to any OpenAI-style chat server on this machine (torc-server, or another engine's
// own server for a fair comparison). One user message: a run line (so no server can reuse a cached prompt from the
// run before), the opening of Alice's Adventures in Wonderland chapter I (Project Gutenberg #11, public domain,
// lb_alice.txt, ~2400 characters), then "Continue the story." Greedy, thinking off, 128 tokens out (LB_MAX).
// Streams the reply and times it on the client:
//   read  = prompt tokens / time to the first reply token
//   write = (reply tokens - 1) / time from the first to the last reply token
//   node lbench.js <port> <label> [model]
const http = require('http');
const fs = require('fs');
const path = require('path');
const [port, label, model = 'local'] = process.argv.slice(2);
const text = fs.readFileSync(path.join(__dirname, 'lb_alice.txt'), 'utf8').trim();
// LB_PROMPT=file: that file's text as the request instead (after the same run line), e.g. a code task
const content = process.env.LB_PROMPT
  ? `Run ${label} ${Date.now()}\n\n${fs.readFileSync(process.env.LB_PROMPT, 'utf8').trim()}`
  : `Run ${label} ${Date.now()}\n\n${text}\n\nContinue the story.`;
const body = JSON.stringify({
  model, max_tokens: Number(process.env.LB_MAX || 128), temperature: 0, top_k: 1, stream: true,
  stream_options: { include_usage: true }, cache_prompt: false,
  messages: [{ role: 'user', content }], chat_template_kwargs: { enable_thinking: false },
});
const t0 = process.hrtime.bigint();
let tFirst = 0n, tLast = 0n, chunks = 0, usage = null, timings = null, buf = '', out = '';
const ms = (a, b) => Number(b - a) / 1e6;
const req = http.request({ host: '127.0.0.1', port: Number(port), path: '/v1/chat/completions', method: 'POST',
  headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => {
  res.setEncoding('utf8');
  res.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line.startsWith('data:')) continue;
      const p = line.slice(5).trim();
      if (p === '[DONE]') continue;
      let j;
      try { j = JSON.parse(p); } catch (_e) { continue; }
      if (j.usage) usage = j.usage;
      if (j.timings) timings = j.timings;
      const c = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
      if (c) {
        const now = process.hrtime.bigint();
        if (!tFirst) tFirst = now;
        tLast = now;
        out += c;
        chunks++;
      }
    }
  });
  res.on('end', () => {
    const pt = usage ? usage.prompt_tokens : (timings ? timings.prompt_n : NaN);
    const ct = usage ? usage.completion_tokens : (timings ? timings.predicted_n : chunks);
    fs.writeFileSync(`lbench_${label}.txt`, out);
    console.log(JSON.stringify({
      label, status: res.statusCode, prompt_tokens: pt, reply_tokens: ct,
      ttft_s: +(ms(t0, tFirst) / 1000).toFixed(2), read_tps: +(pt / (ms(t0, tFirst) / 1000)).toFixed(1),
      write_tps: +((ct - 1) / (ms(tFirst, tLast) / 1000)).toFixed(1),
    }));
  });
});
req.on('error', (e) => { console.log(JSON.stringify({ label, error: e.message })); });
req.setTimeout(1800000, () => req.destroy(new Error('timeout')));
req.end(body);
