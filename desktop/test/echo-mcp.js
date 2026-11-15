// echo-mcp.js - the smallest MCP server over stdio, for the harness: one tool, shout, which gives back its text in
// capitals. node test/echo-mcp.js <pid file> writes its process id there, so the harness can check it is gone after
// the server is removed.
const fs = require('fs');

if (process.argv[2]) fs.writeFileSync(process.argv[2], String(process.pid));

const tool = {
  name: 'shout',
  description: 'Gives back the text in capital letters.',
  inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
};

function answer(id, result) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n');
}

let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => {
  buf += d;
  let cut;
  while ((cut = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, cut).trim();
    buf = buf.slice(cut + 1);
    if (!line) continue;
    const m = JSON.parse(line);
    if (m.id === undefined) continue;
    if (m.method === 'initialize') {
      const version = m.params.protocolVersion;
      const serverInfo = { name: 'echo', version: '1' };
      answer(m.id, { protocolVersion: version, capabilities: { tools: {} }, serverInfo });
    } else if (m.method === 'tools/list') {
      answer(m.id, { tools: [tool] });
    } else if (m.method === 'tools/call') {
      const text = String((m.params.arguments || {}).text || '');
      answer(m.id, { content: [{ type: 'text', text: text.toUpperCase() }] });
    } else {
      const error = { code: -32601, message: 'no such method' };
      process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: m.id, error }) + '\n');
    }
  }
});
// a server stays up until it is stopped; the harness checks that removing it stops it
setInterval(() => {}, 1 << 30);
