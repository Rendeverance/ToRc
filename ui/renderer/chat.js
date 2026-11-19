// chat.js - the conversation: drawing it in both views, sending a message, and compacting.

const COMPACT_AT = 0.9; // auto compact once the context is this full
const WARN_AT = 0.8; // from here the user is told the context is filling
const FULL_AT = 0.9; // from here the warning is red
// the most tokens a reply may take, thinking included (Longest reply, on the Load a model sheet)
const replyMax = () => Math.max(256, Math.round(Number(S.settings.replyMax) || 16384));
const CHARS_PER_TOKEN = 3; // a cautious guess, for checking a message fits before it is sent
const WORDS_PER_TOKEN = 0.75;

// What the model is asked for when a chat is compacted: a summary of the user's share of the context the chat
// holds (15% unless they changed it), never under 150 words or over their ceiling. The length is an aim.
function compactRequest(used) {
  const s = S.settings;
  const aim = Math.round(((used || 0) * (s.compactShare / 100) * WORDS_PER_TOKEN) / 50) * 50;
  // and never more than the room left in the context can take
  const fits = Math.floor((ctxSize() - (used || 0) - 512) * WORDS_PER_TOKEN * 0.9);
  const words = Math.max(100, Math.min(Math.max(150, aim), s.compactMaxWords, fits));
  return (
    'Write a summary of this conversation so far that is enough to carry it on without the messages above. ' +
    'Keep the facts, names, numbers, decisions, what I asked you to do and how, and anything still open. ' +
    `Aim for about ${words} words. Write the summary only.`
  );
}

// The nodes of the reply being written, so each new piece of text touches only them.
let live = null; // { turn, body, think, line }

function render() {
  document.body.classList.toggle('terminal', S.view === 'terminal');
  $('chat').hidden = S.view !== 'chat';
  $('term').hidden = S.view !== 'terminal';
  for (const b of document.querySelectorAll('#view button')) b.classList.toggle('on', b.dataset.v === S.view);
  live = null;
  if (S.view === 'chat') renderChat();
  else renderTerminal();
  renderPane();
  showThinking();
  contextNotice();
  const box = scroller();
  box.scrollTop = box.scrollHeight;
}

// What scrolls: the stage in the chat view, the terminal's own box in the terminal view.
function scroller() {
  return S.view === 'terminal' ? $('term') : $('stage');
}

// How full the chat's context is: its share, and '' (fine), 'warn' or 'full'.
function contextLevel() {
  const share = (S.chat.used || 0) / ctxSize();
  return { share, level: share >= FULL_AT ? 'full' : share >= WARN_AT ? 'warn' : '' };
}

// What to do about a filling context, in words that fit the settings.
function contextAdvice() {
  const { share } = contextLevel();
  if (S.settings.autoCompact && share < COMPACT_AT) return 'It compacts by itself at 90%.';
  return 'Compact it, or start a new chat.';
}

// The warning that the context is filling. In the chat view: a bar over the message box, with a Compact button.
// In both views: the share in the bottom strip. The terminal view also gets a line in the transcript.
function contextNotice() {
  const { share, level } = contextLevel();
  const percent = Math.round(share * 100);
  const note = $('ctxnote');
  note.textContent = level ? `context ${percent}% full` : '';
  note.className = `num ${level}`;
  const bar = $('ctxbar');
  bar.hidden = !level || S.view !== 'chat';
  if (bar.hidden) return;
  bar.className = `ctxbar ${level}`;
  const now = el('button', 'btn', 'Compact');
  now.onclick = () => compact(false);
  bar.replaceChildren(el('span', '', `This chat’s context is ${percent}% full. ${contextAdvice()}`), now);
}

function renderChat() {
  const box = $('chat');
  const chat = S.chat;
  box.replaceChildren();
  if (!chat.turns.length) {
    box.append(emptyState());
    return;
  }
  chat.turns.forEach((turn, i) => {
    if (chat.compact && i === chat.compact.upto) box.append(compactMark(chat.compact));
    box.append(turnNode(turn, chat.compact && i < chat.compact.upto, i === chat.turns.length - 1));
  });
  if (chat.compact && chat.compact.upto >= chat.turns.length) box.append(compactMark(chat.compact));
}

function emptyState() {
  const e = el('div', 'empty');
  const mark = el('img', 'mark-big');
  mark.src = '../assets/mark.png';
  mark.alt = '';
  e.append(mark);
  if (S.loaded) {
    e.append(el('h2', '', 'Model loaded'), el('p', '', 'Write a message below. The model runs on this machine.'));
    return e;
  }
  e.append(el('h2', '', 'Load a model to begin'));
  e.append(el('p', '', 'Pick a .torc file and ToRc starts the engine for it. The model runs on this machine.'));
  const load = el('button', 'btn primary', 'Load a model');
  load.onclick = openLoader;
  e.append(load);
  return e;
}

function turnNode(turn, old, last) {
  const user = turn.role === 'user';
  const d = el('div', `turn ${turn.role}${old ? ' old' : ''}`);
  d.append(el('div', 'who', user ? 'You' : S.chat.model || S.loaded || 'Model'));
  for (const file of turn.files || []) d.append(el('span', 'badge file', file));
  for (const use of turn.used || []) d.append(toolUseNode(use));
  let think = null;
  if (turn.thinking || (last && !user && S.busy)) {
    const folded = el('details', 'think thinking');
    think = document.createTextNode(turn.thinking || '');
    folded.append(el('summary', '', 'Thinking'), think);
    folded.hidden = !turn.thinking;
    // the Thinking folds open and close together, the ones still to come too, and the app remembers which
    folded.open = !!S.settings.thinkingOpen;
    folded.ontoggle = () => foldThinking(folded.open);
    d.append(folded);
  }
  const body = user ? el('div', 'body', turn.text) : rich(turn.text);
  d.append(body);
  if (turn.speeds) d.append(el('div', 'speeds num', turn.speeds));
  if (last && !user && S.busy) live = { turn, body, think, node: d };
  return d;
}

// A tool the model used: one line when its result is short, folded away when it is long.
function toolUseNode(use) {
  const args = use.args && use.args !== '{}' ? ` ${use.args}` : '';
  const head = `used ${use.name}${args}`;
  if (String(use.result).length < 100) return el('div', 'speeds num', `${head} → ${use.result}`);
  const folded = el('details', 'think');
  folded.append(el('summary', '', head), document.createTextNode(use.result));
  return folded;
}

function compactMark(compact) {
  const d = el('details', 'compacted');
  d.append(el('summary', '', 'Compacted here. The model now works from a summary of everything above.'));
  d.append(document.createTextNode(compact.note));
  return d;
}

function renderTerminal() {
  const box = $('term');
  const chat = S.chat;
  box.replaceChildren();
  const line = (cls, text) => {
    const l = el('div', `ln ${cls}`, text);
    box.append(l);
    return l;
  };
  const mark = () => {
    line('d', '-- compacted: the model now works from a summary of everything above --');
    line('d', '');
  };
  line('d', `ToRc ${S.info.version}  ${S.loaded || 'no model loaded'}   /new  /compact  /load  /unload  /primer`);
  line('d', '');
  chat.turns.forEach((turn, i) => {
    if (chat.compact && i === chat.compact.upto) mark();
    if (turn.role === 'user') {
      line('p', `> ${turn.text}`);
      for (const file of turn.files || []) line('d', `  + ${file}`);
      return;
    }
    for (const use of turn.used || []) line('d', `[${use.name}] ${use.result}`);
    const thinking = line('d', turn.thinking || '');
    thinking.hidden = !turn.thinking;
    const text = line('', turn.text);
    const writing = S.busy && i === chat.turns.length - 1;
    if (writing) {
      text.classList.add('cur');
      live = { turn, line: text, thinkLine: thinking };
    }
    if (turn.speeds) line('d', turn.speeds);
    line('d', '');
  });
  if (chat.compact && chat.compact.upto >= chat.turns.length) mark();
  const { share, level } = contextLevel();
  if (level)
    line(
      level === 'full' ? 'b' : 'w',
      `-- context ${Math.round(share * 100)}% full. ${contextAdvice()} /compact or /new --`,
    );
}

// A piece of the reply has arrived: only the reply's own nodes change, and the view follows the text unless the
// user has scrolled up to read.
function onDelta(delta) {
  // the shell's mark that the turn's last piece has been sent (see send)
  if (delta.end) return S.onEnd && S.onEnd();
  const turn = S.chat.turns[S.chat.turns.length - 1];
  if (!turn || turn.role !== 'assistant' || !S.busy) return;
  turn.text += delta.text;
  turn.thinking += delta.thinking;
  setStatus('Writing', true);
  if (!live || live.turn !== turn) return render();
  const stage = scroller();
  const following = stage.scrollHeight - stage.scrollTop - stage.clientHeight < 80;
  if (live.line) {
    live.line.textContent = turn.text;
    live.thinkLine.textContent = turn.thinking;
    live.thinkLine.hidden = !turn.thinking;
  } else {
    const body = rich(turn.text);
    live.body.replaceWith(body);
    live.body = body;
    if (live.think) {
      live.think.textContent = turn.thinking;
      live.think.parentNode.hidden = !turn.thinking;
    }
  }
  if (following) stage.scrollTop = stage.scrollHeight;
}

function onToolUse(use) {
  const turn = S.chat.turns[S.chat.turns.length - 1];
  if (!turn || turn.role !== 'assistant') return;
  turn.used.push(use);
  render();
}

// What the model is sent: the primer first, and after a compact the summary in place of everything before it. The
// summary goes at the head of the first user message, not with the primer: it was written from pages and tool
// results, and the server makes a user's text safe where it leaves the system prompt as it is. It goes in that
// message rather than one of its own, as two user messages in a row can leave a model unsure which to answer.
function history() {
  const chat = S.chat;
  const from = chat.compact ? chat.compact.upto : 0;
  const messages = [];
  const primer = primerOf(chat);
  if (primer) messages.push({ role: 'system', content: primer.text.trim() });
  const head = messages.length;
  for (const turn of chat.turns.slice(from)) {
    // a reply that used tools goes back as it happened: the calls, what the tools said, then the answer
    if (turn.trail && turn.trail.length) {
      messages.push(...turn.trail.map((m) => ({ ...m })));
      messages.push({ role: 'assistant', content: turn.answer || '' });
    } else if (turn.role === 'assistant') {
      // kept even when empty (a reply that ran out of room while thinking), so two user messages never follow each
      // other
      messages.push({ role: 'assistant', content: turn.text || '' });
    } else if (turn.text || turn.content) messages.push({ role: turn.role, content: turn.content || turn.text });
  }
  if (chat.compact) {
    const note = `Summary of the conversation so far:${NL}${chat.compact.note}`;
    const first = messages[head];
    if (!first || first.role !== 'user') messages.splice(head, 0, { role: 'user', content: note });
    else if (Array.isArray(first.content)) first.content = [{ type: 'text', text: note + NL + NL }, ...first.content];
    else first.content = `${note}${NL}${NL}${first.content}`;
  }
  return messages;
}

function command(text) {
  const name = text.split(/\s+/)[0];
  const commands = {
    '/compact': () => compact(false),
    '/new': startNew,
    '/load': openLoader,
    '/unload': unload,
    '/primer': () => openPrimers(),
  };
  if (!commands[name]) return toast(`${name} is not a command. There are /new, /compact, /load, /unload and /primer.`);
  $('input').value = '';
  commands[name]();
}

async function send() {
  const text = $('input').value.trim();
  if (!text || S.busy) return;
  if (S.job) return toast(WAIT_FOR_JOB);
  if (text.startsWith('/')) return command(text);
  if (!S.loaded) return openLoader();
  if (S.asWorker) return toast('This is the second machine: chat on the first one.');

  const chat = S.chat;
  const files = S.files.splice(0);
  const content = files.map((f) => `File ${f.name}:${NL}${f.text}${NL}${NL}`).join('') + text;
  // room for the reply is what is left of the context after this message
  const room = ctxSize() - (chat.used || 0) - Math.ceil(content.length / CHARS_PER_TOKEN) - 256;
  if (room < 256) {
    S.files = files;
    return toast(
      'That is more than this chat’s context has room for. Compact the chat, start a new one, or load the model ' +
      'with a bigger context.',
    );
  }

  chat.turns.push({ role: 'user', text, files: files.map((f) => f.name), content });
  if (!chat.title) chat.title = text.length > 48 ? `${text.slice(0, 47)}…` : text;
  const messages = history();
  const reply = { role: 'assistant', text: '', thinking: '', used: [] };
  chat.turns.push(reply);
  $('input').value = '';
  $('files').replaceChildren();
  setBusy(true, 'Reading the prompt');
  render();

  // the answer can come back before the last pieces of the reply have reached the page: the shell sends a mark after
  // the last one, so the turn ends only once everything before it is in
  const ended = new Promise((resolve) => (S.onEnd = resolve));
  let r;
  try {
    // the Thinking button: as the model's maker set it (nothing sent), on, or off
    const t = S.settings.thinking;
    const think = t === 'on' ? true : t === 'off' ? false : undefined;
    const opts = { max: Math.min(replyMax(), room), ...samplingFor(think !== false) };
    if (think !== undefined) opts.think = think;
    r = await window.torc.chat(messages, opts);
    await Promise.race([ended, new Promise((resolve) => setTimeout(resolve, 2000))]);
  } catch (e) {
    // the shell itself failed: the turn is undone below, as for any failed reply, rather than left busy
    r = { ok: false, error: String(e.message || e) };
  }
  S.onEnd = null;
  setBusy(false, 'Idle');
  // stopped before any answer or tool call came back: kept, it would leave two user messages in a row next time
  const nothing = r.ok && r.stopped && !reply.text && !r.text && !(r.trail && r.trail.length);
  if (!r.ok || nothing) {
    // the turn did not happen: it leaves the chat, and the message goes back in the box to send again
    chat.turns.splice(chat.turns.length - 2, 2);
    if (!chat.turns.length) chat.title = '';
    $('input').value = text;
    // the turn's files go back in front of any attached while it ran
    S.files = files.concat(S.files);
    $('files').replaceChildren(...S.files.map((f) => el('span', 'badge file', f.name)));
    setStatus(nothing ? 'Stopped' : `The engine could not answer: ${r.error}`, false);
    return render();
  }
  // the shell's own copy of the answer is the whole of it, whatever pieces reached the page on the way
  if (!reply.used.length && r.text && r.text.length > reply.text.length) reply.text = r.text;
  // with tools: the calls and their results, and the answer that followed, for the next turn (history)
  if (r.trail && r.trail.length) {
    reply.trail = r.trail;
    reply.answer = r.text || '';
  }
  reply.speeds = speedsLine(r);
  $('speeds').textContent = r.timings ? reply.speeds : '';
  if (r.usage) chat.used = (r.usage.prompt_tokens || 0) + (r.usage.completion_tokens || 0);
  if (r.stopped) setStatus('Stopped', false);
  await keep();
  render();
  const full = chat.used > COMPACT_AT * ctxSize();
  if (!r.stopped && S.settings.autoCompact && full) await compact(true);
  // the model's own requests: compact this chat now, and wake it later
  else if (!r.stopped && r.compact) await compact(false);
  for (const w of r.wake || []) setWake(chat.id, w.minutes, w.message);
}

// A wake-up the model set: at its time the message goes into its chat as a new turn, once nothing else is running.
// At most WAKES_MAX wait at once; they live only as long as the app is open.
const WAKES_MAX = 3;
S.wakes = 0;
function setWake(chatId, minutes, message) {
  if (S.wakes >= WAKES_MAX) return toast('The model asked for a wake-up, but 3 are already waiting');
  S.wakes++;
  const fire = async () => {
    if (S.busy || S.job) return setTimeout(fire, 10000);
    S.wakes--;
    // the model that set it has gone: another one would get a note it never wrote
    if (!S.loaded) return toast('A wake-up was dropped: the model that set it is no longer loaded');
    if (S.chat.id !== chatId) await openChat(chatId);
    if (S.chat.id !== chatId) return toast('A wake-up was dropped: its chat could not be opened');
    // whatever is half typed in the box, and any files waiting to go, wait for the wake-up's turn
    const draft = $('input').value;
    const files = S.files.splice(0);
    const ago = minutes === 1 ? '1 minute' : `${minutes} minutes`;
    $('input').value = `[A wake-up you set ${ago} ago. Your note to yourself:] ${message}`;
    await send();
    if (!$('input').value) $('input').value = draft;
    S.files.push(...files);
  };
  setTimeout(fire, minutes * 60000);
}

// The line under a reply: what was read and written and how fast, by the engine's own count, and how it ended.
function speedsLine(r) {
  const parts = [];
  const t = r.timings;
  if (t) {
    parts.push(`read ${num(t.prompt_n)} new tokens at ${Math.round(t.prompt_per_second)} tok/s`);
    parts.push(`wrote ${num(t.predicted_n)} at ${Math.round(t.predicted_per_second)} tok/s`);
  }
  if (r.stopped) parts.push('stopped');
  if (r.finish === 'length') parts.push('stopped at the length limit');
  if (r.note) parts.push(r.note);
  return parts.join(' · ');
}

function setBusy(busy, status) {
  S.busy = busy;
  $('btnSend').textContent = busy ? 'Stop' : 'Send';
  setStatus(status, busy);
}

// Compact: the model writes a summary of the chat so far, and from then on it is sent the summary in place of the
// messages before it. Nothing is deleted: the whole chat stays on screen and on disk.
async function compact(auto) {
  const chat = S.chat;
  const from = chat.compact ? chat.compact.upto : 0;
  if (S.busy) return;
  if (S.job) return toast(WAIT_FOR_JOB);
  if (!S.loaded) return toast('Load a model first');
  if (chat.turns.length - from < 2) return toast('There is nothing to compact yet');
  setBusy(true, auto ? 'The context is nearly full: compacting' : 'Compacting');
  const messages = history().concat({ role: 'user', content: compactRequest(chat.used) });
  // the summary is written into what is left of the context, without thinking, so it fits even at the auto point
  const room = Math.max(256, ctxSize() - (chat.used || 0) - 512);
  const opts = { quiet: true, plain: true, think: false, max: Math.min(replyMax(), room), ...samplingFor(false) };
  // a failure in the shell itself ends the compact like any other failed one, rather than leaving the app busy
  const r = await window.torc.chat(messages, opts).catch((e) => ({ ok: false, error: String(e.message || e) }));
  setBusy(false, 'Idle');
  const note = r.ok && !r.stopped ? (r.text || '').trim() : '';
  if (note) {
    chat.compact = { upto: chat.turns.length, note, at: Date.now() };
    chat.used = r.usage ? r.usage.completion_tokens || 0 : 0;
    await keep();
    setStatus('Compacted', false);
  } else {
    setStatus(r.ok ? 'The compact did not finish' : `The compact failed: ${r.error}`, false);
  }
  render();
}

// Saves the chat, when saving is on, and brings the list up to date. A save that fails (the folder gone, the file
// locked) is said in the status bar, and the chat stays open as it is.
async function keep() {
  const chat = S.chat;
  chat.updated = Date.now();
  chat.model = S.loaded || chat.model;
  if (!S.settings.saveChats) return;
  try {
    await window.torc.chatPut(chat);
    S.chats = await window.torc.chats();
  } catch (e) {
    setStatus(`Could not save this chat: ${e.message || e}`);
  }
}

// Attaches text files by where they are on disk.
async function attachPaths(paths) {
  for (const p of paths) {
    const r = await window.torc.readText(p);
    if (!r.ok) {
      toast(`${baseName(p)}: ${r.error}`);
      continue;
    }
    S.files.push(r);
    $('files').append(el('span', 'badge file', r.name));
  }
}

// A tool server the user added wants to run a tool: they say yes or no, having seen every argument it would get.
// `done` takes the question down (Stop refused it).
function onAsk(ask) {
  if (ask.done) {
    for (const b of $('asks').querySelectorAll('.ask')) if (b.dataset.id === String(ask.id)) b.remove();
    return;
  }
  const bar = el('div', 'ask');
  bar.dataset.id = String(ask.id);
  bar.append(
    el('span', '', `${ask.server} wants to run ${ask.tool}`),
    el('pre', 'ask-args', JSON.stringify(ask.args || {}, null, 2)),
  );
  const settle = (allowed) => {
    window.torc.answer(ask.id, allowed);
    bar.remove();
  };
  const allow = el('button', 'btn primary', 'Allow');
  allow.onclick = () => settle(true);
  const refuse = el('button', 'btn', 'Refuse');
  refuse.onclick = () => settle(false);
  bar.append(allow, refuse);
  $('asks').append(bar);
}

// The Thinking button's label, from the setting.
function showThinking() {
  const t = (S.settings && S.settings.thinking) || 'default';
  $('btnThink').textContent = `Thinking: ${t}`;
  // ToRc runs Gemma 4 without thinking and LFM2's chat format has no switch for it, so for them it would do nothing
  const arch = (S.modelArch && S.modelArch.arch) || '';
  $('btnThink').hidden = /^(gemma|lfm2)/.test(arch);
}

// Opens or closes every Thinking fold in the chat, and remembers it for the folds still to come.
function foldThinking(open) {
  for (const d of document.querySelectorAll('details.thinking')) if (d.open !== open) d.open = open;
  if (!!S.settings.thinkingOpen === open) return;
  // set here at once, so a fold drawn before the save comes back opens the new way, not the old
  S.settings.thinkingOpen = open;
  setSettings({ thinkingOpen: open }).catch(() => {});
}
