// app.js - the page starts here: it asks the shell for its state, wires the buttons and the shell's events, and
// draws the first screen.

async function start() {
  const torc = window.torc;
  [S.info, S.settings, S.switches, S.basicTools, S.chats, S.primers] = await Promise.all([
    torc.info(),
    torc.settings(),
    torc.switches(),
    torc.tools(),
    torc.chats(),
    torc.primers(),
  ]);
  S.view = S.settings.firstView || 'chat';
  S.chat = newChat();

  torc.on('delta', onDelta);
  torc.on('tooluse', onToolUse);
  torc.on('ask', onAsk);
  torc.on('joblog', (text) => {
    if (S.onJobLine) S.onJobLine(text);
  });
  torc.on('log', (text) => {
    S.log.push(text);
    if (S.log.length > 2000) S.log.splice(0, S.log.length - 2000);
  });
  // the engine has stopped by itself
  torc.on('state', () => {
    S.loaded = null;
    S.asWorker = false;
    setStatus('The engine stopped. Settings has its log', false);
    render();
  });

  $('btnLoad').onclick = openLoader;
  $('btnPrimers').onclick = () => openPrimers();
  $('btnTools').onclick = openTools;
  $('btnSettings').onclick = openSettings;
  $('btnSend').onclick = () => (S.busy ? torc.stop() : send());
  $('btnAttach').onclick = async () => attachPaths(await torc.pick('text'));
  $('btnThink').onclick = () => {
    const next = { default: 'on', on: 'off', off: 'default' }[S.settings.thinking || 'default'];
    // set here at once, so a second press before the save comes back moves on from this one, not the old
    S.settings.thinking = next;
    showThinking();
    setSettings({ thinking: next }).catch(() => {});
  };
  showThinking();
  $('input').onkeydown = (event) => {
    if (event.key !== 'Enter' || event.shiftKey) return;
    event.preventDefault();
    send();
  };
  for (const b of document.querySelectorAll('#view button')) {
    b.onclick = () => {
      S.view = b.dataset.v;
      render();
    };
  }
  // a file dropped anywhere on the window is attached
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => {
    event.preventDefault();
    attachPaths([...event.dataTransfer.files].map((f) => torc.pathOf(f)).filter(Boolean));
  });

  setStatus('Idle', false);
  render();
  refreshServers();
}

start();
