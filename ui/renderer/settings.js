// settings.js - the Settings sheet: chats, appearance, where the engine is, the benchmark and the log.

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
  const modelSettings = el('button', 'btn', 'Open');
  modelSettings.onclick = openLoader;
  const advancedSettings = el('button', 'btn', 'Open');
  advancedSettings.onclick = () => openAdvanced();
  body.append(
    group('Model', [
      row(
        'Context, threads, NPU and GPU',
        'These are set with the model, in Load a model',
        modelSettings,
        'The context, the CPU threads, the NPU and GPU switches and Auto-tune belong to the model you load, so ' +
          'they live on the Load a model sheet. A change there takes effect when you press Load.',
      ),
      row(
        'Advanced settings',
        'The engine’s switches, for the model you load next',
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
  const log = el('button', 'btn', 'Show the log');
  log.onclick = () => showLog();
  body.append(
    group('Engine', [
      row(
        'Engine folder',
        'Where torc-server is',
        engineDir,
        'The folder holding torc-server. After "cargo build --release" that is target/release in the ToRc folder.',
      ),
      row(
        'Port',
        'The local port the engine listens on',
        port,
        'ToRc talks to the engine on this machine only. Change the port if something else is using 8080.',
      ),
      row('Log', 'What the engine and the tool servers have printed', log),
    ]),
  );

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
          'file holds your hardware’s names and the speeds, no host name and no paths.',
      ),
    ]),
  );
}
