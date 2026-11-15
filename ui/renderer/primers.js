// primers.js - the Primers sheet. A primer is what the model is told before a chat starts. A chat refers to its primer
// by its id, so editing a primer changes it for every chat that uses it.

const primerOf = (chat) => S.primers.find((p) => p.id === chat.primerId) || null;

// `editId`: the primer to open in the editor; none for a new one.
function openPrimers(editId) {
  const lead =
    'A primer is what the model is told before the chat starts: who it is, how to answer, what to keep in ' +
    'mind. A chat uses its primer as it is now, so an edit here reaches every chat that uses it.';
  const body = sheet('Primers', lead, true);
  const rows = S.primers.map((p) => primerRow(p));
  rows.unshift(primerRow(null));
  body.append(group('Your primers', rows));
  body.append(...primerEditor(S.primers.find((p) => p.id === editId) || { id: '', name: '', text: '' }));
}

// One primer in the list, or the "no primer" row when `p` is null.
function primerRow(p) {
  const id = p ? p.id : '';
  const chat = S.chat;
  const control = el('span', 'ctl');
  const forChat = (chat.primerId || '') === id;
  const forNew = (S.settings.primer || '') === id;
  if (forNew) control.append(el('span', 'badge accent', 'new chats'));
  if (forChat) control.append(el('span', 'badge good', 'this chat'));
  if (!forChat) {
    const use = el('button', 'btn', 'Use in this chat');
    use.onclick = () => {
      chat.primerId = id;
      if (chat.turns.length) keep();
      openPrimers();
      renderPane();
    };
    control.append(use);
  }
  if (!forNew) {
    const use = el('button', 'btn', 'Use for new chats');
    use.onclick = async () => {
      await setSettings({ primer: id });
      openPrimers();
    };
    control.append(use);
  }
  if (!p) return row('No primer', 'The model with nothing in front of it', control);
  const edit = el('button', 'btn', 'Edit');
  edit.onclick = () => openPrimers(p.id);
  control.append(edit);
  return row(p.name, p.text.replace(/\s+/g, ' ').slice(0, 110), control);
}

// The editor under the list: a name, the text, and what can be done with them.
function primerEditor(primer) {
  const name = field('text', primer.name, () => {}, { placeholder: 'Name' });
  name.className = 'primer-name';
  const text = el('textarea', 'primer-text');
  text.value = primer.text;
  text.placeholder = 'Write the primer here, or load one from a file.';

  const load = el('button', 'btn', 'Load from a file');
  load.onclick = async () => {
    const [picked] = await window.torc.pick('text');
    if (!picked) return;
    const r = await window.torc.readText(picked);
    if (!r.ok) return toast(r.error);
    text.value = r.text;
    if (!name.value) name.value = r.name.replace(/\.[^.]+$/, '');
  };
  const save = el('button', 'btn primary', primer.id ? 'Save' : 'Add');
  save.onclick = async () => {
    if (!name.value.trim() || !text.value.trim()) return toast('A primer needs a name and some text');
    await window.torc.primerPut({ id: primer.id, name: name.value.trim(), text: text.value });
    S.primers = await window.torc.primers();
    openPrimers();
    renderPane();
  };
  const acts = el('div', 'row-actions');
  acts.append(load);
  if (primer.id) {
    acts.append(
      confirmButton('btn', 'Delete', 'Delete it?', async () => {
        await window.torc.primerDel(primer.id);
        S.primers = await window.torc.primers();
        if (S.settings.primer === primer.id) await setSettings({ primer: '' });
        openPrimers();
        renderPane();
      }),
    );
  }
  acts.append(save);

  const box = el('div', 'rows primer-edit');
  box.append(name, text);
  return [el('h3', '', primer.id ? 'Edit this primer' : 'A new primer'), box, acts];
}
