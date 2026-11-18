// advanced.js - the Advanced sheet: every switch the engine has, read from SWITCHES.md, the useful ones first.

// The settings most worth a look, in the order they are listed. The rest of the settings follow in the file's order.
const FIRST = [
  'TORC_MTP',
  'TORC_DRAFT',
  'TORC_ATTN_FAST',
  'TORC_PIN',
  'TORC_DECODE_THREADS',
  'TORC_BLOCK',
  'TORC_BLOCK_LONG',
  'TORC_CACHE_MB',
  'TORC_WARM',
  'TORC_NPU',
  'TORC_NPU_HOT',
  'TORC_GPU_ATTN',
  'TORC_GPU',
  'TORC_GPU_SHARE',
  'TORC_LARGE_PAGES',
  'TORC_PIPE_BIND',
  'TORC_MARKS',
  'TORC_IDLE_SAVE',
];
const SECTIONS = [
  ['setting', 'Settings'],
  ['diagnostic', 'Diagnostics'],
  ['development', 'Left from development'],
];
const KIND_BADGE = { setting: 'badge accent', diagnostic: 'badge warn', development: 'badge' };

// `openName`: a switch to show with its explanation already open.
function openAdvanced(openName) {
  const lead =
    'Every switch the engine has, the useful ones first. What you set here applies the next time a model ' +
    'loads. Leave a box empty for the default.';
  const body = sheet('Advanced settings', lead, true);
  const search = el('input');
  search.placeholder = 'Search switches';
  const list = el('div');
  const draw = () => list.replaceChildren(...switchGroups(search.value.trim().toLowerCase(), openName));
  const reset = confirmButton('btn', 'Reset all', 'Reset every switch?', async () => {
    await setSettings({ env: {} });
    draw();
    toast('Every switch is back to its default');
  });
  const bar = el('div', 'search');
  bar.append(search, reset);
  search.oninput = draw;
  body.append(bar, list);
  draw();
}

// The switches that match the search, grouped by kind.
function switchGroups(wanted, openName) {
  const rank = (w) => {
    const at = FIRST.indexOf(w.name);
    return at < 0 ? FIRST.length : at;
  };
  const groups = [];
  for (const [kind, title] of SECTIONS) {
    const found = S.switches.filter(
      (w) => w.kind === kind && (!wanted || `${w.name} ${w.text}`.toLowerCase().includes(wanted)),
    );
    if (!found.length) continue;
    if (kind === 'setting') found.sort((a, b) => rank(a) - rank(b));
    groups.push(
      group(
        `${title} (${found.length})`,
        found.map((w) => switchRow(w, w.name === openName)),
      ),
    );
  }
  return groups;
}

function switchRow(w, open) {
  const env = S.settings.env;
  const box = field(
    'text',
    env[w.name] ?? '',
    async (typed) => {
      // " 1" would not count as on in the engine
      const value = typed.trim();
      const next = { ...S.settings.env };
      if (value === '') delete next[w.name];
      else next[w.name] = value;
      await setSettings({ env: next });
      box.classList.toggle('set', value !== '');
    },
    { placeholder: 'default' },
  );
  box.className = env[w.name] === undefined ? 'envbox' : 'envbox set';
  const control = el('span', 'ctl');
  control.append(el('span', KIND_BADGE[w.kind] || 'badge', w.kind), box);

  const text = w.text.replace(/`/g, '');
  const others = w.row.filter((n) => n !== w.name);
  const fallback = others.length ? `${text}. Set together with: ${others.join(', ')}.` : text;
  // a switch that is not a setting also says which part of the engine it belongs to
  const hint = w.kind === 'setting' ? text : `${w.group}: ${text}`;
  const rows = row(w.name, hint, control, window.EXPLAIN[w.name] || fallback);
  rows[0].querySelector('b').classList.add('mono');
  if (open) {
    rows[1].hidden = false;
    rows[0].querySelector('.info').classList.add('on');
    setTimeout(() => rows[0].scrollIntoView({ block: 'center' }), 0);
  }
  return rows;
}
