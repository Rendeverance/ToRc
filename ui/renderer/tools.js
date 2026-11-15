// tools.js - the Tools sheet: the basic tools or ToolFunnel, and the MCP servers.

function openTools() {
  const s = S.settings;
  const viaToolfunnel = s.tools === 'toolfunnel';
  const body = sheet(
    'Tools',
    'What the model may call while it answers. A tool or a server you add can reach the ' +
      'network or your files; the model itself cannot.',
  );
  body.append(toolsChoice());
  // the built-in tools are there either way: they need nothing installed
  const rows = S.basicTools.map((t) => {
    const sw = toggle(s.basic[t.name] ?? t.on, (v) => setSettings({ basic: { ...S.settings.basic, [t.name]: v } }));
    return row(t.name, t.text, sw);
  });
  body.append(group('Built-in tools', rows));
  if (viaToolfunnel) {
    body.append(toolfunnelGroup());
    if (S.tfTools && S.tfTools.length) body.append(toolfunnelToolsGroup());
    checkToolfunnelUpdate();
    loadToolfunnelLists();
  }
  body.append(serversGroup(viaToolfunnel));
}

// The two ways the model gets its tools.
function toolsChoice() {
  const choices = [
    ['basic', 'Basic tools', 'The built-in tools, and MCP servers connected straight to ToRc.'],
    ['toolfunnel', 'ToolFunnel', 'The built-in tools, and a gateway for many more, with a policy check on every call.'],
  ];
  const pick = el('div', 'pick');
  for (const [value, name, about] of choices) {
    const label = el('label');
    const radio = el('input');
    radio.type = 'radio';
    radio.name = 'tools';
    radio.checked = S.settings.tools === value;
    radio.onchange = async () => {
      await setSettings({ tools: value });
      openTools();
      renderPane();
      refreshServers();
    };
    label.append(radio, el('b', '', name), el('span', '', about));
    pick.append(label);
  }
  return pick;
}

function toolfunnelGroup() {
  const s = S.settings;
  const have = !!S.toolfunnel;
  const state = S.servers.find((k) => k.name === 'toolfunnel');

  const control = el('span', 'ctl');
  if (have && S.toolfunnelLatest && S.toolfunnelLatest !== S.toolfunnel) {
    const update = el('button', 'btn primary', 'Update');
    update.onclick = installToolfunnel;
    control.append(el('span', 'badge warn', `Update available: ${S.toolfunnelLatest}`), update);
  }
  if (have) {
    control.append(
      confirmButton('btn ghost', 'Uninstall', 'Uninstall it?', async () => {
        await window.torc.tfUninstall();
        S.settings = await window.torc.settings();
        await refreshServers();
        renderPane();
        openTools();
      }),
    );
  }
  const install = el('button', have ? 'btn' : 'btn primary', have ? 'Installed' : 'Install ToolFunnel');
  install.disabled = have;
  install.onclick = installToolfunnel;
  control.append(install);

  let running = 'Installs the current release from npm';
  if (have && state && state.state === 'connected')
    running = `Running, ${state.tools.length} tools offered to the model`;
  else if (have && state && state.error) running = `Not running: ${state.error}`;
  else if (have) running = 'Starting';

  const warning = el(
    'div',
    'note warn',
    'Only have this on while you are setting ToolFunnel up. With it on, the model ' +
      'can add tools and change what they are allowed to do.',
  );
  warning.hidden = !s.agentSetup;
  const agent = toggle(s.agentSetup, async (v) => {
    await setSettings({ agentSetup: v });
    warning.hidden = !v;
  });
  return group('ToolFunnel', [
    row(
      have ? `ToolFunnel ${S.toolfunnel}` : 'Not installed',
      running,
      control,
      'ToolFunnel is a separate open-source gateway by the same author. It puts many tools and MCP servers ' +
      'behind one connection and checks every call against a policy before it runs. It is not bundled: ToRc ' +
      'downloads the current release from npm when you ask, checks the download against the hash npm publishes, ' +
      'and keeps it in its own folder in mcps. With ToolFunnel chosen, every MCP server you add on this screen is ' +
      'added to ToolFunnel, and the model reaches it through ToolFunnel.',
    ),
    row(
      'Allow the agent to configure ToolFunnel',
      'Lets the model set ToolFunnel up for you in a chat',
      agent,
      'With this on you can ask the model to add a tool or an MCP server and it does the setup in ToolFunnel ' +
      'itself. Left on, the model could change its own permissions, so turn it off when the setup is done.',
    ),
    [warning],
  ]);
}

// Asks npm, once per sheet, whether there is a newer ToolFunnel than the installed one.
function checkToolfunnelUpdate() {
  if (!S.toolfunnel) return;
  window.torc.tfLatest().then((version) => {
    if (!version || version === S.toolfunnelLatest) return;
    S.toolfunnelLatest = version;
    if (sheetIs('Tools')) openTools();
  });
}

async function installToolfunnel() {
  const body = sheet(
    'Install ToolFunnel',
    'Downloads the current release from npm, checks it against the hash npm ' +
      'publishes, and unpacks it into its own folder in mcps.',
    true,
  );
  const log = el('pre', 'joblog');
  body.append(log);
  S.onJobLine = (text) => {
    log.textContent += text;
  };
  const r = await window.torc.tfInstall();
  S.onJobLine = null;
  const acts = el('div', 'row-actions');
  const back = el('button', 'btn primary', 'Back to Tools');
  back.onclick = openTools;
  acts.append(back);
  body.append(acts);
  if (!r.ok) return;
  S.settings = await window.torc.settings();
  await refreshServers();
}

// Runs one of ToolFunnel's management tools for the user and returns what the tool printed, parsed.
async function tfRun(name, args) {
  const r = await window.torc.tfManage(name, args);
  if (!r.ok) return { ok: false, error: r.error || 'ToolFunnel refused it' };
  try {
    return JSON.parse(JSON.parse(r.text).stdout);
  } catch (_) {
    return { ok: false, error: 'ToolFunnel gave an answer ToRc could not read' };
  }
}

// What ToolFunnel holds, its own tools and the servers behind it, read from ToolFunnel each time the sheet opens.
async function loadToolfunnelLists() {
  const tf = S.servers.find((k) => k.name === 'toolfunnel' && k.state === 'connected');
  if (!S.toolfunnel || !tf) return;
  const [tools, mcps] = await Promise.all([tfRun('tf_list', { kind: 'tools' }), tfRun('tf_list', { kind: 'mcps' })]);
  const next = {
    tools: tools.ok ? tools.items.filter((t) => t.category !== 'management') : [],
    mcps: mcps.ok ? mcps.items : [],
  };
  if (JSON.stringify(next) === JSON.stringify({ tools: S.tfTools, mcps: S.tfMcps })) return;
  S.tfTools = next.tools;
  S.tfMcps = next.mcps;
  if (sheetIs('Tools')) openTools();
}

// ToolFunnel's own tools, each one on or off for the model.
function toolfunnelToolsGroup() {
  const rows = S.tfTools.map((t) => {
    const sw = toggle(t.enabled !== false, async (v) => {
      const r = await tfRun('tf_tool_set', { id: t.id, action: v ? 'enable' : 'disable' });
      if (!r.ok) return toast(r.error);
      t.enabled = v;
    });
    return row(t.name, t.summary, sw);
  });
  return group('ToolFunnel tools', rows);
}

// A server behind ToolFunnel: on or off, its tools, and Remove.
function toolfunnelServerRow(u) {
  const control = el('span', 'ctl');
  const tools = (u.exposed || []).map((e) => e.name || e.tool).join(', ');
  const sw = toggle(u.enabled !== false, async (v) => {
    const r = await tfRun('tf_mcp_set', { id: u.id, action: v ? 'enable' : 'disable' });
    if (!r.ok) return toast(r.error);
    u.enabled = v;
  });
  const remove = el('button', 'btn ghost', 'Remove');
  remove.onclick = () => removeServer({ name: u.id, via: 'ToolFunnel' });
  control.append(sw, remove);
  return row(u.id, tools ? `Tools: ${tools}` : 'No tools exposed', control);
}

// The MCP servers that belong to the chosen way of working, and the form that adds one. Through ToolFunnel the list
// is ToolFunnel's own, whoever added the server: the user here, or the agent in a chat.
function serversGroup(viaToolfunnel) {
  const mine = S.settings.mcp.filter((m) => (m.via === 'direct') !== viaToolfunnel);
  const rows = viaToolfunnel && S.tfMcps ? S.tfMcps.map((u) => toolfunnelServerRow(u)) : mine.map((m) => serverRow(m));

  const form = el('div', 'row addform');
  form.hidden = true;
  const name = field('text', '', () => {}, { placeholder: 'Name' });
  const example = 'Command or URL, for example: npx -y @modelcontextprotocol/server-filesystem "C:\\My Documents"';
  const target = field('text', '', () => {}, { placeholder: example });
  const add = el('button', 'btn primary', 'Add');
  add.onclick = () => addServer(name.value.trim(), target.value.trim(), viaToolfunnel);
  form.append(name, target, add);

  const show = el('button', 'btn', 'Add a server');
  show.onclick = () => {
    form.hidden = !form.hidden;
    if (!form.hidden) name.focus();
  };
  const how = viaToolfunnel
    ? 'Added to ToolFunnel, with its policy checks'
    : 'Connected straight to ToRc. You approve each call';
  rows.push(
    row(
      rows.length ? 'Another server' : 'No MCP servers yet',
      how,
      show,
      'An MCP server gives the model more tools: files, a browser, a database. With Basic tools chosen a server ' +
      'is connected straight to ToRc, runs with your rights, and ToRc asks you before each call it makes. With ' +
      'ToolFunnel chosen it is added to ToolFunnel, which checks each call against its policy.',
    ),
  );
  rows.push([form]);
  return group('MCP servers', rows);
}

function serverRow(m) {
  const control = el('span', 'ctl');
  const state = S.servers.find((k) => k.name === m.name);
  if (m.via !== 'direct') control.append(el('span', 'badge', 'through ToolFunnel'));
  else if (!state || state.state === 'connecting') control.append(el('span', 'badge', 'connecting'));
  else if (state.state === 'connected') control.append(el('span', 'badge good', `${state.tools.length} tools`));
  else {
    const failed = el('span', 'badge warn', 'not connected');
    failed.title = state.error;
    control.append(failed);
  }
  const remove = el('button', 'btn ghost', 'Remove');
  remove.onclick = () => removeServer(m);
  control.append(remove);
  return row(m.name, m.target, control);
}

async function addServer(name, target, viaToolfunnel) {
  if (!name || !target) return toast('Give the server a name and its command or URL');
  if (S.settings.mcp.some((m) => m.name === name)) return toast('There is already a server with that name');
  if (viaToolfunnel) {
    if (!S.toolfunnel) return toast('Install ToolFunnel first');
    if (/^https?:/i.test(target))
      return toast('Through ToolFunnel a server is added by its command. Add a URL server with Basic tools chosen.');
    // ToolFunnel shows only the tools it is told to expose: the server is started once to ask which it has
    const found = await window.torc.mcpDiscover(target);
    if (!found.ok) return toast(found.error || 'The server did not start');
    const expose = found.tools.map((tool) => ({ tool }));
    // and it runs only what is inside its own folder: the server's script is copied in there
    const placed = await window.torc.tfPlace(name, target);
    if (!placed.ok) return toast(placed.error);
    const add = { id: name, command: placed.command, args: placed.args, expose };
    const r = await window.torc.tfManage('tf_mcp_add', add);
    if (!r.ok) return toast(r.error || 'ToolFunnel refused it');
  }
  const via = viaToolfunnel ? 'ToolFunnel' : 'direct';
  await setSettings({ mcp: [...S.settings.mcp, { name, target, via }] });
  S.tfMcps = null;
  openTools();
  refreshServers();
}

async function removeServer(m) {
  if (m.via !== 'direct') {
    const r = await window.torc.tfManage('tf_mcp_set', { id: m.name, action: 'remove' });
    if (!r.ok) return toast(r.error || 'ToolFunnel refused it');
  }
  await setSettings({ mcp: S.settings.mcp.filter((k) => k !== m && k.name !== m.name) });
  S.tfMcps = null;
  openTools();
  refreshServers();
}

// Brings the tool servers in line with the settings and redraws the sheet if it is open.
async function refreshServers() {
  S.toolfunnel = await window.torc.tfHave();
  S.servers = await window.torc.mcp();
  if (sheetIs('Tools')) openTools();
}
