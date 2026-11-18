// bridge.js - window.torc for the page: each named call goes to the Rust shell, and its events come back the same way.
// Run before the page's own scripts.
(() => {
  const CALLS = [
    'info', 'settings', 'settingsSet', 'switches', 'pick', 'readText', 'load', 'unload', 'chat', 'answer', 'job',
    'chats', 'chatGet', 'chatPut', 'chatDel', 'primers', 'primerPut', 'primerDel', 'tools', 'mcp', 'tfHave',
    'tfLatest', 'tfInstall', 'tfUninstall', 'tfManage', 'addresses', 'mcpDiscover', 'tfPlace', 'ggufCheck',
    'modelArch', 'saveText', 'engineSettings', 'engineSettingsSet',
  ];
  const EVENTS = ['delta', 'tooluse', 'log', 'state', 'joblog', 'ask'];
  const tauri = () => window.__TAURI__;
  const api = {
    stop: () => tauri().core.invoke('stop'),
    jobStop: () => tauri().core.invoke('job_stop'),
    // a dropped file's path comes with Tauri's own drop event, not with the File object
    pathOf: () => '',
    on: (name, handler) => {
      if (EVENTS.includes(name)) tauri().event.listen(`torc:${name}`, (e) => handler(e.payload));
    },
  };
  // the commands take one argument, `args`: the call's arguments as a list
  for (const name of CALLS) api[name] = (...args) => tauri().core.invoke('call', { name, args });
  window.torc = api;
  // a file dropped on the window comes as its path, from Tauri's own drop event, and is attached as the page does it
  const onDrop = () => tauri().event.listen('torc:drop', (e) => window.attachPaths && window.attachPaths(e.payload));
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', onDrop);
  else onDrop();
})();
