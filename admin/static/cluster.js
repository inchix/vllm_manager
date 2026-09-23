/* ============================================================================
 * cluster.js — control-plane cluster UI for vllm-multi-gpu (v0.4.0)
 *
 * Renders two panels on cluster.html:
 *   (A) LIVE CLUSTER MONITOR — nodes grouped by host, role + state badges,
 *       per-GPU util/mem/temp/power, replicas and mounts. Polls every ~3s.
 *   (B) CLUSTER CONFIGURATION — cluster defaults + one card per node, each
 *       setting showing DETECTED (read-only) vs OVERRIDE (input), a merged
 *       effective-config preview, and a Save button.
 *
 * ---------------------------------------------------------------------------
 * API CONTRACT (assumed — endpoints may not exist yet; the UI degrades
 * gracefully to "control plane offline" on 404/network error):
 *
 *   GET /api/cluster/nodes  ->  [
 *     {
 *       node_id, hostname, roles:[...], state,   // state: READY|SERVING|DEGRADED|DOWN
 *       gpus:    [{ index, model, util, mem_used, mem_total, temp,
 *                   power_draw, power_limit }],
 *       replicas:[{ id, state, port }],
 *       mounts:  [{ path, source, ok }]
 *     }, ...
 *   ]
 *
 *   GET /api/cluster/config ->  {
 *     defaults: { <key>: <value>, ... },
 *     nodes: {
 *       <node_id>: { detected:{...}, overrides:{...}, effective:{...} }
 *     }
 *   }
 *
 *   POST /api/cluster/config   body:  {
 *     defaults: {...},
 *     nodes: { <node_id>: { overrides: {...} } }
 *   }
 *
 * Every request carries the admin key in an `X-API-Key` header (see apiFetch).
 * A 401 redirects to /login, matching the existing admin UI.
 *
 * ?mock=1  renders representative sample data (COVID {admin,participant,
 *          storage} + ebola {participant}) so the UI can be reviewed with no
 *          backend running.
 * ==========================================================================*/

const $ = id => document.getElementById(id);
const MOCK = new URLSearchParams(location.search).has('mock');
const POLL_MS = 3000;

/* -------------------------------------------------------------------------
 * Auth-aware fetch. Mirrors index.html's apiFetch (401 -> /login) and adds
 * the admin key as X-API-Key on every request, sourced from the key input
 * (persisted in localStorage so the standalone page is usable on its own).
 * ---------------------------------------------------------------------- */
function getApiKey() {
  try { return localStorage.getItem('adminApiKey') || ''; } catch (e) { return ''; }
}
function setApiKey(k) {
  try { k ? localStorage.setItem('adminApiKey', k) : localStorage.removeItem('adminApiKey'); } catch (e) {}
}

async function apiFetch(url, opts) {
  opts = opts || {};
  const headers = Object.assign({}, opts.headers);
  const key = getApiKey();
  if (key) headers['X-API-Key'] = key;
  const res = await fetch(url, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    // Fall through to the login page like the rest of the admin UI.
    window.location.href = '/login';
    throw new Error('unauthorized');
  }
  return res;
}

/* -------------------------------------------------------------------------
 * Small shared helpers (toast, theme, escaping) — kept consistent with
 * index.html so behaviour matches the rest of the UI.
 * ---------------------------------------------------------------------- */
function showToast(msg, type) {
  type = type || 'error';
  const c = $('toast-container');
  const t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = typeof msg === 'string' ? msg : JSON.stringify(msg);
  const dismiss = () => { t.style.animation = 'toast-out 0.2s ease forwards'; setTimeout(() => t.remove(), 200); };
  t.onclick = dismiss;
  c.appendChild(t);
  setTimeout(() => { if (t.parentNode) dismiss(); }, 5000);
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML;
}

function toggleTheme() {
  document.body.classList.toggle('light-theme');
  const isLight = document.body.classList.contains('light-theme');
  $('theme-toggle').textContent = isLight ? 'Dark Mode' : 'Light Mode';
  try { localStorage.setItem('theme', isLight ? 'light' : 'dark'); } catch (e) {}
}
(function initTheme() {
  try {
    if (localStorage.getItem('theme') === 'light') {
      document.body.classList.add('light-theme');
      const b = $('theme-toggle'); if (b) b.textContent = 'Dark Mode';
    }
  } catch (e) {}
})();

/* -------------------------------------------------------------------------
 * Tab switching.
 * ---------------------------------------------------------------------- */
function showTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
  if (name === 'config' && !configLoaded) loadConfig();
}

/* =========================================================================
 * SETTINGS CATALOG (from docs/07-configuration.md)
 * Each field: key, label, type (text|number|select|checkbox), options?,
 * placeholder?, group. `scope` marks whether a field appears in the
 * cluster-defaults section ('cluster'), per-node cards ('node'), or both.
 * ======================================================================= */
const SETTING_GROUPS = [
  {
    id: 'identity', title: 'Identity & roles',
    fields: [
      { key: 'roles', label: 'roles', type: 'text', placeholder: 'admin,participant,storage', scope: 'node' },
      { key: 'CLUSTER_ID', label: 'CLUSTER_ID', type: 'text', placeholder: 'default', scope: 'both' },
      { key: 'RAY_NODE_IP', label: 'RAY_NODE_IP', type: 'text', placeholder: 'fabric/mgmt IP', scope: 'node' },
      { key: 'RAY_HEAD_HOST', label: 'RAY_HEAD_HOST', type: 'text', placeholder: 'head host:port', scope: 'node' },
    ],
  },
  {
    id: 'fabric', title: 'Fabric / NCCL',
    fields: [
      { key: 'NCCL_SOCKET_IFNAME', label: 'NCCL_SOCKET_IFNAME', type: 'text', placeholder: 'ens2,enp196s0 (local first)', scope: 'node' },
      { key: 'GLOO_SOCKET_IFNAME', label: 'GLOO_SOCKET_IFNAME', type: 'text', placeholder: 'single local NIC', scope: 'node' },
      { key: 'NCCL_IB_HCA', label: 'NCCL_IB_HCA', type: 'text', placeholder: 'mlx4_0:1', scope: 'node' },
      { key: 'NCCL_IB_GID_INDEX', label: 'NCCL_IB_GID_INDEX', type: 'number', placeholder: 'e.g. 3', scope: 'node' },
      { key: 'NCCL_IB_DISABLE', label: 'NCCL_IB_DISABLE', type: 'select', options: ['', '0', '1'], scope: 'node' },
      { key: 'NCCL_P2P_DISABLE', label: 'NCCL_P2P_DISABLE', type: 'select', options: ['', '0', '1'], scope: 'node' },
    ],
  },
  {
    id: 'execution', title: 'Execution / vLLM',
    fields: [
      { key: 'gpu_memory_utilization', label: 'gpu_memory_utilization', type: 'number', placeholder: '0.85', scope: 'both' },
      { key: 'MULTIGPU_EXECUTOR', label: 'MULTIGPU_EXECUTOR', type: 'select', options: ['', 'mp', 'ray'], scope: 'both' },
      { key: 'DISABLE_CUSTOM_ALL_REDUCE', label: 'DISABLE_CUSTOM_ALL_REDUCE', type: 'select', options: ['', '0', '1'], scope: 'both' },
      { key: 'enforce_eager', label: 'enforce_eager', type: 'select', options: ['', 'true', 'false'], scope: 'both' },
      { key: 'dtype', label: 'dtype', type: 'select', options: ['', 'auto', 'float16', 'bfloat16'], scope: 'both' },
      { key: 'max_model_len', label: 'max_model_len', type: 'number', placeholder: 'auto', scope: 'both' },
      { key: 'port_range', label: 'port range', type: 'text', placeholder: '8000-8099', scope: 'both' },
      { key: 'compile_cache_dir', label: 'compile cache dir', type: 'text', placeholder: '/var/cache/vllm-compile', scope: 'both' },
    ],
  },
  {
    id: 'storage', title: 'Storage',
    fields: [
      { key: 'storage_export_dir', label: 'export dir (storage)', type: 'text', placeholder: '/export/models', scope: 'node' },
      { key: 'storage_listen', label: 'listen fabric IP:port', type: 'text', placeholder: '10.0.0.1:2049', scope: 'node' },
      { key: 'storage_allow', label: 'client allow-list', type: 'text', placeholder: '10.0.0.0/24', scope: 'node' },
      { key: 'storage_readahead', label: 'readahead (KB)', type: 'number', placeholder: '4096', scope: 'node' },
      { key: 'mount_canonical_path', label: 'client mount path', type: 'text', placeholder: '/export/models', scope: 'node' },
      { key: 'mount_opts', label: 'mount opts', type: 'text', placeholder: 'ro,vers=4.2', scope: 'node' },
      { key: 'mount_transport', label: 'transport', type: 'select', options: ['', 'tcp', 'rdma'], scope: 'node' },
    ],
  },
  {
    id: 'hardware', title: 'Hardware guards',
    fields: [
      { key: 'gpu_power_cap_w', label: 'GPU power cap (W)', type: 'number', placeholder: 'e.g. 300', scope: 'node' },
      { key: 'gpu_persistence_mode', label: 'persistence mode', type: 'checkbox', scope: 'node' },
    ],
  },
  {
    id: 'control', title: 'Control plane',
    fields: [
      { key: 'CCP_HEARTBEAT_SEC', label: 'CCP_HEARTBEAT_SEC', type: 'number', placeholder: '5', scope: 'cluster' },
      { key: 'CCP_TELEMETRY_SEC', label: 'CCP_TELEMETRY_SEC', type: 'number', placeholder: '10', scope: 'cluster' },
      { key: 'CCP_HEARTBEAT_MISS', label: 'CCP_HEARTBEAT_MISS', type: 'number', placeholder: '3', scope: 'cluster' },
    ],
  },
];

// Flat lookup for effective-config source labelling.
const ALL_FIELDS = SETTING_GROUPS.flatMap(g => g.fields);

/* =========================================================================
 * (A) LIVE CLUSTER MONITOR
 * ======================================================================= */
let monitorTimer = null;

async function loadNodes() {
  if (MOCK) { renderMonitor(MOCK_NODES); return; }
  try {
    const res = await apiFetch('/api/cluster/nodes');
    if (res.status === 404) { renderOffline(); return; }
    if (!res.ok) { renderOffline('control plane returned ' + res.status); return; }
    const nodes = await res.json();
    renderMonitor(Array.isArray(nodes) ? nodes : []);
  } catch (e) {
    // Network error or endpoint missing — degrade, don't throw.
    renderOffline();
  }
}

function renderOffline(detail) {
  const el = $('monitor-content');
  el.innerHTML =
    '<div class="banner banner-offline">Control plane offline' +
    (detail ? ' — ' + escapeHtml(detail) : '') +
    '. No telemetry available. (Open with <code>?mock=1</code> to preview with sample data.)</div>';
}

function stateClass(state) {
  const s = (state || '').toLowerCase();
  if (['ready', 'serving', 'degraded', 'down'].includes(s)) return 'state-' + s;
  return 'state-unknown';
}

function tempClass(t) {
  if (t == null) return 'temp-ok';
  if (t >= 85) return 'temp-hot';
  if (t >= 70) return 'temp-warm';
  return 'temp-ok';
}

function pct(a, b) { return b > 0 ? Math.max(0, Math.min(100, (a / b) * 100)) : 0; }
function fmtMb(mb) {
  if (mb == null) return '—';
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB';
}

function renderMonitor(nodes) {
  const el = $('monitor-content');
  if (!nodes.length) {
    el.innerHTML = '<div class="banner banner-empty">No nodes registered with the control plane yet.</div>';
    return;
  }

  // Group by hostname (a host may register more than one node record).
  const byHost = {};
  nodes.forEach(n => {
    const h = n.hostname || n.node_id || 'unknown';
    (byHost[h] = byHost[h] || []).push(n);
  });

  el.innerHTML = Object.keys(byHost).map(host => {
    const hostNodes = byHost[host];
    return hostNodes.map(n => renderNode(host, n)).join('');
  }).join('');
}

function renderNode(host, n) {
  const roles = (n.roles || []).map(r => `<span class="badge role-badge">${escapeHtml(r)}</span>`).join('');
  const st = (n.state || 'unknown').toUpperCase();
  const stCls = stateClass(n.state);
  const serving = stCls === 'state-serving' ? ' state-serving-anim' : '';
  const stateBadge = `<span class="badge state-badge ${stCls}${serving}"><span class="state-dot"></span>${escapeHtml(st)}</span>`;

  const gpuRows = (n.gpus || []).length
    ? `<div class="gpu-table">
         <div class="gpu-row header"><span>#</span><span>Model</span><span>Utilisation</span><span>Memory</span><span>Temp</span><span>Power</span></div>
         ${(n.gpus || []).map(renderGpuRow).join('')}
       </div>`
    : '<div class="muted">No GPU telemetry.</div>';

  const replicas = (n.replicas || []).length
    ? `<div class="sub-section"><div class="sub-title">Replicas</div><div class="chip-row">${
        n.replicas.map(r => `<span class="chip">${escapeHtml(r.id)} <span class="dim">${escapeHtml(r.state || '')}${r.port ? ' :' + r.port : ''}</span></span>`).join('')
      }</div></div>`
    : '';

  const mounts = (n.mounts || []).length
    ? `<div class="sub-section"><div class="sub-title">Mounts</div><div class="chip-row">${
        n.mounts.map(m => `<span class="chip"><span class="${m.ok ? 'ok' : 'bad'}">${m.ok ? '✔' : '✗'}</span> ${escapeHtml(m.path)} <span class="dim">${escapeHtml(m.source || '')}</span></span>`).join('')
      }</div></div>`
    : '';

  return `
    <div class="host-group">
      <div class="host-head">
        <span class="host-name">${escapeHtml(host)}</span>
        <span class="host-id">${escapeHtml(n.node_id || '')}</span>
        ${roles}
        ${stateBadge}
      </div>
      ${gpuRows}
      ${replicas}
      ${mounts}
    </div>`;
}

function renderGpuRow(g) {
  const utilPct = g.util == null ? 0 : Math.max(0, Math.min(100, g.util));
  const memPct = pct(g.mem_used, g.mem_total);
  const powerPct = pct(g.power_draw, g.power_limit);
  return `
    <div class="gpu-row">
      <span class="gpu-idx">${escapeHtml(g.index)}</span>
      <span class="gpu-model" title="${escapeHtml(g.model || '')}">${escapeHtml(g.model || '—')}</span>
      <span class="metric">
        <span class="metric-val">${g.util == null ? '—' : utilPct.toFixed(0) + '%'}</span>
        <span class="bar"><span class="bar-fill bar-util" style="width:${utilPct}%"></span></span>
      </span>
      <span class="metric">
        <span class="metric-val">${fmtMb(g.mem_used)} <span class="unit">/ ${fmtMb(g.mem_total)}</span></span>
        <span class="bar"><span class="bar-fill bar-mem" style="width:${memPct}%"></span></span>
      </span>
      <span class="metric-val ${tempClass(g.temp)}">${g.temp == null ? '—' : g.temp + '°C'}</span>
      <span class="metric-val">${g.power_draw == null ? '—' : Math.round(g.power_draw) + ' <span class="unit">/ ' + (g.power_limit == null ? '?' : Math.round(g.power_limit)) + ' W</span>'}
        ${g.power_limit ? `<span class="bar"><span class="bar-fill bar-util" style="width:${powerPct}%"></span></span>` : ''}
      </span>
    </div>`;
}

/* =========================================================================
 * (B) CLUSTER CONFIGURATION
 * ======================================================================= */
let configLoaded = false;
let configState = { defaults: {}, nodes: {} };   // authoritative snapshot from server/mock

async function loadConfig() {
  if (MOCK) { configState = deepClone(MOCK_CONFIG); renderConfig(); configLoaded = true; return; }
  try {
    const res = await apiFetch('/api/cluster/config');
    if (res.status === 404) { renderConfigOffline(); return; }
    if (!res.ok) { renderConfigOffline('control plane returned ' + res.status); return; }
    const data = await res.json();
    configState = { defaults: data.defaults || {}, nodes: data.nodes || {} };
    renderConfig();
    configLoaded = true;
  } catch (e) {
    renderConfigOffline();
  }
}

function renderConfigOffline(detail) {
  $('config-content').innerHTML =
    '<div class="banner banner-offline">Control plane offline' +
    (detail ? ' — ' + escapeHtml(detail) : '') +
    '. Cannot load cluster configuration. (Open with <code>?mock=1</code> to preview with sample data.)</div>';
}

function deepClone(o) { return JSON.parse(JSON.stringify(o)); }

function renderConfig() {
  const el = $('config-content');
  const nodeIds = Object.keys(configState.nodes);

  const defaultsCard = `
    <div class="card">
      <div class="card-head"><h2>Cluster defaults</h2><span class="muted">applies to all nodes</span></div>
      <div class="cfg-section-note">Baseline for every node. A per-node override below wins over these.</div>
      ${renderDefaultsFields()}
    </div>`;

  const nodeCards = nodeIds.length
    ? nodeIds.map(renderNodeCard).join('')
    : '<div class="banner banner-empty">No nodes registered — nothing to configure yet.</div>';

  el.innerHTML = defaultsCard + nodeCards + `
    <div class="save-row">
      <button class="btn btn-save" id="btn-save-config" onclick="saveConfig()">Save configuration</button>
      <span class="save-note">Launch-time settings apply on the next replica start; live-safe settings (e.g. power cap) apply immediately.</span>
    </div>`;
}

function renderDefaultsFields() {
  return SETTING_GROUPS.map(group => {
    const fields = group.fields.filter(f => f.scope === 'cluster' || f.scope === 'both');
    if (!fields.length) return '';
    return `
      <div class="cfg-group">
        <h3>${escapeHtml(group.title)}</h3>
        <div class="cfg-defaults-grid">
          ${fields.map(f => `
            <label class="cfg-field-label" for="def-${f.key}"><code>${escapeHtml(f.label)}</code></label>
            <div class="cfg-override">${inputHtml('def-' + f.key, f, configState.defaults[f.key], 'onDefaultInput(\'' + f.key + '\')')}</div>
          `).join('')}
        </div>
      </div>`;
  }).join('');
}

function renderNodeCard(nodeId) {
  const node = configState.nodes[nodeId] || {};
  const detected = node.detected || {};
  const overrides = node.overrides || {};
  const roles = detected.roles || overrides.roles || '';
  const roleBadges = String(roles).split(',').filter(Boolean)
    .map(r => `<span class="badge role-badge">${escapeHtml(r.trim())}</span>`).join('');

  const groups = SETTING_GROUPS.map(group => {
    const fields = group.fields;   // node cards show the full catalog
    return `
      <div class="cfg-group">
        <h3>${escapeHtml(group.title)}</h3>
        <div class="cfg-grid">
          <span class="col-head">setting</span>
          <span class="col-head">detected</span>
          <span class="col-head">override</span>
          ${fields.map(f => renderNodeField(nodeId, f, detected, overrides)).join('')}
        </div>
      </div>`;
  }).join('');

  return `
    <div class="card node-card" data-node="${escapeHtml(nodeId)}">
      <div class="card-head">
        <div class="node-title">
          <h2>${escapeHtml(node.hostname || nodeId)}</h2>
          <span class="host-id">${escapeHtml(nodeId)}</span>
          ${roleBadges}
        </div>
      </div>
      ${groups}
      <details class="effective" open>
        <summary>Effective config (merged, read-only preview)</summary>
        <pre id="eff-${cssId(nodeId)}">${renderEffective(nodeId)}</pre>
      </details>
    </div>`;
}

function renderNodeField(nodeId, f, detected, overrides) {
  const det = detected[f.key];
  const detShown = (det === undefined || det === null || det === '')
    ? '<span class="cfg-detected auto">(auto)</span>'
    : `<span class="cfg-detected">${escapeHtml(det)}</span>`;
  const inputId = 'ov-' + cssId(nodeId) + '-' + f.key;
  const onIn = `onOverrideInput('${nodeId}','${f.key}')`;
  return `
    <span class="cfg-field-label"><code>${escapeHtml(f.label)}</code></span>
    ${detShown}
    <div class="cfg-override">${inputHtml(inputId, f, overrides[f.key], onIn)}</div>`;
}

// Build a form control for a field. `val` is the current stored value ('' = unset).
function inputHtml(id, f, val, onHandler) {
  const v = (val === undefined || val === null) ? '' : val;
  const handler = onHandler ? ` oninput="${onHandler}" onchange="${onHandler}"` : '';
  if (f.type === 'select') {
    const opts = f.options.map(o =>
      `<option value="${escapeHtml(o)}"${String(v) === String(o) ? ' selected' : ''}>${o === '' ? '(use default)' : escapeHtml(o)}</option>`).join('');
    return `<select id="${id}"${handler}>${opts}</select>`;
  }
  if (f.type === 'checkbox') {
    // Tri-state via a select keeps "unset" distinct from explicit true/false.
    const cur = v === true || v === 'true' ? 'true' : (v === false || v === 'false' ? 'false' : '');
    return `<select id="${id}"${handler}>
      <option value=""${cur === '' ? ' selected' : ''}>(use default)</option>
      <option value="true"${cur === 'true' ? ' selected' : ''}>on</option>
      <option value="false"${cur === 'false' ? ' selected' : ''}>off</option>
    </select>`;
  }
  const type = f.type === 'number' ? 'number' : 'text';
  const ph = f.placeholder ? ` placeholder="${escapeHtml(f.placeholder)}"` : '';
  const step = f.type === 'number' ? ' step="any"' : '';
  return `<input type="${type}" id="${id}"${step} value="${escapeHtml(v)}"${ph}${handler}>`;
}

// cssId — make a node_id safe for use inside element ids.
function cssId(s) { return String(s).replace(/[^a-zA-Z0-9_-]/g, '_'); }

/* --- Live editing: keep configState in sync and refresh effective preview --- */
function readControl(el, field) {
  if (!el) return '';
  const raw = el.value;
  if (raw === '') return '';
  return raw;
}

function onDefaultInput(key) {
  const f = ALL_FIELDS.find(x => x.key === key);
  const el = $('def-' + key);
  const val = readControl(el, f);
  if (val === '') delete configState.defaults[key];
  else configState.defaults[key] = val;
  // Defaults feed every node's effective preview.
  Object.keys(configState.nodes).forEach(refreshEffective);
}

function onOverrideInput(nodeId, key) {
  const f = ALL_FIELDS.find(x => x.key === key);
  const el = $('ov-' + cssId(nodeId) + '-' + key);
  const val = readControl(el, f);
  const node = configState.nodes[nodeId];
  node.overrides = node.overrides || {};
  if (val === '') delete node.overrides[key];
  else node.overrides[key] = val;
  refreshEffective(nodeId);
}

function refreshEffective(nodeId) {
  const pre = $('eff-' + cssId(nodeId));
  if (pre) pre.innerHTML = renderEffective(nodeId);
}

/* Merge precedence (docs/07): image defaults < detected < cluster defaults <
 * per-node override. We don't hold image defaults client-side, so the preview
 * merges detected -> cluster defaults -> override and labels each value's
 * winning source. */
function computeEffective(nodeId) {
  const node = configState.nodes[nodeId] || {};
  const detected = node.detected || {};
  const defaults = configState.defaults || {};
  const overrides = node.overrides || {};
  const keys = new Set([...Object.keys(detected), ...Object.keys(defaults), ...Object.keys(overrides)]);
  const out = {};
  keys.forEach(k => {
    if (overrides[k] !== undefined && overrides[k] !== '') out[k] = { val: overrides[k], src: 'override' };
    else if (defaults[k] !== undefined && defaults[k] !== '') out[k] = { val: defaults[k], src: 'default' };
    else if (detected[k] !== undefined && detected[k] !== '') out[k] = { val: detected[k], src: 'detected' };
  });
  return out;
}

function renderEffective(nodeId) {
  const eff = computeEffective(nodeId);
  const keys = Object.keys(eff).sort();
  if (!keys.length) return '<span class="eff-src">(nothing set)</span>';
  return keys.map(k => {
    const { val, src } = eff[k];
    const srcCls = src === 'override' ? 'eff-override' : 'eff-src';
    return `<span class="eff-key">${escapeHtml(k)}</span> = ${escapeHtml(val)}  <span class="${srcCls}"># ${src}</span>`;
  }).join('\n');
}

/* --- Save --- */
async function saveConfig() {
  const payload = { defaults: configState.defaults, nodes: {} };
  Object.keys(configState.nodes).forEach(id => {
    payload.nodes[id] = { overrides: configState.nodes[id].overrides || {} };
  });

  if (MOCK) { showToast('Mock mode: configuration not sent (would POST /api/cluster/config).', 'info'); return; }

  const btn = $('btn-save-config');
  if (btn) btn.disabled = true;
  try {
    const res = await apiFetch('/api/cluster/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { showToast(data.error || ('Save failed (' + res.status + ')'), 'error'); }
    else {
      showToast('Configuration saved', 'success');
      // Reload to pick up server-recomputed effective config.
      configLoaded = false;
      loadConfig();
    }
  } catch (e) {
    showToast('Save request failed: ' + e.message, 'error');
  }
  if (btn) btn.disabled = false;
}

/* =========================================================================
 * MOCK / SAMPLE DATA (?mock=1) — representative populated cluster.
 * ======================================================================= */
const MOCK_NODES = [
  {
    node_id: 'covid-a1b2', hostname: 'covid',
    roles: ['admin', 'participant', 'storage'], state: 'SERVING',
    gpus: [
      { index: 0, model: 'Tesla V100-SXM2-32GB', util: 96, mem_used: 30210, mem_total: 32768, temp: 71, power_draw: 288, power_limit: 300 },
      { index: 1, model: 'Tesla V100-SXM2-32GB', util: 92, mem_used: 29880, mem_total: 32768, temp: 68, power_draw: 271, power_limit: 300 },
      { index: 2, model: 'Tesla V100-SXM2-32GB', util: 4, mem_used: 512, mem_total: 32768, temp: 41, power_draw: 52, power_limit: 300 },
      { index: 3, model: 'Tesla V100-SXM2-32GB', util: 0, mem_used: 3, mem_total: 32768, temp: 38, power_draw: 46, power_limit: 300 },
    ],
    replicas: [{ id: 'devstral-r0', state: 'HEALTHY', port: 8001 }],
    mounts: [{ path: '/export/models', source: 'local (served)', ok: true }],
  },
  {
    node_id: 'ebola-c3d4', hostname: 'ebola',
    roles: ['participant'], state: 'DEGRADED',
    gpus: [
      { index: 0, model: 'Tesla V100-PCIE-16GB', util: 88, mem_used: 15100, mem_total: 16384, temp: 86, power_draw: 148, power_limit: 150 },
      { index: 1, model: 'Tesla V100-PCIE-16GB', util: 0, mem_used: 4, mem_total: 16384, temp: 44, power_draw: 33, power_limit: 150 },
    ],
    replicas: [{ id: 'devstral-r0', state: 'FAILED', port: 8001 }],
    mounts: [{ path: '/export/models', source: 'covid:/export/models', ok: false }],
  },
];

const MOCK_CONFIG = {
  defaults: {
    CLUSTER_ID: 'default',
    gpu_memory_utilization: '0.85',
    MULTIGPU_EXECUTOR: 'mp',
    DISABLE_CUSTOM_ALL_REDUCE: '1',
    enforce_eager: 'true',
    dtype: 'auto',
    port_range: '8000-8099',
    compile_cache_dir: '/var/cache/vllm-compile',
    CCP_HEARTBEAT_SEC: '5',
    CCP_TELEMETRY_SEC: '10',
    CCP_HEARTBEAT_MISS: '3',
  },
  nodes: {
    'covid-a1b2': {
      hostname: 'covid',
      detected: {
        roles: 'admin,participant,storage',
        CLUSTER_ID: 'default',
        RAY_NODE_IP: '10.10.0.1',
        NCCL_SOCKET_IFNAME: 'enp196s0,ens2',
        GLOO_SOCKET_IFNAME: 'enp196s0',
        NCCL_IB_HCA: 'mlx5_0:1',
        NCCL_IB_GID_INDEX: '3',
        NCCL_P2P_DISABLE: '1',
        gpu_power_cap_w: '300',
        storage_export_dir: '/export/models',
      },
      overrides: {
        NCCL_P2P_DISABLE: '1',
      },
    },
    'ebola-c3d4': {
      hostname: 'ebola',
      detected: {
        roles: 'participant',
        CLUSTER_ID: 'default',
        RAY_NODE_IP: '10.10.0.2',
        RAY_HEAD_HOST: '10.10.0.1:6379',
        NCCL_SOCKET_IFNAME: 'enp196s0,ens2',
        GLOO_SOCKET_IFNAME: 'ens2',
        NCCL_IB_HCA: 'mlx4_0:1',
        NCCL_IB_GID_INDEX: '1',
        gpu_power_cap_w: '300',
        mount_canonical_path: '/export/models',
        mount_transport: 'rdma',
      },
      // ebola needs its local NIC first + the workaround power cap.
      overrides: {
        NCCL_SOCKET_IFNAME: 'ens2,enp196s0',
        gpu_power_cap_w: '150',
      },
    },
  },
};

/* =========================================================================
 * INIT
 * ======================================================================= */
function initKeybar() {
  const input = $('api-key-input');
  if (!input) return;
  input.value = getApiKey();
  const status = $('key-status');
  const update = () => {
    setApiKey(input.value.trim());
    if (status) status.textContent = input.value.trim() ? 'key set' : 'no key';
  };
  input.addEventListener('input', update);
  update();
}

function init() {
  initKeybar();
  document.querySelectorAll('.tab-btn').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
  if (MOCK) {
    const badge = $('mock-badge');
    if (badge) badge.style.display = 'inline-flex';
  }
  // Monitor is the default tab; start polling.
  loadNodes();
  monitorTimer = setInterval(loadNodes, POLL_MS);
}

document.addEventListener('DOMContentLoaded', init);
