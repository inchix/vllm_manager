/* ============================================================================
 * cluster.js — control-plane cluster UI for vllm-multi-gpu (v0.4.0)
 *
 * Renders four tabs on cluster.html, organized BY ROLE:
 *   (A) SUMMARY (default)  — cluster totals from GET /api/cluster/summary plus
 *       a compact one-row-per-node table. Every node appears once.
 *   (B) WORKERS            — nodes whose roles include `participant`: one card
 *       per node with full per-GPU util/mem/temp/power telemetry, instances,
 *       AVAILABLE SHARES (every cluster share with a Mount checkbox for THAT
 *       node) and a collapsible NODE DETAIL (NICs, RDMA fabric, mounts).
 *       Mounting is a per-node action, so the checkbox lives on the node it
 *       acts on rather than in a cluster-wide matrix. Polls ~3s.
 *   (C) STORAGE            — nodes whose roles include `storage`: the NIC the
 *       node serves shares on, then its VOLUMES (path/fs/size/used/free) each
 *       with a Share checkbox. No GPUs here — a storage node that also
 *       participates shows its GPUs under Workers. Polls ~3s.
 *   (D) CLUSTER CONFIGURATION — cluster defaults + one card per node: a
 *       NETWORKS block (one NIC picker per traffic type) then each setting
 *       showing DETECTED (read-only) vs OVERRIDE (input), a merged
 *       effective-config preview, and a Save button.
 *
 * Role filtering is by MEMBERSHIP, not exclusivity: a node with roles
 * {participant,storage} shows under BOTH Workers and Storage.
 *
 * Role ids are wire values (`participant`, `storage`, `admin`) and are what
 * every filter matches on; what people read comes from summary.role_labels
 * via roleLabel() — `participant` displays as "GPU Worker".
 *
 * ---------------------------------------------------------------------------
 * API CONTRACT (assumed — endpoints may not exist yet; the UI degrades
 * gracefully to "control plane offline" on 404/network error):
 *
 *   GET /api/cluster/nodes  ->  [
 *     {
 *       node_id, hostname, roles:[...], state,   // state: READY|SERVING|DEGRADED|DOWN
 *       addresses:{ mgmt, fabric:[...] },
 *       interfaces:[{ name, ip, netmask, cidr, rdma }],   // may be absent
 *       rdma:    [{ hca, ports:[...], gid_index, netdev, fabric_ip }],
 *       gpus:    [{ index, model, util, mem_used, mem_total, temp,
 *                   power_draw, power_limit }],
 *       instances:[{ id, state, port }],
 *       mounts:  [{ path, source, ok }]
 *     }, ...
 *   ]
 *   interfaces/rdma/mounts may be missing entirely (older agents) — every
 *   renderer treats them as empty rather than throwing.
 *
 *   GET /api/cluster/config ->  {
 *     defaults: { <key>: <value>, ... },
 *     nodes: {
 *       <node_id>: { detected:{...}, overrides:{...}, effective:{...} }
 *     },
 *     settings: { <key>: { group, widget, label, help } }
 *   }
 *   `settings` is the server's setting catalog. Every entry with
 *   group === "network" is rendered in the NETWORKS block (Configuration tab,
 *   per-node card; the storage row is mirrored on the Storage tab), driven
 *   entirely off that metadata so new network settings appear on their own.
 *   widget decides what a picked interface contributes as the value:
 *     "ip"       -> interfaces[].ip     (e.g. ray_node_ip, storage_bind_ip)
 *     "nic"      -> interfaces[].name   (e.g. gloo_socket_ifname)
 *     "nic-list" -> interfaces[].name   (server-side a comma list; the UI sets
 *                                        the local NIC as the single value)
 *   If a backend does not publish `settings`, NETWORK_SETTINGS_FALLBACK below
 *   supplies the same four keys so the block still works.
 *
 *   POST /api/cluster/config   body (one key per change, from Storage tab):
 *     { "nodes": { "<node_id>": { "overrides": { "storage_bind_ip": "1.2.3.4" } } } }
 *   The Configuration tab batches every pending override into the same shape
 *   when "Save configuration" is pressed.
 *
 *   POST /api/cluster/config   body:  {
 *     defaults: {...},
 *     nodes: { <node_id>: { overrides: {...} } }
 *   }
 *
 *   GET /api/cluster/shares ->  {
 *     shares: [{ id, node_id, path, endpoint, fstype,
 *                total, used, free, ok,                    // sizes in BYTES
 *                mounted_by:[{ node_id, path, ok, error? }] }],
 *     volumes_by_node: {
 *       <node_id>: [{ path, fstype, total, used, free, shared, endpoint }]
 *     }
 *   }
 *   A 404 here means an older backend: the UI shows "shares unavailable"
 *   instead of erroring, and the rest of the page keeps working.
 *
 *   POST /api/cluster/share  body: { node_id, path, enabled }
 *       -> { ok: true } | { ok: false, error: "..." }
 *       `path` is any directory on that node — a discovered volume root or a
 *       user-typed subdirectory. The backend validates it.
 *
 *   POST /api/cluster/mount  body: { node_id, share_id, enabled, mount_path? }
 *       -> { ok: true } | { ok: false, error: "..." }
 *
 * Every request carries the admin key in an `X-API-Key` header (see apiFetch).
 * A 401 redirects to /login, matching the existing admin UI.
 *
 * ?mock=1  renders representative sample data (COVID {admin,participant,
 *          storage} + ebola {participant}) so the UI can be reviewed with no
 *          backend running. The mock share/mount/add-share controls mutate the
 *          in-memory sample data, so the interactions are clickable offline.
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

/* Escape for both text content and quoted attribute values. Note innerHTML
 * serialization of a text node escapes & < > but NOT quotes, so we add those
 * ourselves — otherwise a value containing " would break out of an attribute
 * (and out of the inline on* handlers the share/mount checkboxes use). */
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : String(s);
  return d.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// "/" is the main UI now, so it needs its own sign-out (matches index.html).
async function logout() {
  try { await fetch('/api/auth/logout', { method: 'POST' }); } catch (e) {}
  window.location.href = '/login';
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
 * LIVE TABS — Summary / Workers / Storage
 *
 * All three are driven from one poll: GET /api/cluster/nodes (+ /summary for
 * the totals). Nodes are filtered BY ROLE MEMBERSHIP, not exclusivity — a node
 * whose roles include both `participant` and `storage` appears under Workers
 * AND Storage; Summary lists every node exactly once.
 * ======================================================================= */
let liveTimer = null;

/* Shares/volumes snapshot from GET /api/cluster/shares.
 * status: 'loading' | 'ok' | 'unavailable' (404 — older backend) | 'error' */
let sharesState = { status: 'loading', shares: [], volumesByNode: {}, detail: '' };

/* While a share/mount POST is in flight we skip the poll re-render so the
 * optimistic checkbox state isn't clobbered mid-request. */
let pendingOps = 0;

/* Text typed into a node's "Add share" box, kept across re-renders. */
const addShareDrafts = {};

/* Last node list seen by the poll, by node_id. The Configuration tab needs
 * each node's `interfaces` (which only GET /api/cluster/nodes reports) to
 * build the Networks selects; the poll starts on page load, so the map is
 * populated well before that tab is opened. */
let nodesById = {};

/* Open/closed state of each worker card's "Node detail" disclosure, keyed by
 * node_id — the 3s poll replaces the panel's innerHTML, so without this the
 * section would snap shut on every refresh. */
const nodeDetailOpen = {};

function hasRole(n, role) {
  return (n.roles || []).map(r => String(r).toLowerCase()).includes(role);
}

/* -------------------------------------------------------------------------
 * ROLE LABELS — presentation only.
 *
 * The wire ids (`participant`, `storage`, `admin`) are what .env files, saved
 * per-node config and join tokens carry, so every hasRole() filter keeps using
 * them. What people READ comes from GET /api/cluster/summary -> role_labels,
 * which is how `participant` displays as "GPU Worker"; a future rename lands
 * in the UI with no code change. The fallback map keeps an older control plane
 * that sends no role_labels reading correctly, and an unknown role falls back
 * to its raw id.
 * ---------------------------------------------------------------------- */
const ROLE_LABELS_FALLBACK = { admin: 'Admin', participant: 'GPU Worker', storage: 'Storage' };
let roleLabels = {};

function roleLabel(r) {
  const raw = String(r == null ? '' : r).trim();
  const lk = raw.toLowerCase();
  return roleLabels[lk] || roleLabels[raw] || ROLE_LABELS_FALLBACK[lk] || raw;
}

// Role badges for a node, labelled for display.
function roleBadges(roles) {
  return (roles || []).map(r =>
    `<span class="badge role-badge" title="${escapeHtml(String(r))}">${escapeHtml(roleLabel(r))}</span>`).join(' ');
}

/* -------------------------------------------------------------------------
 * NETWORK INTERFACES
 *
 * Nodes report every interface the agent can see, container plumbing
 * included. Podman/Docker/CNI bridges, veth pairs and `br-*` bridges are
 * never the right answer for cluster, compute or storage traffic, so they are
 * filtered out of both the worker detail list and the Networks selects.
 * ---------------------------------------------------------------------- */
const BRIDGE_IFACE_RE = /^(podman|docker|cni|veth|br-)/i;

function isBridgeIface(name) {
  return BRIDGE_IFACE_RE.test(String(name || ''));
}

// Real (non-bridge) interfaces of a node, always an array.
function usableInterfaces(n) {
  const list = (n && Array.isArray(n.interfaces)) ? n.interfaces : [];
  return list.filter(i => i && !isBridgeIface(i.name));
}

// The node's RDMA fabric interface, if it reported one.
function rdmaIface(n) {
  return usableInterfaces(n).find(i => i.rdma) || null;
}

// "enp196s0 — 172.16.254.201 (172.16.254.0/24) · RDMA"
function ifaceLabel(i) {
  const sub = i.cidr || i.netmask || '';
  return (i.name || '?') +
    ' — ' + (i.ip || 'no address') +
    (sub ? ' (' + sub + ')' : '') +
    (i.rdma ? ' · RDMA' : '');
}

async function loadLive() {
  if (pendingOps > 0) return;               // a toggle is mid-flight
  if (MOCK) {
    sharesState = mockShares();
    configSnapshot = mockConfig();
    renderLive(MOCK_SUMMARY, MOCK_NODES);
    return;
  }
  try {
    const res = await apiFetch('/api/cluster/nodes');
    if (res.status === 404) { renderLiveOffline(); return; }
    if (!res.ok) { renderLiveOffline('control plane returned ' + res.status); return; }
    const nodes = await res.json();
    const nodeArr = Array.isArray(nodes) ? nodes : [];
    // Summary totals come from /api/cluster/summary; fall back to a client-side
    // derivation if that endpoint is unavailable.
    let summary = null;
    try {
      const sres = await apiFetch('/api/cluster/summary');
      if (sres.ok) summary = await sres.json();
    } catch (e) { /* fall through to derived summary */ }
    if (!summary) summary = deriveSummary(nodeArr);
    // Shares ride the same refresh cycle. A 404 (older backend) is not an
    // error: the volumes/shares sections just say so and everything else works.
    await loadShares();
    // …and so does the config snapshot, which the Storage tab's network row
    // reads (effective.storage_bind_ip) and the Configuration tab reuses.
    await loadConfigSnapshot(nodeArr);
    renderLive(summary, nodeArr);
  } catch (e) {
    // Network error or endpoint missing — degrade, don't throw.
    renderLiveOffline();
  }
}

async function loadShares() {
  try {
    const res = await apiFetch('/api/cluster/shares');
    if (res.status === 404) {
      sharesState = { status: 'unavailable', shares: [], volumesByNode: {}, detail: '' };
      return;
    }
    if (!res.ok) {
      sharesState = { status: 'error', shares: [], volumesByNode: {}, detail: 'HTTP ' + res.status };
      return;
    }
    const data = await res.json();
    sharesState = {
      status: 'ok',
      shares: Array.isArray(data && data.shares) ? data.shares : [],
      volumesByNode: (data && data.volumes_by_node) || {},
      detail: '',
    };
  } catch (e) {
    sharesState = { status: 'error', shares: [], volumesByNode: {}, detail: e.message || 'request failed' };
  }
}

/* Read-only snapshot of GET /api/cluster/config, refreshed on the live poll.
 * Deliberately separate from `configState` (the Configuration tab's editable
 * copy) so a poll can never clobber unsaved edits; loadConfig() adopts this
 * snapshot instead of issuing a second request. */
let configSnapshot = null;

async function loadConfigSnapshot(nodes) {
  // Only the Storage tab needs it on every cycle. With no storage node the
  // Configuration tab fetches it itself when opened.
  if (!nodes.some(n => hasRole(n, 'storage'))) return;
  try {
    const res = await apiFetch('/api/cluster/config');
    if (!res.ok) return;                       // keep the last good snapshot
    const data = await res.json();
    if (data && typeof data === 'object') configSnapshot = data;
  } catch (e) { /* keep the last good snapshot */ }
}

// Force an immediate refresh (used after a successful share/mount toggle).
function refreshLive() { loadLive(); }

function renderLive(summary, nodes) {
  nodesById = {};
  nodes.forEach(n => { if (n && n.node_id) nodesById[n.node_id] = n; });
  // Display names for the wire role ids; kept from the last summary that
  // carried them so every tab (Configuration included) labels roles alike.
  if (summary && summary.role_labels && typeof summary.role_labels === 'object') {
    roleLabels = summary.role_labels;
  }
  const workersRoleEl = $('workers-role-label');
  if (workersRoleEl) workersRoleEl.textContent = roleLabel('participant') + ' nodes';
  renderSummary(summary, nodes);
  renderWorkers(nodes);
  renderStorage(nodes);
}

/* =========================================================================
 * NETWORKS — one picker per traffic type, driven by the server's setting
 * catalog (GET /api/cluster/config -> settings) rather than a hardcoded list,
 * so a new network setting shows up here without a UI change. Rendered in the
 * Configuration tab's per-node card; the storage row is mirrored on the
 * Storage tab, where it is the setting people actually go looking for.
 * ======================================================================= */

/* Used only when the backend publishes no `settings` catalog (older control
 * plane). Same keys, same widgets the server tags. */
const NETWORK_SETTINGS_FALLBACK = {
  ray_node_ip: {
    group: 'network', widget: 'ip', label: 'Cluster / control network',
    help: 'Control-plane and Ray traffic; also this node’s identity in the cluster.',
  },
  nccl_socket_ifname: {
    group: 'network', widget: 'nic-list', label: 'Compute (NCCL) network',
    help: 'NCCL collectives between instances. The local NIC is set first.',
  },
  gloo_socket_ifname: {
    group: 'network', widget: 'nic', label: 'Compute (Gloo) NIC',
    help: 'Gloo control collectives — a single local NIC.',
  },
  storage_bind_ip: {
    group: 'network', widget: 'ip', label: 'Storage network',
    help: 'Shares are served on this NIC. The port is chosen automatically and advertised to the cluster.',
  },
};

// The settings catalog currently in hand (snapshot first, then the tab's copy).
function settingsCatalog() {
  const fromSnap = configSnapshot && configSnapshot.settings;
  if (fromSnap && typeof fromSnap === 'object') return fromSnap;
  const fromState = configState && configState.settings;
  if (fromState && typeof fromState === 'object') return fromState;
  return null;
}

/* Every network-group setting as [{ key, widget, label, help }], in catalog
 * order. Falls back to NETWORK_SETTINGS_FALLBACK when the server tags none. */
function networkSettings() {
  const cat = settingsCatalog();
  const out = [];
  if (cat) {
    Object.keys(cat).forEach(k => {
      const m = cat[k] || {};
      if (String(m.group || '').toLowerCase() === 'network') {
        out.push({ key: k, widget: m.widget || 'ip', label: m.label || k, help: m.help || '' });
      }
    });
  }
  if (out.length) return out;
  return Object.keys(NETWORK_SETTINGS_FALLBACK).map(k =>
    Object.assign({ key: k }, NETWORK_SETTINGS_FALLBACK[k]));
}

function networkSetting(key) {
  return networkSettings().find(s => s.key === key) || null;
}

/* Keys owned by the Networks block, lowercased. The legacy SETTING_GROUPS
 * catalog below still lists some of these (as RAY_NODE_IP etc.); they are
 * dropped from those groups so one setting never gets two controls. */
function networkKeySet() {
  const s = new Set();
  networkSettings().forEach(n => s.add(String(n.key).toLowerCase()));
  return s;
}

// What a picked interface contributes for this setting: its IP or its name.
function ifaceValue(iface, setting) {
  const w = String(setting.widget || 'ip').toLowerCase();
  return String((w === 'nic' || w === 'nic-list') ? (iface.name || '') : (iface.ip || ''));
}

/* Preselection: an explicit override wins, then the server's effective value,
 * then what the agent detected, then a sensible guess from the node's NICs —
 * the RDMA fabric interface for compute/storage, the mgmt address for the
 * cluster/control network. */
function networkValue(node, entry, setting) {
  const key = setting.key;
  const srcs = [(entry && entry.overrides) || {}, (entry && entry.effective) || {}, (entry && entry.detected) || {}];
  for (let i = 0; i < srcs.length; i++) {
    const v = srcs[i][key];
    if (v !== undefined && v !== null && v !== '') {
      // nic-list values are comma lists server-side; the local NIC is first.
      return String(v).split(',')[0].trim();
    }
  }
  return networkFallback(node, setting);
}

function networkFallback(node, setting) {
  const ifaces = usableInterfaces(node);
  const rdma = rdmaIface(node);
  const mgmt = ((node && node.addresses) || {}).mgmt || '';
  const isControl = /ray|control|mgmt/i.test(String(setting.key));
  if (isControl) {
    const mgmtIface = ifaces.find(i => i.ip && i.ip === mgmt);
    if (mgmtIface) return ifaceValue(mgmtIface, setting);
    if (String(setting.widget).indexOf('nic') === 0) return rdma ? ifaceValue(rdma, setting) : '';
    return mgmt || (rdma ? ifaceValue(rdma, setting) : '');
  }
  if (rdma) return ifaceValue(rdma, setting);
  const first = ifaces.find(i => ifaceValue(i, setting));
  return first ? ifaceValue(first, setting) : '';
}

/* <option> list for one setting on one node. `cur` is preselected; a current
 * value that matches no reported NIC (stale config, or an agent that doesn't
 * report interfaces) is kept as its own option so nothing is silently lost. */
function networkOptions(node, setting, cur) {
  const ifaces = usableInterfaces(node).filter(i => ifaceValue(i, setting));
  let matched = false;
  const opts = ifaces.map(i => {
    const val = ifaceValue(i, setting);
    let sel = '';
    if (!matched && val === cur) { matched = true; sel = ' selected'; }
    return `<option value="${escapeHtml(val)}"${sel}>${escapeHtml(ifaceLabel(i))}</option>`;
  });
  if (!matched && cur) {
    opts.unshift(`<option value="${escapeHtml(cur)}" selected>${escapeHtml(cur)} — current (no matching NIC reported)</option>`);
  }
  if (!matched && !cur) {
    opts.unshift('<option value="" selected>(not set)</option>');
  }
  return opts.join('');
}

// The config entry (detected/overrides/effective) for a node, from either copy.
function configEntry(nodeId, preferState) {
  const state = (configState && configState.nodes) || {};
  const snap = (configSnapshot && configSnapshot.nodes) || {};
  const a = preferState ? state[nodeId] : snap[nodeId];
  return a || (preferState ? snap[nodeId] : state[nodeId]) || {};
}

/* Don't re-render a panel while the user is typing in it (the add-share box)
 * or has a <select> focused (the network pickers) — the 3s poll would
 * otherwise drop focus, the caret mid-path, or an open dropdown. Checkboxes
 * are left re-renderable: the poll re-asserting server truth is what we want. */
function panelBusy(id) {
  const el = $(id);
  const a = document.activeElement;
  if (!el || !a) return false;
  const tag = (a.tagName || '').toUpperCase();
  const isText = tag === 'INPUT' && String(a.type || 'text').toLowerCase() === 'text';
  return (isText || tag === 'SELECT') && el.contains(a);
}

// Derive a summary from the node list when /api/cluster/summary is unavailable.
function deriveSummary(nodes) {
  const roles = { admin: 0, participant: 0, storage: 0 };
  const instances = {};
  let gpus = 0, alive = 0;
  nodes.forEach(n => {
    if (hasRole(n, 'admin')) roles.admin++;
    if (hasRole(n, 'participant')) roles.participant++;
    if (hasRole(n, 'storage')) roles.storage++;
    gpus += (n.gpus || []).length;
    if (n.connected || ['ready', 'serving', 'degraded'].includes(String(n.state || '').toLowerCase())) alive++;
    (n.instances || []).forEach(r => {
      const st = (r.state || 'unknown').toUpperCase();
      instances[st] = (instances[st] || 0) + 1;
    });
  });
  return { nodes_total: nodes.length, nodes_alive: alive, gpus_total: gpus, roles, instances };
}

function renderLiveOffline(detail) {
  const html = offlineBanner(detail);
  ['summary-content', 'workers-content', 'storage-content'].forEach(id => {
    const el = $(id); if (el) el.innerHTML = html;
  });
}

function offlineBanner(detail) {
  return '<div class="banner banner-offline">Control plane offline' +
    (detail ? ' — ' + escapeHtml(detail) : '') +
    '. No telemetry available. (Open with <code>?mock=1</code> to preview with sample data.)</div>';
}

/* ---- Per-node aggregate (used by the Summary table) ---- */
function nodeAgg(n) {
  const gpus = n.gpus || [];
  let used = 0, total = 0, utilSum = 0, utilN = 0;
  gpus.forEach(g => {
    if (g.mem_used != null) used += g.mem_used;
    if (g.mem_total != null) total += g.mem_total;
    if (g.util != null) { utilSum += g.util; utilN++; }
  });
  return { count: gpus.length, memUsed: used, memTotal: total, util: utilN ? utilSum / utilN : null };
}

/* ---- (A) SUMMARY ---- */
function renderSummary(summary, nodes) {
  const el = $('summary-content');
  const s = summary || {};
  const roles = s.roles || {};
  const instanceTotal = Object.values(s.instances || {}).reduce((a, b) => a + (Number(b) || 0), 0);

  const tile = (val, label) =>
    `<div class="stat-tile"><span class="stat-val">${val}</span><span class="stat-label">${escapeHtml(label)}</span></div>`;
  const num = v => (v == null ? '—' : v);

  const tiles = `
    <div class="stat-row">
      <div class="stat-tile">
        <span class="stat-val">${num(s.nodes_alive)}<span class="stat-sub"> / ${num(s.nodes_total)}</span></span>
        <span class="stat-label">Nodes alive</span>
      </div>
      ${tile(num(s.gpus_total), 'GPUs total')}
      ${tile(roles.participant != null ? roles.participant : 0, roleLabel('participant'))}
      ${tile(roles.storage != null ? roles.storage : 0, roleLabel('storage'))}
      ${tile(roles.admin != null ? roles.admin : 0, roleLabel('admin'))}
      ${tile(instanceTotal, 'Instances')}
    </div>`;

  let table;
  if (!nodes.length) {
    table = '<div class="banner banner-empty">No nodes registered with the control plane yet.</div>';
  } else {
    const rows = nodes.map(n => {
      const a = nodeAgg(n);
      const badges = roleBadges(n.roles);
      const st = (n.state || 'unknown').toUpperCase();
      const stCls = stateClass(n.state);
      const memCell = a.memTotal > 0 ? `${fmtMb(a.memUsed)} / ${fmtMb(a.memTotal)}` : '—';
      const utilCell = a.util == null ? '—' : Math.round(a.util) + '%';
      return `
        <tr>
          <td class="mono">${escapeHtml(n.node_id || '')}</td>
          <td>${escapeHtml(n.hostname || '')}</td>
          <td>${badges || '<span class="muted">—</span>'}</td>
          <td><span class="badge state-badge ${stCls}"><span class="state-dot"></span>${escapeHtml(st)}</span></td>
          <td class="num">${a.count}</td>
          <td class="mono">${utilCell} util · ${memCell}</td>
        </tr>`;
    }).join('');
    table = `
      <div class="table-wrap">
        <table class="summary-table">
          <thead>
            <tr><th>Node</th><th>Host</th><th>Roles</th><th>State</th><th class="num">GPUs</th><th>Mem / Util</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  el.innerHTML = tiles + table;
}

/* ---- (B) WORKERS (participant role) ---- */
function renderWorkers(nodes) {
  const el = $('workers-content');
  if (panelBusy('workers-content')) return;
  const workers = nodes.filter(n => hasRole(n, 'participant'));
  if (!workers.length) {
    el.innerHTML = '<div class="banner banner-empty">No ' + escapeHtml(roleLabel('participant')) +
      ' nodes registered.</div>';
    return;
  }
  el.innerHTML = workers.map(n => renderNode(n.hostname || n.node_id || 'unknown', n)).join('');
}

/* ---- Available shares, per worker card ----
 * Every share advertised anywhere in the cluster, listed on the node that
 * would mount it: mounting is a per-node action, so the checkbox lives on the
 * node it acts on. A share served BY this node needs no mount — that row says
 * "local" instead. */
function renderNodeShares(n) {
  const nodeId = n.node_id || '';
  const wrap = body =>
    `<div class="sub-section node-shares"><div class="sub-title">Available shares</div>${body}</div>`;

  if (sharesState.status === 'unavailable') {
    return wrap('<div class="muted">Shares unavailable &mdash; this control plane does not expose ' +
      '<code>GET /api/cluster/shares</code>.</div>');
  }
  if (sharesState.status === 'error') {
    return wrap('<div class="muted">Shares unavailable' +
      (sharesState.detail ? ' &mdash; ' + escapeHtml(sharesState.detail) : '') + '.</div>');
  }
  if (sharesState.status === 'loading') {
    return wrap('<div class="muted">Loading shares&hellip;</div>');
  }
  const shares = sharesState.shares || [];
  if (!shares.length) {
    return wrap('<div class="muted">No shares available &mdash; tick <strong>Share</strong> on a volume in the ' +
      '<strong>Storage</strong> tab to publish one.</div>');
  }
  return wrap(`
    <div class="share-list">
      <div class="share-row header"><span>Mount</span><span>Share</span><span>Size</span><span>State</span></div>
      ${shares.map(s => renderNodeShareRow(nodeId, s)).join('')}
    </div>`);
}

function renderNodeShareRow(nodeId, s) {
  const src = nodesById[s.node_id] || {};
  const srcName = src.hostname || src.node_id || s.node_id || '?';
  const srcTxt = srcName + (s.endpoint ? ' (' + s.endpoint + ')' : '');
  const okMark = s.ok === false ? ' <span class="bad" title="share is not healthy">✗</span>' : '';
  const size = fmtBytes(s.total) +
    (s.free == null ? '' : ` <span class="dim">· ${fmtBytes(s.free)} free</span>`);

  let control, state;
  if (s.node_id === nodeId) {
    // Served from this node — it reads the directory directly.
    control = '<span class="muted">local</span>';
    state = `<span class="mount-state muted">served by this node</span>`;
  } else {
    const m = (s.mounted_by || []).find(x => x && x.node_id === nodeId);
    const mountPath = (m && m.path) || s.path || '';
    const args = [nodeId, s.id, mountPath].map(jsArg).join(', ');
    control = `<label class="chk"><input type="checkbox"${m ? ' checked' : ''} onchange="onMountToggle(this, ${args})">
      <span class="chk-label">mount</span></label>`;
    if (!m) state = '<span class="mount-state muted">not mounted</span>';
    else if (m.ok === false) state = `<span class="mount-state bad">${escapeHtml(m.error || 'mount failed')}</span>`;
    else state = `<span class="mount-state mono">${escapeHtml(m.path || s.path || '')}</span>`;
  }

  return `
    <div class="share-row">
      <span class="share-mount">${control}</span>
      <span class="share-what">
        <span class="mono">${escapeHtml(s.path || '')}</span>${okMark}
        <span class="share-src dim">${escapeHtml(srcTxt)}</span>
      </span>
      <span class="share-size mono">${size}</span>
      <span class="share-state">${state}</span>
    </div>`;
}

/* ---- (C) STORAGE (storage role) ----
 * Volumes only. A storage node that also participates shows its GPUs under
 * Workers, never here. */
function renderStorage(nodes) {
  const el = $('storage-content');
  if (panelBusy('storage-content')) return;
  const storageNodes = nodes.filter(n => hasRole(n, 'storage'));
  if (!storageNodes.length) {
    el.innerHTML = '<div class="banner banner-empty">No storage nodes registered.</div>';
    return;
  }
  el.innerHTML = storageNodes.map(renderStorageNode).join('');
}

/* Rows for one storage node: every discovered volume, plus any share whose
 * path isn't a discovered volume root (a hand-added subdirectory). */
function storageRowsFor(nodeId) {
  const vols = (sharesState.volumesByNode || {})[nodeId] || [];
  const rows = vols.map(v => ({
    path: v.path, fstype: v.fstype, total: v.total, used: v.used, free: v.free,
    shared: !!v.shared, endpoint: v.endpoint, custom: false,
  }));
  const seen = new Set(rows.map(r => r.path));
  (sharesState.shares || []).forEach(s => {
    if (s.node_id !== nodeId || seen.has(s.path)) return;
    seen.add(s.path);
    rows.push({
      path: s.path, fstype: s.fstype, total: s.total, used: s.used, free: s.free,
      shared: true, endpoint: s.endpoint, custom: true,
    });
  });
  return rows;
}

function renderStorageNode(n) {
  const host = n.hostname || n.node_id || 'unknown';
  const nodeId = n.node_id || '';
  const roles = roleBadges(n.roles);
  const st = (n.state || 'unknown').toUpperCase();
  const stCls = stateClass(n.state);
  const stateBadge = `<span class="badge state-badge ${stCls}"><span class="state-dot"></span>${escapeHtml(st)}</span>`;

  let body;
  if (sharesState.status === 'unavailable') {
    body = '<div class="muted">Volumes unavailable &mdash; this control plane does not expose ' +
      '<code>GET /api/cluster/shares</code>.</div>';
  } else if (sharesState.status === 'error') {
    body = '<div class="muted">Volumes unavailable' +
      (sharesState.detail ? ' &mdash; ' + escapeHtml(sharesState.detail) : '') + '.</div>';
  } else if (sharesState.status === 'loading') {
    body = '<div class="muted">Loading volumes&hellip;</div>';
  } else {
    const rows = storageRowsFor(nodeId);
    body = rows.length
      ? `<div class="table-wrap">
           <table class="summary-table vol-table">
             <thead><tr><th>Volume</th><th>Filesystem</th><th class="num">Size</th><th class="num">Used</th><th class="num">Free</th><th class="num">Share</th></tr></thead>
             <tbody>${rows.map(v => renderVolumeRow(nodeId, v)).join('')}</tbody>
           </table>
         </div>`
      : '<div class="muted">No volumes reported on this node.</div>';
    body += renderAddShare(nodeId);
  }

  return `
    <div class="host-group">
      <div class="host-head">
        <span class="host-name">${escapeHtml(host)}</span>
        <span class="host-id">${escapeHtml(nodeId)}</span>
        ${addrHtml(n)}
        ${roles}
        ${stateBadge}
      </div>
      <div class="storage-note">Ticking <strong>Share</strong> exports that directory read-only over the fabric (modelfsd, NFSv3/TCP).</div>
      ${renderStorageNetworkRow(n)}
      ${body}
    </div>`;
}

/* ---- Storage tab: the storage row of the Networks block ----
 * Same picker as the Configuration tab's Networks block, mirrored here because
 * this is where people are when they think about it. Unlike the config tab it
 * saves on change (no Save button on a live-polling panel). */
function renderStorageNetworkRow(n) {
  const setting = networkSetting('storage_bind_ip');
  if (!setting) return '';
  const nodeId = n.node_id || '';
  const entry = configEntry(nodeId);
  const cur = networkValue(n, entry, setting);
  const id = 'net-live-' + cssId(nodeId) + '-' + cssId(setting.key);
  const ifaces = usableInterfaces(n).filter(i => ifaceValue(i, setting));

  const control = ifaces.length
    ? `<select id="${id}" data-prev="${escapeHtml(cur)}"
               onchange="onNetworkLiveChange(this, ${jsArg(nodeId)}, ${jsArg(setting.key)})">
         ${networkOptions(n, setting, cur)}
       </select>`
    : `<span class="muted">no interfaces reported by this node&rsquo;s agent${
         cur ? ' — currently <code>' + escapeHtml(cur) + '</code>' : ''}</span>`;

  return `
    <div class="net-row net-row-live">
      <label class="net-label" for="${id}">Serve shares on</label>
      <div class="net-control">${control}</div>
      <div class="net-help muted">${escapeHtml(setting.help ||
        'Shares are served on this NIC. The port is chosen automatically and advertised to the cluster.')}
        Changing the NIC only affects shares started afterwards — existing shares keep their current endpoint until re-shared.</div>
    </div>`;
}

/* Live (save-on-change) network picker. Posts exactly one key:
 *   { nodes: { <node_id>: { overrides: { <key>: <value> } } } } */
async function onNetworkLiveChange(sel, nodeId, key) {
  const val = sel.value;
  const prev = sel.dataset ? (sel.dataset.prev || '') : '';
  if (val === prev) return;
  const body = { nodes: {} };
  body.nodes[nodeId] = { overrides: {} };
  body.nodes[nodeId].overrides[key] = val;

  if (MOCK) {
    mockSetOverride(nodeId, key, val);
    if (sel.dataset) sel.dataset.prev = val;
    showToast('Mock mode: ' + key + ' = ' + val + ' on ' + nodeId, 'success');
    renderLive(MOCK_SUMMARY, MOCK_NODES);
    return;
  }

  sel.disabled = true;
  pendingOps++;
  const r = await postAction('/api/cluster/config', body);
  pendingOps--;
  sel.disabled = false;
  if (r.ok) {
    if (sel.dataset) sel.dataset.prev = val;
    showToast('Serving shares on ' + val + ' (' + nodeId + ')', 'success');
    configLoaded = false;              // Configuration tab must re-read
    refreshLive();
  } else {
    sel.value = prev;                  // revert
    showToast('Could not set ' + key + ' on ' + nodeId + ': ' + r.error, 'error');
  }
}

function renderVolumeRow(nodeId, v) {
  const args = [nodeId, v.path].map(jsArg).join(', ');
  const endpoint = (v.shared && v.endpoint)
    ? `<div class="share-endpoint mono">${escapeHtml(v.endpoint)}</div>` : '';
  const custom = v.custom ? ' <span class="badge custom-badge">custom</span>' : '';
  return `
    <tr class="${v.custom ? 'vol-custom' : ''}">
      <td class="mono">${escapeHtml(v.path || '')}${custom}${endpoint}</td>
      <td>${escapeHtml(v.fstype || '—')}</td>
      <td class="num">${fmtBytes(v.total)}</td>
      <td class="num">${fmtBytes(v.used)}</td>
      <td class="num">${fmtBytes(v.free)}</td>
      <td class="num"><label class="chk"><input type="checkbox"${v.shared ? ' checked' : ''} onchange="onShareToggle(this, ${args})"></label></td>
    </tr>`;
}

/* Shares aren't limited to the discovered volume roots — the container only
 * sees what's bind-mounted into it, so any directory can be typed here. It
 * posts to the same POST /api/cluster/share endpoint; the backend validates
 * that the path exists and is a directory. */
function renderAddShare(nodeId) {
  const id = 'addshare-' + cssId(nodeId);
  const draft = addShareDrafts[nodeId] || '';
  const arg = jsArg(nodeId);
  return `
    <div class="add-share">
      <label class="add-share-label" for="${id}">Add share</label>
      <input id="${id}" type="text" spellcheck="false" placeholder="/models/public"
             value="${escapeHtml(draft)}"
             oninput="onAddShareInput(${arg})"
             onkeydown="if(event.key==='Enter'){event.preventDefault();onAddShare(${arg});}">
      <button class="btn btn-neutral" onclick="onAddShare(${arg})">Add</button>
      <span class="muted add-share-note">any directory on this node — typically a subdirectory of a volume above</span>
    </div>`;
}

/* =========================================================================
 * SHARE / MOUNT ACTIONS
 * ======================================================================= */

/* JSON-encode a value as a JS string literal, then HTML-escape it, so it can
 * be inlined into a double-quoted on* attribute. Paths and share ids contain
 * ':' and '/' freely; quotes/backslashes survive the round trip. */
function jsArg(v) {
  return JSON.stringify(v == null ? '' : String(v))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function postAction(url, body) {
  try {
    const res = await apiFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    let data = {};
    try { data = await res.json(); } catch (e) { data = {}; }
    if (res.status === 404) return { ok: false, error: 'endpoint unavailable (' + url + ')' };
    if (!res.ok) return { ok: false, error: (data && data.error) || ('HTTP ' + res.status) };
    if (data && data.ok === false) return { ok: false, error: data.error || 'request failed' };
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message || 'request failed' };
  }
}

async function onShareToggle(cb, nodeId, path) {
  const enabled = !!cb.checked;
  if (MOCK) { mockSetShared(nodeId, path, enabled); renderLive(MOCK_SUMMARY, MOCK_NODES); return; }
  cb.disabled = true;
  pendingOps++;
  const r = await postAction('/api/cluster/share', { node_id: nodeId, path: path, enabled: enabled });
  pendingOps--;
  cb.disabled = false;
  if (r.ok) {
    showToast((enabled ? 'Sharing ' : 'Unshared ') + path + ' on ' + nodeId, 'success');
    refreshLive();
  } else {
    cb.checked = !enabled;                          // revert
    showToast('Share ' + path + ' failed: ' + r.error, 'error');
  }
}

async function onMountToggle(cb, nodeId, shareId, mountPath) {
  const enabled = !!cb.checked;
  const body = { node_id: nodeId, share_id: shareId, enabled: enabled };
  if (enabled && mountPath) body.mount_path = mountPath;
  if (MOCK) { mockSetMounted(shareId, nodeId, enabled, mountPath); renderLive(MOCK_SUMMARY, MOCK_NODES); return; }
  cb.disabled = true;
  pendingOps++;
  const r = await postAction('/api/cluster/mount', body);
  pendingOps--;
  cb.disabled = false;
  if (r.ok) {
    showToast((enabled ? 'Mounted ' : 'Unmounted ') + shareId + ' on ' + nodeId, 'success');
    refreshLive();
  } else {
    cb.checked = !enabled;                          // revert
    showToast('Mount ' + shareId + ' on ' + nodeId + ' failed: ' + r.error, 'error');
  }
}

function onAddShareInput(nodeId) {
  const el = $('addshare-' + cssId(nodeId));
  addShareDrafts[nodeId] = el ? el.value : '';
}

async function onAddShare(nodeId) {
  const el = $('addshare-' + cssId(nodeId));
  const path = el ? el.value.trim() : '';
  if (!path) { showToast('Enter a directory path to share.', 'warning'); if (el) el.focus(); return; }
  addShareDrafts[nodeId] = path;

  if (MOCK) {
    mockSetShared(nodeId, path, true);
    addShareDrafts[nodeId] = '';
    if (el) { el.value = ''; el.blur(); }
    renderLive(MOCK_SUMMARY, MOCK_NODES);
    showToast('Mock mode: shared ' + path, 'success');
    return;
  }

  if (el) el.disabled = true;
  pendingOps++;
  const r = await postAction('/api/cluster/share', { node_id: nodeId, path: path, enabled: true });
  pendingOps--;
  if (el) el.disabled = false;
  if (r.ok) {
    addShareDrafts[nodeId] = '';                    // clear only on success
    if (el) { el.value = ''; el.blur(); }           // blur so the refresh can redraw
    showToast('Sharing ' + path + ' on ' + nodeId, 'success');
    refreshLive();
  } else {
    // Leave the input populated so the path can be corrected.
    showToast('Cannot share ' + path + ': ' + r.error, 'error');
    if (el) el.focus();
  }
}

function addrHtml(n) {
  const a = n.addresses || {};
  const parts = [];
  if (a.mgmt) parts.push('mgmt ' + a.mgmt);
  if (a.fabric && a.fabric.length) parts.push('fabric ' + a.fabric.join(', '));
  return parts.length ? `<span class="host-addr">${escapeHtml(parts.join(' · '))}</span>` : '';
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

/* Volume/share sizes arrive as BYTES. Binary units, one decimal. */
function fmtBytes(b) {
  const n = Number(b);
  if (b == null || !isFinite(n)) return '—';
  const KiB = 1024, MiB = KiB * 1024, GiB = MiB * 1024, TiB = GiB * 1024, PiB = TiB * 1024;
  if (n >= PiB) return (n / PiB).toFixed(1) + ' PiB';
  if (n >= TiB) return (n / TiB).toFixed(1) + ' TiB';
  if (n >= GiB) return (n / GiB).toFixed(1) + ' GiB';
  if (n >= MiB) return (n / MiB).toFixed(1) + ' MiB';
  if (n >= KiB) return (n / KiB).toFixed(1) + ' KiB';
  return Math.round(n) + ' B';
}

function renderNode(host, n) {
  const roles = roleBadges(n.roles);
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

  const instances = (n.instances || []).length
    ? `<div class="sub-section"><div class="sub-title">Instances</div><div class="chip-row">${
        n.instances.map(r => `<span class="chip">${escapeHtml(r.id)} <span class="dim">${escapeHtml(r.state || '')}${r.port ? ' :' + r.port : ''}</span></span>`).join('')
      }</div></div>`
    : '';

  return `
    <div class="host-group">
      <div class="host-head">
        <span class="host-name">${escapeHtml(host)}</span>
        <span class="host-id">${escapeHtml(n.node_id || '')}</span>
        ${addrHtml(n)}
        ${roles}
        ${stateBadge}
      </div>
      ${gpuRows}
      ${instances}
      ${renderNodeShares(n)}
      ${renderNodeDetail(n)}
    </div>`;
}

/* ---- Worker card: node detail (NICs / RDMA fabric / mounted shares) ----
 * Collapsed by default so the GPU rows stay the headline; the open/closed
 * state is remembered across the 3s re-render (nodeDetailOpen). Every field
 * is optional — an older agent reporting none of them still renders. */
function renderNodeDetail(n) {
  const nodeId = n.node_id || '';
  const ifaces = usableInterfaces(n);
  const hcas = Array.isArray(n.rdma) ? n.rdma : [];
  const mounts = Array.isArray(n.mounts) ? n.mounts : [];

  const counts = [
    ifaces.length + ' NIC' + (ifaces.length === 1 ? '' : 's'),
    hcas.length ? hcas.length + ' HCA' + (hcas.length === 1 ? '' : 's') : '',
    mounts.length + ' mount' + (mounts.length === 1 ? '' : 's'),
  ].filter(Boolean).join(' · ');

  const nicBody = ifaces.length
    ? `<div class="nic-list">
         <div class="nic-row header"><span>NIC</span><span>IP</span><span>Subnet</span><span></span></div>
         ${ifaces.map(i => `
           <div class="nic-row">
             <span class="mono nic-name">${escapeHtml(i.name || '?')}</span>
             <span class="mono">${escapeHtml(i.ip || '—')}</span>
             <span class="mono dim">${escapeHtml(i.cidr || i.netmask || '—')}</span>
             <span>${i.rdma ? '<span class="badge rdma-badge">RDMA</span>' : ''}</span>
           </div>`).join('')}
       </div>`
    : '<div class="muted">no interfaces reported</div>';

  const rdmaBody = hcas.length
    ? `<div class="rdma-list">${hcas.map(r => `<div class="rdma-line mono">${rdmaLine(r)}</div>`).join('')}</div>`
    : '<div class="muted">no RDMA devices reported</div>';

  const mountBody = mounts.length
    ? `<div class="mount-list">${mounts.map(m => `
         <div class="mount-row">
           <span class="${m.ok === false ? 'bad' : 'ok'}">${m.ok === false ? '✗' : '✔'}</span>
           <span class="mono">${escapeHtml(m.path || '')}</span>
           <span class="mono dim">${escapeHtml(m.source || '')}</span>
         </div>`).join('')}</div>`
    : '<div class="muted">no shares mounted</div>';

  return `
    <details class="node-detail"${nodeDetailOpen[nodeId] ? ' open' : ''}
             ontoggle="onNodeDetailToggle(this, ${jsArg(nodeId)})">
      <summary>Node detail <span class="dim">${escapeHtml(counts)}</span></summary>
      <div class="node-detail-body">
        <div class="detail-block">
          <div class="sub-title">Network interfaces</div>
          ${nicBody}
        </div>
        <div class="detail-block">
          <div class="sub-title">RDMA fabric</div>
          ${rdmaBody}
        </div>
        <div class="detail-block">
          <div class="sub-title">Mounted shares</div>
          ${mountBody}
        </div>
      </div>
    </details>`;
}

// "mlx4_0 · ports 1,2 · GID 3 · enp196s0 · 172.16.254.201" — parts are skipped
// when the agent doesn't report them.
function rdmaLine(r) {
  r = r || {};
  const ports = Array.isArray(r.ports) ? r.ports : (r.ports == null || r.ports === '' ? [] : [r.ports]);
  const parts = [];
  if (r.hca) parts.push(escapeHtml(r.hca));
  if (ports.length) parts.push('ports ' + escapeHtml(ports.join(',')));
  if (r.gid_index != null && r.gid_index !== '') parts.push('GID ' + escapeHtml(r.gid_index));
  if (r.netdev) parts.push(escapeHtml(r.netdev));
  if (r.fabric_ip) parts.push(escapeHtml(r.fabric_ip));
  return parts.length ? parts.join(' <span class="dim">·</span> ') : '<span class="muted">unknown device</span>';
}

function onNodeDetailToggle(el, nodeId) {
  nodeDetailOpen[nodeId] = !!(el && el.open);
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
let configState = { defaults: {}, nodes: {}, settings: null };   // editable copy

function adoptConfig(data) {
  // Cloned: the poll keeps replacing `configSnapshot`, and edits here must not
  // leak into it (nor be wiped by it).
  configState = {
    defaults: deepClone(data.defaults || {}),
    nodes: deepClone(data.nodes || {}),
    settings: data.settings ? deepClone(data.settings) : null,
  };
}

async function loadConfig() {
  if (MOCK) { adoptConfig(mockConfig()); renderConfig(); configLoaded = true; return; }
  // The live poll already fetches /api/cluster/config whenever a storage node
  // exists — reuse that snapshot rather than issuing a second request.
  if (configSnapshot) { adoptConfig(configSnapshot); renderConfig(); configLoaded = true; return; }
  try {
    const res = await apiFetch('/api/cluster/config');
    if (res.status === 404) { renderConfigOffline(); return; }
    if (!res.ok) { renderConfigOffline('control plane returned ' + res.status); return; }
    const data = await res.json();
    configSnapshot = data;
    adoptConfig(data);
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
      <span class="save-note">Launch-time settings apply on the next instance start; live-safe settings (e.g. power cap) apply immediately.</span>
    </div>`;
}

function renderDefaultsFields() {
  const netKeys = networkKeySet();
  return SETTING_GROUPS.map(group => {
    const fields = group.fields.filter(f =>
      (f.scope === 'cluster' || f.scope === 'both') && !netKeys.has(String(f.key).toLowerCase()));
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
  const badges = roleBadges(String(roles).split(',').map(r => r.trim()).filter(Boolean));

  const netKeys = networkKeySet();
  const groups = SETTING_GROUPS.map(group => {
    // Network settings have their own block above; drop the legacy catalog's
    // copies (matched case-insensitively) so nothing gets two controls.
    const fields = group.fields.filter(f => !netKeys.has(String(f.key).toLowerCase()));
    if (!fields.length) return '';
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
          ${badges}
        </div>
      </div>
      ${renderNetworksBlock(nodeId)}
      ${groups}
      <details class="effective" open>
        <summary>Effective config (merged, read-only preview)</summary>
        <pre id="eff-${cssId(nodeId)}">${renderEffective(nodeId)}</pre>
      </details>
    </div>`;
}

/* ---- Configuration tab: the Networks block ----
 * One picker per network setting the server tags with group "network"
 * (cluster/control, compute NCCL, compute Gloo, storage). Options come from
 * the node's reported interfaces — the same list, bridge-filtered, for all of
 * them; only the value differs (ip vs NIC name, per `widget`).
 *
 * These write into configState.overrides exactly like every other field on
 * this tab, so "Save configuration" batches them into the one POST:
 *   { defaults: {...}, nodes: { <node_id>: { overrides: { ... } } } } */
function renderNetworksBlock(nodeId) {
  const settings = networkSettings();
  if (!settings.length) return '';
  const node = nodesById[nodeId] || {};
  const entry = configEntry(nodeId, true);
  const ifaces = usableInterfaces(node);

  const rows = settings.map(s => {
    const cur = networkValue(node, entry, s);
    const inputId = 'ov-' + cssId(nodeId) + '-' + s.key;
    const onIn = `onOverrideInput('${nodeId}','${s.key}')`;
    const control = ifaces.filter(i => ifaceValue(i, s)).length
      ? `<select id="${inputId}" oninput="${onIn}" onchange="${onIn}">${networkOptions(node, s, cur)}</select>`
      : `<input type="text" id="${inputId}" value="${escapeHtml(cur)}" oninput="${onIn}" onchange="${onIn}"
                placeholder="${escapeHtml(String(s.widget).indexOf('nic') === 0 ? 'NIC name' : 'IP address')}">`;
    return `
      <div class="net-row">
        <label class="net-label" for="${inputId}">${escapeHtml(s.label || s.key)}
          <code class="net-key">${escapeHtml(s.key)}</code></label>
        <div class="net-control">${control}</div>
        ${s.help ? `<div class="net-help muted">${escapeHtml(s.help)}</div>` : ''}
      </div>`;
  }).join('');

  const note = ifaces.length
    ? 'Pick which NIC carries each kind of traffic. Options are this node&rsquo;s interfaces (container bridges excluded).'
    : 'This node reports no interfaces &mdash; values can still be typed by hand.';

  return `
    <div class="cfg-group net-group">
      <h3>Networks</h3>
      <div class="cfg-section-note">${note}</div>
      <div class="net-grid">${rows}</div>
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
      // Reload to pick up server-recomputed effective config. Drop the poll's
      // snapshot first so loadConfig() re-reads instead of adopting pre-save
      // values.
      configSnapshot = null;
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
    roles: ['admin', 'participant', 'storage'], state: 'SERVING', connected: true,
    addresses: { mgmt: '192.168.1.10', fabric: ['172.16.254.201'] },
    // Container bridges are included on purpose: podman0/veth* must never
    // reach the NIC lists or the Networks selects.
    interfaces: [
      { name: 'ens2', ip: '192.168.1.10', netmask: '255.255.255.0', cidr: '192.168.1.0/24', rdma: false },
      { name: 'enp193s0', ip: '172.16.69.201', netmask: '255.255.255.0', cidr: '172.16.69.0/24', rdma: false },
      { name: 'enp196s0', ip: '172.16.254.201', netmask: '255.255.255.0', cidr: '172.16.254.0/24', rdma: true },
      { name: 'enp196s0d1', ip: '172.16.253.201', netmask: '255.255.255.0', cidr: '172.16.253.0/24', rdma: false },
      { name: 'podman0', ip: '10.88.0.1', netmask: '255.255.0.0', cidr: '10.88.0.0/16', rdma: false },
      { name: 'veth9f2c1a', ip: '10.88.0.7', netmask: '255.255.0.0', cidr: '10.88.0.0/16', rdma: false },
    ],
    rdma: [
      { hca: 'mlx4_0', ports: [1, 2], gid_index: 3, netdev: 'enp196s0', fabric_ip: '172.16.254.201' },
    ],
    gpus: [
      { index: 0, model: 'Tesla V100-SXM2-32GB', util: 96, mem_used: 30210, mem_total: 32768, temp: 71, power_draw: 288, power_limit: 300 },
      { index: 1, model: 'Tesla V100-SXM2-32GB', util: 92, mem_used: 29880, mem_total: 32768, temp: 68, power_draw: 271, power_limit: 300 },
      { index: 2, model: 'Tesla V100-SXM2-32GB', util: 4, mem_used: 512, mem_total: 32768, temp: 41, power_draw: 52, power_limit: 300 },
      { index: 3, model: 'Tesla V100-SXM2-32GB', util: 0, mem_used: 3, mem_total: 32768, temp: 38, power_draw: 46, power_limit: 300 },
    ],
    instances: [{ id: 'devstral-r0', state: 'HEALTHY', port: 8001 }],
    mounts: [{ path: '/export/models', source: 'local (served)', ok: true }],
  },
  {
    node_id: 'ebola-c3d4', hostname: 'ebola',
    roles: ['participant'], state: 'DEGRADED', connected: true,
    addresses: { mgmt: '192.168.1.11', fabric: ['172.16.254.202'] },
    interfaces: [
      { name: 'ens2', ip: '192.168.1.11', netmask: '255.255.255.0', cidr: '192.168.1.0/24', rdma: false },
      { name: 'enp193s0', ip: '172.16.69.202', netmask: '255.255.255.0', cidr: '172.16.69.0/24', rdma: false },
      { name: 'enp196s0', ip: '172.16.254.202', netmask: '255.255.255.0', cidr: '172.16.254.0/24', rdma: true },
      { name: 'enp196s0d1', ip: '172.16.253.202', netmask: '255.255.255.0', cidr: '172.16.253.0/24', rdma: false },
      { name: 'cni-podman0', ip: '10.88.0.1', netmask: '255.255.0.0', cidr: '10.88.0.0/16', rdma: false },
    ],
    rdma: [
      { hca: 'mlx4_0', ports: [1, 2], gid_index: 3, netdev: 'enp196s0', fabric_ip: '172.16.254.202' },
    ],
    gpus: [
      { index: 0, model: 'Tesla V100-PCIE-16GB', util: 88, mem_used: 15100, mem_total: 16384, temp: 86, power_draw: 148, power_limit: 150 },
      { index: 1, model: 'Tesla V100-PCIE-16GB', util: 0, mem_used: 4, mem_total: 16384, temp: 44, power_draw: 33, power_limit: 150 },
    ],
    instances: [{ id: 'devstral-r0', state: 'FAILED', port: 8001 }],
    mounts: [
      { path: '/models', source: 'covid:/models', ok: true },
      { path: '/export/models', source: 'covid:/export/models', ok: false },
    ],
  },
];

/* Sample GET /api/cluster/shares payload. COVID has two discovered volumes
 * (/models shared, /scratch not) plus one hand-added subdirectory share
 * (/models/public) that is NOT a volume root, so it renders as a "custom" row.
 * ebola has /models mounted; the custom share is advertised but unmounted, so
 * the Workers matrix shows both a ticked and an untickable-yet cell. */
const MOCK_SHARES = {
  shares: [
    {
      id: 'covid-a1b2:/models', node_id: 'covid-a1b2', path: '/models',
      endpoint: '172.16.254.201:2049', fstype: 'ext4',
      total: 983000000000, used: 442000000000, free: 541000000000, ok: true,
      mounted_by: [{ node_id: 'ebola-c3d4', path: '/models', ok: true }],
    },
    {
      id: 'covid-a1b2:/models/public', node_id: 'covid-a1b2', path: '/models/public',
      endpoint: '172.16.254.201:2050', fstype: 'ext4',
      total: 983000000000, used: 442000000000, free: 541000000000, ok: true,
      mounted_by: [],
    },
  ],
  volumes_by_node: {
    'covid-a1b2': [
      {
        path: '/models', fstype: 'ext4',
        total: 983000000000, used: 442000000000, free: 541000000000,
        shared: true, endpoint: '172.16.254.201:2049',
      },
      {
        path: '/scratch', fstype: 'xfs',
        total: 2000000000000, used: 118000000000, free: 1882000000000,
        shared: false,
      },
    ],
  },
};

// Live mock state (mutated by the Share / Mount checkboxes so ?mock=1 is
// interactive without a backend).
let mockShareData = null;
function mockShares() {
  if (!mockShareData) mockShareData = deepClone(MOCK_SHARES);
  return {
    status: 'ok',
    shares: mockShareData.shares,
    volumesByNode: mockShareData.volumes_by_node,
    detail: '',
  };
}

function mockSetShared(nodeId, path, enabled) {
  const d = mockShareData || (mockShareData = deepClone(MOCK_SHARES));
  const vols = d.volumes_by_node[nodeId] || [];
  const vol = vols.find(v => v.path === path);
  const id = nodeId + ':' + path;
  if (vol) vol.shared = enabled;
  if (enabled) {
    if (!d.shares.some(s => s.id === id)) {
      d.shares.push({
        id: id, node_id: nodeId, path: path,
        endpoint: (vol && vol.endpoint) || '172.16.254.201:2049',
        fstype: (vol && vol.fstype) || 'ext4',
        total: vol ? vol.total : null, used: vol ? vol.used : null, free: vol ? vol.free : null,
        ok: true, mounted_by: [],
      });
    }
  } else {
    d.shares = d.shares.filter(s => s.id !== id);
  }
  sharesState = mockShares();
}

function mockSetMounted(shareId, nodeId, enabled, mountPath) {
  const d = mockShareData || (mockShareData = deepClone(MOCK_SHARES));
  const s = d.shares.find(x => x.id === shareId);
  if (!s) return;
  s.mounted_by = s.mounted_by || [];
  if (enabled) {
    if (!s.mounted_by.some(m => m.node_id === nodeId)) {
      s.mounted_by.push({ node_id: nodeId, path: mountPath || s.path, ok: true });
    }
  } else {
    s.mounted_by = s.mounted_by.filter(m => m.node_id !== nodeId);
  }
  sharesState = mockShares();
}

// Totals as they'd come from GET /api/cluster/summary for the mock cluster.
const MOCK_SUMMARY = {
  nodes_total: 2,
  nodes_alive: 2,
  gpus_total: 6,
  roles: { admin: 1, participant: 2, storage: 1 },
  // Display names for the wire role ids, as the control plane now sends them.
  role_labels: { admin: 'Admin', participant: 'GPU Worker', storage: 'Storage' },
  instances: { HEALTHY: 1, FAILED: 1 },
};

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
  // Setting catalog as the control plane publishes it. Only the `network`
  // group is consumed by the UI today (it drives the Networks block); the
  // rest of the tab still renders from SETTING_GROUPS.
  settings: {
    ray_node_ip: {
      group: 'network', widget: 'ip', label: 'Cluster / control network',
      help: 'Control-plane and Ray traffic; also this node’s identity in the cluster.',
    },
    nccl_socket_ifname: {
      group: 'network', widget: 'nic-list', label: 'Compute (NCCL) network',
      help: 'NCCL collectives between instances. The local NIC is set first; the cluster appends the rest.',
    },
    gloo_socket_ifname: {
      group: 'network', widget: 'nic', label: 'Compute (Gloo) NIC',
      help: 'Gloo control collectives — a single local NIC.',
    },
    storage_bind_ip: {
      group: 'network', widget: 'ip', label: 'Storage network',
      help: 'Shares are served on this NIC. The port is chosen automatically and advertised to the cluster.',
    },
  },
  nodes: {
    'covid-a1b2': {
      hostname: 'covid',
      detected: {
        roles: 'admin,participant,storage',
        CLUSTER_ID: 'default',
        ray_node_ip: '172.16.69.201',
        nccl_socket_ifname: 'enp196s0,ens2',
        gloo_socket_ifname: 'enp196s0',
        NCCL_IB_HCA: 'mlx4_0:1',
        NCCL_IB_GID_INDEX: '3',
        NCCL_P2P_DISABLE: '1',
        gpu_power_cap_w: '300',
        storage_export_dir: '/export/models',
      },
      overrides: {
        NCCL_P2P_DISABLE: '1',
        storage_bind_ip: '172.16.254.201',
      },
      effective: {
        roles: 'admin,participant,storage',
        CLUSTER_ID: 'default',
        ray_node_ip: '172.16.69.201',
        nccl_socket_ifname: 'enp196s0,ens2',
        gloo_socket_ifname: 'enp196s0',
        storage_bind_ip: '172.16.254.201',
        gpu_power_cap_w: '300',
      },
    },
    'ebola-c3d4': {
      hostname: 'ebola',
      detected: {
        roles: 'participant',
        CLUSTER_ID: 'default',
        ray_node_ip: '172.16.69.202',
        RAY_HEAD_HOST: '172.16.69.201:6379',
        nccl_socket_ifname: 'enp196s0,ens2',
        gloo_socket_ifname: 'ens2',
        NCCL_IB_HCA: 'mlx4_0:1',
        NCCL_IB_GID_INDEX: '1',
        gpu_power_cap_w: '300',
        mount_canonical_path: '/export/models',
        mount_transport: 'rdma',
      },
      // ebola needs its local NIC first + the workaround power cap.
      overrides: {
        nccl_socket_ifname: 'enp196s0d1,enp196s0',
        gpu_power_cap_w: '150',
      },
      effective: {
        roles: 'participant',
        CLUSTER_ID: 'default',
        ray_node_ip: '172.16.69.202',
        RAY_HEAD_HOST: '172.16.69.201:6379',
        nccl_socket_ifname: 'enp196s0d1,enp196s0',
        gloo_socket_ifname: 'ens2',
        gpu_power_cap_w: '150',
        // storage_bind_ip deliberately unset: ebola has no storage role, so
        // its picker demonstrates the RDMA-interface fallback.
      },
    },
  },
};

/* Live mock config state — the network pickers mutate it so ?mock=1 round-trips
 * a change without a backend. */
let mockConfigData = null;
function mockConfig() {
  if (!mockConfigData) mockConfigData = deepClone(MOCK_CONFIG);
  return mockConfigData;
}

function mockSetOverride(nodeId, key, val) {
  const d = mockConfig();
  const node = d.nodes[nodeId] || (d.nodes[nodeId] = { overrides: {}, detected: {}, effective: {} });
  node.overrides = node.overrides || {};
  node.effective = node.effective || {};
  if (val === '' || val == null) { delete node.overrides[key]; delete node.effective[key]; }
  else { node.overrides[key] = val; node.effective[key] = val; }
  configSnapshot = d;
  configLoaded = false;                 // Configuration tab re-reads on open
}

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
  // Summary is the default tab; the live poll feeds Summary/Workers/Storage.
  loadLive();
  liveTimer = setInterval(loadLive, POLL_MS);
}

document.addEventListener('DOMContentLoaded', init);
