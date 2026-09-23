/* ============================================================================
 * serving.js — the SERVING panel for the v0.4.0 cluster UI (vllm-multi-gpu)
 *
 * The old single-node landing page's job, re-cast as one more tab next to
 * Summary / Workers / Storage / Configuration. Four sections, top to bottom:
 *
 *   (A) CLUSTER GPU UTILISATION — an at-a-glance strip across the whole
 *       cluster (GPUs total / busy / idle, aggregate memory) plus a compact
 *       per-GPU bar grid grouped by node. Polled ~3s from
 *       GET /api/cluster/nodes.
 *   (B) MODELS — the model library from GET /api/models as a dropdown with a
 *       Delete action on the selected model (POST /api/models/delete), plus a
 *       Hugging Face download form
 *       (POST /api/download, progress from GET /api/download/status). Carried
 *       over from index.html, same endpoints and field names.
 *   (C) LAUNCH A INSTANCE — pick a model, then tick GPUs ANYWHERE in the
 *       cluster. The layout is derived live from the ticks:
 *           TP = GPUs selected per node,  PP = number of distinct nodes
 *       (docs/04 — TP within a node, PP across nodes). An uneven selection is
 *       refused with an inline warning. Preview -> POST /api/cluster/plan
 *       renders the real layout / pp_layer_partition / per-node vllm_args
 *       before anything is committed; Launch -> POST /api/cluster/launch.
 *   (D) RUNNING INSTANCES — from GET /api/cluster/summary (`instances`) merged
 *       with each node's `instances[]` in GET /api/cluster/nodes. Stop ->
 *       POST /api/cluster/stop { instance_id, ray: true } (docs/04: ray:true is
 *       the default teardown — it bounces the Ray runtime and avoids the
 *       placement-group leak). Every row also has a Logs button -> (E).
 *   (E) CLUSTER LOG VIEWER — a modal over GET /api/cluster/logs that tails one
 *       instance's output from EVERY node running it, node-tagged and merged.
 *       A distributed launch fails on whichever node it fails on, and until
 *       this the only way to see that was to shell into a container: the real
 *       case that motivated it is a launch that hangs forever on
 *       `ray_utils.py: The number of required GPUs exceeds the total number of
 *       available GPUs` + endless `Waiting for creating a placement group`,
 *       which is invisible from the instance table alone. Follow/live polls
 *       every 2s, tail size 200/500/2000, per-node filter, Copy. It opens
 *       automatically on Launch so a coming-up (or failing) instance is
 *       watched live — see attachLogsToLaunch() for how the brand-new
 *       instance id is discovered.
 *
 * ---------------------------------------------------------------------------
 * ENTRY POINT
 *
 *   initServing(rootEl)     // builds the panel inside rootEl and starts polling
 *
 * Exported on `window`. It is also auto-wired to #serving-content on
 * DOMContentLoaded if that element exists, so dropping the <script> in is
 * enough; a second call on an already-initialised element is a no-op.
 *
 * Everything lives inside one IIFE and every generated class is prefixed
 * `sv-`: this file is loaded ALONGSIDE cluster.js on the same document, so it
 * must not redeclare `$`, `apiFetch`, `showToast`, `MOCK`, … at top level (a
 * duplicate top-level `const` is a SyntaxError that would break both scripts).
 * For the same reason there are no inline on* handlers — all interaction goes
 * through delegated listeners on the root element.
 *
 * ---------------------------------------------------------------------------
 * REQUEST BODIES (exact)
 *
 *   POST /api/cluster/plan   and   POST /api/cluster/launch
 *     {
 *       model:              "<model name from /api/models>",
 *       node_ids:           ["covid-a1b2", "ebola-c3d4"],   // distinct nodes
 *       port:               8001,
 *       served_model_name?: "devstral",
 *       max_model_len?:     32768,
 *       extra_args?:        ["--foo", "bar"],
 *       gpu_memory_utilization?: 0.85,          // forward-compatible, see below
 *       gpu_ids_by_node:    { "covid-a1b2": [0,1], "ebola-c3d4": [0,1] }
 *     }
 *   The scheduler API takes WHOLE NODES today (`node_ids`); it derives
 *   TP from each node's full GPU count. `gpu_ids_by_node` and
 *   `gpu_memory_utilization` are sent anyway and are forward-compatible: the
 *   current backend ignores unknown keys (pydantic drops extras; the hub reads
 *   named kwargs only), so nothing breaks, and the day the scheduler honours a
 *   GPU subset / a per-launch memory fraction the UI already sends them.
 *   Until then the picker's per-GPU ticks are how the USER expresses intent
 *   and how TP/PP are derived, and gpu_memory_utilization comes from each
 *   node's effective config — the form says so inline.
 *
 *   POST /api/cluster/stop     { instance_id: "instance-1", ray: true }
 *   GET  /api/cluster/logs?instance=<id>&tail=<n>[&node=<node_id>]
 *     -> { ok, instance_id, state, model,
 *          nodes: { "<node_id>": { ok, path, lines: [...] } },
 *          merged: [ { node_id, line }, ... ] }
 *     A node with nothing written yet answers ok:true / lines:[] /
 *     detail:"no log yet"; a node that could not be reached answers ok:false
 *     with an error — both are shown, never hidden.
 *   POST /api/download         { repo_id: "org/model" }  (+ revision when typed)
 *   POST /api/models/delete    { model: "<model name>" }
 *
 * Every endpoint may 404 on an older backend. Nothing here throws on that: the
 * section degrades to muted "unavailable" text and the other sections keep
 * working.
 *
 * ROLE NAMES are presentation-only: the wire id `participant` is displayed as
 * "GPU Worker" via roleLabel(), which reads GET /api/cluster/summary's
 * `role_labels` map and falls back to a built-in one. Every filter, request
 * and comparison still uses the raw wire ids.
 *
 * ?mock=1 (or window.SERVING_MOCK = true, which serving-preview.html sets)
 * renders a built-in sample cluster — covid 2×V100-32GB + ebola 2×V100-16GB,
 * two models, one running instance — with clickable preview/launch/stop, so the
 * panel can be reviewed with no backend running.
 * ==========================================================================*/
(function () {
  'use strict';

  var POLL_MS = 3000;

  var MOCK = (function () {
    try {
      if (window.SERVING_MOCK === true) return true;
      return new URLSearchParams(window.location.search).has('mock');
    } catch (e) { return false; }
  })();

  /* =======================================================================
   * Shared helpers — same behaviour as cluster.js / index.html, scoped here.
   * ===================================================================== */

  function getApiKey() {
    try { return localStorage.getItem('adminApiKey') || ''; } catch (e) { return ''; }
  }

  /* Cookie session (sent by default, same-origin) + the admin key as
   * X-API-Key when one is stored. 401 falls through to /login like the rest
   * of the admin UI. */
  async function apiFetch(url, opts) {
    opts = opts || {};
    var headers = Object.assign({}, opts.headers);
    var key = getApiKey();
    if (key) headers['X-API-Key'] = key;
    var res = await fetch(url, Object.assign({}, opts, { headers: headers }));
    if (res.status === 401) {
      window.location.href = '/login';
      throw new Error('unauthorized');
    }
    return res;
  }

  /* Toasts reuse the page's #toast-container (cluster.html / index.html) and
   * the shared .toast classes; one is created if the host page has none. */
  function showToast(msg, type) {
    type = type || 'error';
    var c = document.getElementById('toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'toast-container';
      c.className = 'toast-container';
      document.body.appendChild(c);
    }
    var t = document.createElement('div');
    t.className = 'toast toast-' + type;
    t.textContent = typeof msg === 'string' ? msg : JSON.stringify(msg);
    var dismiss = function () {
      t.style.animation = 'toast-out 0.2s ease forwards';
      setTimeout(function () { t.remove(); }, 200);
    };
    t.onclick = dismiss;
    c.appendChild(t);
    setTimeout(function () { if (t.parentNode) dismiss(); }, 5000);
  }

  // Escapes for text AND for quoted attribute values (innerHTML leaves quotes).
  function esc(s) {
    var d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function pct(a, b) { return b > 0 ? Math.max(0, Math.min(100, (a / b) * 100)) : 0; }

  // GPU memory arrives in MB (see GET /api/cluster/nodes).
  function fmtMb(mb) {
    if (mb == null) return '—';
    var n = Number(mb);
    if (!isFinite(n)) return '—';
    return n >= 1024 ? (n / 1024).toFixed(1) + ' GB' : Math.round(n) + ' MB';
  }

  // Download byte counters — decimal units, matching index.html's fmtBytes.
  function fmtBytes(b) {
    var n = Number(b);
    if (b == null || !isFinite(n)) return '—';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB';
    if (n >= 1e3) return (n / 1e3).toFixed(0) + ' KB';
    return Math.round(n) + ' B';
  }

  function hasRole(n, role) {
    return ((n && n.roles) || []).map(function (r) { return String(r).toLowerCase(); }).includes(role);
  }

  /* Role WIRE IDS never change (`participant` stays `participant` in every
   * filter, request and comparison) — only how they are SHOWN. The control
   * plane publishes the display names in GET /api/cluster/summary's
   * `role_labels`; an older backend that doesn't gets the built-in map, and an
   * unknown role falls back to its raw id. */
  var ROLE_LABEL_FALLBACK = { admin: 'Admin', participant: 'GPU Worker', storage: 'Storage' };

  function roleLabel(r) {
    var id = String(r == null ? '' : r);
    var fromServer = state.summary && state.summary.role_labels;
    if (fromServer && typeof fromServer === 'object' && fromServer[id]) return String(fromServer[id]);
    var key = id.toLowerCase();
    if (fromServer && typeof fromServer === 'object' && fromServer[key]) return String(fromServer[key]);
    return ROLE_LABEL_FALLBACK[key] || id;
  }

  function nodeName(n) { return (n && (n.hostname || n.node_id)) || 'unknown'; }

  /* A GPU counts as BUSY when it is doing work or holding weights: ≥10% util
   * or ≥10% of its memory in use. Idle is everything else (a few hundred MB of
   * driver/context memory must not read as busy). */
  function gpuBusy(g) {
    if (!g) return false;
    if (g.util != null && Number(g.util) >= 10) return true;
    if (g.mem_total > 0 && (Number(g.mem_used) / Number(g.mem_total)) >= 0.10) return true;
    return false;
  }

  function gpuKey(nodeId, index) { return String(nodeId) + '/' + String(index); }

  /* Split an extra-args string into argv. Honours single/double quotes so
   * `--rope-scaling '{"a":1}'` survives; everything else splits on whitespace. */
  function splitArgs(raw) {
    var out = [];
    var re = /"([^"]*)"|'([^']*)'|(\S+)/g;
    var m;
    while ((m = re.exec(String(raw || ''))) !== null) {
      out.push(m[1] !== undefined ? m[1] : (m[2] !== undefined ? m[2] : m[3]));
    }
    return out;
  }

  /* =======================================================================
   * PANEL STATE
   * ===================================================================== */

  var root = null;
  var timer = null;

  var state = {
    // Cluster telemetry (A, C, D)
    nodes: [],
    summary: null,
    clusterStatus: 'loading',        // loading | ok | unavailable | error
    clusterDetail: '',

    // Model library (B)
    models: [],
    modelsStatus: 'loading',         // loading | ok | unavailable | error
    modelsDetail: '',
    modelsSig: null,                 // last rendered library signature (B)

    // Download (B)
    download: null,
    downloadStatus: 'loading',       // loading | ok | unavailable
    dlLastState: '',
    dlLastBytes: 0,
    dlLastTime: 0,

    // GPU picker (C)
    selected: new Set(),             // "<node_id>/<gpu index>"
    pickerSig: '',                   // topology signature — rebuild vs update
    plan: null,                      // last /api/cluster/plan response
    planError: '',
    planBusy: false,
    launchBusy: false,

    // (E) log viewer
    logs: {
      open: false,
      instanceId: null,              // null while waiting for a launch to name one
      pending: false,                // launch submitted, instance id not known yet
      pendingNote: '',
      follow: true,                  // live: poll every 2s
      tail: 500,
      nodeFilter: '',                // '' = all nodes
      data: null,                    // last /api/cluster/logs body
      status: 'idle',                // idle | loading | ok | unavailable | error
      detail: '',
      busy: false,                   // a fetch is in flight
      stick: true,                   // auto-scroll armed (false once the user scrolls up)
      unseen: false,                 // new output arrived while un-stuck
      sig: '',                       // last rendered line signature
      nodeSig: '',                   // last rendered node-filter option set
      timer: null,                   // follow poll
      attachTimer: null,             // "which instance did my launch create?" poll
      attachUntil: 0,
      knownIds: null,                // instance ids that existed before the launch
    },
  };

  /* =======================================================================
   * SKELETON — built once. Anything the user types into (the download repo,
   * the launch form) lives in this static markup and is NEVER re-rendered by
   * the poll; only the list/readout containers are redrawn.
   * ===================================================================== */

  function skeleton() {
    return '' +
      // ---- (A) cluster GPU utilisation -------------------------------
      '<section class="sv-block">' +
        '<div class="card-head">' +
          '<h2 class="sv-h2">Cluster GPU utilisation</h2>' +
          '<span class="muted">polling <code>GET /api/cluster/nodes</code> every 3s</span>' +
        '</div>' +
        '<div id="sv-util"><div class="banner banner-empty">Loading cluster GPUs&hellip;</div></div>' +
      '</section>' +

      // ---- (B) models -------------------------------------------------
      '<section class="sv-block">' +
        '<div class="card-head">' +
          '<h2 class="sv-h2">Models</h2>' +
          '<span class="muted"><code>GET /api/models</code> &middot; the admin node&rsquo;s model directory</span>' +
        '</div>' +
        '<div class="sv-card">' +
          '<div class="sv-dl">' +
            '<label class="sv-dl-label" for="sv-dl-repo">Download from Hugging Face</label>' +
            '<input id="sv-dl-repo" type="text" spellcheck="false" placeholder="org/model — e.g. mistralai/Devstral-Small-2507">' +
            '<input id="sv-dl-rev" type="text" spellcheck="false" placeholder="revision (optional)">' +
            '<button class="btn btn-save" data-act="download" id="sv-dl-btn">Download</button>' +
            '<span class="muted sv-dl-note">Downloads land in the model directory and appear below when complete.</span>' +
          '</div>' +
          '<div id="sv-dl-progress"></div>' +
        '</div>' +
        '<div id="sv-models"><div class="banner banner-empty">Loading models&hellip;</div></div>' +
      '</section>' +

      // ---- (C) launch an instance ---------------------------------------
      '<section class="sv-block">' +
        '<div class="card-head">' +
          '<h2 class="sv-h2">Launch an instance</h2>' +
          '<span class="muted">TP within a node &middot; PP across nodes (docs/04)</span>' +
        '</div>' +
        '<div class="sv-card">' +
          '<div class="sv-form">' +
            '<div class="sv-field sv-field-wide">' +
              '<label for="sv-model">Model</label>' +
              '<select id="sv-model"><option value="">loading&hellip;</option></select>' +
            '</div>' +
            '<div class="sv-field">' +
              '<label for="sv-served-name">Served model name</label>' +
              '<input id="sv-served-name" type="text" spellcheck="false" placeholder="defaults to the model name">' +
            '</div>' +
            '<div class="sv-field">' +
              '<label for="sv-port">Port</label>' +
              '<input id="sv-port" type="number" min="1" max="65535" step="1" value="8001">' +
            '</div>' +
            '<div class="sv-field">' +
              '<label for="sv-max-len">Max model len</label>' +
              '<input id="sv-max-len" type="number" min="1" step="1" placeholder="model default">' +
            '</div>' +
            '<div class="sv-field">' +
              '<label for="sv-gpu-util">GPU memory utilization</label>' +
              '<input id="sv-gpu-util" type="number" min="0.1" max="1" step="0.01" placeholder="0.85">' +
            '</div>' +
            '<div class="sv-field sv-field-wide">' +
              '<label for="sv-extra-args">Extra vLLM args <span class="muted">(optional)</span></label>' +
              '<input id="sv-extra-args" type="text" spellcheck="false" placeholder="--swap-space 4 --tool-call-parser hermes">' +
            '</div>' +
          '</div>' +

          '<div class="sv-sub-title">GPUs — tick the ones this instance should use</div>' +
          '<div id="sv-picker"><div class="muted">Loading cluster GPUs&hellip;</div></div>' +

          '<div id="sv-layout"></div>' +

          '<div class="sv-actions">' +
            '<button class="btn btn-neutral" data-act="preview" id="sv-btn-preview" disabled>Preview plan</button>' +
            '<button class="btn btn-save" data-act="launch" id="sv-btn-launch" disabled>Launch</button>' +
            '<button class="btn btn-neutral" data-act="clear-sel" id="sv-btn-clear">Clear selection</button>' +
            '<span class="muted sv-actions-note">Preview is a dry run &mdash; <code>POST /api/cluster/plan</code> computes the layout without starting anything.</span>' +
          '</div>' +
          '<div id="sv-plan"></div>' +
        '</div>' +
      '</section>' +

      // ---- (D) running instances ---------------------------------------
      '<section class="sv-block">' +
        '<div class="card-head">' +
          '<h2 class="sv-h2">Running instances</h2>' +
          '<span class="muted"><code>GET /api/cluster/summary</code> + per-node <code>instances[]</code></span>' +
        '</div>' +
        '<div id="sv-instances"><div class="banner banner-empty">Loading instances&hellip;</div></div>' +
      '</section>' +

      // ---- (E) log viewer (modal, hidden until opened) -------------------
      // Built once and kept in the DOM: the controls the user touches (follow,
      // tail, node filter) must never be re-created by a poll, or a click
      // would land on a replaced element. Only the header readout and the
      // line list are redrawn.
      '<div class="sv-log-overlay" id="sv-log-modal" hidden>' +
        '<div class="sv-log-backdrop" data-act="log-close"></div>' +
        '<div class="sv-log-panel" role="dialog" aria-modal="true" aria-label="Instance logs">' +
          '<div class="sv-log-head">' +
            '<div class="sv-log-title">' +
              '<span class="sv-log-id mono" id="sv-log-id">&mdash;</span>' +
              '<span id="sv-log-state"></span>' +
              '<span class="muted sv-log-model mono" id="sv-log-model"></span>' +
            '</div>' +
            '<button class="btn btn-neutral sv-log-x" data-act="log-close" title="Close (Esc)">Close</button>' +
          '</div>' +
          '<div class="sv-log-nodes chip-row" id="sv-log-nodebar"></div>' +
          '<div class="sv-log-controls">' +
            '<button class="btn btn-save sv-log-follow" data-act="log-follow" id="sv-log-follow" aria-pressed="true">&#9679; Following</button>' +
            '<label class="sv-log-ctl"><span class="muted">tail</span>' +
              '<select id="sv-log-tail">' +
                '<option value="200">200</option>' +
                '<option value="500" selected>500</option>' +
                '<option value="2000">2000</option>' +
              '</select></label>' +
            '<label class="sv-log-ctl"><span class="muted">node</span>' +
              '<select id="sv-log-node"><option value="">All nodes</option></select></label>' +
            '<button class="btn btn-neutral" data-act="log-refresh" id="sv-log-refresh">Refresh</button>' +
            '<button class="btn btn-neutral" data-act="log-copy">Copy</button>' +
            '<span class="muted sv-log-count" id="sv-log-count"></span>' +
          '</div>' +
          '<div class="sv-log-scroll" id="sv-log-scroll" tabindex="0">' +
            '<div class="sv-log-lines" id="sv-log-lines"></div>' +
          '</div>' +
          '<button class="sv-log-jump" data-act="log-jump" id="sv-log-jump" hidden>&darr; new output &mdash; jump to latest</button>' +
          '<div class="sv-log-foot muted" id="sv-log-foot"></div>' +
        '</div>' +
      '</div>';
  }

  function el(id) { return root ? root.querySelector('#' + id) : null; }

  function val(id) {
    var e = el(id);
    return e ? String(e.value || '').trim() : '';
  }

  /* =======================================================================
   * POLLING — one cycle feeds every section. Each request is independent:
   * a 404 on one endpoint never stops the others.
   * ===================================================================== */

  async function poll() {
    if (MOCK) {
      mockTick();
      renderAll();
      return;
    }
    await Promise.all([loadCluster(), loadModels(), loadDownload()]);
    renderAll();
  }

  async function loadCluster() {
    try {
      var res = await apiFetch('/api/cluster/nodes');
      if (res.status === 404) {
        state.clusterStatus = 'unavailable'; state.nodes = []; state.summary = null;
        return;
      }
      if (!res.ok) {
        state.clusterStatus = 'error'; state.clusterDetail = 'HTTP ' + res.status;
        return;
      }
      var nodes = await res.json();
      state.nodes = Array.isArray(nodes) ? nodes : [];
      state.clusterStatus = 'ok';
      state.clusterDetail = '';
      try {
        var sres = await apiFetch('/api/cluster/summary');
        state.summary = sres.ok ? await sres.json() : null;
      } catch (e) { state.summary = null; }
    } catch (e) {
      state.clusterStatus = 'error';
      state.clusterDetail = (e && e.message) || 'request failed';
    }
    pruneSelection();
  }

  async function loadModels() {
    try {
      var res = await apiFetch('/api/models');
      if (res.status === 404) { state.modelsStatus = 'unavailable'; state.models = []; return; }
      if (!res.ok) { state.modelsStatus = 'error'; state.modelsDetail = 'HTTP ' + res.status; return; }
      var data = await res.json();
      // GET /api/models returns a JSON array of model-name strings.
      state.models = Array.isArray(data) ? data.filter(function (m) { return typeof m === 'string'; }) : [];
      state.modelsStatus = 'ok';
      state.modelsDetail = '';
    } catch (e) {
      state.modelsStatus = 'error';
      state.modelsDetail = (e && e.message) || 'request failed';
    }
  }

  async function loadDownload() {
    try {
      var res = await apiFetch('/api/download/status');
      if (!res.ok) { state.downloadStatus = 'unavailable'; state.download = null; return; }
      state.download = await res.json();
      state.downloadStatus = 'ok';
    } catch (e) {
      state.downloadStatus = 'unavailable';
      state.download = null;
    }
  }

  // Drop ticks for GPUs that no longer exist (node left, GPU disappeared).
  function pruneSelection() {
    if (!state.selected.size) return;
    var live = new Set();
    state.nodes.forEach(function (n) {
      (n.gpus || []).forEach(function (g) { live.add(gpuKey(n.node_id, g.index)); });
    });
    Array.from(state.selected).forEach(function (k) {
      if (!live.has(k)) state.selected.delete(k);
    });
  }

  function renderAll() {
    renderUtil();
    renderModels();
    renderDownload();
    renderModelSelect();
    renderPicker();
    renderLayout();
    renderInstances();
  }

  /* =======================================================================
   * (A) CLUSTER GPU UTILISATION
   * ===================================================================== */

  function clusterUnavailableHtml(what) {
    if (state.clusterStatus === 'unavailable') {
      return '<div class="muted">' + what + ' unavailable &mdash; this control plane does not expose ' +
        '<code>GET /api/cluster/nodes</code>.</div>';
    }
    if (state.clusterStatus === 'error') {
      return '<div class="muted">' + what + ' unavailable' +
        (state.clusterDetail ? ' &mdash; ' + esc(state.clusterDetail) : '') + '.</div>';
    }
    if (state.clusterStatus === 'loading') return '<div class="muted">Loading&hellip;</div>';
    return null;
  }

  function renderUtil() {
    var host = el('sv-util');
    if (!host) return;
    var un = clusterUnavailableHtml('Cluster GPU telemetry');
    if (un) { host.innerHTML = un; return; }

    var gpuNodes = state.nodes.filter(function (n) { return (n.gpus || []).length; });
    if (!gpuNodes.length) {
      host.innerHTML = '<div class="banner banner-empty">No GPUs reported by any node yet.</div>';
      return;
    }

    var total = 0, busy = 0, memUsed = 0, memTotal = 0, utilSum = 0, utilN = 0;
    gpuNodes.forEach(function (n) {
      (n.gpus || []).forEach(function (g) {
        total++;
        if (gpuBusy(g)) busy++;
        if (g.mem_used != null) memUsed += Number(g.mem_used);
        if (g.mem_total != null) memTotal += Number(g.mem_total);
        if (g.util != null) { utilSum += Number(g.util); utilN++; }
      });
    });
    var idle = total - busy;
    var avg = utilN ? Math.round(utilSum / utilN) : null;

    var tiles = '<div class="sv-stat-row">' +
      tile(String(total), 'GPUs in cluster') +
      tile('<span class="sv-busy">' + busy + '</span>', 'Busy') +
      tile('<span class="sv-idle">' + idle + '</span>', 'Idle') +
      tile(avg == null ? '—' : avg + '%', 'Mean utilisation') +
      tile(fmtMb(memUsed) + '<span class="stat-sub"> / ' + fmtMb(memTotal) + '</span>', 'GPU memory') +
      tile(String(gpuNodes.length), 'Nodes with GPUs') +
      '</div>';

    var grid = gpuNodes.map(function (n) {
      var bars = (n.gpus || []).map(gpuBar).join('');
      return '<div class="sv-node-strip">' +
        '<div class="sv-node-strip-head">' +
          '<span class="sv-node-name">' + esc(nodeName(n)) + '</span>' +
          '<span class="host-id">' + esc(n.node_id || '') + '</span>' +
          '<span class="muted">' + (n.gpus || []).length + ' GPU' + ((n.gpus || []).length === 1 ? '' : 's') + '</span>' +
        '</div>' +
        '<div class="sv-bar-grid">' + bars + '</div>' +
      '</div>';
    }).join('');

    host.innerHTML = tiles +
      '<div class="sv-legend muted">A GPU counts as <strong>busy</strong> at &ge;10% utilisation or &ge;10% memory in use.</div>' +
      grid;
  }

  function tile(valueHtml, label) {
    return '<div class="stat-tile"><span class="stat-val">' + valueHtml + '</span>' +
      '<span class="stat-label">' + esc(label) + '</span></div>';
  }

  function gpuBar(g) {
    var u = g.util == null ? 0 : Math.max(0, Math.min(100, Number(g.util)));
    var m = pct(g.mem_used, g.mem_total);
    return '<div class="sv-gpu-bar' + (gpuBusy(g) ? ' sv-gpu-busy' : '') + '" title="' +
        esc((g.model || 'GPU') + ' — ' + (g.util == null ? 'util ?' : 'util ' + u + '%') +
            ', ' + fmtMb(g.mem_used) + ' / ' + fmtMb(g.mem_total)) + '">' +
      '<span class="sv-gpu-bar-head">' +
        '<span class="gpu-idx">' + esc(g.index) + '</span>' +
        '<span class="sv-gpu-pct">' + (g.util == null ? '—' : u.toFixed(0) + '%') + '</span>' +
      '</span>' +
      '<span class="bar"><span class="bar-fill bar-util" style="width:' + u + '%"></span></span>' +
      '<span class="bar"><span class="bar-fill bar-mem" style="width:' + m + '%"></span></span>' +
      '<span class="sv-gpu-mem">' + fmtMb(g.mem_used) + ' <span class="unit">/ ' + fmtMb(g.mem_total) + '</span></span>' +
    '</div>';
  }

  /* =======================================================================
   * (B) MODELS + DOWNLOAD
   * ===================================================================== */

  /* The library is a <select> + Delete (it used to be a row-per-model table).
   * Redrawn only when the library itself changes — otherwise the 3s poll would
   * reset the chosen model every cycle. The picked model is carried across a
   * rebuild when it survived, so a download landing elsewhere in the list does
   * not move the selection out from under a pending Delete. */
  function modelsSig() {
    return [state.modelsStatus, state.modelsDetail, state.models.join('\u0000')].join('|');
  }

  function renderModels() {
    var host = el('sv-models');
    if (!host) return;
    var sig = modelsSig();
    if (sig === state.modelsSig) return;
    state.modelsSig = sig;

    if (state.modelsStatus === 'unavailable') {
      host.innerHTML = '<div class="muted">Model library unavailable &mdash; this admin does not expose ' +
        '<code>GET /api/models</code>.</div>';
      return;
    }
    if (state.modelsStatus === 'error') {
      host.innerHTML = '<div class="muted">Model library unavailable' +
        (state.modelsDetail ? ' &mdash; ' + esc(state.modelsDetail) : '') + '.</div>';
      return;
    }
    if (state.modelsStatus === 'loading') {
      host.innerHTML = '<div class="banner banner-empty">Loading models&hellip;</div>';
      return;
    }
    if (!state.models.length) {
      host.innerHTML = '<div class="banner banner-empty">No models found. Download one above.</div>';
      return;
    }

    var prevSel = el('sv-model-list');
    var prev = prevSel ? prevSel.value : '';
    var opts = state.models.map(function (m) {
      return '<option value="' + esc(m) + '"' + (m === prev ? ' selected' : '') + '>' + esc(m) + '</option>';
    }).join('');

    host.innerHTML = '<div class="sv-card sv-library">' +
        '<div class="sv-library-row">' +
          '<div class="sv-field">' +
            '<label for="sv-model-list">Installed models</label>' +
            '<select id="sv-model-list">' + opts + '</select>' +
          '</div>' +
          '<button class="btn btn-danger" data-act="delete-model">Delete</button>' +
        '</div>' +
        '<div class="muted sv-library-note">' + state.models.length + ' model' +
          (state.models.length === 1 ? '' : 's') + ' in the model directory. ' +
          'Delete removes the selected one from disk.</div>' +
      '</div>';
  }

  function renderDownload() {
    var host = el('sv-dl-progress');
    var btn = el('sv-dl-btn');
    if (!host) return;
    if (state.downloadStatus !== 'ok' || !state.download) {
      host.innerHTML = '';
      if (btn) btn.disabled = false;
      return;
    }
    var d = state.download;
    var st = String(d.status || 'idle');

    if (st === 'idle') {
      host.innerHTML = '';
      if (btn) btn.disabled = false;
      state.dlLastBytes = 0; state.dlLastTime = 0;
    } else if (st === 'downloading') {
      if (btn) btn.disabled = true;
      var body = '<div class="sv-dl-state">Downloading <span class="mono">' + esc(d.repo_id || '') + '</span>&hellip;</div>';
      if (d.total_bytes > 0) {
        var p = Math.min(100, (Number(d.downloaded_bytes) / Number(d.total_bytes)) * 100);
        var now = Date.now();
        var speed = '';
        if (state.dlLastTime > 0 && now > state.dlLastTime) {
          var delta = Number(d.downloaded_bytes) - state.dlLastBytes;
          var dt = (now - state.dlLastTime) / 1000;
          if (delta > 0 && dt > 0) speed = ' — ' + fmtBytes(delta / dt) + '/s';
        }
        state.dlLastBytes = Number(d.downloaded_bytes) || 0;
        state.dlLastTime = now;
        body += '<div class="sv-progress"><span class="sv-progress-fill" style="width:' + p.toFixed(1) + '%"></span></div>' +
          '<div class="muted">' + p.toFixed(1) + '% — ' + fmtBytes(d.downloaded_bytes) +
          ' / ' + fmtBytes(d.total_bytes) + esc(speed) + '</div>';
      }
      host.innerHTML = body;
    } else if (st === 'complete') {
      if (btn) btn.disabled = false;
      host.innerHTML = '<div class="sv-dl-state sv-ok">Download complete: <span class="mono">' +
        esc(d.repo_id || '') + '</span>' + (d.total_bytes > 0 ? ' — ' + fmtBytes(d.total_bytes) : '') + '</div>';
      if (state.dlLastState === 'downloading') showToast('Download complete: ' + (d.repo_id || ''), 'success');
      state.dlLastBytes = 0; state.dlLastTime = 0;
    } else if (st === 'error') {
      if (btn) btn.disabled = false;
      host.innerHTML = '<div class="sv-dl-state sv-bad">Download failed: ' + esc(d.error || 'unknown error') + '</div>';
      state.dlLastBytes = 0; state.dlLastTime = 0;
    }
    state.dlLastState = st;
  }

  async function onDownload() {
    var repo = val('sv-dl-repo');
    var revision = val('sv-dl-rev');
    if (!repo) { showToast('Enter a Hugging Face repo ID', 'warning'); var r = el('sv-dl-repo'); if (r) r.focus(); return; }
    // POST /api/download { repo_id } (+ revision when one was typed — the
    // current backend's DownloadRequest ignores unknown keys).
    var body = { repo_id: repo };
    if (revision) body.revision = revision;

    if (MOCK) { mockStartDownload(repo, revision); renderDownload(); return; }

    var btn = el('sv-dl-btn');
    if (btn) btn.disabled = true;
    var r2 = await post('/api/download', body);
    if (!r2.ok) {
      showToast('Could not start download: ' + r2.error, 'error');
      if (btn) btn.disabled = false;
      return;
    }
    showToast('Downloading ' + repo, 'info');
    loadDownload().then(renderDownload);
  }

  // The Delete button acts on whatever the library <select> currently shows.
  function selectedLibraryModel() {
    var sel = el('sv-model-list');
    return sel ? String(sel.value || '') : '';
  }

  async function onDeleteModel(model) {
    if (!model) { showToast('Select a model to delete', 'warning'); return; }
    if (!MOCK && !window.confirm('Delete "' + model + '"? This cannot be undone.')) return;
    if (MOCK) {
      state.models = state.models.filter(function (m) { return m !== model; });
      showToast('Mock mode: deleted ' + model, 'success');
      renderModels(); renderModelSelect();
      return;
    }
    // POST /api/models/delete { model }
    var r = await post('/api/models/delete', { model: model });
    if (!r.ok) { showToast('Delete failed: ' + r.error, 'error'); return; }
    showToast('Model deleted: ' + model, 'success');
    await loadModels();
    renderModels(); renderModelSelect();
  }

  /* =======================================================================
   * (C) LAUNCH — model select, cluster-wide GPU picker, derived layout
   * ===================================================================== */

  // The <select> is rebuilt only when the option set changes, so a chosen
  // model survives the 3s poll.
  function renderModelSelect() {
    var sel = el('sv-model');
    if (!sel) return;
    var want = state.models.slice();
    if (!want.length) {
      var placeholder = state.modelsStatus === 'loading' ? 'loading…'
        : (state.modelsStatus === 'ok' ? 'no models available' : 'model list unavailable');
      if (sel.options.length !== 1 || sel.options[0].textContent !== placeholder) {
        sel.innerHTML = '<option value="">' + esc(placeholder) + '</option>';
      }
      return;
    }
    var have = Array.prototype.map.call(sel.options, function (o) { return o.value; }).filter(Boolean);
    if (have.length === want.length && have.every(function (v, i) { return v === want[i]; })) return;
    var current = sel.value;
    sel.innerHTML = want.map(function (m) {
      return '<option value="' + esc(m) + '"' + (m === current ? ' selected' : '') + '>' + esc(m) + '</option>';
    }).join('');
  }

  // Nodes that can actually run an instance, in a stable order.
  function participantNodes() {
    return state.nodes.filter(function (n) {
      return (n.gpus || []).length && (!(n.roles || []).length || hasRole(n, 'participant'));
    });
  }

  /* Topology signature: rebuild the picker markup only when the set of nodes
   * or GPUs changes. Otherwise the poll just refreshes the numbers in place,
   * so a checkbox never loses focus and no tick flickers mid-click. */
  function pickerSignature(nodes) {
    return nodes.map(function (n) {
      return n.node_id + '[' + (n.roles || []).map(roleLabel).join('+') + ']:' +
        (n.gpus || []).map(function (g) { return g.index + '/' + (g.model || ''); }).join(',');
    }).join('|');
  }

  function renderPicker() {
    var host = el('sv-picker');
    if (!host) return;
    var un = clusterUnavailableHtml('GPU picker');
    if (un) { host.innerHTML = un; state.pickerSig = ''; return; }

    var nodes = participantNodes();
    if (!nodes.length) {
      host.innerHTML = '<div class="muted">No ' + esc(roleLabel('participant')) +
        ' node is reporting GPUs &mdash; nothing to select.</div>';
      state.pickerSig = '';
      return;
    }

    var sig = pickerSignature(nodes);
    if (sig !== state.pickerSig) {
      host.innerHTML = nodes.map(renderPickerNode).join('');
      state.pickerSig = sig;
    } else {
      updatePickerMetrics(nodes);
    }
  }

  function renderPickerNode(n) {
    var gpus = (n.gpus || []).map(function (g) {
      var key = gpuKey(n.node_id, g.index);
      return '<label class="sv-gpu-pick' + (gpuBusy(g) ? ' sv-gpu-pick-busy' : '') + '" data-gpu="' + esc(key) + '">' +
        '<input type="checkbox" data-act="pick" data-key="' + esc(key) + '"' +
          (state.selected.has(key) ? ' checked' : '') + '>' +
        '<span class="sv-pick-body">' +
          '<span class="sv-pick-top">' +
            '<span class="gpu-idx">GPU ' + esc(g.index) + '</span>' +
            '<span class="sv-pick-state">' + (gpuBusy(g) ? 'busy' : 'idle') + '</span>' +
          '</span>' +
          '<span class="sv-pick-model" title="' + esc(g.model || '') + '">' + esc(g.model || '—') + '</span>' +
          '<span class="sv-pick-mem muted">' + fmtMb(g.mem_used) + ' / ' + fmtMb(g.mem_total) +
            ' · ' + (g.util == null ? '—' : Math.round(g.util) + '% util') + '</span>' +
        '</span>' +
      '</label>';
    }).join('');

    return '<div class="sv-pick-node">' +
      '<div class="sv-pick-node-head">' +
        '<label class="chk"><input type="checkbox" data-act="pick-node" data-node="' + esc(n.node_id) + '"' +
          (allPicked(n) ? ' checked' : '') + '><span class="sv-node-name">' + esc(nodeName(n)) + '</span></label>' +
        '<span class="host-id">' + esc(n.node_id || '') + '</span>' +
        // Roles are shown by their display label (participant -> "GPU Worker");
        // the wire id is what every filter above still matches on.
        (n.roles || []).map(function (r) {
          return '<span class="badge role-badge">' + esc(roleLabel(r)) + '</span>';
        }).join('') +
        '<span class="badge state-badge ' + stateClass(n.state) + '"><span class="state-dot"></span>' +
          esc(String(n.state || 'unknown').toUpperCase()) + '</span>' +
      '</div>' +
      '<div class="sv-pick-grid">' + gpus + '</div>' +
    '</div>';
  }

  function allPicked(n) {
    var gpus = n.gpus || [];
    return gpus.length > 0 && gpus.every(function (g) { return state.selected.has(gpuKey(n.node_id, g.index)); });
  }

  // Same markup, fresh numbers — no DOM replacement, so focus/ticks survive.
  function updatePickerMetrics(nodes) {
    nodes.forEach(function (n) {
      (n.gpus || []).forEach(function (g) {
        var key = gpuKey(n.node_id, g.index);
        var card = root.querySelector('.sv-gpu-pick[data-gpu="' + cssEscape(key) + '"]');
        if (!card) return;
        card.classList.toggle('sv-gpu-pick-busy', gpuBusy(g));
        var st = card.querySelector('.sv-pick-state');
        if (st) st.textContent = gpuBusy(g) ? 'busy' : 'idle';
        var mem = card.querySelector('.sv-pick-mem');
        if (mem) {
          mem.textContent = fmtMb(g.mem_used) + ' / ' + fmtMb(g.mem_total) +
            ' · ' + (g.util == null ? '—' : Math.round(g.util) + '% util');
        }
      });
      var nodeBox = root.querySelector('input[data-act="pick-node"][data-node="' + cssEscape(n.node_id) + '"]');
      if (nodeBox && document.activeElement !== nodeBox) nodeBox.checked = allPicked(n);
    });
  }

  // Minimal attribute-selector escaping — node ids and keys contain '/' and ':'.
  function cssEscape(s) { return String(s).replace(/(["\\])/g, '\\$1'); }

  function stateClass(s) {
    var v = String(s || '').toLowerCase();
    return ['ready', 'serving', 'degraded', 'down'].includes(v) ? 'state-' + v : 'state-unknown';
  }

  /* ---- Layout derivation -------------------------------------------------
   * Group the ticked GPUs by node, in the node order the cluster reports:
   *     TP = GPUs selected per node        (tensor parallel, within a node)
   *     PP = number of distinct nodes      (pipeline parallel, across nodes)
   * vLLM requires the SAME tensor-parallel size on every pipeline stage, so an
   * uneven selection (covid 2, ebola 1) is not a layout at all — it is flagged
   * inline and Preview/Launch stay disabled. */
  function selectionByNode() {
    var out = [];
    state.nodes.forEach(function (n) {
      var idxs = (n.gpus || [])
        .map(function (g) { return g.index; })
        .filter(function (i) { return state.selected.has(gpuKey(n.node_id, i)); });
      if (idxs.length) out.push({ node: n, gpus: idxs });
    });
    return out;
  }

  function deriveLayout() {
    var sel = selectionByNode();
    var counts = sel.map(function (s) { return s.gpus.length; });
    var tp = counts.length ? counts[0] : 0;
    var even = counts.every(function (c) { return c === tp; });
    return {
      sel: sel,
      counts: counts,
      pp: sel.length,
      tp: even ? tp : null,
      world: even ? tp * sel.length : null,
      even: even,
      empty: sel.length === 0,
    };
  }

  function renderLayout() {
    var host = el('sv-layout');
    if (!host) return;
    var d = deriveLayout();
    var model = val('sv-model');
    var ok = !d.empty && d.even && !!model;

    var preview = el('sv-btn-preview');
    var launch = el('sv-btn-launch');
    if (preview) preview.disabled = !ok || state.planBusy;
    if (launch) launch.disabled = !ok || state.launchBusy;

    if (d.empty) {
      host.innerHTML = '<div class="sv-layout sv-layout-idle muted">' +
        'Tick one or more GPUs above. <strong>TP</strong> = GPUs per node, <strong>PP</strong> = number of nodes.</div>';
      return;
    }

    var chips = d.sel.map(function (s) {
      return '<span class="chip">' + esc(nodeName(s.node)) +
        ' <span class="dim">GPU ' + s.gpus.join(', ') + '</span></span>';
    }).join('');

    if (!d.even) {
      var detail = d.sel.map(function (s) { return nodeName(s.node) + ' ' + s.gpus.length; }).join(', ');
      host.innerHTML = '<div class="sv-layout sv-layout-bad">' +
        '<div class="sv-layout-warn">uneven selection: ' + esc(detail) +
          ' &mdash; TP must match across nodes</div>' +
        '<div class="muted">Every pipeline stage runs the same tensor-parallel size. Select the same number of GPUs on each node, or drop a node from the selection.</div>' +
        '<div class="chip-row sv-layout-chips">' + chips + '</div>' +
      '</div>';
      return;
    }

    var executor = d.pp > 1 ? 'ray' : 'mp';
    var notes = [];
    notes.push('executor <code>' + executor + '</code>' + (d.pp > 1 ? ' (multi-node)' : ' (single node)'));
    if (d.pp > 1) notes.push('<code>--enforce-eager</code> is forced for cluster PP (docs/04)');
    if (d.world > 1) notes.push('<code>--disable-custom-all-reduce</code> on this hardware');
    if (!model) notes.push('<span class="sv-warn-text">pick a model to enable Launch</span>');

    host.innerHTML = '<div class="sv-layout sv-layout-ok">' +
      '<div class="sv-layout-row">' +
        '<span class="sv-kpi"><span class="sv-kpi-val">' + d.tp + '</span><span class="sv-kpi-label">TP<span class="muted"> / node</span></span></span>' +
        '<span class="sv-kpi-x">×</span>' +
        '<span class="sv-kpi"><span class="sv-kpi-val">' + d.pp + '</span><span class="sv-kpi-label">PP<span class="muted"> nodes</span></span></span>' +
        '<span class="sv-kpi-x">=</span>' +
        '<span class="sv-kpi"><span class="sv-kpi-val">' + d.world + '</span><span class="sv-kpi-label">world size</span></span>' +
      '</div>' +
      '<div class="chip-row sv-layout-chips">' + chips + '</div>' +
      '<div class="muted sv-layout-notes">' + notes.join(' · ') + '</div>' +
    '</div>';
  }

  /* ---- Plan / launch bodies --------------------------------------------- */

  function launchBody() {
    var d = deriveLayout();
    var body = {
      model: val('sv-model'),
      node_ids: d.sel.map(function (s) { return s.node.node_id; }),
      port: parseInt(val('sv-port'), 10) || 8001,
    };
    var served = val('sv-served-name');
    if (served) body.served_model_name = served;
    var maxLen = parseInt(val('sv-max-len'), 10);
    if (maxLen > 0) body.max_model_len = maxLen;
    var extra = splitArgs(val('sv-extra-args'));
    if (extra.length) body.extra_args = extra;
    var util = parseFloat(val('sv-gpu-util'));
    // Forward-compatible: today the scheduler reads gpu_memory_utilization from
    // each node's effective config and ignores this key.
    if (isFinite(util) && util > 0 && util <= 1) body.gpu_memory_utilization = util;
    // Forward-compatible: the plan/launch API schedules WHOLE nodes today, but
    // the user picked individual GPUs — send them so a future scheduler can
    // honour the subset without a UI change.
    body.gpu_ids_by_node = {};
    d.sel.forEach(function (s) { body.gpu_ids_by_node[s.node.node_id] = s.gpus.slice(); });
    return body;
  }

  async function onPreview() {
    var d = deriveLayout();
    if (d.empty || !d.even) return;
    var body = launchBody();
    if (!body.model) { showToast('Select a model first', 'warning'); return; }

    state.planBusy = true; state.planError = ''; renderLayout();
    var r = MOCK ? mockPlan(body) : await post('/api/cluster/plan', body, true);
    state.planBusy = false;
    if (!r.ok) {
      state.plan = null;
      state.planError = r.error || 'plan failed';
      showToast('Plan failed: ' + state.planError, 'error');
    } else {
      state.plan = r.data;
      state.planError = '';
    }
    renderLayout();
    renderPlan();
  }

  async function onLaunch() {
    var d = deriveLayout();
    if (d.empty || !d.even) return;
    var body = launchBody();
    if (!body.model) { showToast('Select a model first', 'warning'); return; }

    state.launchBusy = true; renderLayout();

    /* Open the viewer BEFORE the POST and start looking for the instance it
     * creates — the POST does not return until every node has finished
     * starting (minutes for a large model), and everything worth seeing
     * happens in that window. See startAttachTimer() for the two prongs:
     * the id in the eventual POST response, and the id that appears in
     * GET /api/cluster/nodes seconds after the control plane registers it. */
    var before = knownInstanceIds();
    openLogs(null, {
      pending: true,
      follow: true,
      note: body.model + ' on ' + body.node_ids.join(', '),
    });
    startAttachTimer(before);

    var r = MOCK ? mockLaunch(body) : await post('/api/cluster/launch', body, true);
    state.launchBusy = false;
    if (!r.ok) {
      showToast('Launch failed: ' + (r.error || 'unknown error'), 'error');
      // Keep the viewer open if it managed to attach — the logs say WHY.
      if (state.logs.open && !state.logs.instanceId) {
        stopAttachTimer();
        state.logs.pending = false;
        state.logs.pendingNote = '';
        state.logs.status = 'error';
        state.logs.detail = r.error || 'launch failed';
        state.logs.sig = '';
        renderLogs();
      }
    } else {
      var rid = (r.data && r.data.instance_id) || 'instance';
      showToast('Launched ' + rid + ' — ' + body.model + ' on ' + body.node_ids.join(', '), 'success');
      // Prong 1: the authoritative id, if the attach poll has not found it.
      if (r.data && r.data.instance_id) attachLogs(r.data.instance_id);
      if (MOCK) { mockTick(); renderAll(); } else { poll(); }
    }
    renderLayout();
  }

  function renderPlan() {
    var host = el('sv-plan');
    if (!host) return;
    if (state.planError) {
      host.innerHTML = '<div class="sv-plan"><div class="sv-layout-warn">' + esc(state.planError) + '</div></div>';
      return;
    }
    var p = state.plan;
    if (!p) { host.innerHTML = ''; return; }

    var lay = p.layout || {};
    var head = '<div class="sv-plan-head">' +
      '<span class="badge role-badge">TP ' + esc(lay.tp) + '</span>' +
      '<span class="badge role-badge">PP ' + esc(lay.pp) + '</span>' +
      '<span class="badge role-badge">world ' + esc(lay.world) + '</span>' +
      (lay.head_id ? '<span class="badge role-badge">head ' + esc(lay.head_id) + '</span>' : '') +
      (p.port ? '<span class="badge role-badge">port ' + esc(p.port) + '</span>' : '') +
    '</div>';

    var order = Array.isArray(lay.ordered_ids) ? lay.ordered_ids : [];
    var orderLine = order.length
      ? '<div class="sv-plan-line"><span class="sv-plan-key">ordered_ids</span> <span class="mono">' +
        esc(order.join(' → ')) + '</span> <span class="muted">driver first</span></div>'
      : '';
    var partLine = '<div class="sv-plan-line"><span class="sv-plan-key">pp_layer_partition</span> <span class="mono">' +
      (p.pp_layer_partition ? esc(p.pp_layer_partition) : '—') + '</span> ' +
      '<span class="muted">' + (p.pp_layer_partition ? 'memory-weighted layer split' : 'not applicable (PP = 1, or layer count unknown)') + '</span></div>';

    var cmds = Array.isArray(p.commands) ? p.commands : [];
    var perNode = cmds.length
      ? cmds.map(function (c) {
          var b = (c && c.body) || {};
          var l = b.layout || {};
          var args = Array.isArray(b.vllm_args) ? b.vllm_args : [];
          var envKeys = Object.keys(b.env || {});
          var lines = [
            'role            ' + (b.role_in_instance || '?'),
            'model           ' + (b.model || ''),
            'port            ' + (b.port == null ? '(worker — no API port)' : b.port),
            'tp / pp         ' + l.tp + ' / ' + l.pp + '   executor=' + (l.executor || '?'),
            'pp_partition    ' + (l.pp_layer_partition || '—'),
            'max_model_len   ' + (b.max_model_len == null ? '(model default)' : b.max_model_len),
            'gpu_mem_util    ' + (b.gpu_memory_utilization == null ? '(node default)' : b.gpu_memory_utilization),
            'dtype           ' + (b.dtype || 'auto'),
            'vllm_args       ' + (args.length ? args.join(' ') : '(none)'),
            'env             ' + (envKeys.length ? envKeys.map(function (k) { return k + '=' + b.env[k]; }).join('\n                ') : '(none)'),
          ].join('\n');
          return '<div class="sv-plan-node">' +
            '<div class="sv-plan-node-head"><span class="sv-node-name">' + esc(c.node_id) + '</span>' +
              '<span class="badge role-badge">' + esc(b.role_in_instance || 'node') + '</span></div>' +
            '<pre class="sv-pre">' + esc(lines) + '</pre>' +
          '</div>';
        }).join('')
      : '<div class="muted">The plan returned no per-node commands.</div>';

    host.innerHTML = '<div class="sv-plan">' +
      '<div class="sv-sub-title">Plan preview <span class="muted">— read-only, nothing has been started</span></div>' +
      head + orderLine + partLine +
      '<div class="sv-plan-nodes">' + perNode + '</div>' +
    '</div>';
  }

  /* =======================================================================
   * (D) RUNNING INSTANCES
   * ===================================================================== */

  /* Merge two views of the same thing:
   *   - GET /api/cluster/summary -> instances: { "<instance_id>": "SERVING" }
   *     (older/derived summaries may instead count by state — those values are
   *     numbers and carry no instance id, so they are ignored here)
   *   - GET /api/cluster/nodes -> each node's instances[]: { id, state, port, role }
   */
  function collectInstances() {
    var byId = {};
    var sum = (state.summary && state.summary.instances) || {};
    Object.keys(sum).forEach(function (rid) {
      var v = sum[rid];
      if (typeof v !== 'string') return;             // a state->count map, not ids
      byId[rid] = { id: rid, state: v, nodes: [], port: null, roles: {} };
    });
    state.nodes.forEach(function (n) {
      (n.instances || []).forEach(function (r) {
        if (!r || !r.id) return;
        var e = byId[r.id] || (byId[r.id] = { id: r.id, state: r.state || 'UNKNOWN', nodes: [], port: null, roles: {} });
        e.nodes.push(nodeName(n));
        if (r.port) e.port = r.port;
        if (r.role || r.role_in_instance) e.roles[nodeName(n)] = r.role || r.role_in_instance;
        // A per-node FAILED beats an optimistic cluster-level state.
        if (String(r.state || '').toUpperCase() === 'FAILED') e.state = 'FAILED';
        else if (!e.state) e.state = r.state;
      });
    });
    return Object.keys(byId).map(function (k) { return byId[k]; });
  }

  function renderInstances() {
    var host = el('sv-instances');
    if (!host) return;
    var un = clusterUnavailableHtml('Instance list');
    if (un) { host.innerHTML = un; return; }

    var reps = collectInstances();
    if (!reps.length) {
      host.innerHTML = '<div class="banner banner-empty">No instances running. Launch one above.</div>';
      return;
    }
    var rows = reps.map(function (r) {
      var st = String(r.state || 'unknown').toUpperCase();
      var nodes = r.nodes.length
        ? r.nodes.map(function (h) {
            var role = r.roles[h];
            return '<span class="chip">' + esc(h) + (role ? ' <span class="dim">' + esc(role) + '</span>' : '') + '</span>';
          }).join('')
        : '<span class="muted">—</span>';
      return '<tr>' +
        '<td class="mono">' + esc(r.id) + '</td>' +
        '<td><span class="badge state-badge ' + instanceStateClass(st) + '"><span class="state-dot"></span>' + esc(st) + '</span></td>' +
        '<td><div class="chip-row">' + nodes + '</div></td>' +
        '<td class="mono">' + (r.port ? ':' + esc(r.port) : '—') + '</td>' +
        '<td class="num"><div class="sv-row-actions">' +
          '<button class="btn btn-neutral" data-act="logs-instance" data-instance="' + esc(r.id) + '">Logs</button>' +
          '<button class="btn btn-danger" data-act="stop-instance" data-instance="' + esc(r.id) + '">Stop</button>' +
        '</div></td>' +
      '</tr>';
    }).join('');

    host.innerHTML = '<div class="table-wrap"><table class="summary-table">' +
      '<thead><tr><th>Instance</th><th>State</th><th>Nodes</th><th>Port</th><th class="num">Actions</th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table></div>' +
      '<div class="muted sv-legend"><strong>Logs</strong> tails this instance on every node that runs it (<code>GET /api/cluster/logs</code>) — the only place a multi-node failure is visible. ' +
      'Stop sends <code>{ instance_id, ray: true }</code> — bouncing the Ray runtime with the instance is the default teardown (docs/04: it avoids the placement-group leak).</div>';
  }

  function instanceStateClass(st) {
    var s = String(st).toLowerCase();
    if (s === 'serving' || s === 'healthy' || s === 'running') return 'state-serving';
    if (s === 'starting' || s === 'pending') return 'state-ready';
    if (s === 'failed' || s === 'stopped') return 'state-down';
    if (s === 'degraded') return 'state-degraded';
    return 'state-unknown';
  }

  async function onStopInstance(rid) {
    if (!rid) return;
    if (!MOCK && !window.confirm('Stop instance "' + rid + '"? Ray will be bounced on its nodes.')) return;
    if (MOCK) { mockStopInstance(rid); renderAll(); showToast('Mock mode: stopped ' + rid, 'success'); return; }
    // POST /api/cluster/stop { instance_id, ray: true }
    var r = await post('/api/cluster/stop', { instance_id: rid, ray: true }, true);
    if (!r.ok) { showToast('Stop failed: ' + r.error, 'error'); return; }
    showToast('Stopped ' + rid, 'success');
    poll();
  }

  /* =======================================================================
   * (E) CLUSTER LOG VIEWER
   *
   * One instance, every node that runs it, merged and node-tagged:
   *   GET /api/cluster/logs?instance=<id>&tail=<n>
   *
   * The per-node filter is applied CLIENT-SIDE even though the endpoint takes
   * &node=<node_id>: the response already carries every node, so flipping the
   * filter is instant and never loses the other nodes' backlog. The tail size
   * does go to the server (it is what bounds the read).
   * ===================================================================== */

  var LOG_POLL_MS = 2000;
  var LOG_COLOURS = 6;               // sv-log-c0 … sv-log-c5 in serving.css

  // Node ids are long ("covid-a1b2"); the tag shows the hostname when the
  // cluster poll knows it, so head vs worker reads at a glance.
  function logNodeLabel(nodeId) {
    var id = String(nodeId == null ? '' : nodeId);
    var n = state.nodes.find(function (x) { return x.node_id === id; });
    if (n) return nodeName(n);
    return id || 'unknown';
  }

  // Stable colour per node: index into the sorted node key list of THIS
  // response, so covid is always one colour and ebola another.
  function logNodeKeys() {
    var d = state.logs.data;
    var keys = d && d.nodes ? Object.keys(d.nodes) : [];
    if (!keys.length && d && Array.isArray(d.merged)) {
      d.merged.forEach(function (m) {
        if (m && m.node_id && keys.indexOf(m.node_id) === -1) keys.push(m.node_id);
      });
    }
    return keys.slice().sort();
  }

  function logNodeColour(nodeId) {
    var i = logNodeKeys().indexOf(String(nodeId));
    return 'sv-log-c' + (i < 0 ? 0 : (i % LOG_COLOURS));
  }

  /* Log lines are raw process output: strip ANSI colour codes and any other
   * C0 control characters (tab survives) so nothing can corrupt the markup. */
  var ANSI_RE = new RegExp(String.fromCharCode(27) + '\\[[0-9;?]*[ -\\/]*[@-~]', 'g');

  function cleanLogLine(s) {
    var str = String(s == null ? '' : s).replace(ANSI_RE, '');
    var out = '';
    for (var i = 0; i < str.length; i++) {
      var c = str.charCodeAt(i);
      if (c === 9) { out += '    '; continue; }          // tab -> spaces
      if (c < 32 || c === 127) continue;                 // drop every other C0
      out += str.charAt(i);
    }
    return out;
  }

  // The lines currently on screen, after the node filter.
  function logEntries() {
    var d = state.logs.data;
    if (!d) return [];
    var list = [];
    if (Array.isArray(d.merged) && d.merged.length) {
      list = d.merged.map(function (m) {
        return { node_id: (m && m.node_id) || '', line: (m && m.line) || '' };
      });
    } else {
      var nodes = (d && d.nodes) || {};
      Object.keys(nodes).forEach(function (nid) {
        (((nodes[nid] || {}).lines) || []).forEach(function (line) {
          list.push({ node_id: nid, line: line });
        });
      });
    }
    var f = state.logs.nodeFilter;
    if (f) list = list.filter(function (e) { return e.node_id === f; });
    return list;
  }

  function openLogs(instanceId, opts) {
    var L = state.logs;
    opts = opts || {};
    var changed = L.instanceId !== instanceId;
    L.open = true;
    L.instanceId = instanceId || null;
    L.pending = !!opts.pending;
    L.pendingNote = opts.note || '';
    if (opts.follow != null) L.follow = !!opts.follow;
    if (changed) {
      L.data = null; L.status = instanceId ? 'loading' : 'idle';
      L.detail = ''; L.sig = ''; L.nodeSig = '';
      L.nodeFilter = ''; L.stick = true; L.unseen = false;
    }
    var modal = el('sv-log-modal');
    if (modal) modal.hidden = false;
    var nodeSel = el('sv-log-node');
    if (nodeSel && changed) nodeSel.value = '';
    syncFollowButton();
    renderLogs();
    document.addEventListener('keydown', onLogKeydown);
    if (L.instanceId) fetchLogs();
    startLogTimer();
  }

  function closeLogs() {
    var L = state.logs;
    L.open = false;
    L.pending = false;
    stopLogTimer();
    stopAttachTimer();
    document.removeEventListener('keydown', onLogKeydown);
    var modal = el('sv-log-modal');
    if (modal) modal.hidden = true;
  }

  function onLogKeydown(ev) {
    if (ev.key === 'Escape' && state.logs.open) closeLogs();
  }

  function startLogTimer() {
    stopLogTimer();
    if (!state.logs.follow) return;
    state.logs.timer = setInterval(function () {
      if (!state.logs.open) { stopLogTimer(); return; }
      if (state.logs.instanceId) fetchLogs();
    }, LOG_POLL_MS);
  }

  function stopLogTimer() {
    if (state.logs.timer) { clearInterval(state.logs.timer); state.logs.timer = null; }
  }

  function syncFollowButton() {
    var b = el('sv-log-follow');
    if (!b) return;
    var on = state.logs.follow;
    b.textContent = on ? '● Following' : '■ Paused';
    b.className = 'btn sv-log-follow ' + (on ? 'btn-save' : 'btn-neutral');
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
    var r = el('sv-log-refresh');
    if (r) r.disabled = on;
  }

  async function fetchLogs() {
    var L = state.logs;
    if (!L.open || !L.instanceId || L.busy) return;
    L.busy = true;
    if (L.status === 'idle') L.status = 'loading';
    try {
      if (MOCK) {
        L.data = mockLogs(L.instanceId, L.tail);
        L.status = 'ok'; L.detail = '';
      } else {
        var url = '/api/cluster/logs?instance=' + encodeURIComponent(L.instanceId) +
          '&tail=' + encodeURIComponent(L.tail);
        var res = await apiFetch(url);
        if (res.status === 404) {
          L.status = 'unavailable'; L.detail = '';
        } else if (!res.ok) {
          L.status = 'error'; L.detail = 'HTTP ' + res.status;
        } else {
          var d = await res.json();
          if (d && d.ok === false) {
            L.status = 'error'; L.detail = d.error || 'request failed';
          } else {
            L.data = d; L.status = 'ok'; L.detail = '';
          }
        }
      }
    } catch (e) {
      L.status = 'error';
      L.detail = (e && e.message) || 'request failed';
    }
    L.busy = false;
    renderLogs();
  }

  function renderLogs() {
    var L = state.logs;
    if (!L.open) return;
    var host = el('sv-log-lines');
    if (!host) return;

    var d = L.data || {};
    // ---- header readout ------------------------------------------------
    var idEl = el('sv-log-id');
    if (idEl) idEl.textContent = L.instanceId || (L.pending ? 'starting…' : '—');
    var stEl = el('sv-log-state');
    if (stEl) {
      var st = String(d.state || (L.pending ? 'STARTING' : '')).toUpperCase();
      stEl.innerHTML = st
        ? '<span class="badge state-badge ' + instanceStateClass(st) + '"><span class="state-dot"></span>' + esc(st) + '</span>'
        : '';
    }
    var mdEl = el('sv-log-model');
    if (mdEl) mdEl.textContent = d.model ? String(d.model) : '';

    renderLogNodeBar();

    // ---- body ----------------------------------------------------------
    if (L.status === 'unavailable') {
      host.innerHTML = '<div class="sv-log-msg muted">Log viewer unavailable on this control plane ' +
        '&mdash; it does not expose <code>GET /api/cluster/logs</code>.</div>';
      L.sig = 'unavailable';
      setLogFoot('');
      setLogCount('');
      return;
    }

    var entries = logEntries();
    var sig = [L.status, L.nodeFilter, entries.length,
               entries.length ? entries[entries.length - 1].line : ''].join('|');

    if (sig !== L.sig) {
      if (!entries.length) {
        var msg;
        if (L.pending) {
          msg = 'Launch submitted &mdash; waiting for the control plane to register the instance&hellip;' +
            (L.pendingNote ? '<div class="muted">' + L.pendingNote + '</div>' : '');
        } else if (L.status === 'loading') {
          msg = 'Loading&hellip;';
        } else if (L.status === 'error') {
          msg = 'Could not read logs' + (L.detail ? ' &mdash; ' + esc(L.detail) : '') + '.';
        } else if (L.nodeFilter) {
          msg = 'No output from <span class="mono">' + esc(logNodeLabel(L.nodeFilter)) + '</span> yet.';
        } else {
          msg = 'Waiting for output&hellip; <span class="muted">(normal for the first seconds of a launch)</span>';
        }
        host.innerHTML = '<div class="sv-log-msg muted">' + msg + '</div>';
      } else {
        host.innerHTML = entries.map(function (e) {
          return '<div class="sv-log-line ' + logNodeColour(e.node_id) + '">' +
            '<span class="sv-log-tag">' + esc(logNodeLabel(e.node_id)) + '</span>' +
            '<span class="sv-log-text">' + esc(cleanLogLine(e.line)) + '</span>' +
          '</div>';
        }).join('');
      }
      // New output while the user has scrolled up: flag it, never yank.
      if (L.sig && L.sig !== sig && !L.stick && entries.length) L.unseen = true;
      L.sig = sig;
      if (L.stick) scrollLogsToBottom();
    }
    syncJumpButton();

    setLogCount(entries.length
      ? entries.length + ' line' + (entries.length === 1 ? '' : 's') +
        (L.nodeFilter ? ' · ' + logNodeLabel(L.nodeFilter) : ' · all nodes')
      : '');

    // Footer: log paths per node + any transport error on the request itself.
    var foot = [];
    var nodes = d.nodes || {};
    Object.keys(nodes).forEach(function (nid) {
      var v = nodes[nid] || {};
      if (v.path) foot.push(esc(logNodeLabel(nid)) + ': <span class="mono">' + esc(v.path) + '</span>');
    });
    if (L.status === 'error' && L.detail && entries.length) {
      foot.unshift('<span class="sv-bad">last refresh failed: ' + esc(L.detail) + '</span>');
    }
    setLogFoot(foot.join(' · '));
  }

  function setLogCount(txt) {
    var c = el('sv-log-count');
    if (c) c.textContent = txt;
  }

  function setLogFoot(html) {
    var f = el('sv-log-foot');
    if (f) f.innerHTML = html;
  }

  /* Which nodes are reporting — and, crucially, which are NOT: a node that
   * answered ok:false shows its error here instead of silently vanishing, and
   * one with nothing written yet says so. */
  function renderLogNodeBar() {
    var bar = el('sv-log-nodebar');
    if (!bar) return;
    var d = state.logs.data || {};
    var nodes = d.nodes || {};
    var keys = Object.keys(nodes);
    if (!keys.length) {
      bar.innerHTML = state.logs.pending
        ? '<span class="muted">no node is reporting this instance yet</span>' : '';
    } else {
      bar.innerHTML = keys.map(function (nid) {
        var v = nodes[nid] || {};
        var lines = (v.lines || []).length;
        var cls = 'chip sv-log-chip ' + logNodeColour(nid);
        if (v.ok === false) {
          return '<span class="' + cls + ' sv-log-chip-bad" title="' + esc(v.error || 'node error') + '">' +
            esc(logNodeLabel(nid)) + ' <span class="dim">' + esc(v.error || 'unreachable') + '</span></span>';
        }
        var note = lines ? lines + ' lines' : (v.detail || 'no log yet');
        return '<span class="' + cls + '">' + esc(logNodeLabel(nid)) +
          ' <span class="dim">' + esc(note) + '</span></span>';
      }).join('');
    }
    // Keep the filter <select> in step with the nodes actually present.
    var sel = el('sv-log-node');
    if (!sel) return;
    var sig = keys.join(',');
    if (sig === state.logs.nodeSig) return;
    state.logs.nodeSig = sig;
    var cur = state.logs.nodeFilter;
    sel.innerHTML = '<option value="">All nodes</option>' + keys.map(function (nid) {
      return '<option value="' + esc(nid) + '"' + (nid === cur ? ' selected' : '') + '>' +
        esc(logNodeLabel(nid)) + '</option>';
    }).join('');
    if (keys.indexOf(cur) === -1) { state.logs.nodeFilter = ''; sel.value = ''; }
  }

  function scrollLogsToBottom() {
    var sc = el('sv-log-scroll');
    if (!sc) return;
    sc.scrollTop = sc.scrollHeight;
    state.logs.stick = true;
    state.logs.unseen = false;
    syncJumpButton();
  }

  function syncJumpButton() {
    var b = el('sv-log-jump');
    if (b) b.hidden = !(state.logs.unseen && !state.logs.stick);
  }

  /* Auto-scroll is armed only while the view is already at the bottom. Scroll
   * up and it disarms — the poll keeps running, the text stays put, and the
   * "jump to latest" button appears. */
  function onLogScroll(ev) {
    var sc = ev.currentTarget;
    var atBottom = (sc.scrollHeight - sc.scrollTop - sc.clientHeight) < 24;
    state.logs.stick = atBottom;
    if (atBottom) state.logs.unseen = false;
    syncJumpButton();
  }

  function onLogCopy() {
    var entries = logEntries();
    if (!entries.length) { showToast('Nothing to copy yet', 'warning'); return; }
    var text = entries.map(function (e) {
      return '[' + logNodeLabel(e.node_id) + '] ' + cleanLogLine(e.line);
    }).join('\n');
    copyText(text).then(function (ok) {
      showToast(ok ? ('Copied ' + entries.length + ' lines') : 'Copy failed — select the text instead',
        ok ? 'success' : 'error');
    });
  }

  /* navigator.clipboard is unavailable on insecure origins (and on the file://
   * preview), so fall back to the old textarea + execCommand path. */
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(function () { return true; },
        function () { return legacyCopy(text); });
    }
    return Promise.resolve(legacyCopy(text));
  }

  function legacyCopy(text) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-1000px';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand('copy');
      ta.remove();
      return ok;
    } catch (e) { return false; }
  }

  /* ---- attaching the viewer to a brand-new instance ----------------------
   * STRATEGY (two-pronged, because POST /api/cluster/launch does not answer
   * until every node has finished starting — minutes, for a big model):
   *
   *   1. The POST's own response is authoritative when it eventually lands:
   *      it carries `instance_id`, so onLaunch() attaches with it if the
   *      viewer has not attached already.
   *   2. Long before that, the control plane has already REGISTERED the
   *      instance (hub.launch() writes it as STARTING before it sends a
   *      single node command), so its id shows up in GET /api/cluster/nodes
   *      (per-node `instances[]`) and in GET /api/cluster/summary
   *      (`instances` map) within a poll or two. We snapshot the set of known
   *      instance ids immediately BEFORE the POST, then poll the cluster
   *      every 1.5s and attach to the first id that was not in that snapshot.
   *
   * Prong 2 is what normally wins, which is the point: the user watches the
   * instance come up (or hang on a placement group) while the POST is still
   * in flight. The attach poll gives up after 5 minutes, or as soon as the
   * viewer is closed. */
  function knownInstanceIds() {
    var s = new Set();
    collectInstances().forEach(function (r) { s.add(r.id); });
    return s;
  }

  function startAttachTimer(before) {
    stopAttachTimer();
    var L = state.logs;
    L.knownIds = before;
    L.attachUntil = Date.now() + 5 * 60 * 1000;
    L.attachTimer = setInterval(async function () {
      if (!state.logs.open || state.logs.instanceId) { stopAttachTimer(); return; }
      if (Date.now() > state.logs.attachUntil) {
        stopAttachTimer();
        state.logs.pendingNote = 'No new instance appeared in 5 minutes — the launch may have been rejected.';
        renderLogs();
        return;
      }
      if (!MOCK) await loadCluster();
      renderInstances();
      var fresh = null;
      collectInstances().forEach(function (r) {
        if (!state.logs.knownIds.has(r.id) && !fresh) fresh = r.id;
      });
      if (fresh) { stopAttachTimer(); attachLogs(fresh); }
    }, 1500);
  }

  function stopAttachTimer() {
    if (state.logs.attachTimer) { clearInterval(state.logs.attachTimer); state.logs.attachTimer = null; }
  }

  function attachLogs(instanceId) {
    if (!instanceId || !state.logs.open) return;
    if (state.logs.instanceId === instanceId) return;
    stopAttachTimer();
    openLogs(instanceId, { follow: true });
  }

  /* =======================================================================
   * POST helper — never throws; 404 reads as "endpoint unavailable".
   * `wantData` returns the parsed body alongside ok.
   * ===================================================================== */
  async function post(url, body, wantData) {
    try {
      var res = await apiFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      var data = {};
      try { data = await res.json(); } catch (e) { data = {}; }
      if (res.status === 404) return { ok: false, error: 'endpoint unavailable (' + url + ')' };
      if (!res.ok) return { ok: false, error: (data && data.error) || ('HTTP ' + res.status) };
      if (data && data.ok === false) return { ok: false, error: data.error || 'request failed' };
      return wantData ? { ok: true, data: data } : { ok: true };
    } catch (e) {
      return { ok: false, error: (e && e.message) || 'request failed' };
    }
  }

  /* =======================================================================
   * EVENT DELEGATION — no inline on* handlers, so nothing needs to be global.
   * ===================================================================== */

  function onClick(ev) {
    var t = ev.target.closest('[data-act]');
    if (!t || !root.contains(t)) return;
    var act = t.dataset.act;
    if (act === 'download') { ev.preventDefault(); onDownload(); }
    else if (act === 'delete-model') { ev.preventDefault(); onDeleteModel(t.dataset.model || selectedLibraryModel()); }
    else if (act === 'preview') { ev.preventDefault(); onPreview(); }
    else if (act === 'launch') { ev.preventDefault(); onLaunch(); }
    else if (act === 'stop-instance') { ev.preventDefault(); onStopInstance(t.dataset.instance); }
    else if (act === 'logs-instance') { ev.preventDefault(); openLogs(t.dataset.instance, { follow: true }); }
    else if (act === 'log-close') { ev.preventDefault(); closeLogs(); }
    else if (act === 'log-copy') { ev.preventDefault(); onLogCopy(); }
    else if (act === 'log-refresh') { ev.preventDefault(); fetchLogs(); }
    else if (act === 'log-jump') { ev.preventDefault(); scrollLogsToBottom(); }
    else if (act === 'log-follow') {
      ev.preventDefault();
      state.logs.follow = !state.logs.follow;
      syncFollowButton();
      if (state.logs.follow) { scrollLogsToBottom(); fetchLogs(); startLogTimer(); }
      else stopLogTimer();
    }
    else if (act === 'clear-sel') {
      ev.preventDefault();
      state.selected.clear();
      state.plan = null; state.planError = '';
      state.pickerSig = '';                       // force a redraw of the ticks
      renderPicker(); renderLayout(); renderPlan();
    }
  }

  /* Any change to the selection or the launch form invalidates the preview —
   * a plan on screen must always describe what the form currently says. */
  function invalidatePlan(t) {
    if (!state.plan && !state.planError) return;
    if (t.dataset.act === 'pick' || t.dataset.act === 'pick-node' || (t.closest && t.closest('.sv-form'))) {
      state.plan = null; state.planError = '';
      renderPlan();
    }
  }

  function onChange(ev) {
    var t = ev.target;
    if (!t || !t.dataset) return;
    invalidatePlan(t);
    if (t.dataset.act === 'pick') {
      if (t.checked) state.selected.add(t.dataset.key);
      else state.selected.delete(t.dataset.key);
      syncNodeBoxes();
      renderLayout();
    } else if (t.dataset.act === 'pick-node') {
      var node = state.nodes.find(function (n) { return n.node_id === t.dataset.node; });
      if (node) {
        (node.gpus || []).forEach(function (g) {
          var key = gpuKey(node.node_id, g.index);
          if (t.checked) state.selected.add(key); else state.selected.delete(key);
          var box = root.querySelector('input[data-act="pick"][data-key="' + cssEscape(key) + '"]');
          if (box) box.checked = t.checked;
        });
      }
      renderLayout();
    } else if (t.id === 'sv-model') {
      renderLayout();            // Launch is gated on a model being chosen
    } else if (t.id === 'sv-log-tail') {
      // tail is a server-side bound — refetch.
      state.logs.tail = parseInt(t.value, 10) || 500;
      state.logs.sig = '';
      state.logs.stick = true;
      fetchLogs();
    } else if (t.id === 'sv-log-node') {
      // the node filter is client-side — the response already has every node.
      state.logs.nodeFilter = String(t.value || '');
      state.logs.sig = '';
      state.logs.stick = true;
      renderLogs();
    }
  }

  function syncNodeBoxes() {
    state.nodes.forEach(function (n) {
      var box = root.querySelector('input[data-act="pick-node"][data-node="' + cssEscape(n.node_id) + '"]');
      if (box) box.checked = allPicked(n);
    });
  }

  /* =======================================================================
   * MOCK DATA (?mock=1 / window.SERVING_MOCK) — covid 2×V100-32GB +
   * ebola 2×V100-16GB, two models, one instance already serving.
   * ===================================================================== */

  var MOCK_NODES = [
    {
      node_id: 'covid-a1b2', hostname: 'covid',
      roles: ['admin', 'participant', 'storage'], state: 'SERVING', connected: true,
      addresses: { mgmt: '192.168.1.10', fabric: ['172.16.254.201'] },
      gpus: [
        { index: 0, model: 'Tesla V100-SXM2-32GB', util: 94, mem_used: 30210, mem_total: 32768, temp: 71, power_draw: 288, power_limit: 300 },
        { index: 1, model: 'Tesla V100-SXM2-32GB', util: 91, mem_used: 29880, mem_total: 32768, temp: 68, power_draw: 271, power_limit: 300 },
      ],
      instances: [{ id: 'devstral-r0', state: 'SERVING', port: 8001, role: 'head' }],
    },
    {
      node_id: 'ebola-c3d4', hostname: 'ebola',
      roles: ['participant'], state: 'SERVING', connected: true,
      addresses: { mgmt: '192.168.1.11', fabric: ['172.16.254.202'] },
      gpus: [
        { index: 0, model: 'Tesla V100-PCIE-16GB', util: 88, mem_used: 15100, mem_total: 16384, temp: 79, power_draw: 148, power_limit: 150 },
        { index: 1, model: 'Tesla V100-PCIE-16GB', util: 2, mem_used: 312, mem_total: 16384, temp: 44, power_draw: 33, power_limit: 150 },
      ],
      instances: [{ id: 'devstral-r0', state: 'SERVING', port: 8001, role: 'worker' }],
    },
  ];

  var MOCK_MODELS = ['Devstral-Small-2507', 'Qwen2.5-7B-Instruct'];

  var mockData = null;

  function mockInit() {
    if (mockData) return mockData;
    mockData = {
      nodes: JSON.parse(JSON.stringify(MOCK_NODES)),
      models: MOCK_MODELS.slice(),
      instanceStates: { 'devstral-r0': 'SERVING' },
      download: { status: 'idle', repo_id: null, downloaded_bytes: 0, total_bytes: 0, error: null },
      nextInstance: 1,
    };
    return mockData;
  }

  // One poll tick against the sample data: jitter the idle GPUs a little and
  // advance any in-flight download so the progress bar actually moves.
  function mockTick() {
    var d = mockInit();
    d.nodes.forEach(function (n) {
      (n.gpus || []).forEach(function (g) {
        if (g.util != null && g.util > 20) {
          g.util = Math.max(60, Math.min(100, g.util + (Math.random() * 6 - 3)));
        } else if (g.util != null) {
          g.util = Math.max(0, Math.min(8, g.util + (Math.random() * 4 - 2)));
        }
      });
    });
    if (d.download.status === 'downloading') {
      d.download.downloaded_bytes = Math.min(d.download.total_bytes,
        d.download.downloaded_bytes + d.download.total_bytes * 0.18);
      if (d.download.downloaded_bytes >= d.download.total_bytes) {
        d.download.status = 'complete';
        var name = String(d.download.repo_id || '').split('/').pop();
        if (name && d.models.indexOf(name) === -1) d.models.push(name);
      }
    }
    state.nodes = d.nodes;
    state.clusterStatus = 'ok';
    state.models = d.models;
    state.modelsStatus = 'ok';
    state.download = d.download;
    state.downloadStatus = 'ok';
    state.summary = {
      nodes_total: d.nodes.length,
      nodes_alive: d.nodes.length,
      gpus_total: d.nodes.reduce(function (a, n) { return a + (n.gpus || []).length; }, 0),
      roles: { admin: 1, participant: 2, storage: 1 },
      // The control plane publishes display names for the role wire ids.
      role_labels: { admin: 'Admin', participant: 'GPU Worker', storage: 'Storage' },
      instances: Object.assign({}, d.instanceStates),
    };
    pruneSelection();
  }

  function mockStartDownload(repo, revision) {
    var d = mockInit();
    d.download = {
      status: 'downloading', repo_id: repo + (revision ? '@' + revision : ''),
      downloaded_bytes: 0, total_bytes: 14.2e9, error: null,
    };
    state.download = d.download;
    showToast('Mock mode: downloading ' + repo + (revision ? ' @ ' + revision : ''), 'info');
  }

  /* A local stand-in for POST /api/cluster/plan so ?mock=1 shows a real-shaped
   * preview: driver-first ordering by node memory, memory-weighted PP split
   * over a 40-layer model, and the hardware args docs/04 mandates. */
  function mockPlan(body) {
    var d = mockInit();
    var picked = body.node_ids.map(function (id) {
      return d.nodes.find(function (n) { return n.node_id === id; });
    }).filter(Boolean);
    if (!picked.length) return { ok: false, error: 'no such participant nodes' };

    var gpuIds = body.gpu_ids_by_node || {};
    function nodeMem(n) {
      var idxs = gpuIds[n.node_id] || (n.gpus || []).map(function (g) { return g.index; });
      return (n.gpus || []).filter(function (g) { return idxs.includes(g.index); })
        .reduce(function (a, g) { return a + (g.mem_total || 0); }, 0);
    }
    var ordered = picked.slice().sort(function (a, b) { return nodeMem(b) - nodeMem(a); });
    var tp = (gpuIds[ordered[0].node_id] || ordered[0].gpus).length;
    var pp = ordered.length;
    var NUM_LAYERS = 40;

    var partition = null;
    if (pp > 1) {
      var weights = ordered.map(nodeMem);
      var total = weights.reduce(function (a, b) { return a + b; }, 0);
      var parts = weights.map(function (w) { return Math.max(1, Math.round(NUM_LAYERS * w / total)); });
      var drift = NUM_LAYERS - parts.reduce(function (a, b) { return a + b; }, 0);
      parts[0] += drift;
      partition = parts.join(',');
    }

    var args = [].concat(body.extra_args || []);
    if (pp > 1 && args.indexOf('--enforce-eager') === -1) args.push('--enforce-eager');
    if (tp * pp > 1 && args.indexOf('--disable-custom-all-reduce') === -1) args.push('--disable-custom-all-reduce');

    var commands = ordered.map(function (n, i) {
      return {
        node_id: n.node_id,
        body: {
          instance_id: 'plan-preview',
          role_in_instance: i === 0 ? 'head' : 'worker',
          ray: { head_addr: (ordered[0].addresses.fabric || [])[0], port: 6379 },
          model: body.model,
          layout: { tp: tp, pp: pp, pp_layer_partition: partition, executor: pp > 1 ? 'ray' : 'mp' },
          port: i === 0 ? body.port : null,
          served_model_name: body.served_model_name || null,
          max_model_len: body.max_model_len || null,
          gpu_memory_utilization: body.gpu_memory_utilization || 0.85,
          dtype: 'auto',
          vllm_args: args,
          env: {
            NCCL_SOCKET_IFNAME: 'enp196s0,ens2',
            GLOO_SOCKET_IFNAME: 'enp196s0',
            NCCL_IB_HCA: 'mlx4_0:1',
            NCCL_IB_GID_INDEX: '3',
            NCCL_P2P_DISABLE: '1',
            RAY_NODE_IP: (n.addresses.fabric || [])[0],
          },
        },
      };
    });

    return {
      ok: true,
      data: {
        ok: true,
        layout: {
          tp: tp, pp: pp, world: tp * pp,
          head_id: ordered[0].node_id,
          ordered_ids: ordered.map(function (n) { return n.node_id; }),
        },
        port: body.port,
        pp_layer_partition: partition,
        commands: commands,
      },
    };
  }

  function mockLaunch(body) {
    var d = mockInit();
    var p = mockPlan(body);
    if (!p.ok) return p;
    var rid = 'instance-' + (++d.nextInstance);
    d.instanceStates[rid] = 'SERVING';
    p.data.layout.ordered_ids.forEach(function (nid, i) {
      var n = d.nodes.find(function (x) { return x.node_id === nid; });
      if (!n) return;
      n.instances = n.instances || [];
      n.instances.push({ id: rid, state: 'SERVING', port: i === 0 ? body.port : null, role: i === 0 ? 'head' : 'worker' });
    });
    return { ok: true, data: { ok: true, instance_id: rid, layout: p.data.layout } };
  }

  /* A stand-in for GET /api/cluster/logs, so the viewer can be reviewed with no
   * backend: two nodes, node-tagged, plus a "Waiting for creating a placement
   * group" line that repeats on every 2s poll — which is exactly the real
   * failure this viewer was built for (a launch that hangs forever because
   * ray_utils cannot satisfy the placement group). Follow, the node filter and
   * Copy are all visibly live against it.
   *
   * A freshly launched mock instance walks through the three states the real
   * endpoint produces, one after another, so none of them needs contriving:
   *   polls 1-2   ebola -> ok:true, lines:[], detail "no log yet"
   *   polls 3-4   ebola -> ok:false, a CCP timeout, shown inline
   *   polls 5+    ebola -> streaming
   * The pre-existing `devstral-r0` instance skips straight to streaming. */
  function mockLogs(instanceId, tail) {
    var d = mockInit();
    d.logTicks = d.logTicks || {};
    var tick = (d.logTicks[instanceId] = (d.logTicks[instanceId] || 0) + 1);
    var fresh = instanceId !== 'devstral-r0';

    var covid = [
      'INFO 09-23 19:02:11 api_server.py:1032] vLLM API server version 0.10.1',
      'INFO 09-23 19:02:11 api_server.py:1033] args: Namespace(model=\'/models/Devstral-Small-2507\', tensor_parallel_size=2, pipeline_parallel_size=2, distributed_executor_backend=\'ray\', enforce_eager=True)',
      'INFO 09-23 19:02:12 ray_utils.py:284] Starting Ray head at 172.16.254.201:6379 (NCCL_IB_HCA=mlx4_0:1, NCCL_IB_GID_INDEX=3)',
      'INFO 09-23 19:02:18 ray_utils.py:212] Ray cluster ready: 2 nodes, 3 GPUs visible',
      'ERROR 09-23 19:02:19 ray_utils.py:236] The number of required GPUs exceeds the total number of available GPUs in the placement group.',
    ];
    for (var i = 0; i < tick + 1; i++) {
      covid.push('INFO 09-23 19:02:' + String(20 + i * 10).slice(-2) +
        ' ray_utils.py:242] Waiting for creating a placement group of specs for ' +
        ((i + 1) * 10) + ' seconds. specs=[{\'node:172.16.254.201\': 0.001, \'GPU\': 1.0}, {\'GPU\': 1.0}]. ' +
        'Check `ray status` to see if you have enough resources, and make sure the IP addresses are correct.');
    }

    var ebola = [
      'INFO 09-23 19:02:16 ray_worker.py:88] joining Ray head 172.16.254.201:6379 over enp196s0',
      'INFO 09-23 19:02:16 ray_worker.py:94] RoCE fabric up — GID index 3, mlx4_0:1',
      'WARNING 09-23 19:02:18 ray_worker.py:121] only 1 of 2 GPUs free on this node (GPU 0 is held by instance devstral-r0)',
      'INFO 09-23 19:02:21 ray_worker.py:130] registered with the Ray cluster; 1 GPU offered',
    ];
    for (var j = 0; j < tick; j++) {
      ebola.push('INFO 09-23 19:02:' + String(25 + j * 10).slice(-2) +
        ' ray_utils.py:242] Waiting for creating a placement group of specs for ' +
        ((j + 1) * 10) + ' seconds.');
    }

    var nodes = {
      'covid-a1b2': { ok: true, path: '/models/.vllm-manager/logs/' + instanceId + '.log', lines: covid.slice(-tail) },
    };
    if (fresh && tick <= 2) {
      nodes['ebola-c3d4'] = { ok: true, path: null, lines: [], detail: 'no log yet' };
    } else if (fresh && tick <= 4) {
      nodes['ebola-c3d4'] = { ok: false, error: 'node did not answer within 20s (CCP timeout)' };
    } else {
      nodes['ebola-c3d4'] = { ok: true, path: '/models/.vllm-manager/logs/' + instanceId + '.log', lines: ebola.slice(-tail) };
    }

    // The real hub appends node by node, in its own target order — mirror that
    // rather than inventing a global timestamp sort the backend does not do.
    var merged = [];
    Object.keys(nodes).forEach(function (nid) {
      (nodes[nid].lines || []).forEach(function (line) { merged.push({ node_id: nid, line: line }); });
    });

    return {
      ok: true,
      instance_id: instanceId,
      state: d.instanceStates[instanceId] || 'STARTING',
      model: '/models/Devstral-Small-2507',
      nodes: nodes,
      merged: merged,
    };
  }

  function mockStopInstance(rid) {
    var d = mockInit();
    delete d.instanceStates[rid];
    d.nodes.forEach(function (n) {
      n.instances = (n.instances || []).filter(function (r) { return r.id !== rid; });
    });
    mockTick();
  }

  /* =======================================================================
   * INIT
   * ===================================================================== */

  function initServing(rootEl) {
    if (!rootEl) return;
    if (rootEl.dataset && rootEl.dataset.svInit === '1') return;   // already wired
    if (rootEl.dataset) rootEl.dataset.svInit = '1';

    root = rootEl;
    root.classList.add('sv-root');
    root.innerHTML = skeleton();
    root.addEventListener('click', onClick);
    root.addEventListener('change', onChange);
    root.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter' && ev.target && ev.target.id === 'sv-dl-repo') {
        ev.preventDefault();
        onDownload();
      }
    });

    // Log viewer: the scroll container owns "is the user at the bottom?".
    var sc = el('sv-log-scroll');
    if (sc) sc.addEventListener('scroll', onLogScroll);
    var tailSel = el('sv-log-tail');
    if (tailSel) tailSel.value = String(state.logs.tail);
    syncFollowButton();

    poll();
    if (timer) clearInterval(timer);
    timer = setInterval(poll, POLL_MS);
  }

  window.initServing = initServing;

  // Convenience auto-wire: drop the <script> in next to a #serving-content
  // div and the panel comes up on its own. An explicit initServing() call
  // from the host page still wins (the dataset guard makes it idempotent).
  document.addEventListener('DOMContentLoaded', function () {
    var host = document.getElementById('serving-content');
    if (host) initServing(host);
  });
})();
