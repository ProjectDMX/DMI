/*
 * DMI-configurator front end.
 *
 * The browser holds no DMI semantics. Client state *is* the configuration
 * document -- the same shape the YAML file has -- so serializing, parsing, and
 * validating are all server round-trips through the same Python code the
 * runtime uses. What the UI calls valid is what DMI calls valid.
 */
(function () {
  "use strict";

  var model = null;
  var layout = null;
  var state = null;
  var focusedNode = null;
  var updateTimer = null;

  /* Workload and ring sizes are UI-local on purpose. They describe the
   * serving traffic and the transport, not the capture, and DMIConfig has no
   * runtime block -- exposing a subset of RingConfig through the saved YAML
   * would imply a support contract that does not exist. */
  var workload = {
    batch_size: 8,
    prompt_tokens: 2048,
    decode_tokens: 256,
    decode_steps_per_second: 0,
    tensor_parallel_size: 1,
    pipeline_parallel_size: 1,
    dtype: "float16",
    packed: true
  };

  var ring = { payload_mib: 4096, pinned_mib: 4096 };

  var dom = {};

  function $(id) { return document.getElementById(id); }

  function cacheDom() {
    ["model-summary", "status", "architecture", "detail-title", "detail-body",
     "layer-start", "layer-end", "layer-rail", "layer-readout", "yaml-preview",
     "issues", "issue-count", "toast", "file-input", "step-stride",
     "request-stride", "capture-prefill", "capture-decode", "step-offset",
     "warmup-steps", "request-offset", "warmup-requests", "policy",
     "btn-open", "btn-save", "btn-copy",
     "wl-batch", "wl-prompt", "wl-decode", "wl-rate", "wl-tp", "wl-pp",
     "wl-dtype", "wl-packed", "ring-payload", "ring-pinned", "ring-meter",
     "ring-fill", "ring-detail", "est-rank", "est-ranks", "est-ranks-wrap",
     "est-notes", "fig-peak", "fig-decode", "fig-request", "fig-sustained",
     "fig-day"].forEach(function (id) {
      dom[id] = $(id);
    });
  }

  var MIB = 1024 * 1024;

  /* Binary units throughout: DMI sizes its rings in MiB. */
  function formatBytes(value) {
    if (value === null || value === undefined) return "—";
    if (!value) return "0 B";
    var units = ["B", "KiB", "MiB", "GiB", "TiB"];
    var scaled = value;
    var index = 0;
    while (scaled >= 1024 && index < units.length - 1) {
      scaled /= 1024;
      index += 1;
    }
    var digits = index === 0 || scaled >= 100 ? 0 : (scaled >= 10 ? 1 : 2);
    return scaled.toFixed(digits) + " " + units[index];
  }

  function formatRate(value) {
    return value === null || value === undefined
      ? "—"
      : formatBytes(value) + "/s";
  }

  function toast(message) {
    dom.toast.textContent = message;
    dom.toast.hidden = false;
    clearTimeout(dom.toast._timer);
    dom.toast._timer = setTimeout(function () { dom.toast.hidden = true; }, 2600);
  }

  async function api(path, options) {
    var response = await fetch(path, options);
    var payload = await response.json().catch(function () { return {}; });
    if (!response.ok) {
      throw new Error(payload.detail || payload.message || response.statusText);
    }
    return payload;
  }

  function selectedHooks() {
    return new Set(state.observations.hooks);
  }

  function defaultState() {
    return {
      version: 1,
      observations: { hooks: [], layers: null },
      schedule: {
        step_stride: 1,
        request_stride: 1,
        capture_prefill: true,
        capture_decode: true,
        step_offset: 0,
        warmup_steps: 0,
        request_offset: 0,
        warmup_requests: 0
      },
      policy: { objective: "balanced" }
    };
  }

  /* ---------- rendering ---------- */

  function renderArchitecture() {
    // The renderer replaces the whole SVG, which drops keyboard focus. Note
    // which node held it and put it back, so tabbing to a block and pressing
    // Enter does not dump the user back at the top of the document.
    var active = document.activeElement;
    var refocus =
      active && dom.architecture.contains(active)
        ? active.closest("[data-node-id]")
        : null;
    var refocusId = refocus ? refocus.dataset.nodeId : null;

    DMIArchitecture.render(dom.architecture, layout, {
      selected: selectedHooks(),
      focused: focusedNode,
      onSelect: function (nodeId) {
        focusedNode = nodeId;
        renderArchitecture();
        renderDetail();
      }
    });

    if (refocusId) {
      var restored = dom.architecture.querySelector(
        '[data-node-id="' + refocusId + '"]'
      );
      if (restored) restored.focus();
    }
  }

  function findNode(nodeId) {
    return layout.nodes.filter(function (n) { return n.id === nodeId; })[0] || null;
  }

  function renderDetail() {
    var node = focusedNode ? findNode(focusedNode) : null;
    if (!node) {
      dom["detail-title"].textContent = "Selected component";
      dom["detail-body"].replaceChildren(
        Object.assign(document.createElement("p"), {
          className: "hint",
          textContent: "Select a block in the architecture to choose observations."
        })
      );
      return;
    }

    dom["detail-title"].textContent = node.label;

    var list = document.createElement("ul");
    list.className = "hook-list";
    var chosen = selectedHooks();

    node.hooks.forEach(function (hook) {
      var item = document.createElement("li");
      item.className = "hook-item" +
        (hook.available ? "" : " is-unavailable") +
        (hook.per_layer ? "" : " is-global");

      var label = document.createElement("label");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = chosen.has(hook.id);
      box.disabled = !hook.available;
      box.addEventListener("change", function () {
        toggleHook(hook.id, box.checked);
      });

      var body = document.createElement("span");
      var name = document.createElement("span");
      name.className = "hook-label";
      name.textContent = hook.label + " ";
      var id = document.createElement("span");
      id.className = "hook-id";
      id.textContent = hook.id;
      body.append(name, id);

      if (!hook.per_layer) {
        var scope = document.createElement("span");
        scope.className = "hook-scope";
        scope.textContent = "  (model-wide, ignores layer range)";
        body.append(scope);
      }
      if (!hook.available && hook.reason) {
        var reason = document.createElement("span");
        reason.className = "hook-reason";
        reason.textContent = hook.reason;
        body.append(reason);
      }

      label.append(box, body);
      item.append(label);
      list.append(item);
    });

    dom["detail-body"].replaceChildren(list);
  }

  function renderLayers() {
    var total = layout.num_layers;
    var range = state.observations.layers;

    dom["layer-start"].max = total - 1;
    dom["layer-end"].max = total - 1;
    dom["layer-start"].value = range ? range.start : 0;
    dom["layer-end"].value = range ? range.end : total - 1;

    var rail = document.createDocumentFragment();
    for (var i = 0; i < total; i += 1) {
      var tick = document.createElement("span");
      tick.className = "tick" + (!range || (i >= range.start && i <= range.end) ? " is-on" : "");
      tick.title = "Layer " + i;
      tick.dataset.layer = String(i);
      rail.append(tick);
    }
    dom["layer-rail"].replaceChildren(rail);

    dom["layer-readout"].textContent = range
      ? "Layers " + range.start + "–" + range.end +
        " (" + (range.end - range.start + 1) + " of " + total + ")"
      : "All layers (0–" + (total - 1) + ")";
  }

  function renderPolicy() {
    var objective = state.policy ? state.policy.objective : null;
    Array.prototype.forEach.call(
      dom.policy.querySelectorAll("input[name=policy]"),
      function (input) { input.checked = input.value === objective; }
    );
  }

  function renderSchedule() {
    dom["step-stride"].value = state.schedule.step_stride;
    dom["request-stride"].value = state.schedule.request_stride;
    dom["capture-prefill"].checked = state.schedule.capture_prefill;
    dom["capture-decode"].checked = state.schedule.capture_decode;
    dom["step-offset"].value = state.schedule.step_offset;
    dom["warmup-steps"].value = state.schedule.warmup_steps;
    dom["request-offset"].value = state.schedule.request_offset;
    dom["warmup-requests"].value = state.schedule.warmup_requests;
  }

  function renderEstimate(payload) {
    dom["fig-peak"].textContent = formatBytes(payload.peak_step_bytes);
    dom["fig-decode"].textContent = formatBytes(payload.decode_step_bytes);
    dom["fig-request"].textContent = formatBytes(payload.bytes_per_request);
    dom["fig-sustained"].textContent = formatRate(
      payload.sustained_bytes_per_second
    );
    dom["fig-day"].textContent = formatBytes(payload.bytes_per_day);

    dom["est-rank"].textContent = payload.peak_step_rank
      ? "peak on " + payload.peak_step_rank
      : "";

    renderRingFit(payload.ring_fit);
    renderRanks(payload);
    renderNotes(payload);
  }

  function renderRingFit(fit) {
    if (!fit) {
      dom["ring-fill"].style.width = "0%";
      dom["ring-meter"].removeAttribute("data-tone");
      dom["ring-detail"].textContent = "—";
      dom["ring-detail"].removeAttribute("data-tone");
      return;
    }

    var percent = Math.max(0, Math.min(100, fit.occupancy_percent));
    dom["ring-fill"].style.width = percent + "%";
    dom["ring-meter"].setAttribute(
      "aria-label",
      "Ring occupancy " + fit.occupancy_percent.toFixed(0) + " percent"
    );

    var tone = null;
    if (!fit.fits) tone = "over";
    else if (fit.occupancy_percent >= 50) tone = "warning";

    if (tone) dom["ring-meter"].dataset.tone = tone;
    else dom["ring-meter"].removeAttribute("data-tone");

    dom["ring-detail"].textContent = fit.detail;
    if (fit.fits) dom["ring-detail"].removeAttribute("data-tone");
    else dom["ring-detail"].dataset.tone = "over";
  }

  function renderRanks(payload) {
    var ranks = payload.ranks || [];
    // One rank is the single-GPU case; the breakdown says nothing extra.
    var interesting = ranks.length > 1;
    dom["est-ranks-wrap"].hidden = !interesting;
    if (!interesting) return;

    var frag = document.createDocumentFragment();
    ranks.forEach(function (rank) {
      var row = document.createElement("tr");
      if (rank.label === payload.peak_step_rank) row.dataset.worst = "true";
      [
        rank.label,
        formatBytes(rank.prefill_step_bytes),
        formatBytes(rank.decode_step_bytes),
        String(rank.prefill_hooks)
      ].forEach(function (value) {
        var cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      frag.append(row);
    });
    dom["est-ranks"].replaceChildren(frag);
  }

  function renderNotes(payload) {
    var frag = document.createDocumentFragment();
    (payload.warnings || []).forEach(function (message) {
      frag.append(noteElement(message, "warning"));
    });
    (payload.assumptions || []).forEach(function (message) {
      frag.append(noteElement(message, "assumption"));
    });
    dom["est-notes"].replaceChildren(frag);
  }

  function noteElement(message, kind) {
    var note = document.createElement("p");
    note.className = "est-note";
    note.dataset.kind = kind;
    var body = document.createElement("span");
    body.textContent = message;
    note.append(body);
    return note;
  }

  function clearEstimate(message) {
    ["fig-peak", "fig-decode", "fig-request", "fig-sustained", "fig-day"]
      .forEach(function (id) { dom[id].textContent = "—"; });
    dom["est-rank"].textContent = "";
    dom["est-ranks-wrap"].hidden = true;
    renderRingFit(null);
    dom["est-notes"].replaceChildren(
      noteElement(message, "warning")
    );
  }

  function renderIssues(issues) {
    var errors = issues.filter(function (i) { return i.severity === "error"; });
    var warnings = issues.filter(function (i) { return i.severity === "warning"; });

    dom["issue-count"].textContent = String(issues.length);
    dom["issue-count"].removeAttribute("data-tone");
    if (errors.length) dom["issue-count"].dataset.tone = "error";
    else if (warnings.length) dom["issue-count"].dataset.tone = "warning";

    if (errors.length) {
      dom.status.dataset.state = "invalid";
      dom.status.textContent = errors.length === 1 ? "1 issue" : errors.length + " issues";
    } else if (warnings.length) {
      dom.status.dataset.state = "warning";
      dom.status.textContent = warnings.length === 1 ? "1 warning" : warnings.length + " warnings";
    } else {
      dom.status.dataset.state = "valid";
      dom.status.textContent = "Valid";
    }

    if (!issues.length) {
      dom.issues.replaceChildren(
        Object.assign(document.createElement("p"), {
          className: "hint",
          textContent: "No issues. This configuration is ready to run."
        })
      );
      return;
    }

    var frag = document.createDocumentFragment();
    issues.forEach(function (issue) {
      var box = document.createElement("div");
      box.className = "issue";
      box.dataset.severity = issue.severity;
      var field = document.createElement("span");
      field.className = "issue-field";
      field.textContent = issue.field;
      var message = document.createElement("span");
      message.textContent = issue.message;
      box.append(field, message);
      frag.append(box);
    });
    dom.issues.replaceChildren(frag);
  }

  /* ---------- state changes ---------- */

  function toggleHook(hookId, on) {
    var hooks = state.observations.hooks.filter(function (h) { return h !== hookId; });
    if (on) hooks.push(hookId);
    state.observations.hooks = hooks;
    renderArchitecture();
    scheduleUpdate();
  }

  function setLayers(range) {
    state.observations.layers = range;
    renderLayers();
    scheduleUpdate();
  }

  function scheduleUpdate() {
    clearTimeout(updateTimer);
    updateTimer = setTimeout(refreshOutput, 140);
  }

  // Same stale-response guard as refreshEstimate. Two of these can be in
  // flight at once -- the debounce shortens the window but does not close it
  // -- and without a stamp a slow earlier reply paints last, leaving the YAML
  // preview and Issues tab describing a configuration the user has already
  // changed away from.
  var outputRequestId = 0;

  async function refreshOutput() {
    var requestId = (outputRequestId += 1);
    try {
      var results = await Promise.all([
        api("/api/config/serialize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config: state })
        }),
        api("/api/validate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config: state })
        })
      ]);
      if (requestId !== outputRequestId) return;
      dom["yaml-preview"].textContent = results[0].yaml;
      renderIssues(results[1].issues);
    } catch (error) {
      if (requestId !== outputRequestId) return;
      dom.status.dataset.state = "invalid";
      dom.status.textContent = "Error";
      renderIssues([{ severity: "error", field: "request", message: error.message }]);
    }
    // Estimated separately: a workload the estimator rejects should not blank
    // the YAML preview, and an unserializable config should not blank this.
    refreshEstimate();
  }

  // Responses are not guaranteed to arrive in the order they were sent, and
  // the estimate endpoint walks every rank, so a slow earlier request can
  // land after a newer one and paint figures for a workload the user has
  // already changed. Stamp each request and drop anything but the latest.
  var estimateRequestId = 0;
  var copyRequestId = 0;

  async function refreshEstimate() {
    var requestId = (estimateRequestId += 1);
    try {
      var payload = await api("/api/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config: state,
          workload: workload,
          ring: {
            payload_bytes: ring.payload_mib * MIB,
            pinned_bytes: ring.pinned_mib * MIB
          }
        })
      });
      if (requestId !== estimateRequestId) return;
      renderEstimate(payload);
    } catch (error) {
      if (requestId !== estimateRequestId) return;
      clearEstimate("Could not estimate: " + error.message);
    }
  }

  function applyState(next) {
    // Section-by-section, not one shallow assign: defaultState() carries a
    // balanced policy, and a shallow merge would inject it into a config
    // that deliberately has none -- load then save would change the file's
    // meaning and break the advertised round-trip. Policy comes from next
    // only, so its absence stays absent.
    state = Object.assign({}, next, { version: 1 });
    state.schedule = Object.assign(defaultState().schedule, next.schedule || {});
    state.observations = Object.assign({ hooks: [], layers: null }, next.observations || {});
    state.policy = next.policy ? { objective: next.policy.objective } : undefined;
    renderSchedule();
    renderPolicy();
    renderLayers();
    renderArchitecture();
    renderDetail();
    scheduleUpdate();
  }

  /* ---------- wiring ---------- */

  // Listeners bind before startup resolves the model; every handler below
  // reads state/layout, so until boot fills them the event is dropped
  // rather than crashing (a dropped keystroke is recovered by applyState's
  // render; a TypeError mid-applyState would discard a loaded file).
  var uiReady = false;

  function bindSchedule() {
    var numeric = {
      "step-stride": "step_stride",
      "request-stride": "request_stride",
      "step-offset": "step_offset",
      "warmup-steps": "warmup_steps",
      "request-offset": "request_offset",
      "warmup-requests": "warmup_requests"
    };
    Object.keys(numeric).forEach(function (id) {
      dom[id].addEventListener("input", function () {
        if (!uiReady) return;
        var value = parseInt(dom[id].value, 10);
        if (Number.isNaN(value)) return;
        state.schedule[numeric[id]] = value;
        scheduleUpdate();
      });
    });

    dom["capture-prefill"].addEventListener("change", function () {
      if (!uiReady) return;
      state.schedule.capture_prefill = dom["capture-prefill"].checked;
      scheduleUpdate();
    });
    dom["capture-decode"].addEventListener("change", function () {
      if (!uiReady) return;
      state.schedule.capture_decode = dom["capture-decode"].checked;
      scheduleUpdate();
    });
  }

  function bindWorkload() {
    var numeric = {
      "wl-batch": "batch_size",
      "wl-prompt": "prompt_tokens",
      "wl-decode": "decode_tokens",
      "wl-rate": "decode_steps_per_second",
      "wl-tp": "tensor_parallel_size",
      "wl-pp": "pipeline_parallel_size"
    };
    Object.keys(numeric).forEach(function (id) {
      dom[id].addEventListener("input", function () {
        // The decode rate is a float on the server (0.5 steps/s is valid);
        // everything else is a count.
        var value = id === "wl-rate"
          ? parseFloat(dom[id].value)
          : parseInt(dom[id].value, 10);
        if (Number.isNaN(value)) return;
        workload[numeric[id]] = value;
        scheduleEstimate();
      });
    });

    dom["wl-dtype"].addEventListener("change", function () {
      workload.dtype = dom["wl-dtype"].value;
      scheduleEstimate();
    });
    dom["wl-packed"].addEventListener("change", function () {
      workload.packed = dom["wl-packed"].value === "packed";
      scheduleEstimate();
    });

    var ringInputs = { "ring-payload": "payload_mib", "ring-pinned": "pinned_mib" };
    Object.keys(ringInputs).forEach(function (id) {
      dom[id].addEventListener("input", function () {
        var value = parseInt(dom[id].value, 10);
        if (Number.isNaN(value) || value < 0) return;
        ring[ringInputs[id]] = value;
        scheduleEstimate();
      });
    });
  }

  var estimateTimer = null;

  function scheduleEstimate() {
    clearTimeout(estimateTimer);
    estimateTimer = setTimeout(refreshEstimate, 140);
  }

  function bindLayers() {
    function readInputs() {
      var start = parseInt(dom["layer-start"].value, 10);
      var end = parseInt(dom["layer-end"].value, 10);
      if (Number.isNaN(start) || Number.isNaN(end)) return;
      if (end < start) end = start;
      setLayers({ start: start, end: end });
    }
    dom["layer-start"].addEventListener("change", function (event) {
      if (!uiReady) return;
      readInputs();
    });
    dom["layer-end"].addEventListener("change", function (event) {
      if (!uiReady) return;
      readInputs();
    });

    // Click picks one layer; shift-click extends from the current start.
    dom["layer-rail"].addEventListener("click", function (event) {
      if (!uiReady) return;
      var target = event.target.closest(".tick");
      if (!target) return;
      var layer = parseInt(target.dataset.layer, 10);
      var current = state.observations.layers;
      if (event.shiftKey && current) {
        setLayers({
          start: Math.min(current.start, layer),
          end: Math.max(current.start, layer)
        });
      } else {
        setLayers({ start: layer, end: layer });
      }
    });

    document.querySelectorAll("[data-layers]").forEach(function (chip) {
      chip.addEventListener("click", function () {
        var total = layout.num_layers;
        var third = Math.max(1, Math.floor(total / 3));
        // Clamp so tiny models (1-2 layers) never produce start > end,
        // which LayerSelection rejects.
        var middleStart = Math.min(third, total - 1);
        var presets = {
          all: null,
          first: { start: 0, end: third - 1 },
          middle: {
            start: middleStart,
            end: Math.max(middleStart, Math.min(total - 1, 2 * third - 1))
          },
          last: { start: Math.max(0, total - third), end: total - 1 }
        };
        setLayers(presets[chip.dataset.layers]);
      });
    });
  }

  function bindTabs() {
    document.querySelectorAll(".tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        document.querySelectorAll(".tab").forEach(function (other) {
          other.classList.toggle("is-active", other === tab);
          other.setAttribute("aria-selected", other === tab ? "true" : "false");
        });
        document.querySelectorAll("[data-tab-panel]").forEach(function (panel) {
          panel.hidden = panel.dataset.tabPanel !== tab.dataset.tab;
        });
      });
    });
  }

  function bindPolicy() {
    dom.policy.addEventListener("change", function (event) {
      if (event.target.name !== "policy") return;
      state.policy = { objective: event.target.value };
      scheduleUpdate();
    });
  }

  function bindActions() {
    dom["btn-copy"].addEventListener("click", async function () {
      if (!uiReady) return;
      // Same stale-response guard as refreshOutput: two rapid Copies with
      // an edit between them can complete out of order (sync endpoints run
      // in a threadpool), and the older response must not win the clipboard.
      var requestId = (copyRequestId += 1);
      try {
        // Serialize the CURRENT state, not the preview element: edits reach
        // the preview only after debounce + request latency, so a Copy made
        // right after a change would otherwise export the previous
        // configuration with a success toast.
        var payload = await api("/api/config/serialize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config: state })
        });
        if (requestId !== copyRequestId) return;
        await navigator.clipboard.writeText(payload.yaml);
        toast("YAML copied to clipboard");
      } catch (error) {
        if (requestId !== copyRequestId) return;
        toast("Could not copy: " + error.message);
      }
    });

    dom["btn-open"].addEventListener("click", function () { dom["file-input"].click(); });

    dom["file-input"].addEventListener("change", async function () {
      if (!uiReady) return;
      var file = dom["file-input"].files[0];
      if (!file) return;
      try {
        var payload = await api("/api/config/parse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ yaml: await file.text() })
        });
        applyState(payload.config);
        toast("Loaded " + file.name);
      } catch (error) {
        toast("Could not load: " + error.message);
      } finally {
        dom["file-input"].value = "";
      }
    });

    dom["btn-save"].addEventListener("click", async function () {
      if (!uiReady) return;
      try {
        var payload = await api("/api/config/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ config: state })
        });
        toast("Saved to " + payload.path);
      } catch (error) {
        toast("Could not save: " + error.message);
      }
    });
  }

  /* ---------- boot ---------- */

  async function boot() {
    cacheDom();
    bindSchedule();
    bindWorkload();
    bindLayers();
    bindTabs();
    bindPolicy();
    bindActions();

    try {
      model = await api("/api/model");
      layout = model.architecture_layout;
    } catch (error) {
      dom["model-summary"].textContent = "Could not load model: " + error.message;
      dom.status.dataset.state = "invalid";
      dom.status.textContent = "Error";
      return;
    }

    var architecture = model.architecture.replace(/_/g, " ");
    dom["model-summary"].textContent =
      model.name + " · " + architecture + " · " +
      model.topology.num_layers + " layers" +
      (model.topology.num_experts ? " · " + model.topology.num_experts + " experts" : "");
    document.title = "DMI-configurator — " + model.name;

    var initial = await api("/api/config").catch(function () { return { config: null }; });
    applyState(initial.config || defaultState());
    uiReady = true;
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
