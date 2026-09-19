/* ===========================================================================
   Lunar Low-Light Lab - client

   No framework and no build step. The page has one piece of state (the current
   result) and one job: show it accurately. Everything the UI displays comes
   from the server payload, including the list of methods and the upload
   limits, so the interface cannot drift out of sync with the backend.

   Loaded with a strict CSP (script-src 'self'), so there is no inline script
   and no eval anywhere in here.
   =========================================================================== */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);

  const el = {
    engine: $("engine-status"),
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    dzInner: document.querySelector(".dropzone-inner"),
    dzLoaded: $("dz-loaded"),
    thumb: $("thumb"),
    fileName: $("file-name"),
    fileDims: $("file-dims"),
    clearFile: $("clear-file"),
    modeList: $("mode-list"),
    runBtn: $("run-btn"),
    runNote: $("run-note"),
    empty: $("empty-state"),
    result: $("result"),
    busy: $("busy"),
    busyText: $("busy-text"),
    stage: $("stage"),
    compareWrap: $("compare-wrap"),
    singleWrap: $("single-wrap"),
    imgBase: $("img-base"),
    imgTop: $("img-top"),
    imgSingle: $("img-single"),
    compareTop: $("compare-top"),
    handle: $("handle"),
    diffLegend: $("diff-legend"),
    stageTagRight: $("stage-tag-right"),
    verdict: $("verdict"),
    metricsBody: document.querySelector("#metrics-table tbody"),
    histogram: $("histogram"),
    zones: $("zones"),
    methodName: $("method-name"),
    steps: $("steps"),
    caveat: $("caveat"),
    facts: $("facts"),
    dlEnhanced: $("dl-enhanced"),
    dlDiff: $("dl-diff"),
    discard: $("discard"),
    toast: $("toast"),
    toastTitle: $("toast-title"),
    toastBody: $("toast-body"),
    toastClose: $("toast-close"),
  };

  const state = {
    file: null,
    objectUrl: null,
    config: null,
    result: null,
    view: "compare",
    split: 50,
    busy: false,
  };

  /* ------------------------------------------------------------ helpers -- */

  function toast(title, body) {
    el.toastTitle.textContent = title;
    el.toastBody.textContent = body || "";
    el.toast.hidden = false;
  }

  function hideToast() { el.toast.hidden = true; }

  function formatBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }

  function fmt(value, digits) {
    if (value === null || value === undefined) return "\u2014";
    const d = digits === undefined ? 2 : digits;
    if (Math.abs(value) >= 1000) return value.toFixed(0);
    return value.toFixed(d);
  }

  /* Turn any failed response into a message worth reading. The server always
     sends {error:{code,message}}, but a crash or a proxy could send HTML, so
     fall back to the status line rather than printing "[object Object]". */
  async function describeFailure(response) {
    try {
      const data = await response.json();
      if (data && data.error && data.error.message) return data.error.message;
    } catch (err) { /* not JSON */ }
    return "The server responded with " + response.status + " " + response.statusText + ".";
  }

  /* -------------------------------------------------------------- engine -- */

  async function loadConfig() {
    try {
      const [cfgRes, healthRes] = await Promise.all([
        fetch("/api/config"),
        fetch("/api/health"),
      ]);
      if (!cfgRes.ok) throw new Error(await describeFailure(cfgRes));
      state.config = await cfgRes.json();
      renderModes(state.config.modes, state.config.defaults.mode);
      applyDefaults(state.config.defaults);

      if (healthRes.ok) {
        const health = await healthRes.json();
        const neural = health.zero_dce && health.zero_dce.neural_available;
        el.engine.dataset.state = neural ? "neural" : "fallback";
        el.engine.textContent = neural
          ? "Zero-DCE weights loaded"
          : "Zero-DCE running analytic fallback";
        el.engine.title = (health.zero_dce && health.zero_dce.detail) || "";
      }
    } catch (err) {
      el.engine.dataset.state = "down";
      el.engine.textContent = "Engine unreachable";
      toast("Cannot reach the server", String(err.message || err));
    }
  }

  function renderModes(modes, active) {
    el.modeList.textContent = "";
    modes.forEach((mode) => {
      const label = document.createElement("label");
      label.className = "mode" + (mode.id === active ? " is-active" : "");
      label.dataset.mode = mode.id;

      const input = document.createElement("input");
      input.type = "radio";
      input.name = "mode";
      input.value = mode.id;
      input.checked = mode.id === active;

      const name = document.createElement("span");
      name.className = "mode-name";
      name.textContent = mode.name;

      const summary = document.createElement("span");
      summary.className = "mode-sum";
      summary.textContent = mode.summary;

      label.append(input, name, summary);
      input.addEventListener("change", () => selectMode(mode.id));
      el.modeList.appendChild(label);
    });
    syncParams(active);
  }

  function applyDefaults(defaults) {
    if (!defaults) return;
    $("gamma").value = defaults.gamma;
    $("clahe_clip").value = defaults.clahe_clip;
    $("clahe_grid").value = defaults.clahe_grid;
    syncOutputs();
  }

  function currentMode() {
    const checked = el.modeList.querySelector("input:checked");
    return checked ? checked.value : "scientific";
  }

  function selectMode(mode) {
    el.modeList.querySelectorAll(".mode").forEach((node) => {
      node.classList.toggle("is-active", node.dataset.mode === mode);
    });
    syncParams(mode);
  }

  /* Only show the controls that actually affect the chosen method. A gamma
     slider that silently does nothing is worse than no slider. */
  function syncParams(mode) {
    document.querySelectorAll(".param").forEach((node) => {
      const applies = (node.dataset.for || "").split(" ");
      node.hidden = !applies.includes(mode);
    });
  }

  function syncOutputs() {
    $("gamma-out").textContent = Number($("gamma").value).toFixed(2);
    $("clip-out").textContent = Number($("clahe_clip").value).toFixed(2);
    const grid = $("clahe_grid").value;
    $("grid-out").textContent = grid + " \u00d7 " + grid;
  }

  /* ---------------------------------------------------------------- file -- */

  function acceptFile(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      toast("That is not an image", "Choose a PNG, JPEG, TIFF, BMP or WebP file.");
      return;
    }

    const limit = state.config && state.config.limits.max_upload_bytes;
    if (limit && file.size > limit) {
      toast(
        "That file is too large",
        formatBytes(file.size) + " exceeds the " + formatBytes(limit) + " limit."
      );
      return;
    }

    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);

    el.thumb.src = state.objectUrl;
    el.fileName.textContent = file.name;
    el.fileDims.textContent = formatBytes(file.size);

    // Report the real pixel dimensions once the browser has decoded it.
    const probe = new Image();
    probe.onload = () => {
      el.fileDims.textContent =
        probe.naturalWidth + " \u00d7 " + probe.naturalHeight + " \u00b7 " + formatBytes(file.size);
    };
    probe.src = state.objectUrl;

    el.dzInner.hidden = true;
    el.dzLoaded.hidden = false;
    el.dropzone.classList.add("has-file");
    el.runBtn.disabled = false;
    el.runNote.textContent = "Ready.";
    hideToast();
  }

  function clearFile() {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = null;
    state.objectUrl = null;
    el.fileInput.value = "";
    el.dzInner.hidden = false;
    el.dzLoaded.hidden = true;
    el.dropzone.classList.remove("has-file");
    el.runBtn.disabled = true;
    el.runNote.textContent = "Choose a frame to begin.";
  }

  /* ------------------------------------------------------------- request -- */

  async function runEnhancement() {
    if (!state.file || state.busy) return;

    const body = new FormData();
    body.append("image", state.file);
    body.append("mode", currentMode());
    body.append("gamma", $("gamma").value);
    body.append("clahe_clip", $("clahe_clip").value);
    body.append("clahe_grid", $("clahe_grid").value);
    body.append("denoise", $("denoise").checked ? "1" : "0");
    body.append("preserve_highlights", $("preserve_highlights").checked ? "1" : "0");

    setBusy(true, currentMode() === "zerodce"
      ? "Estimating tone curves"
      : "Processing frame");

    try {
      const response = await fetch("/api/enhance", { method: "POST", body: body });
      if (!response.ok) {
        toast("Enhancement failed", await describeFailure(response));
        return;
      }
      render(await response.json());
      hideToast();
    } catch (err) {
      toast("Request failed", "Could not reach the server. Is it still running?");
    } finally {
      setBusy(false);
    }
  }

  function setBusy(busy, message) {
    state.busy = busy;
    el.busy.hidden = !busy;
    el.runBtn.disabled = busy || !state.file;
    if (message) el.busyText.textContent = message;
    el.runNote.textContent = busy ? "Working\u2026" : (state.file ? "Ready." : "Choose a frame to begin.");
  }

  /* -------------------------------------------------------------- render -- */

  function render(payload) {
    state.result = payload;
    el.empty.hidden = true;
    el.result.hidden = false;

    // Cache-bust so a second run does not show the previous result's pixels.
    const stamp = "?t=" + Date.now();
    // The clipped layer covers the LEFT of the frame, so it must hold the
    // original for the "Original | Enhanced" tags to be accurate.
    el.imgBase.src = payload.images.enhanced + stamp;
    el.imgTop.src = payload.images.original + stamp;
    el.dlEnhanced.href = payload.images.enhanced + "?download=1";
    el.dlDiff.href = payload.images.difference + "?download=1";

    renderVerdict(payload);
    renderMetrics(payload.comparison);
    renderHistogram(payload.histogram);
    renderZones(payload.telemetry);
    renderMethod(payload);

    setView("compare");
    setSplit(50);
    sizeCompareLayer();
  }

  function renderVerdict(payload) {
    const c = payload.comparison;
    const entropy = c.entropy || {};
    const shadows = c.shadow_clipping || {};
    const noise = c.noise_sigma || {};
    const diff = payload.difference || {};

    const recovered = (shadows.before || 0) - (shadows.after || 0);
    const parts = [];

    parts.push(
      "Detail recovered: <strong>" + fmt(entropy.before) + " \u2192 " +
      fmt(entropy.after) + " bits</strong> of tonal information."
    );

    if (recovered > 0.05) {
      parts.push(
        "<strong>" + fmt(recovered, 1) + "%</strong> of the frame came back out of pure black."
      );
    } else if (recovered < -0.05) {
      parts.push(
        "<strong>" + fmt(-recovered, 1) + "%</strong> more of the frame was crushed to black."
      );
    }

    if (noise.after !== null && noise.before !== null) {
      const factor = noise.before > 0.001 ? noise.after / noise.before : null;
      if (factor && factor > 1.15) {
        parts.push(
          "Estimated noise rose <strong>" + fmt(factor, 1) +
          "\u00d7</strong>, which is the cost of the lift."
        );
      }
    }

    parts.push(
      "The change map covers <strong>" + fmt(diff.pixels_changed_pct, 1) +
      "%</strong> of pixels."
    );

    el.verdict.innerHTML = parts.join(" ");
  }

  function renderMetrics(comparison) {
    el.metricsBody.textContent = "";

    Object.keys(comparison).forEach((key) => {
      const m = comparison[key];
      const tr = document.createElement("tr");

      const nameCell = document.createElement("td");
      const nameWrap = document.createElement("span");
      nameWrap.className = "m-name";

      const label = document.createElement("span");
      label.textContent = m.label || key;
      nameWrap.appendChild(label);

      if (m.kind === "estimated") {
        const tag = document.createElement("span");
        tag.className = "m-est";
        tag.textContent = "estimated";
        tag.title = "Derived from an assumption about the image, not measured directly.";
        nameWrap.appendChild(tag);
      }

      const unit = document.createElement("span");
      unit.className = "m-unit";
      unit.textContent = m.unit || "";
      nameWrap.appendChild(unit);
      nameCell.appendChild(nameWrap);

      const before = document.createElement("td");
      before.className = "num";
      before.textContent = fmt(m.before);

      const after = document.createElement("td");
      after.className = "num";
      after.textContent = fmt(m.after);

      const delta = document.createElement("td");
      delta.className = "num delta " +
        (m.improved === true ? "up" : m.improved === false ? "down" : "flat");
      if (m.delta === null || m.delta === undefined) {
        delta.textContent = "\u2014";
      } else {
        const sign = m.delta > 0 ? "+" : "";
        delta.textContent = sign + fmt(m.delta);
        if (m.delta_pct !== null && m.delta_pct !== undefined) {
          const pct = document.createElement("span");
          pct.className = "delta-pct";
          pct.textContent = (m.delta_pct > 0 ? "+" : "") + fmt(m.delta_pct, 0) + "%";
          delta.appendChild(pct);
        }
      }

      tr.append(nameCell, before, after, delta);
      el.metricsBody.appendChild(tr);
    });
  }

  /* Histogram drawn by hand rather than with a chart library: two filled
     curves, no axes furniture, because the shape is the whole message. */
  function renderHistogram(hist) {
    const canvas = el.histogram;
    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 760;
    const cssHeight = 220;

    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const pad = { top: 12, right: 8, bottom: 22, left: 8 };
    const w = cssWidth - pad.left - pad.right;
    const h = cssHeight - pad.top - pad.bottom;

    // Baseline plus quarter-tone guides, so the eye can locate the midpoint.
    ctx.strokeStyle = "#2a323d";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = pad.left + (w * i) / 4;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + h);
      ctx.stroke();
    }

    const drawCurve = (values, stroke, fill) => {
      if (!values || !values.length) return;
      const step = w / (values.length - 1);
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top + h);
      values.forEach((v, i) => {
        ctx.lineTo(pad.left + i * step, pad.top + h - v * h);
      });
      ctx.lineTo(pad.left + w, pad.top + h);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    };

    drawCurve(hist.before.luma, "#5a6675", "rgba(90, 102, 117, 0.42)");
    drawCurve(hist.after.luma, "#e8b464", "rgba(232, 180, 100, 0.26)");

    ctx.fillStyle = "#6b7685";
    ctx.font = "11px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "top";
    ctx.fillText("black", pad.left, pad.top + h + 6);
    ctx.textAlign = "right";
    ctx.fillText("white", pad.left + w, pad.top + h + 6);
    ctx.textAlign = "left";
  }

  function renderZones(telemetry) {
    el.zones.textContent = "";
    telemetry.zones.forEach((zone) => {
      const row = document.createElement("div");
      row.className = "zone-row";

      const head = document.createElement("div");
      head.className = "zone-head";

      const name = document.createElement("span");
      name.textContent = zone.zone + " (" + zone.range + ")";

      const values = document.createElement("span");
      values.innerHTML = zone.before_pct.toFixed(1) + "% \u2192 <b>" +
        zone.after_pct.toFixed(1) + "%</b>";

      head.append(name, values);

      const bar = document.createElement("div");
      bar.className = "zone-bar";
      const beforeBar = document.createElement("i");
      beforeBar.className = "zb-before";
      beforeBar.style.width = (zone.before_pct / 2) + "%";
      const afterBar = document.createElement("i");
      afterBar.className = "zb-after";
      afterBar.style.width = (zone.after_pct / 2) + "%";
      bar.append(beforeBar, afterBar);

      row.append(head, bar);
      el.zones.appendChild(row);
    });
  }

  function renderMethod(payload) {
    const report = payload.enhancement || {};
    el.methodName.textContent = report.method || "";

    el.steps.textContent = "";
    (report.operations || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      el.steps.appendChild(li);
    });

    el.caveat.textContent = report.caveat || "";

    el.facts.textContent = "";
    const facts = [
      ["Source", payload.source.width + " \u00d7 " + payload.source.height +
        " " + payload.source.detected_format],
      ["Processing time", payload.timings_ms.total_ms + " ms"],
    ];

    if (report.neural === false) {
      facts.push(["Neural weights", "not installed \u2014 analytic fallback"]);
    } else if (report.neural === true) {
      facts.push(["Neural weights", "loaded, " + report.iterations + " iterations"]);
    }
    if (payload.source.resized_for_processing) {
      facts.push(["Note", "downscaled from " + payload.source.source_width +
        " \u00d7 " + payload.source.source_height]);
    }
    if (report.fallback_reason) {
      facts.push(["Reason", report.fallback_reason]);
    }

    facts.forEach(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      el.facts.append(dt, dd);
    });
  }

  /* ---------------------------------------------------------------- view -- */

  function setView(view) {
    state.view = view;
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("is-active", tab.dataset.view === view);
    });

    const isCompare = view === "compare";
    el.compareWrap.hidden = !isCompare;
    el.singleWrap.hidden = isCompare;
    el.diffLegend.hidden = view !== "difference";

    if (!isCompare && state.result) {
      el.imgSingle.src = state.result.images[view] + "?t=" + Date.now();
    }
    if (isCompare) sizeCompareLayer();
  }

  function setSplit(percent) {
    state.split = Math.max(0, Math.min(100, percent));
    el.compareTop.style.width = state.split + "%";
    el.handle.style.left = state.split + "%";
    el.handle.setAttribute("aria-valuenow", Math.round(state.split));
    el.stageTagRight.style.opacity = state.split > 88 ? "0" : "1";
    document.querySelector(".stage-tag-left").style.opacity = state.split < 12 ? "0" : "1";
  }

  /* The clipped layer is narrower than the frame, so its image must be pinned
     to the full frame width or the two halves drift out of alignment. */
  function sizeCompareLayer() {
    const width = el.compareWrap.clientWidth;
    if (width) {
      el.imgTop.style.width = width + "px";
      el.imgTop.style.height = "auto";
    }
  }

  function pointerToSplit(clientX) {
    const rect = el.compareWrap.getBoundingClientRect();
    if (!rect.width) return;
    setSplit(((clientX - rect.left) / rect.width) * 100);
  }

  /* -------------------------------------------------------------- events -- */

  el.dropzone.addEventListener("click", (event) => {
    if (state.file && !event.target.closest(".dropzone-inner")) return;
    el.fileInput.click();
  });

  el.dropzone.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && !state.file) {
      event.preventDefault();
      el.fileInput.click();
    }
  });

  el.fileInput.addEventListener("change", (event) => acceptFile(event.target.files[0]));

  ["dragenter", "dragover"].forEach((name) => {
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.add("is-over");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.remove("is-over");
    });
  });

  el.dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) acceptFile(files[0]);
  });

  // The whole window is a drop target in practice; stop the browser from
  // navigating away to the dropped file if the aim was off.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  el.clearFile.addEventListener("click", (event) => {
    event.stopPropagation();
    clearFile();
  });

  el.runBtn.addEventListener("click", runEnhancement);

  ["gamma", "clahe_clip", "clahe_grid"].forEach((id) => {
    $(id).addEventListener("input", syncOutputs);
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  });

  // Comparison handle: pointer events cover mouse, touch and pen in one path.
  el.handle.addEventListener("pointerdown", (event) => {
    el.handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  el.handle.addEventListener("pointermove", (event) => {
    if (el.handle.hasPointerCapture(event.pointerId)) pointerToSplit(event.clientX);
  });

  el.compareWrap.addEventListener("click", (event) => {
    if (event.target === el.handle || el.handle.contains(event.target)) return;
    pointerToSplit(event.clientX);
  });

  el.handle.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 10 : 2;
    if (event.key === "ArrowLeft") { setSplit(state.split - step); event.preventDefault(); }
    if (event.key === "ArrowRight") { setSplit(state.split + step); event.preventDefault(); }
    if (event.key === "Home") { setSplit(0); event.preventDefault(); }
    if (event.key === "End") { setSplit(100); event.preventDefault(); }
  });

  el.discard.addEventListener("click", async () => {
    if (!state.result) return;
    try {
      await fetch("/api/results/" + state.result.id, { method: "DELETE" });
    } catch (err) { /* the result expires on its own regardless */ }
    state.result = null;
    el.result.hidden = true;
    el.empty.hidden = false;
  });

  el.toastClose.addEventListener("click", hideToast);

  window.addEventListener("resize", () => {
    sizeCompareLayer();
    if (state.result) renderHistogram(state.result.histogram);
  });

  el.imgBase.addEventListener("load", sizeCompareLayer);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) runEnhancement();
    if (event.key === "Escape") hideToast();
  });

  loadConfig();
  syncOutputs();
})();
