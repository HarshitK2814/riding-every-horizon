/* Riding Every Horizon: page logic. The backtest itself runs in worker.js (Pyodide + trend_lite.py). */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const DEFAULTS = { signal: "tanh", use: ["short", "medium", "long"], buffer: 0.10, target_vol: 10, cost_mult: 1, period: "full" };
  // date-defined windows; same as CRISES in trend_lite.py
  const CRISES = [["2008-01-01", "2008-12-31"], ["2018-10-01", "2018-12-31"], ["2020-02-19", "2020-03-23"], ["2022-01-01", "2022-12-31"]];
  const PERIODS = {
    full: { start: null, end: null, hint: "All data from the first live day (warm-up excluded). Seen by the researcher; contaminated for TimesFM." },
    choice: { start: "2001-01-01", end: "2017-12-31", hint: "Development + validation: v1's settings were chosen on these years, so this is in-sample for v1." },
    seen: { start: "2018-01-01", end: "2024-12-31", hint: "v1's pre-registered test period. Read once in v1; seen ever since, and contaminated for TimesFM." },
    crises: { windows: CRISES, hint: "Days inside four date-defined windows (2008, 2018 Q4, Covid 2020, 2022). Positions still built on full history." },
  };
  const SIGNAL_HINT = {
    tanh: "tanh of the trailing-return t-statistic, per horizon, blended with equal weight (v1).",
    sign: "Sign of the trailing return per horizon (v1 EXP-001 variant). Flips on noise; trades more.",
    tsfm: "TimesFM 2.5 zero-shot forecasts of 21/63/252-day returns, blended, refreshed weekly, held up to 5 days. Horizons fixed by design.",
  };
  const LABEL_BANDS = (m) => [
    { key: "--band-c", name: "contaminated", x0: "1990-01-01", x1: m.labels.seen_start },
    { key: "--band-cs", name: "contam. + seen", x0: m.labels.seen_start, x1: nextDay(m.labels.contam_end) },
    { key: "--band-s", name: "seen", x0: nextDay(m.labels.contam_end), x1: m.labels.forward_start },
    { key: "--band-f", name: "forward", x0: m.labels.forward_start, x1: "2030-01-01" },
  ];
  const PILL = { supported: "supported", not_supported: "not supported", pending: "pending" };

  let META = null, last = null, lastDefault = false, sentDefault = new Map(), worker = null, ready = false, inflight = false, queued = null, seq = 0, timer = null;

  function nextDay(d) { const t = new Date(d + "T00:00:00Z"); t.setUTCDate(t.getUTCDate() + 1); return t.toISOString().slice(0, 10); }
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (x, d = 2) => (x == null || !isFinite(x) ? "—" : x.toFixed(d).replace("-", "−"));
  const pct = (x, d = 1) => (x == null || !isFinite(x) ? "—" : (x * 100).toFixed(d).replace("-", "−") + "%");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); });

  /* ---------- verdicts ---------- */
  function renderVerdicts(v) {
    const cols = ["v1", "v2", "v3"].map((k) => v[k]);
    const all = cols.flatMap((c) => c.rows);
    const n = (s) => all.filter((r) => r.status === s).length;
    const pend = all.find((r) => r.status === "pending");
    $("#tally").innerHTML = `<b>${all.length}</b> pre-registered hypotheses · <b>${n("supported")}</b> supported · ` +
      `<b>${n("not_supported")}</b> not supported · <b>${n("pending")}</b> pending` +
      (pend ? ` <span class="nowrap">(TimesFM forward test, first read ${esc((pend.result.match(/first read (\S+)/) || [])[1] || "")})</span>` : "");
    $("#vgrid").innerHTML = cols.map((c) => `
      <article class="vcol">
        <h3>${esc(c.title)}</h3>
        <p class="per">${esc(c.period)}</p>
        ${c.rows.map((r) => `
          <div class="hyp">
            <div class="hyp-top"><span class="claim"><span class="id">${esc(r.id)}</span>${esc(r.claim)}</span>
              <span class="pill ${r.status}">${PILL[r.status]}</span></div>
            <p class="explain">${esc(r.explain)}</p>
            <div class="result">${esc(r.result)}</div>
          </div>`).join("")}
        <a class="src" href="${esc(c.source)}">Full write-up &rarr;</a>
      </article>`).join("");
  }

  /* ---------- forward ---------- */
  function renderForward(f) {
    const today = new Date().toISOString().slice(0, 10);
    const titles = { v3: "v3 · TimesFM vs v1 vs clean model", v2: "v2 · v1 blend vs GARCH risk model" };
    $("#fgrid").innerHTML = ["v3", "v2"].map((k) => {
      const g = f[k];
      const books = Object.entries(g.books);
      const status = today < g.start
        ? `tracking starts <b>${g.start}</b> · first read ${g.first_read} · ${g.days} days tracked`
        : `tracking since <b>${g.start}</b> · first read ${g.first_read} · ${g.days} days tracked`;
      const body = g.days === 0
        ? `<div class="fempty">No forward days yet. The daily job appends one row per book per trading day, and this
             panel updates when it commits. The verdict is read once, on ${g.first_read}, with the pre-registered test.
             <span class="m">tracking starts ${g.start} · first read ${g.first_read} · 0 days tracked</span></div>`
        : `<div class="fchart" id="fchart-${k}" role="img" aria-label="${k} forward cumulative net return by book"></div>`;
      return `<article class="fcard">
        <h3>${titles[k]}</h3>
        <p class="fmeta">${status}${g.last_date ? ` · last ${g.last_date}` : ""}</p>
        ${body}
        ${books.map(([b, d]) => `<div class="fbook"><span>${esc(b)}</span><span>${d.days} d · ${d.cum.length ? pct(d.cum[d.cum.length - 1], 2) : "—"}</span></div>`).join("")}
        <p class="hint">${esc(g.note)} · <a href="${esc(g.source)}">forward.csv</a></p>
      </article>`;
    }).join("");
    if (!window.Plotly) return;
    for (const k of ["v3", "v2"]) {
      const el = $("#fchart-" + k);
      if (!el) continue;
      const traces = Object.entries(f[k].books).filter(([, d]) => d.dates.length).map(([b, d], i) => ({
        x: d.dates, y: d.cum.map((c) => c * 100), name: b, mode: "lines",
        line: { width: i === 0 ? 2 : 1.4, color: i === 0 ? css("--accent") : (i === 1 ? css("--ink") : css("--muted")) },
        hovertemplate: "%{x|%Y-%m-%d}  %{y:.2f}%<extra>" + b + "</extra>",
      }));
      Plotly.react(el, traces, baseLayout({ yaxis: { ticksuffix: "%" }, showlegend: true,
        legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } } }), { displayModeBar: false, responsive: true });
    }
  }

  function baseLayout(extra = {}) {
    const ink = css("--ink-2"), grid = css("--rule-2");
    const L = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "IBM Plex Mono, ui-monospace, monospace", size: 11, color: ink },
      margin: { l: 46, r: 10, t: 26, b: 34 }, hovermode: "x", showlegend: false,
      xaxis: { gridcolor: grid, linecolor: css("--rule"), zeroline: false, fixedrange: false },
      yaxis: { gridcolor: grid, linecolor: css("--rule"), zeroline: false },
      hoverlabel: { font: { family: "IBM Plex Mono, monospace", size: 11 } },
    };
    for (const [k, v] of Object.entries(extra)) L[k] = typeof v === "object" && !Array.isArray(v) && L[k] ? { ...L[k], ...v } : v;
    return L;
  }

  /* ---------- lab ---------- */
  function params() {
    const p = PERIODS[$("input[name=period]:checked").value];
    return {
      signal: $("#signal").value,
      use: $$("input[name=h]:checked").map((i) => i.value),
      buffer: +$("#buffer").value,
      target_vol: +$("#target_vol").value / 100,
      cost_mult: +$("#cost_mult").value,
      start: p.start || null, end: p.end || null, windows: p.windows || null,
    };
  }
  const isDefault = () => {
    const p = params();
    return p.signal === "tanh" && p.use.length === 3 && Math.abs(p.buffer - 0.1) < 1e-9 && Math.abs(p.target_vol - 0.1) < 1e-9 &&
      p.cost_mult === 1 && $("input[name=period]:checked").value === "full";
  };

  function syncControls() {
    $("#buffer-o").textContent = (+$("#buffer").value).toFixed(2);
    $("#target_vol-o").textContent = $("#target_vol").value + "%";
    $("#cost_mult-o").textContent = (+$("#cost_mult").value).toFixed(2) + "×";
    const tsfm = $("#signal").value === "tsfm";
    $("#horizons").classList.toggle("off", tsfm);
    $$("input[name=h]").forEach((i) => { i.disabled = tsfm; });
    const checked = $$("input[name=h]:checked");
    if (checked.length === 1) checked[0].disabled = true;     // at least one horizon
    $("#h-hint").textContent = SIGNAL_HINT[$("#signal").value];
    $("#p-hint").textContent = PERIODS[$("input[name=period]:checked").value].hint;
  }

  function schedule(delay = 350) {
    syncControls();
    if (!ready) return;
    clearTimeout(timer);
    $(".out").classList.add("stale");
    timer = setTimeout(send, delay);
  }

  function send() {
    const p = params();
    if (inflight) { queued = { p, d: isDefault() }; return; }
    inflight = true;
    $("#ctime").textContent = "computing…";
    post(p, isDefault());
  }

  function post(p, d) { const id = ++seq; sentDefault.set(id, d); worker.postMessage({ type: "run", id, params: p }); }

  function onResult(m) {
    inflight = false;
    if (queued) { const q = queued; queued = null; inflight = true; post(q.p, q.d); }
    else $(".out").classList.remove("stale");
    last = m.out; lastDefault = !!sentDefault.get(m.id); sentDefault.delete(m.id);
    $("#ctime").textContent = `computed in ${(m.ms / 1000).toFixed(2)} s`;
    renderResult(m.out);
  }

  function renderResult(o) {
    $("#s-sharpe").textContent = fmt(o.sharpe);
    $("#s-ci").textContent = `95% CI [${fmt(o.sharpe_lo)}, ${fmt(o.sharpe_hi)}]`;
    $("#s-days").textContent = o.days.toLocaleString("en-US");
    $("#s-ret").textContent = pct(o.ann_ret);
    $("#s-vol").textContent = pct(o.ann_vol);
    $("#s-dd").textContent = pct(o.maxdd);
    $("#s-cost").textContent = pct(o.cost_ann, 2);
    const ref = META && META.reference;
    if (ref && lastDefault && o.sharpe != null) {
      const d = Math.abs(o.sharpe - ref.sharpe);
      $("#refcheck").innerHTML = `Default settings vs the paper engine (<span class="m">trend.model.backtest</span>, same data, ${ref.days} days): ` +
        `${o.sharpe.toFixed(3)} here, ${ref.sharpe.toFixed(3)} there, difference ${d.toFixed(4)} ` +
        (d < 0.01 ? `<span class="ok">· match</span>` : `<b>· MISMATCH</b>`);
    } else {
      $("#refcheck").textContent = "Signal at close t, trade t+1. Net of v1 costs × multiplier; no market impact. Exploration on seen data, not evidence.";
    }
    $("#crisis").innerHTML = Object.entries(o.crisis).sort((a, b) => (a[1].start < b[1].start ? -1 : 1)).map(([k, c]) => `<tr><td>${esc(k)}</td><td class="dates">${c.start} – ${c.end}</td>` +
      `<td class="n">${c.days || "—"}</td><td class="n">${c.cum_net_return == null ? "—" : pct(c.cum_net_return)}</td></tr>`).join("");
    drawEquity(o);
  }

  function drawEquity(o) {
    if (!window.Plotly || !o) return;
    const x = [], y = [];
    for (let i = 0; i < o.dates.length; i++) {      // break the line across gaps (crisis windows)
      if (i && (Date.parse(o.dates[i]) - Date.parse(o.dates[i - 1])) > 10 * 864e5) { x.push(null); y.push(null); }
      x.push(o.dates[i]); y.push(o.equity[i]);
    }
    const first = o.dates[0], lastD = o.dates[o.dates.length - 1];
    const atEnd = META && lastD === META.last_date;
    const x1 = atEnd ? "2027-03-31" : lastD;               // show the forward band when the range reaches the end of data
    const bands = META ? LABEL_BANDS(META).filter((b) => b.x1 > first && b.x0 < x1) : [];
    const shapes = bands.map((b) => ({ type: "rect", xref: "x", yref: "paper", x0: b.x0 < first ? first : b.x0, x1: b.x1 > x1 ? x1 : b.x1,
      y0: 0, y1: 1, fillcolor: css(b.key), line: { width: 0 }, layer: "below" }));
    const narrow = window.innerWidth < 600;
    const span = Date.parse(x1) - Date.parse(first);
    const wide = (b) => (Date.parse(b.x1 > x1 ? x1 : b.x1) - Date.parse(b.x0 < first ? first : b.x0)) / span > (narrow ? 0.14 : 0.07);
    const ann = bands.filter(wide).map((b) => ({ xref: "x", yref: "paper", x: b.x0 < first ? first : b.x0, y: 1, yanchor: "bottom", xanchor: "left",
      text: narrow && b.name === "contam. + seen" ? "c+s" : b.name, showarrow: false, font: { size: 10, color: css("--muted") } }));
    Plotly.react("equity", [{
      x, y, mode: "lines", connectgaps: false, line: { width: 1.6, color: css("--accent") },
      hovertemplate: "%{x|%Y-%m-%d}  %{y:.3f}<extra></extra>",
    }], baseLayout({ shapes, annotations: ann, yaxis: { type: "log" }, xaxis: { range: [first, x1] } }),
    { displayModeBar: false, responsive: true });
  }

  /* ---------- loading ---------- */
  function stage(s, detail) {
    const order = ["runtime", "packages", "data", "ready"];
    const i = order.indexOf(s);
    $$(".stages span").forEach((el, j) => { el.classList.toggle("ok", j < i || s === "ready"); el.classList.toggle("on", j === i && s !== "ready"); });
    $("#loader-text").textContent = s === "ready" ? "Ready · runs locally in your browser" : `Loading ${s}: ${detail}…`;
    if (s === "ready") $("#loader").classList.add("done");
  }

  function startWorker() {
    if (!window.Worker) { $("#loader-text").textContent = "This browser has no Web Worker support; the lab cannot run."; return; }
    worker = new Worker("worker.js");
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "stage") stage(m.stage, m.detail);
      else if (m.type === "ready") { ready = true; stage("ready"); $("#run").disabled = false; send(); }
      else if (m.type === "result") onResult(m);
      else if (m.type === "error") {
        inflight = false;
        $("#loader").classList.add("err");
        $("#loader-text").textContent = "Error: " + m.message.split("\n").slice(-2).join(" ");
        console.error(m.message);
      }
    };
    worker.onerror = (e) => { $("#loader").classList.add("err"); $("#loader-text").textContent = "Worker failed to start: " + (e.message || e); };
  }

  function wire() {
    const form = $("#controls");
    form.addEventListener("input", () => schedule());
    form.addEventListener("change", () => schedule());
    form.addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(timer); syncControls(); if (ready) send(); });
    $("#reset").addEventListener("click", () => {
      $("#signal").value = DEFAULTS.signal;
      $$("input[name=h]").forEach((i) => { i.checked = true; });
      $("#buffer").value = DEFAULTS.buffer; $("#target_vol").value = DEFAULTS.target_vol; $("#cost_mult").value = DEFAULTS.cost_mult;
      $(`input[name=period][value=${DEFAULTS.period}]`).checked = true;
      schedule(0);
    });
    let rz;
    const redraw = () => { clearTimeout(rz); rz = setTimeout(() => { drawEquity(last); if (window.__fwd) renderForward(window.__fwd); }, 150); };
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    window.addEventListener("resize", redraw);
    syncControls();
  }

  async function init() {
    wire();
    startWorker();
    try {
      const [v, f, m] = await Promise.all([getJSON("data/verdicts.json"), getJSON("data/forward.json"), getJSON("data/meta.json")]);
      META = m;
      $("#vintage").textContent = m.vintage; $("#lastdate").textContent = m.last_date;
      renderVerdicts(v);
      window.__fwd = f; renderForward(f);
      if (last) renderResult(last);
    } catch (e) {
      $("#tally").textContent = "Could not load data files: " + e.message;
      console.error(e);
    }
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
