/* Riding Every Horizon: page logic. The backtest runs in worker.js (Pyodide + trend_lite.py), started only when the
   lab comes near the viewport or the reader asks for a run. Until then the lab shows data/default_run.json, which is
   the same code's output for the default settings, computed at export. Charts and theme come from kit/kit.js. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const DAY = 864e5;
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
  const PILL = { supported: ["supported", "k-pill--yes"], not_supported: ["not supported", ""], pending: ["pending", "k-pill--wait"] };
  const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
  const word = (n, cap) => { const w = WORDS[n] || String(n); return cap ? w[0].toUpperCase() + w.slice(1) : w; };

  let META = null, FIRST_READ = "2027-11-30", last = null, lastDefault = false, lastDisjoint = false, lastLive = false;
  let worker = null, started = false, ready = false, inflight = false, queued = null, seq = 0, timer = null;
  const sentDefault = new Map();

  function nextDay(d) { const t = new Date(d + "T00:00:00Z"); t.setUTCDate(t.getUTCDate() + 1); return t.toISOString().slice(0, 10); }
  const fmt = (x, d = 2) => (x == null || !isFinite(x) ? "—" : x.toFixed(d).replace("-", "−"));
  const pct = (x, d = 1) => (x == null || !isFinite(x) ? "—" : (x * 100).toFixed(d).replace("-", "−") + "%");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); });

  const bands = (m, accentLabel) => [
    { fill: "var(--k-band-1)", label: "contaminated", x0: "1990-01-01", x1: m.labels.seen_start },
    { fill: "var(--k-band-2)", label: "contaminated + seen", short: "c + s", x0: m.labels.seen_start, x1: nextDay(m.labels.contam_end) },
    { fill: "var(--k-band-3)", label: "seen", x0: nextDay(m.labels.contam_end), x1: m.labels.forward_start },
    { fill: "var(--k-accent)", accent: true, label: accentLabel, short: "forward", anchor: "end", x0: m.labels.forward_start, x1: "2030-01-01" },
  ];

  /* ---------- verdicts ---------- */
  function renderVerdicts(v) {
    const cols = ["v1", "v2", "v3"].map((k) => ({ k, ...v[k] }));
    const all = cols.flatMap((c) => c.rows.map((r) => ({ ...r, round: c.k })));
    const n = (s) => all.filter((r) => r.status === s).length;
    const pend = all.find((r) => r.status === "pending");
    const pendRead = pend ? ((pend.result.match(/first read (\S+)/) || [])[1] || "") : "";
    if (pendRead) FIRST_READ = pendRead;
    $("#tally").innerHTML = `<b>${all.length}</b> pre-registered hypotheses: <b>${n("supported")}</b> supported, ` +
      `<b>${n("not_supported")}</b> not supported, <b>${n("pending")}</b> pending` +
      (pend ? ` (the TimesFM forward test, first read ${esc(pendRead)}).` : ".");
    const strip = $("#tally-strip");
    strip.style.setProperty("--n", all.length);
    strip.innerHTML = all.map((r) => `<div class="k-tally__cell ${r.status === "supported" ? "is-yes" : r.status === "pending" ? "is-wait" : ""}">` +
      `<i></i><span>${esc(r.round)} ${esc(r.id)}</span></div>`).join("");
    // hero and "how" sentences use the same counts
    $("#h-total").textContent = word(all.length, true);
    $("#h-yes").textContent = word(n("supported"), true);
    $("#h-no").textContent = word(n("not_supported"), true);
    $("#h-wait").textContent = word(n("pending"));
    $("#how-no").textContent = word(n("not_supported"), true);
    $("#how-total").textContent = word(all.length);
    $("#vgrid").innerHTML = cols.map((c) => `
      <article class="k-verdict" aria-labelledby="vt-${c.k}">
        <div class="k-verdict__head"><h3 class="k-h3" id="vt-${c.k}">${esc(c.title)}</h3><p class="k-verdict__per">${esc(c.period)}</p></div>
        ${c.rows.map((r) => `
          <div class="k-hyp">
            <div class="k-hyp__top"><p class="k-hyp__claim"><span class="k-hyp__id">${esc(r.id)}</span>${esc(r.claim)}</p>
              <span class="k-pill ${PILL[r.status][1]}">${PILL[r.status][0]}</span></div>
            <p class="k-hyp__explain">${esc(r.explain)}</p>
            <p class="k-hyp__result">${esc(r.result)}</p>
          </div>`).join("")}
        <a class="k-verdict__src" href="${esc(c.source)}">Full ${esc(c.k)} write-up</a>
      </article>`).join("");
  }

  /* ---------- forward ---------- */
  function renderForward(f) {
    const today = new Date().toISOString().slice(0, 10);
    const g3 = f.v3;
    const days = Math.max(0, Math.ceil((Date.parse(g3.first_read) - Date.parse(today)) / DAY));
    $("#clock-date").textContent = g3.first_read;
    $("#clock-unit").textContent = days === 1 ? "day" : "days";
    $("#clock-sub").textContent = (today < g3.start ? `Tracking starts ${g3.start}.` : `Tracking since ${g3.start}.`) +
      " The pre-registered test needs at least 240 trading days.";
    $("#clock-tracked").textContent = g3.days;
    const meter = $("#clock-meter");
    meter.setAttribute("aria-valuenow", Math.min(240, g3.days));
    meter.firstElementChild.style.setProperty("--v", Math.min(100, g3.days / 240 * 100).toFixed(1) + "%");
    meter.firstElementChild.classList.toggle("is-zero", g3.days === 0);
    const num = $("#clock-days");
    num.textContent = days;
    const clock = $("#clock");
    const count = () => window.Kit && Kit.countUp(num, days, { from: Math.min(days + 365, days * 2), duration: 1300 });
    clock.classList.contains("is-in") ? null : clock.addEventListener("kit:reveal", count, { once: true });

    const titles = { v3: "v3: TimesFM vs v1 vs clean model", v2: "v2: v1 blend vs GARCH risk model" };
    $("#fgrid").innerHTML = ["v3", "v2"].map((k) => {
      const g = f[k];
      const status = today < g.start
        ? `Tracking starts ${esc(g.start)}. First read ${esc(g.first_read)}. ${esc(g.days)} days tracked`
        : `Tracking since ${esc(g.start)}. First read ${esc(g.first_read)}. ${esc(g.days)} days tracked`;
      const body = g.days === 0
        ? `<div class="fcard__empty"><p>No forward days yet. The daily job appends one row per book per trading day, and this
             panel updates when it commits. The verdict is read once, on ${esc(g.first_read)}, with the pre-registered test.</p></div>`
        : `<div class="k-chart__plot" id="fchart-${k}" role="group" aria-label="${k} forward cumulative net return by book. Use the arrow keys to read values."></div>
           <p class="k-sr" aria-live="polite" id="flive-${k}"></p>`;
      return `<article class="fcard" aria-labelledby="ft-${k}">
        <h3 class="k-h3" id="ft-${k}">${titles[k]}</h3>
        <p class="fcard__meta">${status}${g.last_date ? `, last ${esc(g.last_date)}` : ""}.</p>
        ${body}
        <dl class="fbooks">${Object.entries(g.books).map(([b, d]) => `<div><dt>${esc(b)}</dt><dd>${esc(d.days)} d, ${d.cum.length ? pct(d.cum[d.cum.length - 1], 2) : "—"}</dd></div>`).join("")}</dl>
        <p class="k-small k-muted">${esc(g.note)}. Data: <a href="${esc(g.source)}">forward.csv</a></p>
      </article>`;
    }).join("");
    for (const k of ["v3", "v2"]) {
      const el = $("#fchart-" + k);
      if (!el || !window.Kit) continue;
      const books = Object.entries(f[k].books).filter(([, d]) => d.dates.length);
      const cls = [null, "k-line k-line--2", "k-line k-line--3"];
      Kit.lineChart(el, {
        series: books.map(([b, d], i) => ({ name: b, x: d.dates, y: d.cum, cls: cls[i], endLabel: true })),
        includeZero: true, yTick: (v) => (v * 100).toFixed(1).replace("-", "−") + "%", live: $("#flive-" + k),
        tip: (i, date) => date + "  " + books.map(([b, d]) => `${b} ${pct(d.cum[d.dates.indexOf(date)], 2)}`).join("  "),
      });
    }
  }

  /* ---------- hero ---------- */
  function renderHero(o) {
    if (!o || !window.Kit) return;
    $("#hs-span").textContent = `${o.dates[0]} to ${o.dates[o.dates.length - 1]}`;
    $("#hs-ci").textContent = `95% CI [${fmt(o.sharpe_lo)}, ${fmt(o.sharpe_hi)}]`;
    Kit.countUp($("#hs-sharpe"), o.sharpe, { format: (v) => fmt(v), label: "Net Sharpe" });
    Kit.countUp($("#hs-dd"), o.maxdd, { format: (v) => pct(v), label: "Max drawdown" });
    Kit.countUp($("#hs-days"), o.days, { format: (v) => Math.round(v).toLocaleString("en-US"), label: "Trading days" });
    drawEquity($("#hero-plot"), o, { animate: true, live: $("#hero-live"), accentLabel: "forward test" });
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
    $$(".k-range").forEach((r) => r.style.setProperty("--pct", ((r.value - r.min) / (r.max - r.min) * 100) + "%"));
    const tsfm = $("#signal").value === "tsfm";
    $("#horizons-set").classList.toggle("is-off", tsfm);
    $$("input[name=h]").forEach((i) => { i.disabled = tsfm; });
    const checked = $$("input[name=h]:checked");
    if (checked.length === 1) checked[0].disabled = true;     // at least one horizon
    $("#h-hint").textContent = SIGNAL_HINT[$("#signal").value];
    $("#p-hint").textContent = PERIODS[$("input[name=period]:checked").value].hint;
  }

  function schedule(delay = 350) {
    syncControls();
    startWorker();
    if (!ready) return;                       // the run happens as soon as the engine is ready
    clearTimeout(timer);
    $("#out").classList.add("is-stale");
    timer = setTimeout(send, delay);
  }

  function send() {
    const p = params();
    if (inflight) { queued = { p, d: isDefault(), j: !!p.windows }; return; }
    inflight = true;
    $("#ctime").textContent = "computing…";
    $("#out").setAttribute("aria-busy", "true");
    post(p, isDefault(), !!p.windows);
  }

  function post(p, d, j) { const id = ++seq; sentDefault.set(id, { d, j }); worker.postMessage({ type: "run", id, params: p }); }

  function onResult(m) {
    inflight = false;
    if (queued) { const q = queued; queued = null; inflight = true; post(q.p, q.d, q.j); }
    else { $("#out").classList.remove("is-stale"); $("#out").setAttribute("aria-busy", "false"); }
    const sent = sentDefault.get(m.id) || {};
    last = m.out; lastDefault = !!sent.d; lastDisjoint = !!sent.j; lastLive = true; sentDefault.delete(m.id);
    $("#ctime").textContent = `computed in ${(m.ms / 1000).toFixed(2)} s`;
    renderResult(m.out);
  }

  function renderResult(o) {
    const dj = lastDisjoint;     // Sharpe, CI and vol are not meaningful on joined, non-contiguous windows
    $("#s-sharpe").textContent = dj ? "—" : fmt(o.sharpe);
    $("#s-ci").textContent = dj ? "Sharpe not meaningful on disjoint windows" : `95% CI [${fmt(o.sharpe_lo)}, ${fmt(o.sharpe_hi)}]`;
    $("#s-days").textContent = o.days.toLocaleString("en-US");
    $("#s-ret").textContent = pct(o.ann_ret);
    $("#s-vol").textContent = dj ? "—" : pct(o.ann_vol);
    $("#s-dd").textContent = pct(o.maxdd);
    $("#s-cost").textContent = pct(o.cost_ann, 2);
    const ref = META && META.reference;
    if (!lastLive) {
      $("#refcheck").innerHTML = `Default settings, precomputed by <code>trend_lite.py</code> when the data was exported. ` +
        `Your browser reruns it once the engine has loaded and checks it against the paper engine.`;
    } else if (ref && lastDefault && o.sharpe != null) {
      const d = Math.abs(o.sharpe - ref.sharpe);
      $("#refcheck").innerHTML = `Default settings vs the paper engine (<code>trend.model.backtest</code>, same data, ${esc(ref.days)} days): ` +
        `${o.sharpe.toFixed(3)} here, ${ref.sharpe.toFixed(3)} there, difference ${d.toFixed(4)} ` +
        (d < 0.01 ? `<span class="ok">match</span>` : `<b>MISMATCH</b>`);
    } else {
      $("#refcheck").textContent = "Signal at close t, trade t+1. Net of v1 costs × multiplier; no market impact. Exploration on seen data, not evidence.";
    }
    $("#crisis").innerHTML = Object.entries(o.crisis).sort((a, b) => (a[1].start < b[1].start ? -1 : 1)).map(([k, c]) =>
      `<tr><th scope="row">${esc(k)}</th><td>${esc(c.start)} – ${esc(c.end)}</td>` +
      `<td class="n">${c.days || "—"}</td><td class="n">${c.cum_net_return == null ? "—" : pct(c.cum_net_return)}</td></tr>`).join("");
    drawEquity($("#equity"), o, { live: $("#equity-live"), accentLabel: "forward" });
  }

  function drawEquity(el, o, opt) {
    if (!window.Kit || !o || !o.dates.length) return;
    const first = o.dates[0], lastD = o.dates[o.dates.length - 1];
    const atEnd = META && lastD === META.last_date;
    const x1 = atEnd ? FIRST_READ : lastD;          // show the forward band when the range reaches the end of data
    Kit.lineChart(el, {
      series: [{ name: "growth of 1", x: o.dates, y: o.equity }],
      log: true, xMin: first, xMax: x1, gapDays: 10, animate: !!opt.animate, live: opt.live,
      bands: META ? bands(META, opt.accentLabel) : [],
      yTick: (v) => (v >= 10 ? v.toFixed(0) : String(+v.toFixed(2))),
      tip: (i, date) => `${date}  growth ${o.equity[i].toFixed(3)}`,
    });
  }

  /* ---------- engine loading ---------- */
  function stage(s, detail) {
    const order = ["runtime", "packages", "data", "ready"];
    const i = order.indexOf(s);
    $$(".k-steps li").forEach((el, j) => { el.classList.toggle("is-ok", j < i || s === "ready"); el.classList.toggle("is-on", j === i && s !== "ready"); });
    const L = $("#loader");
    L.classList.toggle("is-busy", s !== "ready");
    L.classList.toggle("is-ok", s === "ready");
    $("#loader-text").textContent = s === "ready" ? "Ready. The engine runs locally in your browser." : `Loading ${s}: ${detail}…`;
  }

  function startWorker() {
    if (started) return;
    started = true;
    if (!window.Worker) { $("#loader-text").textContent = "This browser has no Web Worker support, so the lab cannot run. The numbers shown are the precomputed default."; return; }
    stage("runtime", "Python runtime (Pyodide 0.26.4)");
    worker = new Worker("worker.js");
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "stage") stage(m.stage, m.detail);
      else if (m.type === "ready") { ready = true; stage("ready"); send(); }
      else if (m.type === "result") onResult(m);
      else if (m.type === "error") {
        const msg = "Error: " + String(m.message).split("\n").slice(-2).join(" ");
        console.error(m.message);
        if (m.id == null) {                       // boot failure: the lab cannot run
          inflight = false; ready = false;
          const L = $("#loader"); L.classList.remove("is-busy"); L.classList.add("is-err");
          $("#loader-text").textContent = msg + " The numbers shown are the precomputed default.";
          return;
        }
        sentDefault.delete(m.id);
        inflight = false;
        $("#ctime").textContent = msg;
        if (queued) { const q = queued; queued = null; inflight = true; post(q.p, q.d, q.j); }
        else { $("#out").classList.remove("is-stale"); $("#out").setAttribute("aria-busy", "false"); }
      }
    };
    worker.onerror = (e) => {
      const L = $("#loader"); L.classList.remove("is-busy"); L.classList.add("is-err");
      $("#loader-text").textContent = "The engine failed to start: " + (e.message || e) + ". The numbers shown are the precomputed default.";
    };
  }

  function wire() {
    const form = $("#controls");
    form.addEventListener("input", () => schedule());
    form.addEventListener("change", () => schedule());
    form.addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(timer); syncControls(); startWorker(); if (ready) send(); });
    $("#reset").addEventListener("click", () => {
      $("#signal").value = DEFAULTS.signal;
      $$("input[name=h]").forEach((i) => { i.checked = true; });
      $("#buffer").value = DEFAULTS.buffer; $("#target_vol").value = DEFAULTS.target_vol; $("#cost_mult").value = DEFAULTS.cost_mult;
      $(`input[name=period][value=${DEFAULTS.period}]`).checked = true;
      schedule(0);
    });
    $("#hero-run").addEventListener("click", () => {
      startWorker();
      setTimeout(() => $("#signal").focus({ preventScroll: true }), Kit.reducedMotion() ? 0 : 600);
    });
    // load the engine a little before the lab scrolls into view
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); startWorker(); } }, { rootMargin: "300px 0px" });
      io.observe($("#lab"));
    }
    syncControls();
  }

  async function init() {
    wire();
    const run = getJSON("data/default_run.json").catch((e) => { console.error(e); return null; });
    try {
      const [v, f, m] = await Promise.all([getJSON("data/verdicts.json"), getJSON("data/forward.json"), getJSON("data/meta.json")]);
      META = m;
      $$(".vintage").forEach((el) => { el.textContent = m.vintage; });
      $("#lastdate").textContent = m.last_date;
      $("#hn-start").textContent = m.labels.forward_start;
      renderVerdicts(v);
      $("#hn-read").textContent = FIRST_READ;
      renderForward(f);
    } catch (e) {
      $("#tally").textContent = "Could not load data files: " + e.message;
      console.error(e);
    }
    const d = await run;
    if (d) {
      renderHero(d);
      if (!last) { last = d; lastDefault = true; lastDisjoint = false; renderResult(d); }
    } else if (!last) {
      $("#refcheck").textContent = "The precomputed default run could not be loaded; the engine will compute it.";
    }
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
