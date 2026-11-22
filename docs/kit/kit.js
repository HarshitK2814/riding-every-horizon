/* Research site kit v1: theme toggle, reveal-on-scroll, count-up numbers, and a small accessible SVG line chart.
   No dependencies. Exposes window.Kit. See kit/README.md. */
(function () {
  "use strict";
  var KEY = "kit-theme";
  var root = document.documentElement;
  var reduced = function () { return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches; };

  /* ---------- theme ---------- */
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function labelToggle(btn) {
    var next = effectiveTheme() === "dark" ? "light" : "dark";
    btn.setAttribute("aria-label", "Switch to " + next + " theme");
    btn.setAttribute("title", "Switch to " + next + " theme");
  }
  function initTheme() {
    var btns = document.querySelectorAll("[data-theme-toggle]");
    Array.prototype.forEach.call(btns, function (btn) {
      labelToggle(btn);
      btn.addEventListener("click", function () {
        var next = effectiveTheme() === "dark" ? "light" : "dark";
        root.classList.add("k-theme-switching");
        root.setAttribute("data-theme", next);
        void getComputedStyle(root).backgroundColor;          // flush styles with transitions off
        requestAnimationFrame(function () { requestAnimationFrame(function () { root.classList.remove("k-theme-switching"); }); });
        try { localStorage.setItem(KEY, next); } catch (e) { /* storage blocked: theme still switches for this visit */ }
        Array.prototype.forEach.call(btns, labelToggle);
        document.dispatchEvent(new CustomEvent("kit:theme", { detail: next }));
      });
    });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var on = function () { Array.prototype.forEach.call(btns, labelToggle); document.dispatchEvent(new CustomEvent("kit:theme", { detail: effectiveTheme() })); };
      mq.addEventListener ? mq.addEventListener("change", on) : mq.addListener(on);
    }
  }

  /* ---------- reveal on scroll (opt-in: [data-reveal]; use for one or two moments, not every section) ---------- */
  function reveal(scope) {
    var els = (scope || document).querySelectorAll("[data-reveal]:not(.is-in)");
    var show = function (el) { el.classList.add("is-in"); el.dispatchEvent(new CustomEvent("kit:reveal", { bubbles: true })); };
    if (!("IntersectionObserver" in window) || reduced()) { Array.prototype.forEach.call(els, show); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); show(e.target); } });
    }, { rootMargin: "0px 0px -12% 0px", threshold: 0.15 });
    Array.prototype.forEach.call(els, function (el) { io.observe(el); });
  }

  /* ---------- count-up: the final text is set at once for assistive tech; only the visible digits animate ---------- */
  function countUp(el, to, opts) {
    opts = opts || {};
    var fmt = opts.format || function (v) { return v.toFixed(opts.decimals || 0); };
    var final = fmt(to);
    el.setAttribute("aria-label", opts.label ? opts.label + " " + final : final);
    if (reduced() || !isFinite(to)) { el.textContent = final; return; }
    var from = opts.from || 0, dur = opts.duration || 1100, t0 = null;
    function step(t) {
      if (t0 === null) t0 = t;
      var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = p < 1 ? fmt(from + (to - from) * e) : final;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ---------- SVG line chart ---------- */
  var NS = "http://www.w3.org/2000/svg";
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var DAY = 864e5;
  function ms(d) { return typeof d === "number" ? d : Date.parse(d.length === 10 ? d + "T00:00:00Z" : d); }
  function iso(t) { return new Date(t).toISOString().slice(0, 10); }
  function mk(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function niceLinear(lo, hi, n) {
    var span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    var step = (f < 1.5 ? 1 : f < 2.25 ? 2 : f < 3.5 ? 2.5 : f < 7 ? 5 : 10) * mag, out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function niceLog(lo, hi) {
    var sets = [[1, 1.5, 2, 3, 4, 5, 7], [1, 2, 3, 5], [1, 2, 5], [1]];
    for (var s = 0; s < sets.length; s++) {
      var out = [];
      for (var k = Math.floor(Math.log10(lo)) - 1; k <= Math.ceil(Math.log10(hi)); k++)
        sets[s].forEach(function (m) { var v = m * Math.pow(10, k); if (v >= lo && v <= hi) out.push(+v.toPrecision(6)); });
      if (out.length <= 7) return out.length >= 2 ? out : niceLinear(lo, hi, 4);
    }
    return [lo, hi];
  }
  function timeTicks(a, b, width) {
    var span = (b - a) / DAY, maxN = Math.max(2, Math.floor(width / 84)), out = [], d, t;
    if (span <= 120) {
      var stepD = [1, 2, 7, 14, 28].filter(function (s) { return span / s <= maxN; })[0] || 28;
      for (t = a; t <= b; t += stepD * DAY) { d = new Date(t); out.push({ t: t, s: MON[d.getUTCMonth()] + " " + d.getUTCDate() }); }
      return out;
    }
    if (span <= 1100) {
      var stepM = [1, 2, 3, 6, 12].filter(function (s) { return span / 30.4 / s <= maxN; })[0] || 12;
      d = new Date(a); d = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1);
      for (t = d; t <= b; ) {
        var dt = new Date(t);
        if (dt.getUTCMonth() % stepM === 0) out.push({ t: t, s: MON[dt.getUTCMonth()] + (dt.getUTCMonth() === 0 ? " " + dt.getUTCFullYear() : "") });
        t = Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth() + 1, 1);
      }
      return out;
    }
    var years = span / 365.25, stepY = [1, 2, 5, 10, 20].filter(function (s) { return years / s <= maxN; })[0] || 20;
    var y0 = new Date(a).getUTCFullYear() + 1, y1 = new Date(b).getUTCFullYear();
    for (var y = Math.ceil(y0 / stepY) * stepY; y <= y1; y += stepY) out.push({ t: Date.UTC(y, 0, 1), s: String(y) });
    return out;
  }

  function lineChart(el, opts) {
    el.__kitChart = opts;
    if (!el.__kitRO && "ResizeObserver" in window) {
      var lastW = el.clientWidth, raf = 0;
      el.__kitRO = new ResizeObserver(function () {
        if (Math.abs(el.clientWidth - lastW) < 2) return;
        lastW = el.clientWidth; cancelAnimationFrame(raf);
        raf = requestAnimationFrame(function () { draw(el, el.__kitChart, false); });
      });
      el.__kitRO.observe(el);
    }
    draw(el, opts, !!opts.animate && !reduced());
  }

  function draw(el, o, animate) {
    var W = el.clientWidth, H = el.clientHeight;
    if (!W || !H) return;
    var narrow = W < 520;
    var series = o.series.filter(function (s) { return s.x && s.x.length; });
    var endLbl = series.some(function (s) { return s.endLabel; });
    var M = { l: o.padLeft || (narrow ? 38 : 46), r: endLbl ? (narrow ? 64 : 84) : 12, t: o.bands && o.bands.length ? 24 : 10, b: 26 };
    var iw = W - M.l - M.r, ih = H - M.t - M.b;
    var X = series.map(function (s) { return s.x.map(ms); });
    var x0 = o.xMin ? ms(o.xMin) : Math.min.apply(null, X.map(function (a) { return a[0]; }));
    var x1 = o.xMax ? ms(o.xMax) : Math.max.apply(null, X.map(function (a) { return a[a.length - 1]; }));
    var ylo = Infinity, yhi = -Infinity;
    series.forEach(function (s) { s.y.forEach(function (v) { if (v != null && isFinite(v)) { if (v < ylo) ylo = v; if (v > yhi) yhi = v; } }); });
    if (!isFinite(ylo)) { ylo = 0; yhi = 1; }
    if (o.includeZero) { ylo = Math.min(ylo, 0); yhi = Math.max(yhi, 0); }
    var log = !!o.log && ylo > 0, f = log ? Math.log10 : function (v) { return v; };
    var a = f(ylo), b = f(yhi), pad = (b - a || 1) * 0.06; a -= pad; b += pad;
    var sx = function (t) { return M.l + (t - x0) / (x1 - x0 || 1) * iw; };
    var sy = function (v) { return M.t + ih - (f(v) - a) / (b - a) * ih; };

    el.innerHTML = "";
    var svg = mk("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, "aria-hidden": "true", focusable: "false" }, el);
    // bands
    (o.bands || []).forEach(function (bd) {
      var bx0 = Math.max(ms(bd.x0), x0), bx1 = Math.min(ms(bd.x1), x1);
      if (bx1 <= bx0) return;
      var r = mk("rect", { x: sx(bx0), y: M.t, width: sx(bx1) - sx(bx0), height: ih, "class": "k-band" + (bd.accent ? " k-band--accent" : ""),
        style: "fill:" + bd.fill }, svg);
      if (bd.accent && animate) r.classList.add("is-draw");
      var lbl = narrow && bd.short ? bd.short : bd.label, endA = bd.anchor === "end";
      if (lbl && (endA || sx(bx1) - sx(bx0) > lbl.length * 6.6 + 12)) {
        var tx = mk("text", { x: endA ? sx(bx1) : sx(bx0) + 4, y: M.t - 8, "text-anchor": endA ? "end" : null,
          "class": "k-band-lbl" + (bd.accent ? " k-band-lbl--accent" : "") }, svg);
        tx.textContent = lbl;
      }
    });
    // axes
    var ax = mk("g", { "class": "k-ax" }, svg);
    var yt = log ? niceLog(Math.pow(10, a), Math.pow(10, b)) : niceLinear(a, b, narrow ? 4 : 5);
    yt.forEach(function (v) {
      var y = sy(v); if (y < M.t - 1 || y > M.t + ih + 1) return;
      mk("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y }, ax);
      var t = mk("text", { x: M.l - 8, y: y + 4, "text-anchor": "end" }, ax);
      t.textContent = (o.yTick || o.yFormat || String)(v);
    });
    mk("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, "class": "k-ax__base" }, ax);
    timeTicks(x0, x1, iw).forEach(function (tk) {
      var x = sx(tk.t); if (x < M.l + 8 || x > M.l + iw - 8) return;
      var t = mk("text", { x: x, y: M.t + ih + 18, "text-anchor": "middle" }, ax);
      t.textContent = tk.s;
    });
    // lines
    var ends = [];
    series.forEach(function (s, si) {
      var d = "", pen = false, xs = X[si], gap = (o.gapDays || 10) * DAY;
      for (var i = 0; i < xs.length; i++) {
        var v = s.y[i];
        if (v == null || !isFinite(v) || (log && v <= 0)) { pen = false; continue; }
        if (i && xs[i] - xs[i - 1] > gap) pen = false;
        d += (pen ? "L" : "M") + sx(xs[i]).toFixed(1) + " " + sy(v).toFixed(1);
        pen = true;
      }
      var p = mk("path", { d: d, "class": s.cls || "k-line", pathLength: animate && !s.cls ? 1 : null }, svg);
      if (animate && !s.cls) p.classList.add("is-draw");
      if (s.endLabel) {
        var li = s.y.length - 1; while (li > 0 && (s.y[li] == null || !isFinite(s.y[li]))) li--;
        ends.push({ y: sy(s.y[li]), x: sx(xs[li]), s: s.endLabel === true ? s.name : s.endLabel });
      }
    });
    ends.sort(function (p, q) { return p.y - q.y; });
    for (var e = 1; e < ends.length; e++) if (ends[e].y - ends[e - 1].y < 13) ends[e].y = ends[e - 1].y + 13;
    ends.forEach(function (en) { var t = mk("text", { x: en.x + 6, y: en.y + 4, "class": "k-end-lbl" }, svg); t.textContent = en.s; });

    // crosshair + tooltip + keyboard
    var base = X[0] || [], cross = mk("g", { style: "display:none" }, svg);
    var cl = mk("line", { y1: M.t, y2: M.t + ih, "class": "k-cross" }, cross);
    var dots = series.map(function () { return mk("circle", { r: 4, "class": "k-dot" }, cross); });
    var tip = document.createElement("div"); tip.className = "k-tip"; el.appendChild(tip);
    var live = o.live || null, liveT = 0, cur = -1;
    function nearest(t) { var lo = 0, hi = base.length - 1; while (hi - lo > 1) { var m = (lo + hi) >> 1; base[m] < t ? lo = m : hi = m; } return Math.abs(base[lo] - t) <= Math.abs(base[hi] - t) ? lo : hi; }
    function show(i) {
      if (i < 0 || i >= base.length) return;
      cur = i;
      var x = sx(base[i]);
      cross.style.display = ""; cl.setAttribute("x1", x); cl.setAttribute("x2", x);
      series.forEach(function (s, si) {
        var j = si === 0 ? i : X[si].indexOf(base[i]), v = j >= 0 ? s.y[j] : null;
        if (v == null || !isFinite(v)) { dots[si].style.display = "none"; return; }
        dots[si].style.display = ""; dots[si].setAttribute("cx", x); dots[si].setAttribute("cy", sy(v));
      });
      var txt = o.tip ? o.tip(i, iso(base[i])) : iso(base[i]) + "  " + (o.yFormat || String)(series[0].y[i]);
      tip.textContent = txt; tip.classList.add("is-on");
      var tw = tip.offsetWidth, left = x + 10 + tw > W ? x - 10 - tw : x + 10;
      tip.style.transform = "translate(" + Math.max(0, left) + "px," + (M.t + 4) + "px)";
      if (live) { clearTimeout(liveT); liveT = setTimeout(function () { live.textContent = txt; }, 220); }
    }
    function hide() { cross.style.display = "none"; tip.classList.remove("is-on"); }
    svg.addEventListener("pointermove", function (ev) {
      var r = svg.getBoundingClientRect(), t = x0 + (ev.clientX - r.left - M.l) / iw * (x1 - x0);
      if (t < base[0] - 30 * DAY || t > base[base.length - 1] + 30 * DAY) { hide(); return; }
      show(nearest(t));
    });
    svg.addEventListener("pointerleave", hide);
    if (!el.__kitKeys) {
      el.__kitKeys = true;
      el.tabIndex = 0;
      el.addEventListener("keydown", function (ev) {
        var c = el.__kitCur; if (!c) return;
        var n = c.len(), i = c.get(), big = Math.max(1, Math.round(n / 40));
        var k = ev.key, to = k === "ArrowRight" ? i + (ev.shiftKey ? big : 1) : k === "ArrowLeft" ? i - (ev.shiftKey ? big : 1)
          : k === "PageUp" ? i + big : k === "PageDown" ? i - big : k === "Home" ? 0 : k === "End" ? n - 1 : null;
        if (to === null) { if (k === "Escape") c.hide(); return; }
        ev.preventDefault(); c.show(Math.max(0, Math.min(n - 1, i < 0 ? n - 1 : to)));
      });
      el.addEventListener("blur", function () { if (el.__kitCur) el.__kitCur.hide(); });
    }
    el.__kitCur = { len: function () { return base.length; }, get: function () { return cur; }, show: show, hide: hide };
  }

  window.Kit = { theme: effectiveTheme, reveal: reveal, countUp: countUp, lineChart: lineChart, reducedMotion: reduced };
  var boot = function () { initTheme(); reveal(); };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", boot) : boot();
})();
