/* Runs trend_lite.py in Pyodide off the main thread. Messages in: {type:"run", id, params}.
   Messages out: {type:"stage", stage, detail} while loading, {type:"ready"}, {type:"result", id, ms, out},
   {type:"error", id, message}. */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null;      // runPython is synchronous; the page sends one request at a time and keeps only the latest

const post = (m) => self.postMessage(m);

async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (HTTP ${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}

async function boot() {
  post({ type: "stage", stage: "runtime", detail: "Python runtime (Pyodide 0.26.4)" });
  py = await loadPyodide({ indexURL: PYODIDE });
  post({ type: "stage", stage: "packages", detail: "numpy, pandas" });
  await py.loadPackage(["numpy", "pandas"]);
  post({ type: "stage", stage: "data", detail: "39 instruments, 1999-2026" });
  const text = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url} (HTTP ${r.status})`);
    return r.text();
  };
  const [src, ret, inst, sig, rf] = await Promise.all([
    text("trend_lite.py"),
    bytes("data/returns.csv.gz"),
    text("data/instruments.json"),
    bytes("data/tsfm_signal.csv.gz"),
    bytes("data/tbill.csv.gz"),
  ]);
  py.FS.writeFile("/home/pyodide/trend_lite.py", src);
  py.FS.writeFile("/tmp/returns.csv.gz", ret);
  py.FS.writeFile("/tmp/tsfm_signal.csv.gz", sig);
  py.FS.writeFile("/tmp/tbill.csv.gz", rf);
  py.globals.set("INST_JSON", inst);
  py.runPython(`
import trend_lite as tl
INP = tl.load_inputs(open("/tmp/returns.csv.gz", "rb").read(), INST_JSON, open("/tmp/tsfm_signal.csv.gz", "rb").read(), open("/tmp/tbill.csv.gz", "rb").read())
CACHE = {}
`);
  post({ type: "ready" });
}

function runOne(req) {
  const t0 = performance.now();
  try {
    py.globals.set("PARAMS", JSON.stringify(req.params));
    const out = JSON.parse(py.runPython("tl.run_json(INP, PARAMS, CACHE)"));
    post({ type: "result", id: req.id, ms: performance.now() - t0, out });
  } catch (e) {
    post({ type: "error", id: req.id, message: String(e && e.message ? e.message : e) });
  }
}

const ready = boot().catch((e) => post({ type: "error", id: null, message: "load failed: " + e }));

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.type !== "run") return;
  await ready;
  if (!py) return;
  runOne(m);
};
