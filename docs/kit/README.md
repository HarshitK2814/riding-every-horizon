# Research site kit

Shared look for the study sites. Static files, no build, no dependencies.

- `kit.css`: tokens and components
- `kit.js`: theme toggle, reveal on scroll, count-up numbers, SVG line chart (`window.Kit`)

## The one rule

Ink on paper, plus **one accent colour used as a highlighter fill**. In this site the accent (yellow) marks
the things that are still open: the forward test, the pending hypothesis, the countdown. Links and focus
rings use it as an underline or halo. Never use the accent as text colour on the page ground.

## Tokens

| Token | Use |
|---|---|
| `--k-bg`, `--k-surface`, `--k-surface-2` | page, panels, pressed or selected |
| `--k-ink`, `--k-ink-2`, `--k-muted` | headings and chart lines, body, secondary text (muted is at least 7:1 on bg and surface) |
| `--k-line`, `--k-line-2` | borders, chart grid |
| `--k-accent`, `--k-accent-ink`, `--k-accent-soft` | highlighter fill, text on it, faint tint |
| `--k-band-1..3`, `--k-band-accent-op` | chart bands from least to most "seen"; opacity of the accent band |
| `--k-font`, `--k-code` | Archivo (variable width axis), system monospace for file names only |
| `--k-s1..9`, `--k-r1..3` | spacing (4 to 112 px), radius (3, 6, 10) |
| `--k-ease`, `--k-fast`, `--k-mid`, `--k-slow` | strong ease-out; 150, 200, 300 ms |

Dark values live in two blocks (`prefers-color-scheme` and `[data-theme="dark"]`). Change both together.

Type voice: headings use the condensed width (`wdth` 78 to 82), numbers use the expanded width
(`.k-figure`, `wdth` 125, tabular). That contrast is the typographic identity, so keep it.

## Components

| Class | What it is |
|---|---|
| `.k-header`, `.k-brand`, `.k-nav`, `.k-theme` | sticky header; nav links with `.k-nav__opt` hide under 760 px |
| `.k-section`, `.k-section__head`, `.k-h2`, `.k-lede` | section with heading left, summary right |
| `.k-display` | hero headline |
| `.k-btn`, `.k-btn--ghost` | primary and secondary button |
| `.k-pill`, `--yes`, `--wait` | status word: not supported (outline), supported (solid), pending (accent) |
| `.k-tally`, `.k-tally__cell.is-yes/.is-wait` | one cell per pre-registered test |
| `.k-verdicts`, `.k-verdict`, `.k-hyp*` | one card per round, one row per hypothesis |
| `.k-stats`, `.k-stat`, `.k-stat--lead`, `.k-figure` | stat tiles; set `style="--cols:N"` |
| `.k-chart`, `.k-chart__plot`, `.k-chart__cap`, `.k-chart__key` | chart frame; set the plot height with `--h` so nothing shifts on load |
| `.k-field`, `.k-label`, `.k-select`, `.k-chip`, `.k-seg`, `.k-range`, `.k-hint` | lab controls |
| `.k-status`, `.k-steps` | loading line with stages; add `.is-busy`, `.is-ok` or `.is-err` |
| `.k-table-wrap`, `.k-table` | numeric table (scrolls inside its frame on phones) |
| `.k-notes`, `.k-note` | method notes as `<details>`: summary line visible, detail and source links inside |
| `.k-meter` | progress bar (`<i style="--v: 40%">`) |
| `.k-mark` | inline highlighter on a word |
| `.k-footer*` | disclaimer, data vintage, repo links |

## kit.js

```js
Kit.lineChart(el, {
  series: [{ name, x: ["2020-01-02", ...], y: [...], cls, endLabel: true }],  // cls: "k-line k-line--2" etc.
  log: true, xMin, xMax, includeZero, gapDays: 10, animate: true,
  bands: [{ x0, x1, fill: "var(--k-band-1)", label, short, accent: true, anchor: "end" }],
  yTick: (v) => v.toFixed(1), tip: (i, isoDate) => "text", live: srOnlyElement,
});
Kit.countUp(el, 0.36, { format: (v) => v.toFixed(2), label: "Net Sharpe" });
Kit.reveal();            // runs on load for every [data-reveal]; fires "kit:reveal" on the element
```

Charts are keyboard readable (Tab to the plot, arrow keys, Home and End) and announce through `live`.
Colours come from tokens, so a theme switch needs no redraw. Put `data-theme-toggle` on the toggle
button, and put this in `<head>` before the CSS so the saved theme applies before first paint:

```html
<script>document.documentElement.classList.add("k-js");try{var t=localStorage.getItem("kit-theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}</script>
```

## Starting a sibling site

1. Copy `kit/` into the new `docs/`. Link `kit/kit.css`, then the page's own `style.css`, then `kit/kit.js` (defer).
2. Pick one accent and override only the accent tokens in the page's `style.css`, in both themes:

   ```css
   :root { --k-accent: #9fe3ff; --k-accent-soft: rgba(159, 227, 255, .5); }
   ```

   Rule: black text on the accent must reach 4.5:1, so pick a light, bright colour (relative luminance above
   about 0.18). Suggested starting points, checked against black text:
   - **Options-collar study**: sky `#9fe3ff` (14.9:1). Use it for whatever that study has not seen yet.
   - **10-K text-signal study**: coral `#ffb199` (12.0:1). Same rule.

   No violet, and not the yellow of this site, so the three sites read as one family with separate
   identities.
3. Keep the rest: same type, greys, radius, motion. Write page layout in `style.css` only.
4. Keep the honesty conventions: every verdict shows its status word and numbers, every number links to the
   file that made it, in-sample work says "exploration, not evidence", and only the forward period is called
   out-of-sample. Footer: "Research artifact, not investment advice. Simulated, net of stated costs."
5. Check at 375 px and 1440 px, in both themes, with reduced motion on.
