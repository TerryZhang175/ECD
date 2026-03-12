---
name: ecd-ui-debug
description: Debug and modify the ECD UI served from /ui, including ui/index.html, ui/app.js, ui/styles.css, and FastAPI static serving. Use when changing frontend behavior, investigating UI/backend mismatches, or when a fix seems correct in code but not visible in the browser. Always check frontend cache-busting and confirm which JS/CSS URLs the browser actually loaded.
---

# ECD UI Debug

Use this skill for UI work in this repository.

## Scope

- `ui/index.html`
- `ui/app.js`
- `ui/styles.css`
- `ecd_api.py` static serving under `/ui`

## Required workflow

1. Confirm whether the issue is in code, cached assets, localStorage state, or an old backend process.
2. If JS or CSS behavior changes, inspect `ui/index.html` and check the asset query strings for `app.js` and `styles.css`.
3. When the browser may be using stale assets, bump the cache-busting version in `ui/index.html`.
4. Verify the browser actually loaded the new URLs, preferably with Playwright request logging or equivalent network inspection.
5. If the UI still looks unchanged, verify the visible result in the browser instead of trusting DOM order alone. Check actual element positions, rendered text, and active classes.
6. If behavior depends on backend data, confirm the active local server is the current code version and not an older process still bound to the same port.

## Checks to run

- Confirm the loaded script and stylesheet URLs include the expected version suffix.
- Confirm the page is being served from `http://127.0.0.1:8001/ui/` unless the user specifies another origin.
- Check whether `localStorage` is affecting panel order, view mode, or similar UI state.
- When a button "does nothing", verify both:
  - the click handler is bound
  - the visible screen position or content actually changes after the click

## Final response expectations

- State whether the problem was code, cache, stale backend, or persisted browser state.
- If you changed cache-busting, mention the new asset version explicitly.
