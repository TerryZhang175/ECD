---
name: ecd-ui-debug
description: Debug and modify the ECD UI served from /ui, including ui/index.html, ui/app.js, ui/styles.css, and FastAPI static serving. Use when changing frontend behavior, investigating UI/backend mismatches, or when a fix seems correct in code but not visible in the browser. Always check frontend cache-busting and confirm which JS/CSS URLs the browser actually loaded. MANDATORY: Every UI modification must be verified with browser screenshot proof before completion.
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

## Post-modification browser verification (MANDATORY)

**EVERY UI MODIFICATION MUST BE VERIFIED WITH A SCREENSHOT.**

After completing ANY UI or backend modification, ALWAYS run browser verification using Playwright and provide a screenshot as proof.

### Step 1: Restart API server (if backend changed)

```bash
# Kill existing server
lsof -i :8001 -t | xargs kill -9 2>/dev/null
sleep 1

# Start server in background
python3 ecd_api.py &
sleep 3
```

### Step 2: Run Playwright verification

Create and run a Playwright script:

```javascript
const { chromium } = require('/Users/terry/Codehub/ECD/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1600, height: 1000 });

  await page.goto('http://127.0.0.1:8001/ui/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Set parameters for your test
  await page.fill('#filePath', '/Users/terry/Codehub/ECD/sample/WT/ZD_21to22Feb_2n5_WTECD/RE 12.txt');
  await page.fill('#peptide', 'KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY');
  await page.fill('#scanSelect', '1');
  await page.selectOption('#modeSelect', 'diagnose'); // or fragments, precursor, etc.

  // Run analysis
  await page.click('#runButton');
  await page.waitForTimeout(15000); // Wait for results

  // Take screenshot
  await page.screenshot({ path: '/tmp/ecd_ui_verification.png', fullPage: false });
  console.log('Screenshot saved to: /tmp/ecd_ui_verification.png');

  await browser.close();
})();
```

### Step 3: Display screenshot

Use the Read tool to display the screenshot:

```bash
# After the Playwright script completes
Read /tmp/ecd_ui_verification.png
```

This will upload and display the screenshot in the conversation.

### Step 4: Report verification results

Always include in your final response:
- ✅/❌ API response contains expected data
- ✅/❌ UI displays expected changes
- **Screenshot with visible proof of the change**

## Sample data files

- Spectrum data: `/Users/terry/Codehub/ECD/sample/WT/ZD_21to22Feb_2n5_WTECD/RE 12.txt`
- Alternative: `/Users/terry/Codehub/ECD/sample/Q10R/Centroid (lock mass)_副本/ECDRE45.txt`
- Q10R2: `/Users/terry/Codehub/ECD/sample/Q10R2/ECDRE0.txt`

## Checks to run

- Confirm the loaded script and stylesheet URLs include the expected version suffix.
- Confirm the page is being served from `http://127.0.0.1:8001/ui/` unless the user specifies another origin.
- Check whether `localStorage` is affecting panel order, view mode, or similar UI state.
- When a button "does nothing", verify both:
  - the click handler is bound
  - the visible screen position or content actually changes after the click

## Final response expectations

- State whether the problem was code, cache, stale backend, or persisted browser state.
- If you changed cache-busting, mention the new asset version explicitly (e.g., `v=20260402-2`).
- **Include a screenshot showing the UI change.**
