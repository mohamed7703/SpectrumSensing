# Research validation progress presentation

Eight slides, dated 6 November 2026. Open `output/Research_Validation_Progress.html` in a modern browser. The HTML contains its complete React and Framer Motion bundle, styles and evidence data, so it runs without an internet connection or local server.

Controls: Left/Right or Page Up/Page Down to navigate, 1–8 to jump, Home/End for first/last, F for fullscreen, N for presenter notes. Touch users can swipe. The deck respects reduced-motion preferences and reflows for narrow screens. Browser printing shows all eight slides.

Sources: the [recovery report](../../recovery/RECOVERY_PROGRESS.md) and [JSON evidence records](../../../results/recovery). `src/evidence.json` captures the original record values, provenance paths and SHA-256 identities. Its absolute paths identify the machine where the measurements ran; they are not required to open the presentation. Speaker notes retain the scope and evidence references. Software success, synthetic performance and unverified research claims remain distinct.

The presentation uses React 18.3.1 and Framer Motion 11.18.2. SVG block diagrams and the evidence chart remain editable in `src/main.jsx`. The final HTML is self-contained. The original research source, models, reports and results are unchanged by presentation creation.

To rebuild with Node.js 18 or later:

```sh
pnpm install --frozen-lockfile
pnpm build
```

`pnpm-lock.yaml` records the exact dependency graph. The workspace explicitly permits esbuild's installation script. Dependency copyright and license notices are in `licenses/`.

GitHub displays the HTML as source. Download `output/Research_Validation_Progress.html` and open it in a browser to play the presentation. This commit does not enable public website hosting.
