# HACRL project page

Static project website for **Heterogeneous Agent Collaborative Reinforcement Learning**, accepted at NeurIPS 2026.

The page includes the current author list, practical motivation, main results, three-agent collaboration, distillation and train-then-distill comparisons, weak-to-strong rollout analysis, and five-seed results. Expandable tables provide the full main and three-agent results; one table reports the final five-seed results.

- `index.html`: project page and manuscript results.
- `static/pdf/HACRL.pdf`: current camera-ready manuscript.
- `static/css/index.css` and `static/js/index.js`: page styles and interactions.

No build step is required. To preview locally from this directory:

```sh
python -m http.server 8000
```

Then open `http://localhost:8000/`. Fonts, icons, and math rendering use the existing external CDN dependencies.
